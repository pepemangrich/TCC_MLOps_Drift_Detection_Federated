# pytorchexample: A Flower / PyTorch app.

import os
import json
import csv
import torch
from typing import List, Tuple, Optional, Dict, Set

from flwr.common import (
    Context,
    Metrics,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    EvaluateRes,
    FitRes,
    EvaluateIns,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from pytorchexample.task import Net, get_weights

import numpy as np


# -------------------------- Utils --------------------------

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": (sum(accuracies) / sum(examples)) if sum(examples) > 0 else 0.0}

def save_client_weights(parameters: Parameters, cid: str, round_number: int):
    weights = parameters_to_ndarrays(parameters)
    state_dict = {f"param_{i}": torch.tensor(w) for i, w in enumerate(weights)}
    folder = f"saved_weights/round_{round_number}"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"client_{cid}.pt")
    torch.save(state_dict, path)
    print(f"[SERVER] Pesos salvos em: {path}")

def _to_py(obj):
    """Converte numpy/containers para tipos JSON-friendly."""
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# -------------------------- Estratégia --------------------------

ROUND_CSV_HEADER = [
    "round",
    "global_accuracy",
    "num_clients",
    "global_drift_flag",     # 0/1 (Wilbik/global)
    "clients_with_drift",    # JSON list de CIDs (detectores locais). Para Wilbik: "[]"
    "delta_t",               # só Wilbik
    "delta_0",               # só Wilbik
    "band_lo",               # só Wilbik
    "band_hi",               # só Wilbik
    "phase",                 # INIT/MONITOR
    "label",
]

PER_CLIENT_HEADER = ["round", "cid", "accuracy", "num_examples", "drift", "label"]


class DriftAwareStrategy(FedAvg):
    def __init__(self, *, run_config: Dict, env_snapshot: Dict[str, str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.clients_with_drift_last_round: Set[str] = set()
        self._run_config = dict(run_config) if run_config is not None else {}
        self._env_snapshot = dict(env_snapshot or {})

        os.makedirs("logs", exist_ok=True)
        self._csv_path = os.path.join("logs", "round_metrics.csv")
        self._jsonl_path = os.path.join("logs", "round_metrics.jsonl")
        self._per_client_csv = os.path.join("logs", "per_client_metrics.csv")

        # ----------------- Wilbik Federated -----------------
        self._wf_enabled: bool = (os.getenv("DRIFT_METHOD", "") == "wilbik_federated")
        # Fases: init_centers -> init_delta0 -> delta (monitor)
        self._wf_state: Optional[str] = "init_centers" if self._wf_enabled else None

        self._wf_K: int = int(os.getenv("WILBIK_K", "3"))
        self._wf_m: float = float(os.getenv("WILBIK_M", "2.0"))
        self._wf_delta_band: float = float(os.getenv("WILBIK_DELTA", "0.1"))
        self._wf_max_samples: int = int(os.getenv("WILBIK_MAX_SAMPLES", "512"))
        self._wf_init_iters: int = int(os.getenv("WILBIK_INIT_ITERS", "5"))
        self._wf_init_iter: int = 0
        self._wf_eps: float = float(os.getenv("WILBIK_EPS", "1e-3"))
        self._wf_q: float = float(os.getenv("WILBIK_Q", "2.0"))

        self._wf_centers: Optional[np.ndarray] = None  # (K,D)
        self._wf_prev_centers: Optional[np.ndarray] = None
        self._wf_delta0: Optional[float] = None
        self._wf_feature_dim: Optional[int] = None

        self._scenario_label = os.getenv("SCENARIO_LABEL", "")

        # Cabeçalhos
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=ROUND_CSV_HEADER).writeheader()
        if not os.path.exists(self._per_client_csv):
            with open(self._per_client_csv, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=PER_CLIENT_HEADER).writeheader()

    # -------------------------- Avaliação --------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:

        self.clients_with_drift_last_round.clear()

        # ---- agregação padrão (global accuracy) ----
        weighted_acc_sum = 0.0
        weighted_n_sum = 0
        for client_proxy, evaluate_res in results:
            cid = client_proxy.cid
            acc = float(evaluate_res.metrics.get("accuracy", 0.0))
            n = int(evaluate_res.num_examples)
            weighted_acc_sum += acc * n
            weighted_n_sum += n

            drift_detected = bool(evaluate_res.metrics.get("drift", False))
            if drift_detected:
                self.clients_with_drift_last_round.add(cid)

            with open(self._per_client_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=PER_CLIENT_HEADER).writerow(
                    {
                        "round": server_round,
                        "cid": cid,
                        "accuracy": acc,
                        "num_examples": n,
                        "drift": drift_detected,
                        "label": self._scenario_label,
                    }
                )

        global_acc = (weighted_acc_sum / weighted_n_sum) if weighted_n_sum > 0 else 0.0

        # Linha base de log por rodada
        row_global = {
            "round": int(server_round),
            "global_accuracy": float(global_acc),
            "num_clients": int(len(results)),
            "global_drift_flag": 0,                    # será atualizado p/ Wilbik
            "clients_with_drift": json.dumps(sorted(self.clients_with_drift_last_round)),  # locais
            "delta_t": "",
            "delta_0": "",
            "band_lo": "",
            "band_hi": "",
            "phase": "INIT" if self._wf_state in {"init_centers", "init_delta0"} else "MONITOR",
            "label": self._scenario_label,
        }

        # ----------------- Wilbik Federated (estados) -----------------
        wilbik_log = None
        if self._wf_enabled:
            eps = 1e-12

            # Acumuladores
            init_snum = None  # lista np.array (K,D)
            init_sden = None  # lista float (K,)

            B = None          # lista float (K,)   (exato)
            W = None          # lista float (K,)   (exato)
            sse_fallback = None  # lista float (K,) (fallback)
            w_fallback = None    # lista float (K,) (fallback)

            for _, evaluate_res in results:
                m = evaluate_res.metrics

                # --- INIT: S_num/S_den ---
                if self._wf_state == "init_centers":
                    local_snum, local_sden, max_d = {}, {}, -1
                    for k, v in m.items():
                        if isinstance(k, str) and k.startswith("wf_init_snum_"):
                            parts = k.split("_")
                            j, d = int(parts[3]), int(parts[4])
                            local_snum.setdefault(j, {})[d] = float(v)
                            max_d = max(max_d, d)
                        elif isinstance(k, str) and k.startswith("wf_init_sden_"):
                            j = int(k.split("_")[3])
                            local_sden[j] = local_sden.get(j, 0.0) + float(v)

                    if max_d >= 0 and local_sden:
                        D = max_d + 1
                        if init_snum is None:
                            init_snum = [np.zeros(D, dtype=np.float64) for _ in range(self._wf_K)]
                            init_sden = [0.0 for _ in range(self._wf_K)]
                            self._wf_feature_dim = D
                        for j in range(self._wf_K):
                            den = float(local_sden.get(j, 0.0))
                            if den > 0.0:
                                vec = np.zeros(D, dtype=np.float64)
                                for d, val in local_snum.get(j, {}).items():
                                    if d < D:
                                        vec[d] = float(val)
                                init_snum[j] += vec
                                init_sden[j] += den

                # --- DELTA0/DELTA: DB fuzzy exato (B/W) ou fallback SSE/W ---
                if self._wf_state in {"init_delta0", "delta"}:
                    has_B = any(isinstance(k, str) and k.startswith("wf_B_") for k in m.keys())
                    has_W = any(isinstance(k, str) and k.startswith("wf_w_") for k in m.keys())
                    if has_B and has_W:
                        if B is None:
                            B = [0.0 for _ in range(self._wf_K)]
                        if W is None:
                            W = [0.0 for _ in range(self._wf_K)]
                        for k, v in m.items():
                            if isinstance(k, str) and k.startswith("wf_B_"):
                                j = int(k.split("_")[2]); B[j] += float(v)
                            elif isinstance(k, str) and k.startswith("wf_w_"):
                                j = int(k.split("_")[2]); W[j] += float(v)
                    else:
                        if sse_fallback is None:
                            sse_fallback = [0.0 for _ in range(self._wf_K)]
                            w_fallback = [0.0 for _ in range(self._wf_K)]
                        for k, v in m.items():
                            if isinstance(k, str) and k.startswith("wf_sse_"):
                                j = int(k.split("_")[2]); sse_fallback[j] += float(v)
                            elif isinstance(k, str) and k.startswith("wf_w_"):
                                j = int(k.split("_")[2]); w_fallback[j] += float(v)

            # ---- Atualiza estado e calcula deltas/banda ----
            if self._wf_state == "init_centers":
                if init_snum is not None and init_sden is not None:
                    centers_new = []
                    for j in range(self._wf_K):
                        den = init_sden[j] if init_sden[j] > eps else eps
                        centers_new.append((init_snum[j] / den).tolist())
                    centers_new = np.array(centers_new, dtype=np.float64)

                    # shift p/ critério de parada
                    shift = None
                    if self._wf_centers is not None:
                        shift = float(np.max(np.linalg.norm(centers_new - self._wf_centers, axis=1)))
                    self._wf_prev_centers = self._wf_centers
                    self._wf_centers = centers_new
                    self._wf_init_iter += 1

                    if (shift is not None and shift < self._wf_eps) or (self._wf_init_iter >= self._wf_init_iters):
                        self._wf_state = "init_delta0"   # próxima rodada calcula Δ0

                    wilbik_log = {
                        "stage": "init_centers",
                        "init_iter": self._wf_init_iter,
                        "init_iters": self._wf_init_iters,
                        "K": self._wf_K,
                        "m": self._wf_m,
                        "D": self._wf_feature_dim,
                        "centers_ready": True,
                        "center_shift": shift,
                        "delta0": None,
                        "delta_t": None,
                        "band": None,
                        "drift_global": False,
                    }

            elif self._wf_state in {"init_delta0", "delta"} and self._wf_centers is not None:
                C = self._wf_centers
                eps = 1e-12
                # Preferir DB exato se B/W chegaram; senão, fallback SSE/W
                if (B is not None) and (W is not None) and (np.sum(W) > eps):
                    B_arr = np.array(B, dtype=np.float64)
                    W_arr = np.array(W, dtype=np.float64)
                    S = B_arr / (W_arr + eps)                                # S_k^{(f)} = B_k / W_k
                    M = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2) + eps
                    R = (S[:, None] + S[None, :]) / M
                    np.fill_diagonal(R, -np.inf)
                    delta_val = float(np.max(R, axis=1).mean())
                    exact_db = True
                else:
                    scatters = np.array([(sse_fallback[j] / (w_fallback[j] + eps)) for j in range(self._wf_K)], dtype=np.float64)
                    M = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2) + eps
                    R = (scatters[:, None] + scatters[None, :]) / M
                    np.fill_diagonal(R, -np.inf)
                    delta_val = float(np.max(R, axis=1).mean())
                    exact_db = False

                if self._wf_state == "init_delta0":
                    self._wf_delta0 = delta_val
                    self._wf_state = "delta"
                    low = (1.0 - self._wf_delta_band) * self._wf_delta0
                    high = (1.0 + self._wf_delta_band) * self._wf_delta0
                    row_global.update({
                        "delta_t": "",
                        "delta_0": float(self._wf_delta0),
                        "band_lo": float(low),
                        "band_hi": float(high),
                        "phase": "INIT",
                    })
                    wilbik_log = {
                        "stage": "init_delta0",
                        "K": self._wf_K, "m": self._wf_m, "D": self._wf_feature_dim,
                        "delta0": self._wf_delta0, "delta_t": None,
                        "band": [low, high], "drift_global": False,
                        "exact_db": bool(exact_db),
                    }
                else:
                    # state == "delta" (monitor)
                    if self._wf_delta0 is None:
                        self._wf_delta0 = delta_val
                    low = (1.0 - self._wf_delta_band) * self._wf_delta0
                    high = (1.0 + self._wf_delta_band) * self._wf_delta0
                    drift_global = int((delta_val < low) or (delta_val > high))

                    # Atualiza campos Wilbik no CSV (sem listar clientes!)
                    row_global.update({
                        "global_drift_flag": drift_global,
                        "clients_with_drift": json.dumps([]),  # Wilbik: sempre lista vazia
                        "delta_t": float(delta_val),
                        "delta_0": float(self._wf_delta0),
                        "band_lo": float(low),
                        "band_hi": float(high),
                        "phase": "MONITOR",
                    })

                    wilbik_log = {
                        "stage": "delta",
                        "K": self._wf_K, "m": self._wf_m, "D": self._wf_feature_dim,
                        "delta0": self._wf_delta0, "delta_t": delta_val,
                        "band": [low, high], "drift_global": bool(drift_global),
                        "exact_db": bool(exact_db),
                    }

        # --- Persistência ---
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=ROUND_CSV_HEADER).writerow(row_global)

        json_obj = {"round": server_round, "label": self._scenario_label, "global": row_global}
        if self._wf_enabled and wilbik_log is not None:
            json_obj["wilbik"] = wilbik_log

        json_obj = _to_py(json_obj)
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(json_obj) + "\n")

        print(
            f"[SERVER] Round {server_round}: GlobalAcc={global_acc:.4f} | "
            f"global_drift_flag={row_global['global_drift_flag']} | "
            f"clients_with_drift={row_global['clients_with_drift']}"
        )
        return super().aggregate_evaluate(server_round, results, failures)

    # -------------------------- Fit / Pesos --------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        # Salva pesos apenas dos clientes com drift local na última rodada de avaliação
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if cid in self.clients_with_drift_last_round:
                save_client_weights(fit_res.parameters, cid, server_round)
        return super().aggregate_fit(server_round, results, failures)

    # -------------------------- Envio de config p/ clientes --------------------------

    def configure_evaluate(self, server_round, parameters, client_manager):
        instructions = super().configure_evaluate(server_round, parameters, client_manager)
        if not self._wf_enabled:
            return instructions

        # Guarda-corpo: se não há centros e o estado não é INIT, retorne para INIT
        if self._wf_state != "init_centers" and (self._wf_centers is None):
            print("[SERVER][Wilbik] Centros ausentes fora da INIT — retornando a 'init_centers'.")
            self._wf_state = "init_centers"
            self._wf_init_iter = 0

        new_instructions = []
        for client, eval_ins in instructions:
            cfg = dict(eval_ins.config) if isinstance(eval_ins.config, dict) else {}
            cfg["wilbik_k"] = str(self._wf_K)
            cfg["wilbik_m"] = str(self._wf_m)
            cfg["wilbik_q"] = str(self._wf_q)
            cfg["wilbik_max_samples"] = str(self._wf_max_samples)
            cfg["wilbik_init_iters"] = str(self._wf_init_iters)
            cfg["server_round"] = str(server_round)

            if self._wf_state == "init_centers":
                cfg["wilbik_stage"] = "init"
                if self._wf_centers is not None:
                    cfg["wilbik_centers"] = json.dumps(self._wf_centers.tolist())
            else:
                cfg["wilbik_stage"] = "delta"
                if self._wf_centers is not None:
                    cfg["wilbik_centers"] = json.dumps(self._wf_centers.tolist())

            new_eval_ins = EvaluateIns(parameters=eval_ins.parameters, config=cfg)
            new_instructions.append((client, new_eval_ins))

        return new_instructions


# -------------------------- ServerApp --------------------------

def server_fn(context: Context):
    env_keys = [
        "DRIFT_METHOD", "DRIFT_WINDOW", "DRIFT_THRESHOLD", "DRIFT_DEBUG", "DRIFT_KSWIN_STAT",
        "NON_IID_ALPHA", "SMOKE_DRIFT", "SCENARIO_LABEL",
        "PARTITION_ID", "NUM_PARTITIONS",
        "WILBIK_K", "WILBIK_M", "WILBIK_DELTA", "WILBIK_MAX_SAMPLES", "WILBIK_INIT_ITERS",
        "WILBIK_EPS", "WILBIK_Q",
    ]
    env_snapshot = {k: os.getenv(k) for k in env_keys if os.getenv(k) is not None}

    num_rounds = context.run_config["num-server-rounds"]
    initial_params = ndarrays_to_parameters(get_weights(Net()))

    strategy = DriftAwareStrategy(
        run_config=dict(context.run_config),
        env_snapshot=env_snapshot,
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_params,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)