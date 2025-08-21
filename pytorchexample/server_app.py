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


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def save_client_weights(parameters: Parameters, cid: str, round_number: int):
    weights = parameters_to_ndarrays(parameters)
    state_dict = {f"param_{i}": torch.tensor(w) for i, w in enumerate(weights)}

    folder = f"saved_weights/round_{round_number}"
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"client_{cid}.pt")

    torch.save(state_dict, path)
    print(f"[SERVER] Pesos salvos em: {path}")


class DriftAwareStrategy(FedAvg):
    def __init__(self, *, run_config: Dict, env_snapshot: Dict[str, str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.clients_with_drift_last_round: Set[str] = set()

        # Guarda config/snapshot para log
        self._run_config = dict(run_config) if run_config is not None else {}
        self._env_snapshot = dict(env_snapshot or {})

        # Logs estruturados
        os.makedirs("logs", exist_ok=True)
        self._csv_path = os.path.join("logs", "round_metrics.csv")
        self._jsonl_path = os.path.join("logs", "round_metrics.jsonl")
        self._per_client_csv = os.path.join("logs", "per_client_metrics.csv")

        # ----------------- Wilbik Federated (NOVO) -----------------
        self._wf_enabled: bool = (os.getenv("DRIFT_METHOD", "") == "wilbik_federated")
        self._wf_state: Optional[str] = "init" if self._wf_enabled else None  # "init" -> agrega M0; depois "delta"
        self._wf_K: int = int(os.getenv("WILBIK_K", "3"))
        self._wf_m: float = float(os.getenv("WILBIK_M", "2.0"))
        self._wf_delta_band: float = float(os.getenv("WILBIK_DELTA", "0.1"))      # δ da banda ±δ
        self._wf_max_samples: int = int(os.getenv("WILBIK_MAX_SAMPLES", "512"))
        self._wf_centers: Optional[np.ndarray] = None  # (K, D) após init
        self._wf_delta0: Optional[float] = None        # baseline DB
        self._wf_feature_dim: Optional[int] = None

        # Label do cenário (preenchido pelo run_experiments.sh)
        self._scenario_label = os.getenv("SCENARIO_LABEL", "")

        # Cabeçalhos dos CSVs (se ainda não existirem)
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.DictWriter(
                    f,
                    fieldnames=["round", "global_accuracy", "num_clients", "clients_with_drift", "label"],
                ).writeheader()

        if not os.path.exists(self._per_client_csv):
            with open(self._per_client_csv, "w", newline="") as f:
                csv.DictWriter(
                    f,
                    fieldnames=["round", "cid", "accuracy", "num_examples", "drift", "label"],
                ).writeheader()

    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:

        self.clients_with_drift_last_round.clear()

        # Coleta/Agrega global + acumula per-client
        weighted_acc_sum = 0.0
        weighted_n_sum = 0
        per_client_rows: List[Dict[str, object]] = []

        for client_proxy, evaluate_res in results:
            cid = client_proxy.cid
            acc = float(evaluate_res.metrics.get("accuracy", 0.0))
            n = int(evaluate_res.num_examples)
            drift_detected = bool(evaluate_res.metrics.get("drift", False))

            if drift_detected:
                self.clients_with_drift_last_round.add(cid)

            weighted_acc_sum += acc * n
            weighted_n_sum += n

            per_client_rows.append({
                "round": server_round,
                "cid": cid,
                "accuracy": acc,
                "num_examples": n,
                "drift": drift_detected,
                "label": self._scenario_label,
            })

        global_acc = (weighted_acc_sum / weighted_n_sum) if weighted_n_sum > 0 else 0.0
        clients_sorted = sorted(self.clients_with_drift_last_round)
        row_global = {
            "round": server_round,
            "global_accuracy": global_acc,
            "num_clients": len(results),
            "clients_with_drift": ";".join(clients_sorted) if clients_sorted else "",
            "label": self._scenario_label,
        }

        # ----------------- Wilbik Federated (NOVO) -----------------
        wilbik_log = None
        if self._wf_enabled:
            init_snum = None  # lista de np.array (K,D)
            init_sden = None  # lista float (K,)
            delta_sse = None  # lista float (K,)
            delta_w   = None  # lista float (K,)

            for _, evaluate_res in results:
                m = evaluate_res.metrics
                if self._wf_state == "init" or self._wf_centers is None:
                    local_snum, local_sden, max_d = {}, {}, -1
                    for k, v in m.items():
                        if isinstance(k, str) and k.startswith("wf_init_snum_"):
                            _, _, _, j, d = k.split("_")
                            j, d = int(j), int(d)
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
                else:
                    if delta_sse is None:
                        delta_sse = [0.0 for _ in range(self._wf_K)]
                        delta_w   = [0.0 for _ in range(self._wf_K)]
                    for k, v in m.items():
                        if isinstance(k, str) and k.startswith("wf_sse_"):
                            j = int(k.split("_")[2])
                            delta_sse[j] += float(v)
                        elif isinstance(k, str) and k.startswith("wf_w_"):
                            j = int(k.split("_")[2])
                            delta_w[j] += float(v)

            if self._wf_state == "init" or self._wf_centers is None:
                if init_snum is not None and init_sden is not None:
                    centers = []
                    for j in range(self._wf_K):
                        den = init_sden[j] if init_sden[j] > 1e-12 else 1e-12
                        centers.append((init_snum[j] / den).tolist())
                    self._wf_centers = np.array(centers, dtype=np.float64)
                    self._wf_state = "delta"
                    wilbik_log = {
                        "stage": "init",
                        "K": self._wf_K,
                        "m": self._wf_m,
                        "D": self._wf_feature_dim,
                        "centers_ready": True,
                        "delta0": None,
                        "delta_t": None,
                        "band": None,
                        "drift_global": False,
                    }
            else:
                if delta_sse is not None and delta_w is not None:
                    scatters = np.array(
                        [(delta_sse[j] / (delta_w[j] + 1e-12)) for j in range(self._wf_K)],
                        dtype=np.float64
                    )
                    C = self._wf_centers
                    dist = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2) + 1e-12
                    R = (scatters[:, None] + scatters[None, :]) / dist
                    np.fill_diagonal(R, -np.inf)
                    delta_t = float(np.max(R, axis=1).mean())

                    if self._wf_delta0 is None:
                        self._wf_delta0 = delta_t
                    low = (1.0 - self._wf_delta_band) * self._wf_delta0
                    high = (1.0 + self._wf_delta_band) * self._wf_delta0
                    drift_global = (delta_t < low) or (delta_t > high)
                    if drift_global:
                        row_global["clients_with_drift"] = "GLOBAL"

                    wilbik_log = {
                        "stage": "delta",
                        "K": self._wf_K,
                        "m": self._wf_m,
                        "D": self._wf_feature_dim,
                        "delta0": self._wf_delta0,
                        "delta_t": delta_t,
                        "band": [low, high],
                        "drift_global": drift_global,
                    }

        # --- Persistência (global) ---
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(
                f,
                fieldnames=["round", "global_accuracy", "num_clients", "clients_with_drift", "label"],
            ).writerow(row_global)

        # --- Persistência (per-client) ---
        if per_client_rows:
            with open(self._per_client_csv, "a", newline="") as f:
                csv.DictWriter(
                    f,
                    fieldnames=["round", "cid", "accuracy", "num_examples", "drift", "label"],
                ).writerows(per_client_rows)

        # --- JSONL (global + per_client + wilbik, com label) ---
        json_obj = {"round": server_round, "label": self._scenario_label, "global": row_global}
        if per_client_rows:
            json_obj["per_client"] = per_client_rows
        if wilbik_log is not None:
            json_obj["wilbik"] = wilbik_log
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(json_obj) + "\n")

        print(f"[SERVER] Round {server_round}: GlobalAcc={global_acc:.4f} | Drift clients = {row_global['clients_with_drift']}")
        return super().aggregate_evaluate(server_round, results, failures)

    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        # Salva pesos apenas dos clientes com drift (seu comportamento atual)
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if cid in self.clients_with_drift_last_round:
                save_client_weights(fit_res.parameters, cid, server_round)
        return super().aggregate_fit(server_round, results, failures)

    # ------------------------------------------------------------------

    def configure_evaluate(self, server_round, parameters, client_manager):
        # A estratégia base retorna List[Tuple[ClientProxy, EvaluateIns]]
        instructions = super().configure_evaluate(server_round, parameters, client_manager)

        if not self._wf_enabled:
            return instructions

        new_instructions = []
        for client, eval_ins in instructions:
            # Copia o config atual (pode vir vazio)
            cfg = dict(eval_ins.config) if isinstance(eval_ins.config, dict) else {}

            # Injeta parâmetros do Wilbik em cada cliente
            cfg["wilbik_k"] = str(self._wf_K)
            cfg["wilbik_m"] = str(self._wf_m)
            cfg["wilbik_max_samples"] = str(self._wf_max_samples)

            if self._wf_state == "init" or self._wf_centers is None:
                cfg["wilbik_stage"] = "init"
            else:
                cfg["wilbik_stage"] = "delta"
                cfg["wilbik_centers"] = json.dumps(self._wf_centers.tolist())

            # Reconstroi o EvaluateIns com o novo config
            new_eval_ins = EvaluateIns(parameters=eval_ins.parameters, config=cfg)
            new_instructions.append((client, new_eval_ins))

        return new_instructions


def server_fn(context: Context):
    # Snapshot de ambiente "relevante" para drift/dados
    env_keys = [
        "DRIFT_METHOD", "DRIFT_WINDOW", "DRIFT_THRESHOLD", "DRIFT_DEBUG", "DRIFT_KSWIN_STAT",
        "NON_IID_ALPHA", "SMOKE_DRIFT",
        # abaixo normalmente são por-cliente, mas guardamos aqui se estiverem setados globalmente
        "PARTITION_ID", "NUM_PARTITIONS",
    ]
    env_snapshot = {k: os.getenv(k) for k in env_keys if os.getenv(k) is not None}

    num_rounds = context.run_config["num-server-rounds"]
    initial_params = ndarrays_to_parameters(get_weights(Net()))

    strategy = DriftAwareStrategy(
        run_config=dict(context.run_config),           # salva configuração da execução
        env_snapshot=env_snapshot,                    # e algumas variáveis de ambiente
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_params,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)