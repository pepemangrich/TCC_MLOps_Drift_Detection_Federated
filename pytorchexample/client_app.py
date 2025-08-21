# pytorchexample: A Flower / PyTorch app.

import os
import numpy as np
import torch
import json
import torch.nn as nn
import flwr as fl
from flwr.client import Client, ClientApp
from flwr.common import (
    Context, EvaluateIns, EvaluateRes, FitIns, FitRes,
    ndarrays_to_parameters, parameters_to_ndarrays, Code, Status
)

from pytorchexample.task import Net, get_weights, load_data, set_weights, test, train
from pytorchexample.drift_detector import DriftDetector


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default

def _extract_prob_features(model, loader, device, max_samples=None):
    """
    Extrai features como vetor de probabilidades (softmax dos logits), robusto
    a diferentes formatos de batch.
    """
    model.eval()
    feats = []
    total = 0

    def _pick_tensor_from_batch(batch):
        if isinstance(batch, dict):
            for k in ("img", "image", "input", "inputs", "x", "data", "features"):
                if k in batch and torch.is_tensor(batch[k]):
                    return batch[k]
            for v in batch.values():
                if torch.is_tensor(v):
                    return v
                if isinstance(v, (list, tuple)):
                    for e in v:
                        if torch.is_tensor(e):
                            return e
        if isinstance(batch, (list, tuple)):
            for e in batch:
                if torch.is_tensor(e):
                    return e
                if isinstance(e, dict):
                    for v in e.values():
                        if torch.is_tensor(v):
                            return v
        if torch.is_tensor(batch):
            return batch
        return None

    with torch.no_grad():
        for batch in loader:
            xb = _pick_tensor_from_batch(batch)
            if xb is None:
                continue
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            feats.append(probs.cpu().numpy())
            total += probs.shape[0]
            if max_samples is not None and total >= max_samples:
                break

    if not feats:
        # fallback robusto: 10 classes (MNIST/CIFAR10);
        num_classes = 10
        return np.full((1, num_classes), 1.0/num_classes, dtype=np.float64)

    X = np.vstack(feats)
    if max_samples is not None:
        X = X[:max_samples]
    return X.astype(np.float64, copy=False)

def _fcm_membership(X, C, m):
    # X: (N,D), C: (K,D)
    dist = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2) + 1e-12
    p = 2.0 / (m - 1.0 + 1e-12)
    inv = 1.0 / dist
    denom = np.sum((inv[:, :, None] / inv[:, None, :]) ** p, axis=2)
    U = 1.0 / denom
    W = U ** m
    return U, W

def _fcm_local_init(X, K, m, steps=5, seed=42):
    """Gera S_num/S_den localmente (sem centros globais), para bootstrap."""
    if X.size == 0:
        D = 1 if X.ndim < 2 else X.shape[1]
        return np.zeros((K, D), dtype=np.float64), np.zeros((K,), dtype=np.float64)
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = rng.choice(N, size=K, replace=(N < K))
    C = X[idx].copy()
    for _ in range(steps):
        _, W = _fcm_membership(X, C, m)
        num = W.T @ X
        den = np.sum(W, axis=0)[:, None] + 1e-12
        C = num / den
    _, W = _fcm_membership(X, C, m)
    S_num = W.T @ X
    S_den = W.sum(axis=0)
    return S_num, S_den


class FlowerClient(Client):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate,
                 partition_id: int, num_partitions: int):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.drift_detector = DriftDetector(num_classes=10, window_size=100, threshold=0.1)
        self.eval_calls = 0
        self.smoke = os.getenv("SMOKE_DRIFT", "0") == "1"

        print(f"[CLIENT] {self.drift_detector} | SMOKE_DRIFT={self.smoke} | "
              f"partition_id={self.partition_id} num_partitions={self.num_partitions}")

    def get_parameters(self):
        return get_weights(self.net)

    def set_parameters(self, parameters):
        set_weights(self.net, parameters)

    def fit(self, ins: FitIns) -> FitRes:
        self.set_parameters(parameters_to_ndarrays(ins.parameters))
        results = train(self.net, self.trainloader, self.valloader, self.local_epochs, self.lr, self.device)

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(get_weights(self.net)),
            num_examples=len(self.trainloader.dataset),
            metrics=results,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # 1) Sincroniza pesos
        self.set_parameters(parameters_to_ndarrays(ins.parameters))

        # 2) Avalia no conjunto de validação
        loss, accuracy = test(self.net, self.valloader, self.device)

        # 3) Drift
        drift_method = os.getenv("DRIFT_METHOD", "entropy_adaptive")
        metrics = {"accuracy": float(accuracy)}

        if drift_method == "wilbik_federated":
            cfg = ins.config

            def cfg_get(k, default=None):
                try:
                    return cfg.get(k, default)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        return dict(cfg).get(k, default)
                    except Exception:
                        return default

            stage = cfg_get("wilbik_stage", "init")
            K = int(cfg_get("wilbik_k", os.getenv("WILBIK_K", "3")))
            m = float(cfg_get("wilbik_m", os.getenv("WILBIK_M", "2.0")))
            q = float(cfg_get("wilbik_q", os.getenv("WILBIK_Q", "2.0")))  # novo (DB exato)
            max_samp = int(cfg_get("wilbik_max_samples", os.getenv("WILBIK_MAX_SAMPLES", "512")))
            init_iters = int(cfg_get("wilbik_init_iters", os.getenv("WILBIK_INIT_ITERS", "5")))

            X = _extract_prob_features(self.net, self.valloader, self.device, max_samples=max_samp)  # (N,D)
            centers_json = cfg_get("wilbik_centers", None)

            if stage == "init":
                # --- INIT federado guiado por centros (se houver) ---
                if centers_json:
                    C = np.array(json.loads(centers_json), dtype=np.float64)  # (K,D)
                    if X.size == 0:
                        S_num = np.zeros((K, C.shape[1]), dtype=np.float64)
                        S_den = np.zeros((K,), dtype=np.float64)
                    else:
                        # membership com os centros globais
                        dist = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2) + 1e-12
                        p = 2.0 / (m - 1.0 + 1e-12)
                        inv = 1.0 / dist
                        denom = np.sum((inv[:, :, None] / inv[:, None, :]) ** p, axis=2)
                        U = 1.0 / denom
                        W = U ** m
                        S_num = W.T @ X
                        S_den = W.sum(axis=0)
                    K_, D_ = S_num.shape
                    for j in range(K_):
                        metrics[f"wf_init_sden_{j}"] = float(S_den[j])
                        for d in range(D_):
                            metrics[f"wf_init_snum_{j}_{d}"] = float(S_num[j, d])
                    metrics["drift"] = 0.0
                else:
                    # primeiro bootstrap local (sem centros)
                    S_num, S_den = _fcm_local_init(X, K=K, m=m, steps=init_iters)
                    K_, D_ = S_num.shape
                    for j in range(K_):
                        metrics[f"wf_init_sden_{j}"] = float(S_den[j])
                        for d in range(D_):
                            metrics[f"wf_init_snum_{j}_{d}"] = float(S_num[j, d])
                    metrics["drift"] = 0.0

            else:
                # --- init_delta0/delta: DB fuzzy exato (A/B/N), com fallback SSE/W ---
                if not centers_json:
                    # segurança: sem centros não há como computar distâncias
                    metrics["wf_N"] = int(X.shape[0])
                    metrics["drift"] = 0.0
                else:
                    C = np.array(json.loads(centers_json), dtype=np.float64)  # (K,D)
                    if X.size == 0:
                        # tudo zero, sem contribuição
                        metrics["wf_N"] = 0
                        for j in range(K):
                            metrics[f"wf_A_{j}"] = 0.0
                            metrics[f"wf_B_{j}"] = 0.0
                            metrics[f"wf_sse_{j}"] = 0.0
                            metrics[f"wf_w_{j}"] = 0.0
                    else:
                        dist = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2) + 1e-12  # (N,K)
                        p = 2.0 / (m - 1.0 + 1e-12)
                        inv = 1.0 / dist
                        denom = np.sum((inv[:, :, None] / inv[:, None, :]) ** p, axis=2)
                        U = 1.0 / denom
                        W = U ** m
                        # DB exato
                        A = U.sum(axis=0)                    # (K,)
                        B = (dist ** q).sum(axis=0)          # (K,)
                        # fallback métricas (mantidas por compatibilidade)
                        sse = (W * dist).sum(axis=0)         # (K,)
                        wsum = W.sum(axis=0)                 # (K,)

                        metrics["wf_N"] = int(X.shape[0])
                        for j in range(K):
                            metrics[f"wf_A_{j}"] = float(A[j])
                            metrics[f"wf_B_{j}"] = float(B[j])
                            metrics[f"wf_sse_{j}"] = float(sse[j])
                            metrics[f"wf_w_{j}"] = float(wsum[j])

                    metrics["drift"] = 0.0  # decisão global no servidor

        else:
            # Detectores locais (como estavam)
            self.net.eval()
            pred_labels = []
            with torch.no_grad():
                for batch in self.valloader:
                    if isinstance(batch, dict) and "img" in batch:
                        inputs = batch["img"].to(self.device)
                    else:
                        inputs = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(self.device)
                    outputs = self.net(inputs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    pred_labels.extend(preds)

            self.eval_calls += 1
            if self.smoke and self.eval_calls >= 2 and (self.partition_id % 2 == 0):
                rng = np.random.default_rng(42 + self.partition_id)
                pred_labels = rng.integers(0, 10, size=len(pred_labels))

            self.drift_detector.update(pred_labels)
            metrics["drift"] = bool(self.drift_detector.detect())

        print(f"[CLIENT] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        if drift_method != "wilbik_federated":
            print(f"[CLIENT] Drift detected? {metrics['drift']}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.valloader.dataset),
            metrics=metrics,
        )


def client_fn(context: Context):
    def _get(key: str, env: str, default: int) -> int:
        if isinstance(context.node_config, dict) and key in context.node_config:
            return int(context.node_config[key])
        return _int_env(env, default)

    partition_id = _get("partition-id", "PARTITION_ID", 0)
    num_partitions = _get("num-partitions", "NUM_PARTITIONS", 4)

    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])

    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate,
                        partition_id, num_partitions)


app = ClientApp(client_fn)