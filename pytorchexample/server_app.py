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
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    EvaluateRes,
    FitRes,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from pytorchexample.task import Net, get_weights


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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clients_with_drift_last_round: Set[str] = set()

        # Logs estruturados
        os.makedirs("logs", exist_ok=True)
        self._csv_path = os.path.join("logs", "round_metrics.csv")
        self._jsonl_path = os.path.join("logs", "round_metrics.jsonl")
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["round", "global_accuracy", "num_clients", "clients_with_drift"]
                )
                writer.writeheader()

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        self.clients_with_drift_last_round.clear()

        # Calcula acurácia global ponderada e coleta clientes com drift
        weighted_acc_sum = 0.0
        weighted_n_sum = 0
        for client_proxy, evaluate_res in results:
            cid = client_proxy.cid
            drift_detected = evaluate_res.metrics.get("drift", False)
            if drift_detected:
                self.clients_with_drift_last_round.add(cid)

            acc = float(evaluate_res.metrics.get("accuracy", 0.0))
            n = int(evaluate_res.num_examples)
            weighted_acc_sum += acc * n
            weighted_n_sum += n

        global_acc = (weighted_acc_sum / weighted_n_sum) if weighted_n_sum > 0 else 0.0
        clients_sorted = sorted(self.clients_with_drift_last_round)
        print(f"[SERVER] Round {server_round}: GlobalAcc={global_acc:.4f} | Drift clients = {clients_sorted}")

        # Persiste CSV/JSONL
        row = {
            "round": server_round,
            "global_accuracy": global_acc,
            "num_clients": len(results),
            "clients_with_drift": ";".join(clients_sorted),
        }
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "global_accuracy", "num_clients", "clients_with_drift"])
            writer.writerow(row)
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        return super().aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        # Salva pesos apenas dos clientes com drift
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if cid in self.clients_with_drift_last_round:
                save_client_weights(fit_res.parameters, cid, server_round)

        return super().aggregate_fit(server_round, results, failures)


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    initial_params = ndarrays_to_parameters(get_weights(Net()))

    strategy = DriftAwareStrategy(
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_params,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)