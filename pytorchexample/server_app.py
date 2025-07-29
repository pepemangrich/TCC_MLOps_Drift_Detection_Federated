# pytorchexample: A Flower / PyTorch app.

import os
import torch
from typing import List, Tuple, Optional, Dict, Set

from flwr.common import Context, Metrics, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, EvaluateRes, FitRes
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager

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

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        self.clients_with_drift_last_round.clear()

        for client_proxy, evaluate_res in results:
            cid = client_proxy.cid
            drift_detected = evaluate_res.metrics.get("drift", False)
            if drift_detected:
                self.clients_with_drift_last_round.add(cid)

        print(f"[SERVER] Round {server_round}: Clientes com drift = {sorted(self.clients_with_drift_last_round)}")
        return super().aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
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