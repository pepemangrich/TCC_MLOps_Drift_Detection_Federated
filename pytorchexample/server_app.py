"""pytorchexample: A Flower / PyTorch server app."""

import flwr as fl
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from pytorchexample.task import Net, get_weights

from typing import List, Tuple, Dict, Optional, Any

class DriftAwareStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drift_clients_per_round = []

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation results and log clients that detected drift."""

        drift_clients = []
        for client_proxy, eval_res in results:
            client_cid = client_proxy.cid
            metrics: Dict[str, Any] = eval_res.metrics
            if "drift" in metrics and metrics["drift"]:
                drift_clients.append(client_cid)

        self.drift_clients_per_round.append((rnd, drift_clients))

        if drift_clients:
            print(f"[Round {rnd}] Drift detected by clients: {drift_clients}")
        else:
            print(f"[Round {rnd}] No drift detected.")

        # Continue with standard evaluation aggregation
        return super().aggregate_evaluate(rnd, results, failures)


def server_fn(context: Context):
    """Construct the ServerAppComponents with a drift-aware strategy."""

    # Lê configurações do pyproject.toml
    num_rounds = context.run_config["num-server-rounds"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define a estratégia personalizada
    strategy = DriftAwareStrategy(
        fraction_fit=1.0,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=10,
        min_evaluate_clients=5,
        min_available_clients=10,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Cria o ServerApp
app = ServerApp(server_fn=server_fn)