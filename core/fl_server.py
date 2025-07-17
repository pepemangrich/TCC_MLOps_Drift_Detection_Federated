import flwr as fl
from core.client_template import FLClient

def run_federated_learning(data, drift_clients=None):
    def client_fn(cid: str):
        idx = int(cid)
        X, y = data[idx]
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        return FLClient(X_train, y_train, X_val, y_val)

    strategy = fl.server.strategy.FedAvg()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(data),
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )