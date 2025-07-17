import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from data.simulate_drift import generate_clients_data
import numpy as np
import os
import json


class FLClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = int(cid)
        all_data = generate_clients_data(n_clients=5, drift_client_ids=[1])
        X, y = all_data[self.cid]
        self.X_train, self.X_val = X[:150], X[150:]
        self.y_train, self.y_val = y[:150], y[150:]
        self.model = LogisticRegression()

    def get_parameters(self, config):
        try:
            return [self.model.coef_, self.model.intercept_]
        except AttributeError:
            n_features = self.X_train.shape[1]
            return [np.zeros((1, n_features)), np.zeros(1)]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        print(f"[CLIENT {self.cid}] Avaliando modelo...")
        self.set_parameters(parameters)
        y_pred = self.model.predict(self.X_val)
        loss = log_loss(self.y_val, self.model.predict_proba(self.X_val))
        accuracy = accuracy_score(self.y_val, y_pred)

        metrics = {
            "round": config.get("round", "unknown"),
            "cid": self.cid,
            "loss": loss,
            "accuracy": accuracy
        }

        os.makedirs("results", exist_ok=True)
        path = f"results/client_{self.cid}_metrics.json"

        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(metrics)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return loss, len(self.X_val), {"accuracy": accuracy}

def client_fn(cid: str):
    print(f"[DEBUG] Inicializando client com cid={cid}")
    return FLClient(cid)

__all__ = ["client_fn"]