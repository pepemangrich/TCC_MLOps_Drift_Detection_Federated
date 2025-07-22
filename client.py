from typing import Callable
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from flwr.client import NumPyClient
from flwr.common import Context
from flwr.client import Client

def load_partition(cid: int):
    rng = np.random.default_rng(cid)
    X = rng.normal(size=(100, 10))
    y = rng.integers(0, 2, size=100)
    return X, y

class FlowerClient(NumPyClient):
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def get_parameters(self, config):
        return [self.model.coef_.copy(), self.model.intercept_.copy()]

    def fit(self, parameters, config):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        self.model.fit(self.X, self.y)
        return self.get_parameters(config), len(self.X), {}

    def evaluate(self, parameters, config):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        loss = log_loss(self.y, self.model.predict_proba(self.X))
        acc = accuracy_score(self.y, self.model.predict(self.X))
        return loss, len(self.X), {"accuracy": acc}

# API nova (usada com `flwr run`)
def client_fn(context: Context) -> Client:
    print("DEBUG context recebido no client_fn:", context)
    cid = int(context.node_config["partition_id"])
    X, y = load_partition(cid)
    model = LogisticRegression()
    model.fit(X, y)  # inicializa coef_ e intercept_
    return FlowerClient(model, X, y).to_client()  # `.to_client()` necessário na nova API