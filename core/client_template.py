import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

class FLClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_val, y_val):
        self.model = LogisticRegression()
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.classes_ = np.unique(y_train)

    def get_parameters(self, config):
        try:
            return [self.model.coef_, self.model.intercept_]
        except AttributeError:
            n_features = self.X_train.shape[1]
            coef = np.zeros((1, n_features))
            intercept = np.zeros(1)
            return [coef, intercept]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        self.model.classes_ = self.classes_

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred = self.model.predict(self.X_val)
        loss = log_loss(self.y_val, self.model.predict_proba(self.X_val))
        accuracy = accuracy_score(self.y_val, y_pred)
        return loss, len(self.X_val), {"accuracy": accuracy}