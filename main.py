from client import FederatedClient
from dataset import client_datasets
from flwr.common import Context

def client_fn(context: Context):
    cid = context.node_config["partition_id"]
    X_train, y_train, X_test, y_test = client_datasets[cid]
    return FederatedClient(str(cid), X_train, y_train, X_test, y_test).to_client()