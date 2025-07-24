"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from pytorchexample.task import Net, get_weights, load_data, set_weights, test, train
from pytorchexample.drift_detector import DriftDetector


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.drift_detector = DriftDetector(num_classes=10, window_size=100, threshold=0.1)

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        self.net.eval()
        pred_labels = []
        with torch.no_grad():
            for batch in self.valloader:
                if isinstance(batch, dict) and "img" in batch:
                    inputs = batch["img"].to(self.device)
                else:
                    print("[DEBUG] Tipo inesperado em inputs:", type(inputs))
                    continue
                outputs = self.net(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                pred_labels.extend(preds)

        self.drift_detector.update(pred_labels)
        drift = self.drift_detector.detect_drift()
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "drift": drift}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate).to_client()


# Flower ClientApp
app = ClientApp(client_fn)