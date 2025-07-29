# pytorchexample: A Flower / PyTorch app.

import torch
import flwr as fl
from flwr.client import Client, ClientApp
from flwr.common import Context, EvaluateIns, EvaluateRes, FitIns, FitRes, ndarrays_to_parameters, parameters_to_ndarrays, Code, Status

from pytorchexample.task import Net, get_weights, load_data, set_weights, test, train
from pytorchexample.drift_detector import DriftDetector


class FlowerClient(Client):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.drift_detector = DriftDetector(num_classes=10, window_size=100, threshold=0.1)

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
        self.set_parameters(parameters_to_ndarrays(ins.parameters))
        loss, accuracy = test(self.net, self.valloader, self.device)

        # Drift detection
        self.net.eval()
        pred_labels = []
        with torch.no_grad():
            for batch in self.valloader:
                if isinstance(batch, dict) and "img" in batch:
                    inputs = batch["img"].to(self.device)
                else:
                    continue
                outputs = self.net(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                pred_labels.extend(preds)

        self.drift_detector.update(pred_labels)
        drift = self.drift_detector.detect_drift()

        print(f"[CLIENT] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"[CLIENT] Drift detected? {drift}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.valloader.dataset),
            metrics={
                "accuracy": float(accuracy),
                "drift": bool(drift),
            },
        )


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    return FlowerClient(trainloader, valloader, local_epochs, learning_rate)


app = ClientApp(client_fn)