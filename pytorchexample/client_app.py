# pytorchexample: A Flower / PyTorch app.

import os
import numpy as np
import torch
import flwr as fl
from flwr.client import Client, ClientApp
from flwr.common import (
    Context, EvaluateIns, EvaluateRes, FitIns, FitRes,
    ndarrays_to_parameters, parameters_to_ndarrays, Code, Status
)

from pytorchexample.task import Net, get_weights, load_data, set_weights, test, train
from pytorchexample.drift_detector import DriftDetector


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


class FlowerClient(Client):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate,
                 partition_id: int, num_partitions: int):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Detector com override via env
        self.drift_detector = DriftDetector(num_classes=10, window_size=100, threshold=0.1)

        # Controle do "smoke drift"
        self.eval_calls = 0
        self.smoke = os.getenv("SMOKE_DRIFT", "0") == "1"

        print(f"[CLIENT] {self.drift_detector} | SMOKE_DRIFT={self.smoke} | "
              f"partition_id={self.partition_id} num_partitions={self.num_partitions}")

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

        # Coleta rótulos preditos (para o detector)
        self.net.eval()
        pred_labels = []
        with torch.no_grad():
            for batch in self.valloader:
                if isinstance(batch, dict) and "img" in batch:
                    inputs = batch["img"].to(self.device)
                    outputs = self.net(inputs)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    pred_labels.extend(preds)

        # ------ SMOKE DRIFT (opcional) ------
        # Se SMOKE_DRIFT=1, a partir da 2ª chamada de evaluate,
        # corrompemos as predições em metade dos clientes (pares)
        # para causar mudança de entropia e validar o pipeline.
        self.eval_calls += 1
        if self.smoke and self.eval_calls >= 2 and (self.partition_id % 2 == 0):
            rng = np.random.default_rng(42 + self.partition_id)
            pred_labels = rng.integers(0, 10, size=len(pred_labels))
        # -------------------------------------

        # Atualiza detector e decide
        self.drift_detector.update(pred_labels)
        drift = self.drift_detector.detect()

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
    # Robustez: tenta node_config; se não houver, cai para env; por fim, defaults
    def _get(key: str, env: str, default: int) -> int:
        if isinstance(context.node_config, dict) and key in context.node_config:
            return int(context.node_config[key])
        return _int_env(env, default)

    partition_id = _get("partition-id", "PARTITION_ID", 0)
    num_partitions = _get("num-partitions", "NUM_PARTITIONS", 4)

    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])

    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate,
                        partition_id, num_partitions)


app = ClientApp(client_fn)