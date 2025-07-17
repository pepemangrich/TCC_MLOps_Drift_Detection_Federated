from data.simulate_drift import generate_clients_data
from core.fl_server import run_federated_learning

def main():
    drift_clients = {
        "data_drift_clients": [1],
        "model_drift_clients": [],
        "input_drift_clients": []
    }

    data = generate_clients_data(
        n_clients=5,
        drift_client_ids=drift_clients["data_drift_clients"]
    )

    run_federated_learning(data, drift_clients=drift_clients)