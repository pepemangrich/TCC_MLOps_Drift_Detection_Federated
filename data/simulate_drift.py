from sklearn.datasets import make_blobs
import numpy as np

def generate_clients_data(n_clients=5, n_samples=200, drift_client_ids=None):
    drift_client_ids = drift_client_ids or []
    base_centers = [(0, 0), (5, 5)]

    data = []
    for cid in range(n_clients):
        if cid in drift_client_ids:
            centers = [(2, 2), (8, 8)]  # desloca os centros → data drift
        else:
            centers = base_centers
        X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1.0)
        data.append((X, y))
    return data