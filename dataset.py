from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

def make_drift_dataset(seed=0):
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        flip_y=0.1,
        class_sep=0.5 + (seed % 3) * 0.5,  # varia separabilidade
    )
    return train_test_split(X, y, test_size=0.2, random_state=seed)

client_datasets = {
    i: make_drift_dataset(seed=i) for i in range(5)
}