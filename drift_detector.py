import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from typing import Literal

def softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def compute_confidence(probs: np.ndarray) -> float:
    return np.mean(np.max(probs, axis=1))

def compute_entropy(probs: np.ndarray) -> float:
    return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

def check_drift(model: BaseEstimator, X: np.ndarray, 
                baseline: float,
                mode: Literal["entropy", "confidence"] = "entropy",
                threshold: float = 0.1) -> tuple[bool, float]:
    logits = model.decision_function(X)
    if logits.ndim == 1:
        logits = np.vstack([-logits, logits]).T
    probs = softmax(logits)

    if mode == "entropy":
        current = compute_entropy(probs)
        drift = abs(current - baseline) > threshold
    elif mode == "confidence":
        current = compute_confidence(probs)
        drift = abs(current - baseline) > threshold
    else:
        raise ValueError("mode must be 'entropy' or 'confidence'")
    
    return drift, current