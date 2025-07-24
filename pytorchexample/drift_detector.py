# drift_detector.py

import numpy as np
from scipy.stats import entropy
from typing import List


class DriftDetector:
    def __init__(self, num_classes: int, window_size: int = 50, threshold: float = 0.1):
        """
        Detector de drift baseado na entropia da distribuição de previsões.

        Args:
            num_classes (int): Número de classes do problema.
            window_size (int): Quantidade de previsões usadas para calcular a entropia.
            threshold (float): Valor mínimo de diferença entre entropias médias para sinalizar drift.
        """
        self.num_classes = num_classes
        self.window_size = window_size
        self.threshold = threshold
        self.past_window: List[List[int]] = []
        self.current_window: List[int] = []

    def update(self, preds: List[int]) -> None:
        """
        Adiciona novas previsões ao detector.

        Args:
            preds (List[int]): Lista de rótulos preditos.
        """
        self.current_window.extend(preds)
        if len(self.current_window) >= self.window_size:
            self._slide_window()

    def _slide_window(self):
        if len(self.past_window) >= 1:
            self.past_window.pop(0)
        self.past_window.append(self.current_window[:self.window_size])
        self.current_window = self.current_window[self.window_size:]

    def _compute_entropy(self, labels: List[int]) -> float:
        counts = np.bincount(labels, minlength=self.num_classes)
        probs = counts / np.sum(counts)
        return float(entropy(probs, base=2))

    def detect_drift(self) -> bool:
        """
        Detecta se houve drift com base na diferença de entropia.

        Returns:
            bool: True se drift for detectado, False caso contrário.
        """
        if len(self.past_window) < 1 or len(self.current_window) < self.window_size:
            return False

        past_labels = self.past_window[-1]
        curr_labels = self.current_window[:self.window_size]

        ent_past = self._compute_entropy(past_labels)
        ent_curr = self._compute_entropy(curr_labels)

        return abs(ent_curr - ent_past) > self.threshold