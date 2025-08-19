# drift_detector.py

import os
from typing import List
import numpy as np
from scipy.stats import entropy


class DriftDetector:
    """
    Detector simples de drift usando diferença de entropia entre
    janelas consecutivas de rótulos preditos.

    Correção: comparamos SEMPRE as duas últimas janelas COMPLETAS
    (em vez de comparar com o resto da janela atual).
    """

    def __init__(self, num_classes: int, window_size: int = 50, threshold: float = 0.1):
        # Overrides via ambiente (opcional)
        env_w = os.getenv("DRIFT_WINDOW")
        env_t = os.getenv("DRIFT_THRESHOLD")
        if env_w is not None:
            try:
                window_size = int(env_w)
            except ValueError:
                pass
        if env_t is not None:
            try:
                threshold = float(env_t)
            except ValueError:
                pass

        self.num_classes = num_classes
        self.window_size = max(2, window_size)
        self.threshold = max(0.0, threshold)

        # Buffer de amostras até completar uma janela
        self._buffer: List[int] = []
        # Sequência de janelas COMPLETAS (mantemos só as N mais recentes)
        self._windows: List[List[int]] = []
        self._max_windows_keep = 50  # para não crescer memória

        # Debug opcional
        self._debug = os.getenv("DRIFT_DEBUG", "0") == "1"

    def __repr__(self) -> str:
        return f"DriftDetector(window_size={self.window_size}, threshold={self.threshold:.4f})"

    def update(self, preds: List[int]) -> None:
        """Alimenta o detector com novas predições (streaming)."""
        self._buffer.extend(int(p) for p in preds)

        # Enquanto houver material para formar janelas completas, formamos
        while len(self._buffer) >= self.window_size:
            win = self._buffer[: self.window_size]
            self._buffer = self._buffer[self.window_size :]
            self._windows.append(win)
            if len(self._windows) > self._max_windows_keep:
                self._windows = self._windows[-self._max_windows_keep :]

    def _entropy(self, labels: List[int]) -> float:
        counts = np.bincount(labels, minlength=self.num_classes)
        probs = counts / max(1, np.sum(counts))
        return float(entropy(probs, base=2))

    def detect(self) -> bool:
        """
        True se |H(win[-1]) - H(win[-2])| > threshold,
        onde win[-1] e win[-2] são DUAS janelas COMPLETAS consecutivas.
        """
        if len(self._windows) < 2:
            return False

        past_labels = self._windows[-2]
        curr_labels = self._windows[-1]

        ent_past = self._entropy(past_labels)
        ent_curr = self._entropy(curr_labels)
        diff = abs(ent_curr - ent_past)

        if self._debug:
            print(f"[DRIFT_DEBUG] H_past={ent_past:.4f} H_curr={ent_curr:.4f} |Δ|={diff:.4f} thr={self.threshold:.4f}")

        return diff > self.threshold