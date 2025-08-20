# drift_detector.py

import os
from typing import List, Optional, Deque
from collections import deque

import numpy as np
from scipy.stats import entropy

# ---- Detectores "river" (opcionais) ----
_HAS_RIVER = True
try:
    from river.drift import KSWIN, ADWIN
except Exception:
    _HAS_RIVER = False


def _bool_env(name: str) -> bool:
    return os.getenv(name, "0") in {"1", "true", "True", "YES", "yes"}


class DriftDetector:
    """
    Detecta drift em 4 modos:

    - 'entropy_fixed'     : |H(win[-1]) - H(win[-2])| > threshold
    - 'entropy_adaptive'  : |ΔH| > (μ_ΔH + 3σ_ΔH) (limiar auto-adaptativo)
    - 'kswin'             : KSWIN (river) em cima do stream de rótulos preditos
    - 'adwin'             : ADWIN (river) em cima do stream de rótulos preditos

    Dica: defina via ambiente:
      DRIFT_METHOD     ∈ {entropy_fixed, entropy_adaptive, kswin, adwin}
      DRIFT_WINDOW     = <int>
      DRIFT_THRESHOLD  = <float>    (usado em entropy_fixed)
      DRIFT_DEBUG      = 1          (loga H_pass, H_curr, |Δ| e thr)
    """

    def __init__(
        self,
        num_classes: int,
        window_size: int = 50,
        threshold: float = 0.1,
        method: Optional[str] = None,
    ):
        # ---- Overrides via ambiente ----
        env_w = os.getenv("DRIFT_WINDOW")
        env_t = os.getenv("DRIFT_THRESHOLD")
        env_m = os.getenv("DRIFT_METHOD")  # e.g., "entropy_adaptive"

        if env_w:
            try:
                window_size = int(env_w)
            except ValueError:
                pass
        if env_t:
            try:
                threshold = float(env_t)
            except ValueError:
                pass
        if env_m:
            method = env_m

        # normalizar método (default: entropy_fixed)
        method = (method or "entropy_fixed").lower().strip()
        if method in {"entropy", "fixed", "entropy-fixed"}:
            method = "entropy_fixed"
        if method in {"entropy_adapt", "adaptive", "entropy-adaptive"}:
            method = "entropy_adaptive"

        self.num_classes = num_classes
        self.window_size = max(2, int(window_size))
        self.threshold = max(0.0, float(threshold))
        self.method = method
        self._debug = _bool_env("DRIFT_DEBUG")

        # ---- Estado para modos de entropia ----
        self._buffer: List[int] = []
        self._windows: Deque[List[int]] = deque(maxlen=50)  # janelas completas
        # mantemos histórico das diferenças de entropia (para limiar adaptativo)
        self._delta_hist: Deque[float] = deque(maxlen=500)

        # ---- Estado para modos "river" ----
        self._river = None
        if self.method in {"kswin", "adwin"}:
            if not _HAS_RIVER:
                raise ImportError(
                    "Modo '%s' requer a biblioteca 'river'. "
                    "Instale com: pip install river"
                    % self.method
                )
            if self.method == "kswin":
                # Permite override via ambiente, mas sempre respeita a restrição:
                # stat_size ≤ floor(window_size/2)
                env_stat = os.getenv("DRIFT_KSWIN_STAT")
                try:
                    stat_size = int(env_stat) if env_stat is not None else 30
                except ValueError:
                    stat_size = 30

                max_allowed = max(2, self.window_size // 2)  # floor(window/2), mínimo 2
                stat_size = max(2, min(stat_size, max_allowed))

                # Observação: se quiser ver mais/menos sensibilidade, aumente/diminua window_size.
                self._river = KSWIN(alpha=0.005, window_size=self.window_size, stat_size=stat_size)
            else:
                # ADWIN: não precisa de janela fixa
                self._river = ADWIN()

    # ---------- Utilidades ----------
    def __repr__(self) -> str:
        extra = ""
        if self.method.startswith("entropy"):
            extra = f", threshold={self.threshold:.4f}"
        return (
            f"DriftDetector(method='{self.method}', window_size={self.window_size}"
            f"{extra})"
        )

    def _entropy(self, labels: List[int]) -> float:
        counts = np.bincount(labels, minlength=self.num_classes)
        total = int(np.sum(counts))
        if total == 0:
            return 0.0
        probs = counts / total
        return float(entropy(probs, base=2))

    # ---------- API pública ----------
    def update(self, preds: List[int]) -> None:
        """
        Alimenta o detector com novas predições (rótulos inteiros).
        Para KSWIN/ADWIN, atualiza um a um.
        Para entropia, criamos janelas completas de tamanho `window_size`.
        """
        if not preds:
            return

        if self.method in {"kswin", "adwin"}:
            # alimentar item a item
            for p in preds:
                x = float(int(p))  # stream numérico
                self._river.update(x)
        else:
            # métodos baseados em entropia
            self._buffer.extend(int(p) for p in preds)
            while len(self._buffer) >= self.window_size:
                win = self._buffer[: self.window_size]
                self._buffer = self._buffer[self.window_size :]
                self._windows.append(win)

                # sempre que fechamos uma nova janela, podemos atualizar o histórico de ΔH
                if len(self._windows) >= 2:
                    h_prev = self._entropy(self._windows[-2])
                    h_curr = self._entropy(self._windows[-1])
                    dh = abs(h_curr - h_prev)
                    self._delta_hist.append(dh)

    def detect(self) -> bool:
        """
        Retorna True se drift for detectado conforme o `method`.
        """
        if self.method == "entropy_fixed":
            return self._detect_entropy_fixed()
        if self.method == "entropy_adaptive":
            return self._detect_entropy_adaptive()
        if self.method in {"kswin", "adwin"}:
            return bool(getattr(self._river, "change_detected", False))
        # fallback seguro
        return False

    # ---------- Implementações por modo ----------
    def _detect_entropy_fixed(self) -> bool:
        # precisa de 2 janelas COMPLETAS
        if len(self._windows) < 2:
            return False
        h_prev = self._entropy(self._windows[-2])
        h_curr = self._entropy(self._windows[-1])
        dh = abs(h_curr - h_prev)

        if self._debug:
            print(
                f"[DRIFT_DEBUG] (fixed) H_past={h_prev:.4f} H_curr={h_curr:.4f} "
                f"|Δ|={dh:.4f} thr={self.threshold:.4f}"
            )
        return dh > self.threshold

    def _detect_entropy_adaptive(self) -> bool:
        # precisa de 2 janelas para ter ΔH e um mínimo de histórico para estatística
        if len(self._windows) < 2:
            return False

        # ΔH atual
        h_prev = self._entropy(self._windows[-2])
        h_curr = self._entropy(self._windows[-1])
        dh = abs(h_curr - h_prev)

        # limiar adaptativo = μ + 3σ no histórico de ΔH (excluindo o atual)
        hist = list(self._delta_hist)[:-1] if len(self._delta_hist) > 1 else list(self._delta_hist)
        if len(hist) >= 10:
            mu = float(np.mean(hist))
            sigma = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.0
            thr = mu + 3.0 * sigma
        else:
            # aquecimento: enquanto não temos histórico suficiente, use threshold fixo
            thr = self.threshold

        if self._debug:
            msg = (
                f"[DRIFT_DEBUG] (adaptive) H_past={h_prev:.4f} H_curr={h_curr:.4f} "
                f"|Δ|={dh:.4f} thr*={thr:.4f}"
            )
            if len(hist) < 10:
                msg += " (warmup: usando threshold fixo)"
            print(msg)

        return dh > thr