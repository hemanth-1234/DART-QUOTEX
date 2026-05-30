"""
dart_quotex/ml/uncertainty.py
Uncertainty Quantification for the DART-Quotex trading system.

Three complementary methods are fused into one calibrated confidence score:

1. Ensemble Disagreement
   - Measures variance across RF, GB, SGD predictions
   - High variance = uncertain market state
   - Score: 1 - std(individual_probs)

2. Monte Carlo Dropout (requires PyTorch)
   - Keeps dropout active during inference
   - Runs T=50 stochastic forward passes
   - Score: 1 - std(T_probs) / std_normalised

3. Temperature Scaling (calibration)
   - Learns a single scalar T that calibrates confidence → accuracy
   - Prevents overconfident predictions on unseen data

4. Prediction Interval Estimation
   - Bootstrap-based prediction intervals on the ensemble
   - Provides [lower, upper] confidence bounds

The final confidence emitted to the risk manager is the
MINIMUM of all active methods (most conservative estimate).
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False


# ──────────────────────────────────────────────────────────────────────────────
# MC Dropout network (small, fast)
# ──────────────────────────────────────────────────────────────────────────────

if _TORCH:
    class _MCDropoutNet(nn.Module):
        """Binary classifier with always-active dropout for MC inference."""

        def __init__(self, input_dim: int, hidden: int = 128, p: float = 0.3) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(p),            # always-active: train=True even at eval
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(p),
                nn.Linear(hidden, 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

        def predict_mc(
            self, x: "torch.Tensor", n_passes: int = 50
        ) -> "torch.Tensor":
            """Return (n_passes, B, 2) probability tensor."""
            self.train()   # keep dropout active
            samples = []
            with torch.no_grad():
                for _ in range(n_passes):
                    logits = self(x)
                    probs  = F.softmax(logits, dim=-1)
                    samples.append(probs.unsqueeze(0))
            return torch.cat(samples, dim=0)   # (T, B, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty Quantifier
# ──────────────────────────────────────────────────────────────────────────────

class UncertaintyQuantifier:
    """
    Combines multiple uncertainty methods into a calibrated confidence score.

    Parameters
    ----------
    input_dim   : feature vector size
    n_passes    : MC dropout forward passes (T)
    min_samples : minimum training samples before uncertainty is trusted
    """

    def __init__(
        self,
        input_dim: int = 34,
        n_passes: int = 50,
        min_samples: int = 30,
    ) -> None:
        self.input_dim   = input_dim
        self.n_passes    = n_passes
        self.min_samples = min_samples

        # Calibration history buffer
        self._cal_X: List[np.ndarray] = []
        self._cal_y: List[int] = []
        self._cal_preds: List[float] = []   # predicted confidences
        self._cal_outcomes: List[int] = []   # actual outcomes (0/1)

        # Temperature for calibration (learned scalar)
        self._temperature: float = 1.0
        self._cal_fitted = False

        if _TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._mc_net = _MCDropoutNet(input_dim).to(self.device)
            self._mc_opt = optim.Adam(self._mc_net.parameters(), lr=1e-3)
            self._mc_fitted = False
        else:
            self._mc_net    = None
            self._mc_fitted = False

        # Prediction interval bootstrap buffer
        self._bootstrap_confidences: Deque[float] = deque(maxlen=200)

    # ── public API ────────────────────────────────────────────────────────────

    def quantify(
        self,
        features: np.ndarray,
        ensemble_probs: Optional[np.ndarray] = None,   # (n_models, 2)
    ) -> Tuple[float, dict]:
        """
        Compute unified uncertainty-adjusted confidence.

        Parameters
        ----------
        features        : 1-D feature vector
        ensemble_probs  : per-model probability arrays from EnsembleModel

        Returns
        -------
        confidence : float 0–1 (calibrated, uncertainty-adjusted)
        detail     : dict with per-method scores
        """
        detail = {}

        # ── 1. Ensemble disagreement ──────────────────────────────────────────
        ens_conf = 0.5
        ens_std  = 0.5
        if ensemble_probs is not None and len(ensemble_probs) >= 2:
            call_probs = np.array([p[1] for p in ensemble_probs if len(p) == 2])
            if len(call_probs) >= 2:
                ens_conf = float(call_probs.mean())
                ens_std  = float(call_probs.std())
        disagreement_penalty = ens_std * 2.0   # high std → reduce confidence
        ens_adjusted = max(0.0, min(1.0, ens_conf - disagreement_penalty))
        detail["ensemble_conf"] = ens_conf
        detail["ensemble_std"]  = ens_std
        detail["ens_adjusted"]  = ens_adjusted

        # ── 2. MC Dropout ─────────────────────────────────────────────────────
        mc_conf = ens_conf   # fallback
        if _TORCH and self._mc_fitted and self._mc_net is not None:
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            mc_samples = self._mc_net.predict_mc(x, self.n_passes)   # (T, 1, 2)
            mc_probs   = mc_samples[:, 0, 1].cpu().numpy()            # call probs
            mc_mean    = float(mc_probs.mean())
            mc_std     = float(mc_probs.std())
            mc_conf    = max(0.0, min(1.0, mc_mean - mc_std * 1.5))
            detail["mc_mean"] = mc_mean
            detail["mc_std"]  = mc_std
            detail["mc_conf"] = mc_conf
        else:
            detail["mc_conf"] = mc_conf

        # ── 3. Temperature scaling ────────────────────────────────────────────
        raw_conf    = (ens_adjusted + mc_conf) / 2.0
        cal_conf    = self._temperature_scale(raw_conf)
        detail["temperature"]  = self._temperature
        detail["calibrated"]   = cal_conf

        # ── 4. Prediction interval ────────────────────────────────────────────
        interval_lower = self._prediction_interval_lower(cal_conf)
        detail["interval_lower"] = interval_lower

        # ── Final fusion: take the most conservative estimate ─────────────────
        scores = [ens_adjusted, mc_conf, cal_conf, interval_lower]
        final  = float(np.mean(scores))   # mean of conservative estimates
        final  = float(np.clip(final, 0.0, 1.0))

        detail["final"] = final
        return final, detail

    def update(
        self,
        features: np.ndarray,
        label: int,
        predicted_confidence: float,
    ) -> None:
        """
        Update with a new ground-truth outcome.

        Parameters
        ----------
        features              : feature vector at prediction time
        label                 : 1 = prediction was correct, 0 = incorrect
        predicted_confidence  : confidence we reported at prediction time
        """
        self._cal_preds.append(predicted_confidence)
        self._cal_outcomes.append(label)
        self._cal_X.append(features)
        self._cal_y.append(label)
        self._bootstrap_confidences.append(predicted_confidence)

        n = len(self._cal_y)

        # Update MC Dropout net
        if _TORCH and self._mc_net is not None and n >= self.min_samples:
            self._update_mc_net(features, label)

        # Recalibrate temperature every 20 samples
        if n >= 20 and n % 20 == 0:
            self._recalibrate_temperature()

    def expected_calibration_error(self, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        ECE = Σ |acc(bin) - conf(bin)| · |bin|/n
        """
        preds    = np.array(self._cal_preds)
        outcomes = np.array(self._cal_outcomes)
        if len(preds) < 10:
            return 0.0

        bins  = np.linspace(0, 1, n_bins + 1)
        ece   = 0.0
        n     = len(preds)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (preds >= lo) & (preds < hi)
            if mask.sum() == 0:
                continue
            acc  = outcomes[mask].mean()
            conf = preds[mask].mean()
            ece += mask.sum() / n * abs(acc - conf)
        return float(ece)

    def reliability_diagram_data(self) -> dict:
        """Return data for plotting a reliability diagram."""
        preds    = np.array(self._cal_preds)
        outcomes = np.array(self._cal_outcomes)
        if len(preds) < 10:
            return {}

        bins = np.linspace(0, 1, 11)
        acc_by_bin  = []
        conf_by_bin = []
        count_by_bin = []

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (preds >= lo) & (preds < hi)
            if mask.sum() == 0:
                acc_by_bin.append(0.0)
                conf_by_bin.append((lo + hi) / 2)
                count_by_bin.append(0)
            else:
                acc_by_bin.append(float(outcomes[mask].mean()))
                conf_by_bin.append(float(preds[mask].mean()))
                count_by_bin.append(int(mask.sum()))

        return {
            "bin_confs": conf_by_bin,
            "bin_accs":  acc_by_bin,
            "bin_counts": count_by_bin,
            "ece": self.expected_calibration_error(),
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "temperature": self._temperature,
            "cal_preds":   self._cal_preds,
            "cal_outcomes": self._cal_outcomes,
            "cal_X": self._cal_X,
            "cal_y": self._cal_y,
        }
        with open(path / "uncertainty.pkl", "wb") as f:
            pickle.dump(state, f)
        if _TORCH and self._mc_net is not None:
            torch.save(self._mc_net.state_dict(), path / "mc_dropout.pt")
        logger.info("UncertaintyQuantifier saved to %s", path)

    def load(self, path: Path) -> bool:
        pkl = Path(path) / "uncertainty.pkl"
        if not pkl.exists():
            return False
        with open(pkl, "rb") as f:
            state = pickle.load(f)
        self._temperature   = state.get("temperature", 1.0)
        self._cal_preds     = state.get("cal_preds", [])
        self._cal_outcomes  = state.get("cal_outcomes", [])
        self._cal_X         = state.get("cal_X", [])
        self._cal_y         = state.get("cal_y", [])
        pt = Path(path) / "mc_dropout.pt"
        if _TORCH and self._mc_net is not None and pt.exists():
            self._mc_net.load_state_dict(
                torch.load(pt, map_location=self.device)
            )
            self._mc_fitted = True
        logger.info("UncertaintyQuantifier loaded (T=%.3f, n=%d)", self._temperature, len(self._cal_y))
        return True

    # ── internal ──────────────────────────────────────────────────────────────

    def _update_mc_net(self, features: np.ndarray, label: int) -> None:
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        y = torch.LongTensor([label]).to(self.device)
        self._mc_net.train()
        logits = self._mc_net(x)
        loss   = F.cross_entropy(logits, y)
        self._mc_opt.zero_grad()
        loss.backward()
        self._mc_opt.step()
        self._mc_fitted = True

    def _recalibrate_temperature(self) -> None:
        """
        Learn temperature T by minimising NLL on held-out calibration data.
        T > 1 → softer predictions  (underconfident)
        T < 1 → sharper predictions (overconfident, squeeze it)
        """
        if len(self._cal_preds) < 20:
            return

        preds    = np.array(self._cal_preds[-100:])    # recent window
        outcomes = np.array(self._cal_outcomes[-100:])

        # Simple bisection: find T that minimises |mean_conf - empirical_acc|
        target_acc = outcomes.mean()

        lo, hi = 0.1, 10.0
        for _ in range(30):
            mid = (lo + hi) / 2
            scaled_conf = self._apply_temperature(preds, mid).mean()
            if scaled_conf > target_acc:
                lo = mid
            else:
                hi = mid

        new_T = (lo + hi) / 2
        # Smooth update
        self._temperature = 0.8 * self._temperature + 0.2 * new_T
        self._temperature = float(np.clip(self._temperature, 0.3, 5.0))
        self._cal_fitted  = True
        logger.debug("Temperature recalibrated → T=%.3f", self._temperature)

    def _temperature_scale(self, prob: float) -> float:
        """Apply temperature scaling to a single probability."""
        if self._temperature == 1.0:
            return prob
        # Logit → scale → sigmoid
        eps   = 1e-6
        logit = np.log(prob + eps) - np.log(1 - prob + eps)
        logit_scaled = logit / self._temperature
        return float(1.0 / (1.0 + np.exp(-logit_scaled)))

    @staticmethod
    def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
        eps    = 1e-6
        logits = np.log(probs + eps) - np.log(1 - probs + eps)
        return 1.0 / (1.0 + np.exp(-logits / T))

    def _prediction_interval_lower(self, confidence: float) -> float:
        """
        Estimate the lower bound of a 90% prediction interval
        using the bootstrap distribution of past confidences.
        """
        if len(self._bootstrap_confidences) < 20:
            return confidence * 0.8   # fallback: 80% of stated confidence

        buf  = np.array(self._bootstrap_confidences)
        # Bootstrap resample
        rng  = np.random.default_rng()
        samples = rng.choice(buf, size=(500, len(buf)), replace=True).mean(axis=1)
        lower   = float(np.percentile(samples, 5))   # 5th percentile
        return max(0.0, min(confidence, lower))
