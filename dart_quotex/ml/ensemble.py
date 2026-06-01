"""
dart_quotex/ml/ensemble.py
Hybrid ensemble classifier for binary options direction prediction.

Architecture
------------
Three base learners vote with learned weights:
  1. RandomForestClassifier      – captures non-linear feature interactions
  2. GradientBoostingClassifier  – sequential error correction
  3. SGDClassifier (log loss)    – supports partial_fit (incremental)

The ensemble meta-weight is updated online after each trade result using
exponential weighting (better recent performers get more weight).

Incremental learning
--------------------
After every trade closes, call `update(features, label)`.
The SGD model updates immediately via partial_fit.
RF / GB are retrained periodically (every N samples) from the growing
history buffer so their "cold" knowledge does not become stale.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────

class EnsembleModel:
    """
    Binary classifier: 1 = CALL (price up), 0 = PUT (price down).

    Parameters
    ----------
    n_estimators   : trees in RF and GB
    retrain_every  : retrain RF/GB after this many new samples
    min_samples    : minimum samples before predictions are trusted
    """

    def __init__(
        self,
        n_estimators: int = 100,
        retrain_every: int = 20,
        min_samples: int = 50,
    ) -> None:
        self.n_estimators = n_estimators
        self.retrain_every = retrain_every
        self.min_samples = min_samples

        # Base learners
        self._rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        )
        self._gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self._sgd = SGDClassifier(
            loss="log_loss",
            max_iter=1,
            tol=None,
            warm_start=True,
            random_state=42,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()

        # Meta-weights: [rf, gb, sgd]
        self._weights = np.array([1.0, 1.0, 1.0])

        # History buffers
        self._X: List[np.ndarray] = []
        self._y: List[int] = []
        self._since_retrain = 0
        self._fitted = {"rf": False, "gb": False, "sgd": False}

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Predict direction and return (direction, confidence).
        direction: 1 = CALL, 0 = PUT
        confidence: 0.0 – 1.0
        """
        if len(self._y) < self.min_samples:
            # Not enough data — return neutral
            return 1, 0.0

        x = self._scaler.transform(features.reshape(1, -1))
        probs = self._weighted_proba(x)[0]   # [prob_put, prob_call]
        prob_call = float(probs[1])
        direction = 1 if prob_call > 0.5 else 0
        confidence = abs(prob_call - 0.5) * 2   # scale 0-1
        return direction, confidence

    def update(self, features: np.ndarray, label: int) -> None:
        """
        Incorporate one new labelled example (called after trade closes).
        label: 1 = trade was correct (won), 0 = trade was wrong (lost)
        """
        self._X.append(features)
        self._y.append(label)
        self._since_retrain += 1

        n = len(self._y)
        X = np.array(self._X)
        y = np.array(self._y)

        # Fit scaler (incremental approximation: refit on full history)
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        # SGD — always update incrementally
        self._sgd.partial_fit(
            X_scaled[-1:], y[-1:], classes=np.array([0, 1])
        )
        self._fitted["sgd"] = True

        # RF / GB — retrain from scratch on full history periodically
        if n >= self.min_samples and self._since_retrain >= self.retrain_every:
            logger.info("Retraining RF and GB on %d samples", n)
            self._rf.fit(X_scaled, y)
            self._gb.fit(X_scaled, y)
            self._fitted["rf"] = True
            self._fitted["gb"] = True
            self._since_retrain = 0
        elif n == self.min_samples:
            # First time we have enough data — do initial fit
            logger.info("Initial fit on %d samples", n)
            self._rf.fit(X_scaled, y)
            self._gb.fit(X_scaled, y)
            self._fitted["rf"] = True
            self._fitted["gb"] = True
            self._since_retrain = 0

        # Update meta-weights based on recent accuracy
        self._update_weights(X_scaled[-min(50, n):], y[-min(50, n):])

    def train_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """
        Bulk training from historical data (backtester / offline training).
        """
        assert len(X) == len(y), "X and y must have the same length"
        self._X = list(X)
        self._y = list(y)

        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        logger.info("Batch training ensemble on %d samples", len(y))
        self._rf.fit(X_scaled, y, sample_weight=sample_weight)
        self._gb.fit(X_scaled, y, sample_weight=sample_weight)
        # SGD: iterate multiple passes
        for _ in range(5):
            self._sgd.partial_fit(X_scaled, y, classes=np.array([0, 1]))
        self._fitted = {"rf": True, "gb": True, "sgd": True}
        self._since_retrain = 0
        logger.info("Batch training complete")

    def is_ready(self) -> bool:
        return len(self._y) >= self.min_samples

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "rf": self._rf,
            "gb": self._gb,
            "sgd": self._sgd,
            "scaler": self._scaler,
            "weights": self._weights,
            "X": self._X,
            "y": self._y,
            "fitted": self._fitted,
            "since_retrain": self._since_retrain,
        }
        with open(path / "ensemble.pkl", "wb") as f:
            pickle.dump(state, f)
        logger.info("Ensemble saved to %s", path)

    def load(self, path: Path) -> bool:
        pkl = Path(path) / "ensemble.pkl"
        if not pkl.exists():
            logger.info("No saved ensemble found at %s", pkl)
            return False
        with open(pkl, "rb") as f:
            state = pickle.load(f)
        self._rf = state["rf"]
        self._gb = state["gb"]
        self._sgd = state["sgd"]
        self._scaler = state["scaler"]
        self._weights = state["weights"]
        self._X = state["X"]
        self._y = state["y"]
        self._fitted = state["fitted"]
        self._since_retrain = state["since_retrain"]
        logger.info(
            "Ensemble loaded from %s (%d samples)", pkl, len(self._y)
        )
        return True

    # ── internal ──────────────────────────────────────────────────────────────

    def _weighted_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return weighted average probability across active models."""
        probas = []
        weights = []

        if self._fitted["rf"]:
            probas.append(self._rf.predict_proba(X_scaled))
            weights.append(self._weights[0])
        if self._fitted["gb"]:
            probas.append(self._gb.predict_proba(X_scaled))
            weights.append(self._weights[1])
        if self._fitted["sgd"]:
            probas.append(self._sgd.predict_proba(X_scaled))
            weights.append(self._weights[2])

        if not probas:
            # Nothing fitted yet — uniform
            return np.array([[0.5, 0.5]] * len(X_scaled))

        w = np.array(weights)
        w = w / w.sum()
        combined = sum(p * wi for p, wi in zip(probas, w))
        return combined

    def _update_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Adjust meta-weights based on per-model accuracy on recent data."""
        if len(y) < 5:
            return

        accs = []
        for key, model in [("rf", self._rf), ("gb", self._gb), ("sgd", self._sgd)]:
            if self._fitted[key]:
                pred = model.predict(X)
                acc = float((pred == y).mean())
            else:
                acc = 0.5
            accs.append(acc)

        accs_arr = np.array(accs) + 1e-6
        # Exponential weighting: reward accuracy above 0.5
        exp_w = np.exp(4 * (accs_arr - 0.5))
        self._weights = exp_w / exp_w.sum() * 3   # keep sum ~3 for scale
