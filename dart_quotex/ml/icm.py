"""
dart_quotex/ml/icm.py
Intrinsic Curiosity Module (ICM) — curiosity-driven exploration.

Architecture
------------
Feature Encoder  φ(s)  : state → feature embedding (no gradient stop)
Forward Model    f     : [φ(s), a] → φ̂(s')    (predicts next state features)
Inverse Model    g     : [φ(s), φ(s')] → â      (predicts action taken)

Intrinsic reward = η · ‖φ(s') − φ̂(s')‖²

The intrinsic reward is added to the extrinsic (P&L) reward before the
SAC update.  High prediction error = unexpected transition = novel state
= exploration bonus.

This prevents the agent from getting stuck repeating the same strategy
when market conditions change.

Reference: Pathak et al. (2017) "Curiosity-driven Exploration by
           Self-Supervised Prediction"
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False
    logger.warning("PyTorch not found — ICM disabled (no intrinsic reward)")


# ──────────────────────────────────────────────────────────────────────────────
# Neural network components
# ──────────────────────────────────────────────────────────────────────────────

if _TORCH:
    class _FeatureEncoder(nn.Module):
        """Maps state → compact feature embedding."""

        def __init__(self, state_dim: int, feat_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ELU(),
                nn.Linear(128, feat_dim),
                nn.LayerNorm(feat_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class _ForwardModel(nn.Module):
        """Predicts φ(s') from [φ(s), a]."""

        def __init__(self, feat_dim: int = 64, action_dim: int = 2) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feat_dim + action_dim, 128),
                nn.ELU(),
                nn.Linear(128, feat_dim),
            )

        def forward(
            self,
            phi_s: "torch.Tensor",
            action: "torch.Tensor",
        ) -> "torch.Tensor":
            x = torch.cat([phi_s, action], dim=-1)
            return self.net(x)

    class _InverseModel(nn.Module):
        """Predicts action from [φ(s), φ(s')]."""

        def __init__(self, feat_dim: int = 64, action_dim: int = 2) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feat_dim * 2, 128),
                nn.ELU(),
                nn.Linear(128, action_dim),
            )

        def forward(
            self,
            phi_s: "torch.Tensor",
            phi_s_next: "torch.Tensor",
        ) -> "torch.Tensor":
            x = torch.cat([phi_s, phi_s_next], dim=-1)
            return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# ICM module
# ──────────────────────────────────────────────────────────────────────────────

class ICM:
    """
    Intrinsic Curiosity Module.

    Parameters
    ----------
    state_dim   : dimension of state (feature) vector
    action_dim  : dimension of action vector
    feat_dim    : size of internal feature embedding
    eta         : intrinsic reward scale (0–1, default 0.1)
    beta        : weight of forward vs inverse loss (default 0.2)
    lr          : learning rate
    """

    def __init__(
        self,
        state_dim: int = 34,
        action_dim: int = 2,
        feat_dim: int = 64,
        eta: float = 0.1,
        beta: float = 0.2,
        lr: float = 1e-3,
    ) -> None:
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.feat_dim   = feat_dim
        self.eta        = eta
        self.beta       = beta

        self._updates   = 0
        self._novelty_buffer: list = []   # recent curiosity scores for normalisation

        if _TORCH:
            self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._encoder = _FeatureEncoder(state_dim, feat_dim).to(self.device)
            self._forward = _ForwardModel(feat_dim, action_dim).to(self.device)
            self._inverse = _InverseModel(feat_dim, action_dim).to(self.device)
            self._opt = optim.Adam(
                list(self._encoder.parameters())
                + list(self._forward.parameters())
                + list(self._inverse.parameters()),
                lr=lr,
            )
            logger.info("ICM initialised (η=%.2f, β=%.2f)", eta, beta)
        else:
            self._encoder = None

    # ── public API ────────────────────────────────────────────────────────────

    def intrinsic_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> float:
        """
        Compute intrinsic (curiosity) reward for a transition.

        reward = η · ½ · ‖φ(s') − φ̂(s')‖²
        """
        if not _TORCH or self._encoder is None or self._updates < 5:
            return 0.0

        s  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        a  = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            phi_s  = self._encoder(s)
            phi_ns = self._encoder(ns)
            phi_ns_pred = self._forward(phi_s, a)
            error   = 0.5 * F.mse_loss(phi_ns_pred, phi_ns).item()

        # Normalise with running buffer
        self._novelty_buffer.append(error)
        if len(self._novelty_buffer) > 100:
            self._novelty_buffer.pop(0)
        mean_e = float(np.mean(self._novelty_buffer))
        std_e  = float(np.std(self._novelty_buffer)) + 1e-8
        normalised = (error - mean_e) / std_e

        return float(self.eta * normalised)

    def update(
        self,
        states: np.ndarray,       # (B, state_dim)
        actions: np.ndarray,      # (B, action_dim)
        next_states: np.ndarray,  # (B, state_dim)
    ) -> dict:
        """
        Perform one gradient update step.
        Returns loss dict.
        """
        if not _TORCH or self._encoder is None:
            return {"fwd": 0.0, "inv": 0.0}
        if len(states) < 2:
            return {"fwd": 0.0, "inv": 0.0}

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.FloatTensor(actions).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)

        phi_s  = self._encoder(s)
        phi_ns = self._encoder(ns)

        # Forward loss: predict φ(s')
        phi_ns_pred  = self._forward(phi_s, a)
        forward_loss = 0.5 * F.mse_loss(phi_ns_pred, phi_ns.detach())

        # Inverse loss: predict action from (φ(s), φ(s'))
        action_pred  = self._inverse(phi_s, phi_ns)
        inverse_loss = F.mse_loss(action_pred, a)

        loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
        self._opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self._encoder.parameters())
            + list(self._forward.parameters())
            + list(self._inverse.parameters()),
            max_norm=0.5,
        )
        self._opt.step()
        self._updates += 1

        return {
            "fwd":   float(forward_loss.item()),
            "inv":   float(inverse_loss.item()),
            "total": float(loss.item()),
        }

    def novelty_score(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """
        Return a 0–1 novelty score (how surprising is this transition?).
        High score → AI should be more exploratory.
        """
        if not _TORCH or self._updates < 10:
            return 0.5

        dummy_action = np.zeros(self.action_dim, dtype=np.float32)
        reward = self.intrinsic_reward(state, dummy_action, next_state)
        # Sigmoid normalisation
        score = 1.0 / (1.0 + np.exp(-reward * 5))
        return float(score)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if _TORCH and self._encoder is not None:
            torch.save({
                "encoder": self._encoder.state_dict(),
                "forward": self._forward.state_dict(),
                "inverse": self._inverse.state_dict(),
                "updates": self._updates,
            }, path / "icm.pt")
        logger.info("ICM saved to %s", path)

    def load(self, path: Path) -> bool:
        pt_file = Path(path) / "icm.pt"
        if not pt_file.exists():
            return False
        if _TORCH and self._encoder is not None:
            ckpt = torch.load(pt_file, map_location=self.device)
            self._encoder.load_state_dict(ckpt["encoder"])
            self._forward.load_state_dict(ckpt["forward"])
            self._inverse.load_state_dict(ckpt["inverse"])
            self._updates = ckpt.get("updates", 0)
        logger.info("ICM loaded from %s (updates=%d)", path, self._updates)
        return True
