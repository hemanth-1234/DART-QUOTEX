"""
dart_quotex/ml/regime_detector.py
VAE-based Market Regime Detector — identifies 7 distinct market conditions.

Architecture
------------
Encoder: OHLCV sequence (T×5) → μ, log_σ² (latent dim=16)
Decoder: z → reconstructed sequence
Regime head: z → softmax over 7 regimes (semi-supervised)

7 Regimes
---------
0  TRENDING_UP      – sustained upward momentum, ADX>25, +DI>-DI
1  TRENDING_DOWN    – sustained downward momentum, ADX>25, -DI>+DI
2  RANGING          – price oscillating in tight band, ADX<20
3  VOLATILE         – high ATR, large wicks, erratic moves
4  BREAKOUT         – price exiting consolidation zone with volume surge
5  REVERSAL         – change of character, RSI divergence
6  CHOPPY           – random walk, no discernible pattern

In live inference, regime drives strategy selection:
  TRENDING_*  → momentum strategies preferred
  RANGING     → mean-reversion preferred
  VOLATILE    → reduce position sizing, tighter confirmation
  BREAKOUT    → aggressive entry on first candle
  REVERSAL    → counter-trend entries
  CHOPPY      → skip / minimum stakes
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

REGIME_NAMES = [
    "TRENDING_UP",
    "TRENDING_DOWN",
    "RANGING",
    "VOLATILE",
    "BREAKOUT",
    "REVERSAL",
    "CHOPPY",
]
N_REGIMES = len(REGIME_NAMES)

# ── Optional PyTorch ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False
    logger.warning("PyTorch not found — RegimeDetector uses rule-based fallback")


# ──────────────────────────────────────────────────────────────────────────────
# Neural network components
# ──────────────────────────────────────────────────────────────────────────────

if _TORCH:
    class _Encoder(nn.Module):
        """Bidirectional GRU encoder → (μ, log_σ²)."""

        def __init__(self, input_dim: int = 5, hidden: int = 64, latent: int = 16) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_dim, hidden, num_layers=2, batch_first=True,
                bidirectional=True, dropout=0.1,
            )
            self.mu_head     = nn.Linear(hidden * 2, latent)
            self.logvar_head = nn.Linear(hidden * 2, latent)

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            _, h = self.gru(x)                  # h: (4, B, hidden)
            h = torch.cat([h[-2], h[-1]], dim=-1)   # (B, hidden*2) — last fwd+bwd
            return self.mu_head(h), self.logvar_head(h)

    class _Decoder(nn.Module):
        """MLP decoder: z → reconstructed sequence."""

        def __init__(self, latent: int = 16, seq_len: int = 30, input_dim: int = 5) -> None:
            super().__init__()
            self.seq_len   = seq_len
            self.input_dim = input_dim
            self.net = nn.Sequential(
                nn.Linear(latent, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, seq_len * input_dim),
            )

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            out = self.net(z)
            return out.view(-1, self.seq_len, self.input_dim)

    class _RegimeHead(nn.Module):
        """Classifier: z → regime probabilities."""

        def __init__(self, latent: int = 16, n_regimes: int = N_REGIMES) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, n_regimes),
            )

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.net(z)    # logits

    class _VAEModel(nn.Module):
        def __init__(
            self, input_dim: int = 5, seq_len: int = 30,
            hidden: int = 64, latent: int = 16,
        ) -> None:
            super().__init__()
            self.encoder     = _Encoder(input_dim, hidden, latent)
            self.decoder     = _Decoder(latent, seq_len, input_dim)
            self.regime_head = _RegimeHead(latent)

        def reparameterise(
            self, mu: "torch.Tensor", logvar: "torch.Tensor"
        ) -> "torch.Tensor":
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(
            self, x: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            mu, logvar = self.encoder(x)
            z          = self.reparameterise(mu, logvar)
            recon      = self.decoder(z)
            logits     = self.regime_head(z)
            return recon, mu, logvar, logits

        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            """Return mean (deterministic) latent vector."""
            mu, _ = self.encoder(x)
            return mu

        def classify(self, x: "torch.Tensor") -> "torch.Tensor":
            """Return regime probabilities."""
            mu = self.encode(x)
            return F.softmax(self.regime_head(mu), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# MarketRegimeDetector
# ──────────────────────────────────────────────────────────────────────────────

class MarketRegimeDetector:
    """
    Detects the current market regime from OHLCV data.

    Parameters
    ----------
    seq_len   : number of candles in each input window
    latent    : VAE latent dimension
    lr        : learning rate for online updates
    """

    def __init__(
        self,
        seq_len: int = 30,
        latent: int = 16,
        lr: float = 1e-3,
    ) -> None:
        self.seq_len = seq_len
        self.latent  = latent
        self._fitted = False
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std:  Optional[np.ndarray] = None

        if _TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = _VAEModel(input_dim=5, seq_len=seq_len, latent=latent).to(self.device)
            self._opt   = optim.Adam(self._model.parameters(), lr=lr)
            logger.info("RegimeDetector on %s", self.device)
        else:
            self._model = None

    # ── public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        candles: np.ndarray,           # (N, 5) — open,high,low,close,volume
    ) -> Tuple[int, str, np.ndarray]:
        """
        Detect regime from the last `seq_len` candles.

        Returns (regime_id, regime_name, probabilities[7])
        """
        if len(candles) < self.seq_len:
            return self._rule_based(candles)

        seq = candles[-self.seq_len:]

        if _TORCH and self._fitted and self._model is not None:
            return self._neural_detect(seq)
        else:
            return self._rule_based(candles)

    def train_online(
        self,
        candles: np.ndarray,           # (N, 5)
        n_epochs: int = 3,
    ) -> float:
        """
        Unsupervised VAE training on a window of candles.
        Returns average reconstruction loss.
        """
        if not _TORCH or self._model is None:
            return 0.0
        if len(candles) < self.seq_len + 10:
            return 0.0

        # Build dataset of overlapping windows
        seqs = self._build_sequences(candles)
        if len(seqs) < 4:
            return 0.0

        X = torch.FloatTensor(seqs).to(self.device)
        total_loss = 0.0
        self._model.train()

        for _ in range(n_epochs):
            for i in range(0, len(X), 32):
                batch = X[i : i + 32]
                recon, mu, logvar, logits = self._model(batch)

                # Reconstruction loss (MSE)
                recon_loss = F.mse_loss(recon, batch)
                # KL divergence
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                # Regime entropy regularisation (encourage diversity)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
                entropy_loss = -0.05 * entropy   # maximise entropy (diversity)

                loss = recon_loss + 0.001 * kl_loss + entropy_loss
                self._opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._opt.step()
                total_loss += loss.item()

        self._fitted = True
        avg = total_loss / max(1, n_epochs * (len(seqs) // 32 + 1))
        return float(avg)

    def regime_for_strategy(self, regime_id: int) -> Dict[str, float]:
        """
        Return strategy multipliers for the given regime.

        Keys: momentum, mean_reversion, position_size_mult, skip
        """
        table = {
            0: {"momentum": 1.4, "mean_reversion": 0.6, "position_size_mult": 1.1, "skip": 0.0},  # TRENDING_UP
            1: {"momentum": 1.4, "mean_reversion": 0.6, "position_size_mult": 1.1, "skip": 0.0},  # TRENDING_DOWN
            2: {"momentum": 0.6, "mean_reversion": 1.4, "position_size_mult": 1.0, "skip": 0.0},  # RANGING
            3: {"momentum": 0.7, "mean_reversion": 0.7, "position_size_mult": 0.6, "skip": 0.2},  # VOLATILE
            4: {"momentum": 1.5, "mean_reversion": 0.5, "position_size_mult": 1.2, "skip": 0.0},  # BREAKOUT
            5: {"momentum": 0.5, "mean_reversion": 1.5, "position_size_mult": 0.9, "skip": 0.0},  # REVERSAL
            6: {"momentum": 0.3, "mean_reversion": 0.3, "position_size_mult": 0.4, "skip": 0.6},  # CHOPPY
        }
        return table.get(regime_id, table[6])

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if _TORCH and self._model is not None:
            torch.save(self._model.state_dict(), path / "regime_vae.pt")
        np.save(path / "regime_scaler.npy",
                np.array([self._scaler_mean or [], self._scaler_std or []], dtype=object))
        logger.info("RegimeDetector saved to %s", path)

    def load(self, path: Path) -> bool:
        path = Path(path)
        pt_file = path / "regime_vae.pt"
        if not pt_file.exists():
            return False
        if _TORCH and self._model is not None:
            self._model.load_state_dict(
                torch.load(pt_file, map_location=self.device)
            )
            self._fitted = True
        sc_file = path / "regime_scaler.npy"
        if sc_file.exists():
            try:
                sc = np.load(sc_file, allow_pickle=True)
                if len(sc) == 2 and len(sc[0]) > 0:
                    self._scaler_mean = sc[0].astype(float)
                    self._scaler_std  = sc[1].astype(float)
            except Exception:
                pass
        logger.info("RegimeDetector loaded from %s", path)
        return True

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_sequences(self, candles: np.ndarray) -> np.ndarray:
        """Build normalised overlapping windows from raw OHLCV."""
        data = candles[:, :5].astype(float)

        # Fit / update running stats
        if self._scaler_mean is None:
            self._scaler_mean = data.mean(axis=0)
            self._scaler_std  = data.std(axis=0) + 1e-8
        else:
            alpha = 0.05
            self._scaler_mean = (1 - alpha) * self._scaler_mean + alpha * data.mean(axis=0)
            self._scaler_std  = (1 - alpha) * self._scaler_std  + alpha * (data.std(axis=0) + 1e-8)

        norm = (data - self._scaler_mean) / self._scaler_std

        seqs = []
        for i in range(self.seq_len, len(norm) + 1):
            seqs.append(norm[i - self.seq_len : i])
        return np.array(seqs, dtype=np.float32)

    def _neural_detect(self, seq: np.ndarray) -> Tuple[int, str, np.ndarray]:
        data = seq[:, :5].astype(float)
        if self._scaler_mean is not None:
            data = (data - self._scaler_mean) / self._scaler_std

        x = torch.FloatTensor(data).unsqueeze(0).to(self.device)  # (1, T, 5)
        self._model.eval()
        with torch.no_grad():
            probs = self._model.classify(x).cpu().numpy()[0]       # (7,)

        regime_id   = int(np.argmax(probs))
        regime_name = REGIME_NAMES[regime_id]
        return regime_id, regime_name, probs

    def _rule_based(
        self, candles: np.ndarray
    ) -> Tuple[int, str, np.ndarray]:
        """
        Fallback rule-based regime detection using ADX, ATR, RSI.
        Produces a hard assignment (one-hot-ish) rather than probabilities.
        """
        if len(candles) < 15:
            probs = np.ones(N_REGIMES) / N_REGIMES
            return 6, "CHOPPY", probs

        closes = candles[:, 3].astype(float)    # col 3 = close
        highs  = candles[:, 1].astype(float)
        lows   = candles[:, 2].astype(float)
        n      = len(closes)

        # ATR
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:]  - closes[:-1]),
            )
        )
        atr = tr[-14:].mean() if len(tr) >= 14 else tr.mean()
        atr_pct = atr / (closes[-1] + 1e-9) * 100

        # ADX (simplified)
        up_moves   = np.diff(highs)
        down_moves = -np.diff(lows)
        dm_plus  = np.where((up_moves > down_moves) & (up_moves > 0), up_moves, 0.0)
        dm_minus = np.where((down_moves > up_moves) & (down_moves > 0), down_moves, 0.0)
        atr_arr  = tr + 1e-9
        di_plus  = 100 * (dm_plus[-14:].sum()  / atr_arr[-14:].sum())
        di_minus = 100 * (dm_minus[-14:].sum() / atr_arr[-14:].sum())
        dx       = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)
        adx      = float(dx)

        # Price direction
        sma_short = closes[-5:].mean()
        sma_long  = closes[-20:].mean() if n >= 20 else closes.mean()
        price_up  = sma_short > sma_long

        # RSI
        delta  = np.diff(closes)
        gain   = np.where(delta > 0, delta, 0.0)[-14:].mean()
        loss   = np.where(delta < 0, -delta, 0.0)[-14:].mean()
        rsi    = 100 - 100 / (1 + gain / (loss + 1e-9))

        # Candle size variance (choppiness indicator)
        body_sizes = np.abs(candles[:, 3] - candles[:, 0])[-20:]
        cv         = body_sizes.std() / (body_sizes.mean() + 1e-9)

        # Range tightness
        recent_range = (highs[-20:].max() - lows[-20:].min()) / (closes[-1] + 1e-9) * 100

        # ── Decision tree ─────────────────────────────────────────────────────
        probs = np.zeros(N_REGIMES)

        if atr_pct > 0.15 and cv > 1.2:
            regime_id = 3   # VOLATILE
            probs[3]  = 0.7; probs[6] = 0.3
        elif adx > 25 and price_up and di_plus > di_minus:
            regime_id = 0   # TRENDING_UP
            probs[0]  = 0.75; probs[4] = 0.25
        elif adx > 25 and not price_up and di_minus > di_plus:
            regime_id = 1   # TRENDING_DOWN
            probs[1]  = 0.75; probs[4] = 0.25
        elif adx < 18 and recent_range < 0.3:
            regime_id = 2   # RANGING
            probs[2]  = 0.7; probs[6] = 0.3
        elif (rsi > 70 and not price_up) or (rsi < 30 and price_up):
            regime_id = 5   # REVERSAL
            probs[5]  = 0.65; probs[2] = 0.35
        elif recent_range < 0.15 and atr_pct > 0.08:
            regime_id = 4   # BREAKOUT (compression before breakout)
            probs[4]  = 0.6; probs[2] = 0.4
        else:
            regime_id = 6   # CHOPPY
            probs[6]  = 0.6; probs[2] = 0.4

        return regime_id, REGIME_NAMES[regime_id], probs
