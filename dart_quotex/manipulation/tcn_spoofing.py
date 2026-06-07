"""
dart_quotex/manipulation/tcn_spoofing.py
Feature 6 — TCN-based Spoofing / Layering Detection
=====================================================
Detects suspicious order book event sequences using a Temporal Convolutional
Network (TCN) and cosine similarity search against a library of known patterns.

Architecture
------------
  Encoder   : 4-layer dilated TCN → 64-dim embedding
  Labelling  : Heuristic weak-supervision (large orders placed and
               reversed within 2 candles, tagged as suspicious)
  Inference  : Embed current sequence → cosine search in pattern library
               → flag if similarity > threshold

Since Quotex does not expose a true order book, this module operates on
OHLCV proxy features (large wick + volume spike + immediate reversal)
as stand-ins for order-book events.

Usage
-----
    detector = TCNSpoofingDetector()
    detector.build_pattern_library(historical_df)   # offline, once
    result = detector.detect(recent_df)
    if result.suspicious:
        # skip trade or reduce size
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    log.warning("PyTorch not installed — TCNSpoofingDetector in rule-based fallback mode")


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpoofingResult:
    suspicious:   bool
    score:        float     # 0 (clean) – 1 (suspicious)
    similarity:   float     # cosine similarity to closest pattern
    pattern_type: str       # "large_wick_reversal" | "volume_spike_reversal" | "clean"
    details:      str = ""


# ──────────────────────────────────────────────────────────────────────────────
# TCN building block
# ──────────────────────────────────────────────────────────────────────────────

if _TORCH:
    class _TCNBlock(nn.Module):
        """Single dilated causal convolution block with residual connection."""
        def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
            super().__init__()
            pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                                  padding=pad, dilation=dilation)
            self.norm = nn.BatchNorm1d(out_ch)
            self.drop = nn.Dropout(0.1)
            self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # Causal: remove future padding
            out = self.conv(x)
            out = out[:, :, :x.shape[2]]
            out = F.relu(self.norm(out))
            out = self.drop(out)
            return F.relu(out + self.res(x))

    class _TCNEncoder(nn.Module):
        """4-layer dilated TCN → mean-pooled 64-dim embedding."""
        def __init__(self, n_features: int = 8, embed_dim: int = 64) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                _TCNBlock(n_features, 32, kernel=3, dilation=1),
                _TCNBlock(32, 32, kernel=3, dilation=2),
                _TCNBlock(32, 64, kernel=3, dilation=4),
                _TCNBlock(64, embed_dim, kernel=3, dilation=8),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, features, seq_len)
            h = self.blocks(x)
            return h.mean(dim=2)   # (batch, embed_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction from OHLCV (order-book proxy)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_features(df: pd.DataFrame, seq_len: int = 30) -> np.ndarray:
    """
    Build 8-feature proxy sequence from OHLCV.
    These mimic order-book signals using only price and volume.

    Features per candle
    -------------------
    0  normalised log return
    1  upper wick / range
    2  lower wick / range
    3  body / range
    4  volume z-score (rolling 20)
    5  range z-score  (rolling 20)
    6  return velocity (1-bar momentum)
    7  return reversal (sign change: was up, now down)
    """
    if len(df) < max(seq_len, 22):
        return np.zeros((8, seq_len), dtype=np.float32)

    df = df.tail(seq_len + 20).copy()
    o  = df["open"].values.astype(float)
    h  = df["high"].values.astype(float)
    l  = df["low"].values.astype(float)
    c  = df["close"].values.astype(float)
    v  = df["volume"].values.astype(float) + 1e-9

    rng  = (h - l) + 1e-9
    body = np.abs(c - o)
    uw   = (h - np.maximum(c, o)) / rng
    lw   = (np.minimum(c, o) - l) / rng
    br   = body / rng
    ret  = np.diff(c, prepend=c[0]) / (c + 1e-9)

    v_mean = np.convolve(v,   np.ones(20)/20, mode='same')
    v_std  = np.array([v[max(0,i-20):i+1].std() for i in range(len(v))]) + 1e-9
    r_mean = np.convolve(rng, np.ones(20)/20, mode='same')
    r_std  = np.array([rng[max(0,i-20):i+1].std() for i in range(len(rng))]) + 1e-9

    v_z   = (v - v_mean) / v_std
    rng_z = (rng - r_mean) / r_std

    vel   = np.diff(ret, prepend=ret[0])
    rev   = (np.sign(ret) != np.sign(np.roll(ret, 1))).astype(float)

    feats = np.stack([
        np.clip(ret, -0.01, 0.01) / 0.01,
        np.clip(uw, 0, 1),
        np.clip(lw, 0, 1),
        np.clip(br, 0, 1),
        np.clip(v_z, -3, 3) / 3,
        np.clip(rng_z, -3, 3) / 3,
        np.clip(vel, -0.005, 0.005) / 0.005,
        rev,
    ], axis=0)   # (8, N)

    # Return last seq_len columns
    return feats[:, -seq_len:].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Weak-supervision labeller
# ──────────────────────────────────────────────────────────────────────────────

def _label_suspicious(df: pd.DataFrame, window: int = 20) -> np.ndarray:
    """
    Heuristically label candles as suspicious (1) or clean (0).

    Suspicious pattern: large volume spike AND large wick AND
    price reverses within 2 candles (simulates order placement + cancellation).
    """
    v   = df["volume"].values.astype(float) + 1e-9
    h   = df["high"].values.astype(float)
    l   = df["low"].values.astype(float)
    c   = df["close"].values.astype(float)
    o   = df["open"].values.astype(float)
    rng = (h - l) + 1e-9
    labels = np.zeros(len(df), dtype=np.float32)

    v_roll = pd.Series(v).rolling(window).mean().values + 1e-9

    for i in range(window, len(df) - 2):
        vol_spike  = v[i] > v_roll[i] * 3
        large_wick = (max(h[i]-max(c[i],o[i]), min(c[i],o[i])-l[i]) / rng[i]) > 0.6
        # reversal within 2 bars
        went_up   = c[i] > o[i]
        reversed_ = (c[i+1] < o[i+1]) if went_up else (c[i+1] > o[i+1])
        if vol_spike and large_wick and reversed_:
            labels[i] = 1.0
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# TCNSpoofingDetector
# ──────────────────────────────────────────────────────────────────────────────

class TCNSpoofingDetector:
    """
    Offline training + online detection pipeline.

    Workflow
    --------
    Offline  (once, before live trading):
        detector = TCNSpoofingDetector()
        detector.build_pattern_library(historical_df)
        detector.save("models/")

    Online   (every bar):
        detector.load("models/")
        result = detector.detect(recent_df)
        if result.suspicious:
            logger.warning("Spoofing detected: %s", result.details)
    """

    def __init__(
        self,
        seq_len:    int   = 30,
        embed_dim:  int   = 64,
        threshold:  float = 0.80,   # cosine similarity threshold
        n_patterns: int   = 500,    # max patterns to store
    ) -> None:
        self.seq_len    = seq_len
        self.embed_dim  = embed_dim
        self.threshold  = threshold
        self.n_patterns = n_patterns

        if _TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._encoder = _TCNEncoder(n_features=8, embed_dim=embed_dim).to(self.device)
            self._opt     = torch.optim.Adam(self._encoder.parameters(), lr=1e-3)
        else:
            self._encoder = None

        # Pattern library: list of (embedding, label, type)
        self._library: List[tuple] = []   # (np.ndarray embed, float label, str type)
        self._trained = False

    # ── offline: build pattern library ───────────────────────────────────────

    def build_pattern_library(
        self,
        df:       pd.DataFrame,
        epochs:   int = 10,
    ) -> None:
        """
        Train TCN encoder on historical data and populate the pattern library.
        Run this offline before live trading.
        """
        if len(df) < self.seq_len + 50:
            log.warning("Insufficient data for TCN training (%d rows)", len(df))
            self._build_rule_based_library(df)
            return

        if not _TORCH or self._encoder is None:
            log.info("PyTorch unavailable — building rule-based pattern library")
            self._build_rule_based_library(df)
            return

        labels_arr = _label_suspicious(df)
        n_sus = int(labels_arr.sum())
        log.info(
            "TCN training: %d candles, %d suspicious labels",
            len(df), n_sus,
        )

        # Build sliding-window dataset
        X_sus, X_clean = [], []
        for i in range(self.seq_len, len(df)):
            window = df.iloc[i - self.seq_len : i]
            feat   = _extract_features(window, self.seq_len)
            if labels_arr[i - 1] == 1.0:
                X_sus.append(feat)
            elif len(X_clean) < len(X_sus) * 3 + 100:
                X_clean.append(feat)

        if not X_sus:
            log.info("No suspicious labels found — library built from rule heuristics only")
            self._build_rule_based_library(df)
            return

        # Simple contrastive-style training: pull suspicious together
        self._encoder.train()
        all_X   = X_sus + X_clean[:len(X_sus) * 2]
        all_lbl = [1.0] * len(X_sus) + [0.0] * len(X_clean[:len(X_sus) * 2])

        tensor_X = torch.FloatTensor(all_X).to(self.device)
        tensor_y = torch.FloatTensor(all_lbl).to(self.device)

        for ep in range(epochs):
            idx  = torch.randperm(len(tensor_X))
            loss_sum = 0.0
            for start in range(0, len(idx), 32):
                batch_idx = idx[start:start + 32]
                xb = tensor_X[batch_idx]
                yb = tensor_y[batch_idx]
                emb  = self._encoder(xb)
                # Cosine similarity loss: push suspicious high, clean low
                logit = emb.norm(dim=1)
                loss  = F.binary_cross_entropy_with_logits(logit, yb)
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()
                loss_sum += loss.item()
            log.debug("TCN epoch %d/%d loss=%.4f", ep + 1, epochs, loss_sum)

        # Populate library with suspicious embeddings
        self._encoder.eval()
        self._library = []
        for feat, lbl in zip(all_X, all_lbl):
            emb = self._embed(feat)
            ptype = "suspicious" if lbl == 1.0 else "clean"
            self._library.append((emb, lbl, ptype))

        # Limit library size
        sus = [(e, l, t) for e, l, t in self._library if t == "suspicious"]
        cln = [(e, l, t) for e, l, t in self._library if t == "clean"]
        self._library = sus[:self.n_patterns // 2] + cln[:self.n_patterns // 2]
        self._trained = True
        log.info(
            "Pattern library built: %d suspicious + %d clean patterns",
            len(sus[:self.n_patterns // 2]),
            len(cln[:self.n_patterns // 2]),
        )

    # ── online: detect ────────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> SpoofingResult:
        """
        Check the most recent candles for suspicious patterns.
        Falls back to rule-based scoring if TCN not trained.
        """
        if not self._trained or not self._library:
            return self._rule_based_detect(df)

        feat = _extract_features(df, self.seq_len)
        emb  = self._embed(feat)

        # Cosine similarity search
        sus_sims, clean_sims = [], []
        for lib_emb, label, ptype in self._library:
            sim = self._cosine(emb, lib_emb)
            if ptype == "suspicious":
                sus_sims.append(sim)
            else:
                clean_sims.append(sim)

        max_sus   = float(max(sus_sims))   if sus_sims   else 0.0
        max_clean = float(max(clean_sims)) if clean_sims else 1.0

        # Score: how much more similar to suspicious than clean
        score = max(0.0, min(1.0, max_sus - max_clean * 0.5))
        suspicious = max_sus > self.threshold and max_sus > max_clean * 0.8

        return SpoofingResult(
            suspicious=suspicious,
            score=round(score, 3),
            similarity=round(max_sus, 3),
            pattern_type="suspicious" if suspicious else "clean",
            details=(
                f"max_sus_sim={max_sus:.3f} "
                f"max_clean_sim={max_clean:.3f} "
                f"threshold={self.threshold}"
            ),
        )

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if _TORCH and self._encoder is not None:
            torch.save(self._encoder.state_dict(), path / "tcn_encoder.pt")
        with open(path / "tcn_library.pkl", "wb") as f:
            pickle.dump({"library": self._library, "trained": self._trained}, f)
        log.info("TCNSpoofingDetector saved to %s (%d patterns)", path, len(self._library))

    def load(self, path: Path) -> bool:
        path = Path(path)
        pkl  = path / "tcn_library.pkl"
        if not pkl.exists():
            return False
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        self._library = data.get("library", [])
        self._trained = data.get("trained", False)
        pt = path / "tcn_encoder.pt"
        if _TORCH and self._encoder is not None and pt.exists():
            self._encoder.load_state_dict(
                torch.load(pt, map_location=self.device)
            )
        log.info("TCNSpoofingDetector loaded (%d patterns)", len(self._library))
        return True

    # ── internal ──────────────────────────────────────────────────────────────

    def _embed(self, feat: np.ndarray) -> np.ndarray:
        if not _TORCH or self._encoder is None:
            return feat.mean(axis=1)   # trivial fallback
        x   = torch.FloatTensor(feat).unsqueeze(0).to(self.device)
        self._encoder.eval()
        with torch.no_grad():
            emb = self._encoder(x).cpu().numpy()[0]
        norm = np.linalg.norm(emb) + 1e-9
        return emb / norm

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def _build_rule_based_library(self, df: pd.DataFrame) -> None:
        """Fallback: build library from pure rule scoring, no TCN."""
        labels = _label_suspicious(df)
        for i in range(self.seq_len, min(len(df), self.seq_len + self.n_patterns)):
            w    = df.iloc[i - self.seq_len : i]
            feat = _extract_features(w, self.seq_len)
            emb  = feat.mean(axis=1)
            lbl  = float(labels[i - 1])
            self._library.append((emb, lbl, "suspicious" if lbl else "clean"))
        self._trained = True

    def _rule_based_detect(self, df: pd.DataFrame) -> SpoofingResult:
        """Pure rule-based fallback (no TCN, no library)."""
        from dart_quotex.signals.manipulation import wick_rejection_trap, volume_anomaly
        wick_s = wick_rejection_trap(df)
        vol_s  = volume_anomaly(df)
        score  = (wick_s + vol_s) / 2
        return SpoofingResult(
            suspicious=(score > 0.6),
            score=round(score, 3),
            similarity=0.0,
            pattern_type="rule_based",
            details=f"wick={wick_s:.2f} vol={vol_s:.2f}",
        )
