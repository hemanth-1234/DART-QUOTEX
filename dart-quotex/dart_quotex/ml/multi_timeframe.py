"""
dart_quotex/ml/multi_timeframe.py
Multi-Timeframe Analysis (MTF).

Resamples base 1-minute OHLCV data to 5m, 15m, and 1h candles
and builds a consolidated feature vector that captures context
across all timeframes simultaneously.

Key insight
-----------
A 1-minute signal is far more reliable when higher timeframes align:
  - 1h trend says UP
  - 15m momentum confirms UP
  - 5m shows pullback to support
  - 1m gives entry trigger

Feature fusion strategies
-------------------------
1. Concatenation: [feat_1m | feat_5m | feat_15m | feat_1h]
   → full information, high dimensionality
2. Agreement Score: percentage of TFs agreeing on direction
3. Confluence Matrix: which TF combinations are aligned
4. Timeframe Divergence: spots when TFs disagree (reversal warning)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Timeframe definitions
# ──────────────────────────────────────────────────────────────────────────────

TIMEFRAMES: Dict[str, int] = {
    "1m":  1,
    "5m":  5,
    "15m": 15,
    "1h":  60,
}


# ──────────────────────────────────────────────────────────────────────────────
# MultiTimeframeAnalyzer
# ──────────────────────────────────────────────────────────────────────────────

class MultiTimeframeAnalyzer:
    """
    Builds multi-timeframe feature vectors from 1-minute base candles.

    Parameters
    ----------
    timeframes     : list of TF names (must be keys in TIMEFRAMES)
    base_gran      : base granularity in minutes (usually 1)
    lookback_bars  : bars of base TF to keep in memory
    """

    def __init__(
        self,
        timeframes: Optional[List[str]] = None,
        base_gran: int = 1,
        lookback_bars: int = 200,
    ) -> None:
        self.timeframes   = timeframes or ["1m", "5m", "15m", "1h"]
        self.base_gran    = base_gran
        self.lookback     = lookback_bars

        # Cached resampled DataFrames per TF
        self._cache: Dict[str, pd.DataFrame] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, df_1m: pd.DataFrame) -> None:
        """
        Update internal cache from the latest 1-minute DataFrame.

        Parameters
        ----------
        df_1m : DataFrame with columns open/high/low/close/volume,
                DatetimeIndex (UTC), 1-minute resolution.
        """
        df = df_1m.tail(self.lookback).copy()
        self._cache["1m"] = df

        for tf_name, minutes in TIMEFRAMES.items():
            if tf_name == "1m":
                continue
            if tf_name not in self.timeframes:
                continue
            resampled = self._resample(df, minutes)
            self._cache[tf_name] = resampled

    def features(self) -> np.ndarray:
        """
        Return consolidated MTF feature vector.
        Shape: (n_tf_features,)  — 13 features × n_timeframes
        """
        if not self._cache:
            return np.zeros(13 * len(self.timeframes), dtype=np.float32)

        parts = []
        for tf_name in self.timeframes:
            df = self._cache.get(tf_name)
            if df is None or len(df) < 5:
                parts.append(np.zeros(13, dtype=np.float32))
            else:
                parts.append(self._tf_features(df))

        vec = np.concatenate(parts, dtype=np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    def confluence(self) -> Tuple[float, str]:
        """
        Compute directional confluence score across all timeframes.

        Returns (score, direction) where:
          score     : fraction of TFs agreeing on direction (0–1)
          direction : "CALL", "PUT", or "NEUTRAL"
        """
        signals = []
        for tf_name in self.timeframes:
            df = self._cache.get(tf_name)
            if df is None or len(df) < 5:
                continue
            sig = self._direction_signal(df)
            signals.append(sig)

        if not signals:
            return 0.0, "NEUTRAL"

        calls = sum(1 for s in signals if s > 0)
        puts  = sum(1 for s in signals if s < 0)
        total = len(signals)

        if calls >= puts:
            score = calls / total
            direction = "CALL" if score > 0.5 else "NEUTRAL"
        else:
            score = puts / total
            direction = "PUT" if score > 0.5 else "NEUTRAL"

        return score, direction

    def divergence(self) -> float:
        """
        Return divergence index: how much do TFs disagree?
        High divergence → possible reversal, reduce confidence.
        0 = all agree, 1 = maximum disagreement.
        """
        signals = []
        for tf_name in self.timeframes:
            df = self._cache.get(tf_name)
            if df is None or len(df) < 5:
                continue
            signals.append(self._direction_signal(df))

        if len(signals) < 2:
            return 0.0

        arr = np.array(signals)
        # Normalised standard deviation of direction signals
        div = float(np.std(arr) / (np.abs(arr).mean() + 1e-8))
        return float(np.clip(div, 0.0, 1.0))

    def tf_summary(self) -> Dict[str, dict]:
        """Return per-timeframe RSI, trend, momentum summary."""
        summary = {}
        for tf_name in self.timeframes:
            df = self._cache.get(tf_name)
            if df is None or len(df) < 5:
                summary[tf_name] = {"trend": "N/A", "rsi": 50.0, "momentum": 0.0}
                continue

            c = df["close"].values.astype(float)
            rsi = float(_rsi_series(c, 14)[-1]) if len(c) >= 14 else 50.0
            momentum = float(c[-1] - c[-5]) / (c[-5] + 1e-9) * 100 if len(c) >= 5 else 0.0
            sma_short = c[-5:].mean()
            sma_long  = c[-20:].mean() if len(c) >= 20 else c.mean()
            trend = "UP" if sma_short > sma_long else "DOWN"

            summary[tf_name] = {
                "trend":    trend,
                "rsi":      round(rsi, 1),
                "momentum": round(momentum, 4),
            }

        return summary

    # ── resampling ────────────────────────────────────────────────────────────

    @staticmethod
    def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        """Resample 1m OHLCV DataFrame to N-minute bars."""
        rule = f"{minutes}min"
        agg = {
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }
        resampled = df.resample(rule, closed="left", label="left").agg(agg)
        resampled.dropna(how="all", inplace=True)
        return resampled

    # ── per-TF features (13 values) ───────────────────────────────────────────

    @staticmethod
    def _tf_features(df: pd.DataFrame) -> np.ndarray:
        """
        Extract 13 normalised features from a single TF DataFrame.
        Returns np.ndarray of shape (13,)
        """
        c = df["close"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        o = df["open"].values.astype(float)
        n = len(c)

        price = c[-1] + 1e-9

        # 1. 1-bar return
        ret1  = (c[-1] - c[-2]) / price if n >= 2 else 0.0
        # 2. 5-bar return
        ret5  = (c[-1] - c[-5]) / price if n >= 5 else 0.0
        # 3. RSI-14 normalised
        rsi   = float(_rsi_series(c, 14)[-1]) / 100.0 if n >= 15 else 0.5
        # 4. SMA crossover (fast-slow normalised by price)
        sma5  = c[-5:].mean() if n >= 5 else c[-1]
        sma20 = c[-20:].mean() if n >= 20 else c.mean()
        sma_x = (sma5 - sma20) / price
        # 5. Body-to-range ratio of last candle
        rng   = (h[-1] - l[-1]) + 1e-9
        body  = abs(c[-1] - o[-1])
        btr   = body / rng
        # 6. Upper wick fraction
        up_wick = (h[-1] - max(c[-1], o[-1])) / rng
        # 7. Lower wick fraction
        lo_wick = (min(c[-1], o[-1]) - l[-1]) / rng
        # 8. ATR normalised
        tr_arr = _true_range(h, l, c)
        atr    = tr_arr[-14:].mean() if len(tr_arr) >= 14 else tr_arr.mean()
        atr_n  = atr / price
        # 9. Bollinger band position (where is close relative to BB)
        if n >= 20:
            sma   = c[-20:].mean()
            std_  = c[-20:].std() + 1e-8
            bb_p  = (c[-1] - (sma - 2 * std_)) / (4 * std_)   # 0=lower, 0.5=mid, 1=upper
        else:
            bb_p  = 0.5
        # 10. MACD histogram sign and normalised value
        ema12 = _ema(c, 12)
        ema26 = _ema(c, 26)
        macd  = (ema12 - ema26) / price
        sig   = _ema_arr(np.array([macd]), 9)
        macd_h = macd - (sig[-1] if len(sig) else 0)
        # 11. Volume z-score (if available)
        v = df["volume"].values.astype(float)
        if v.sum() > 0 and n >= 10:
            vm = v[-10:].mean()
            vs = v[-10:].std() + 1e-8
            vol_z = float(np.clip((v[-1] - vm) / vs, -3, 3))
        else:
            vol_z = 0.0
        # 12. Trend direction binary (+1/-1)
        trend = 1.0 if sma5 > sma20 else -1.0
        # 13. Momentum (5-bar ROC)
        mom = float(c[-1] - c[-5]) / price if n >= 5 else 0.0

        return np.array(
            [ret1, ret5, rsi, sma_x, btr, up_wick, lo_wick,
             atr_n, bb_p, float(macd_h), vol_z, trend, mom],
            dtype=np.float32,
        )

    @staticmethod
    def _direction_signal(df: pd.DataFrame) -> float:
        """
        Return a direction signal in [-1, +1].
        +1 = bullish, -1 = bearish, 0 = neutral
        """
        c = df["close"].values.astype(float)
        if len(c) < 5:
            return 0.0

        sma5  = c[-5:].mean()
        sma20 = c[-20:].mean() if len(c) >= 20 else c.mean()
        rsi   = float(_rsi_series(c, 14)[-1]) if len(c) >= 15 else 50.0

        trend_score = 1.0 if sma5 > sma20 else -1.0
        rsi_score   = 1.0 if rsi < 50 else -1.0
        ret_score   = 1.0 if c[-1] > c[-2] else -1.0

        return float((trend_score + rsi_score + ret_score) / 3.0)


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _rsi_series(c: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(c)
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = np.convolve(gain, np.ones(period) / period, mode="valid")
    avg_l = np.convolve(loss, np.ones(period) / period, mode="valid")
    rs  = avg_g / (avg_l + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return np.clip(rsi, 0, 100)


def _ema(c: np.ndarray, period: int) -> float:
    if len(c) == 0:
        return 0.0
    alpha = 2.0 / (period + 1)
    v = float(c[0])
    for x in c[1:]:
        v = alpha * x + (1 - alpha) * v
    return v


def _ema_arr(c: np.ndarray, period: int) -> np.ndarray:
    if len(c) == 0:
        return np.array([0.0])
    alpha = 2.0 / (period + 1)
    result = [float(c[0])]
    for x in c[1:]:
        result.append(alpha * x + (1 - alpha) * result[-1])
    return np.array(result)


def _true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    if len(h) < 2:
        return h - l
    return np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )
