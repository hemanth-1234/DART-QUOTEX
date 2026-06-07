"""
dart_quotex/signals/manipulation.py
Feature 4 — AIMM-X Market Manipulation Detection
=================================================
Transparent, explainable statistical scoring of price action integrity.

Five sub-scores are combined into one Integrity Score (0 = suspicious,
1 = clean). When the score drops below a dynamic rolling threshold the
module flags the window and instructs the risk manager to reduce size or
skip the trade entirely.

Sub-scores
----------
1. Return Distribution Normality
   Benford-like check — large kurtosis or skew spikes indicate
   non-natural price moves.

2. Volatility Clustering Deviation
   Normal markets show ARCH-like volatility clustering.  Sudden
   volatility regime jumps without catalyst suggest manipulation.

3. Spread / Body Compression
   When candle bodies become unnaturally small for many bars in a row
   (artificially compressed range) a "spring" / manipulation event
   is likely being set up.

4. Volume Anomaly
   Volume z-score versus its rolling mean.  Extreme spikes or
   complete absence of volume (when you'd expect it) are anomalies.

5. Price Autocorrelation Break
   Natural prices exhibit weak mean-reversion autocorrelation.
   Persistent one-directional autocorrelation at unusual magnitude
   suggests tick manipulation.

Integration
-----------
In risk/manager.py, call:
    from dart_quotex.signals.manipulation import ManipulationDetector
    detector = ManipulationDetector()
    result = detector.score(df)
    if result.warning_flag:
        # reduce stake or skip
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ManipulationResult:
    integrity_score:    float               # 0 (suspicious) – 1 (clean)
    warning_flag:       bool                # True if below dynamic threshold
    sub_scores:         Dict[str, float] = field(default_factory=dict)
    anomaly_details:    List[str]         = field(default_factory=list)
    recommended_action: str = "NORMAL"    # "NORMAL" | "REDUCE_SIZE" | "SKIP"

    def __str__(self) -> str:
        subs = "  ".join(f"{k}={v:.2f}" for k, v in self.sub_scores.items())
        return (
            f"IntegrityScore={self.integrity_score:.3f}  "
            f"Flag={self.warning_flag}  "
            f"Action={self.recommended_action}  "
            f"[{subs}]"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Individual sub-score calculators
# ──────────────────────────────────────────────────────────────────────────────

def _score_return_distribution(c: np.ndarray, window: int = 20) -> float:
    """
    Score 1: Is the return distribution in the last `window` bars
    consistent with a normal market?

    Normal market: kurtosis 0-5, skew -1 to +1
    Suspicious: extreme kurtosis (>10) or extreme skew (>3)
    """
    if len(c) < window + 1:
        return 0.5
    rets = np.diff(c[-window - 1:]) / (c[-(window + 1):-1] + 1e-9)
    if len(rets) < 5:
        return 0.5
    k    = float(scipy_stats.kurtosis(rets))
    skew = float(abs(scipy_stats.skew(rets)))

    k_score    = max(0.0, 1.0 - (max(0, k - 5) / 10.0))
    skew_score = max(0.0, 1.0 - (max(0, skew - 1) / 3.0))
    return float((k_score + skew_score) / 2)


def _score_volatility_clustering(c: np.ndarray, window: int = 30) -> float:
    """
    Score 2: Volatility clustering (ARCH effect check).
    Normal markets show positive autocorrelation in squared returns.
    Sudden regime shifts in volatility → anomaly.
    """
    if len(c) < window + 1:
        return 0.5
    rets    = np.diff(c[-(window + 1):]) / (c[-(window + 1):-1] + 1e-9)
    sq_rets = rets ** 2
    if len(sq_rets) < 5:
        return 0.5

    # Autocorrelation of squared returns at lag 1
    autocorr = float(pd.Series(sq_rets).autocorr(lag=1))
    # Negative autocorr = volatility non-clustering = artificial
    if autocorr < -0.3:
        return 0.3
    if autocorr < 0:
        return 0.5
    return min(1.0, 0.5 + autocorr)


def _score_body_compression(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, window: int = 20
) -> float:
    """
    Score 3: Candle body / range ratio.
    Consistently tiny bodies (< 5% of range) for many bars in a row
    is unnatural and may precede a manipulation spike.
    """
    if len(c) < window:
        return 0.5
    bodies = np.abs(c[-window:] - o[-window:])
    ranges = (h[-window:] - l[-window:]) + 1e-9
    ratios = bodies / ranges

    mean_r = float(ratios.mean())
    # Healthy: mean body ratio 20-60%
    if mean_r < 0.05:
        return 0.1    # extremely compressed — very suspicious
    if mean_r < 0.10:
        return 0.3
    if mean_r < 0.15:
        return 0.6
    return min(1.0, 0.5 + mean_r)


def _score_volume_anomaly(v: np.ndarray, window: int = 20) -> float:
    """
    Score 4: Volume z-score.
    Both extreme spikes (> +3σ) and abnormal absence (< -2σ) are anomalous.
    """
    if len(v) < window + 1 or v.sum() < 1e-9:
        return 0.5   # no volume data available
    recent  = v[-window:]
    history = v[-(window * 3):-window] if len(v) > window * 3 else v[:-1]
    if len(history) < 5:
        return 0.5

    mean_v, std_v = history.mean(), history.std() + 1e-9
    z_scores = (recent - mean_v) / std_v
    max_z    = float(np.abs(z_scores).max())

    if max_z > 5:
        return 0.1
    if max_z > 3:
        return 0.4
    if max_z > 2:
        return 0.7
    return 1.0


def _score_autocorrelation(c: np.ndarray, window: int = 30) -> float:
    """
    Score 5: Return autocorrelation.
    Efficient markets: autocorrelation near 0.
    Persistent strong positive autocorrelation = possible painting the tape.
    Persistent strong negative = possible whipsaw manipulation.
    """
    if len(c) < window + 1:
        return 0.5
    rets = pd.Series(np.diff(c[-(window + 1):]) / (c[-(window + 1):-1] + 1e-9))
    ac   = float(rets.autocorr(lag=1))
    if np.isnan(ac):
        return 0.5

    intensity = abs(ac)
    if intensity > 0.5:
        return 0.2
    if intensity > 0.3:
        return 0.5
    return min(1.0, 1.0 - intensity * 1.5)


# ──────────────────────────────────────────────────────────────────────────────
# ManipulationDetector — combines sub-scores + rolling threshold
# ──────────────────────────────────────────────────────────────────────────────

class ManipulationDetector:
    """
    Stateful detector that maintains a rolling history of integrity scores
    to produce a dynamic detection threshold.

    Parameters
    ----------
    window          : bars per scoring window (default 20)
    score_history   : how many past scores to keep for threshold calculation
    percentile      : anomaly threshold percentile (default 10 = lowest 10%)
    weights         : relative weight per sub-score
    skip_threshold  : if score < this, recommend SKIP (default 0.30)
    reduce_threshold: if score < this, recommend REDUCE_SIZE (default 0.50)
    """

    WEIGHTS = {
        "return_dist":     0.25,
        "vol_clustering":  0.20,
        "body_compression":0.25,
        "volume_anomaly":  0.15,
        "autocorrelation": 0.15,
    }

    def __init__(
        self,
        window:           int   = 20,
        score_history:    int   = 288,    # ~2 weeks of 1-min candles
        percentile:       float = 10.0,
        skip_threshold:   float = 0.30,
        reduce_threshold: float = 0.50,
    ) -> None:
        self.window           = window
        self.percentile       = percentile
        self.skip_threshold   = skip_threshold
        self.reduce_threshold = reduce_threshold
        self._history: Deque[float] = deque(maxlen=score_history)

    def score(self, df: pd.DataFrame) -> ManipulationResult:
        """
        Score the most recent `window` candles.

        Parameters
        ----------
        df : OHLCV DataFrame (must have at least window + 30 rows)

        Returns
        -------
        ManipulationResult
        """
        if len(df) < self.window + 5:
            return ManipulationResult(
                integrity_score=0.5,
                warning_flag=False,
                recommended_action="NORMAL",
                anomaly_details=["Insufficient data"],
            )

        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        v = df["volume"].values.astype(float)

        sub: Dict[str, float] = {
            "return_dist":      _score_return_distribution(c, self.window),
            "vol_clustering":   _score_volatility_clustering(c, self.window),
            "body_compression": _score_body_compression(o, h, l, c, self.window),
            "volume_anomaly":   _score_volume_anomaly(v, self.window),
            "autocorrelation":  _score_autocorrelation(c, self.window),
        }

        # Weighted combination
        integrity = float(sum(
            self.WEIGHTS[k] * v for k, v in sub.items()
        ))

        # Update rolling history
        self._history.append(integrity)

        # Dynamic threshold: bottom percentile of own history
        dynamic_threshold = (
            float(np.percentile(list(self._history), self.percentile))
            if len(self._history) >= 20
            else self.reduce_threshold
        )

        warning_flag = integrity < dynamic_threshold

        # Anomaly narration
        details: List[str] = []
        if sub["return_dist"]      < 0.4: details.append("Non-normal return distribution")
        if sub["vol_clustering"]   < 0.4: details.append("Volatility regime break")
        if sub["body_compression"] < 0.3: details.append("Abnormal body compression")
        if sub["volume_anomaly"]   < 0.4: details.append("Volume anomaly detected")
        if sub["autocorrelation"]  < 0.3: details.append("Persistent autocorrelation")

        # Recommended action
        if integrity < self.skip_threshold:
            action = "SKIP"
        elif integrity < self.reduce_threshold or warning_flag:
            action = "REDUCE_SIZE"
        else:
            action = "NORMAL"

        result = ManipulationResult(
            integrity_score=round(integrity, 4),
            warning_flag=warning_flag,
            sub_scores={k: round(v, 3) for k, v in sub.items()},
            anomaly_details=details,
            recommended_action=action,
        )

        if warning_flag:
            log.warning("AIMM-X flag: %s", result)

        return result

    def rolling_mean_score(self) -> float:
        """Average integrity score over the rolling history."""
        if not self._history:
            return 0.5
        return float(np.mean(list(self._history)))

    def is_clean(self, df: pd.DataFrame) -> bool:
        """Quick boolean check: True = market looks clean."""
        return self.score(df).recommended_action == "NORMAL"


# ──────────────────────────────────────────────────────────────────────────────
# Module 3: Specific pre-trade manipulation filter functions
# ──────────────────────────────────────────────────────────────────────────────

def wick_rejection_trap(df: pd.DataFrame, window: int = 5) -> float:
    """
    Detect high wick/body ratio (>3×) with close in opposite direction.
    A large wick followed by close back inside = fake breakout / stop hunt.
    Returns score 0-1 (1 = strong wick rejection trap detected).
    """
    if len(df) < 3:
        return 0.0

    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)

    scores = []
    for i in range(max(0, len(df) - window), len(df)):
        body  = abs(c[i] - o[i]) + 1e-9
        rng   = h[i] - l[i] + 1e-9
        upper = h[i] - max(c[i], o[i])
        lower = min(c[i], o[i]) - l[i]

        # Upper wick rejection: large upper wick + bearish close
        upper_ratio = upper / body
        if upper_ratio > 3 and c[i] < o[i]:
            scores.append(min(1.0, upper_ratio / 6))

        # Lower wick rejection: large lower wick + bullish close
        lower_ratio = lower / body
        if lower_ratio > 3 and c[i] > o[i]:
            scores.append(min(1.0, lower_ratio / 6))

    return float(max(scores)) if scores else 0.0


def volume_anomaly(df: pd.DataFrame, window: int = 20, spike_factor: float = 3.0) -> float:
    """
    Volume spike >3× rolling average but price change <0.05%.
    High volume with almost no price movement = quote stuffing / fake activity.
    Returns score 0-1.
    """
    if len(df) < window + 1:
        return 0.0

    v = df["volume"].values.astype(float)
    c = df["close"].values.astype(float)

    if v.sum() < 1e-6:
        return 0.0   # no volume data

    recent_vol = v[-1]
    avg_vol    = v[-window - 1:-1].mean() + 1e-9
    vol_ratio  = recent_vol / avg_vol

    if vol_ratio < spike_factor:
        return 0.0

    # Check price change
    price_change_pct = abs(c[-1] - c[-2]) / (c[-2] + 1e-9) * 100
    if price_change_pct >= 0.05:
        return 0.0   # price moved — volume is legitimate

    # Volume spike with no price movement
    score = min(1.0, (vol_ratio - spike_factor) / spike_factor)
    return float(score)


def fake_breakout(df: pd.DataFrame, resistance_window: int = 20) -> float:
    """
    Close above recent resistance (20-bar high) but the next candle
    closes back inside the prior range = fake breakout.
    Returns score 0-1.
    """
    if len(df) < resistance_window + 2:
        return 0.0

    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)

    # Prior range (exclude last 2 candles)
    prior_high = h[-(resistance_window + 2):-2].max()
    prior_low  = l[-(resistance_window + 2):-2].min()

    breakout_candle = c[-2]   # candle before most recent
    return_candle   = c[-1]   # most recent candle

    # Bullish fake breakout: broke above resistance, closed back below
    if breakout_candle > prior_high and return_candle < prior_high:
        extension = (breakout_candle - prior_high) / (prior_high + 1e-9)
        return float(min(1.0, 0.5 + extension * 100))

    # Bearish fake breakout: broke below support, closed back above
    if breakout_candle < prior_low and return_candle > prior_low:
        extension = (prior_low - breakout_candle) / (prior_low + 1e-9)
        return float(min(1.0, 0.5 + extension * 100))

    return 0.0


def manipulation_score(
    df: pd.DataFrame,
    weights: Optional[dict] = None,
) -> Tuple[float, str]:
    """
    Weighted average of all three manipulation sub-scores.

    Parameters
    ----------
    df      : OHLCV DataFrame
    weights : optional dict with keys wick/volume/breakout (default equal)

    Returns
    -------
    (score 0-1, description string)
    """
    w = weights or {"wick": 1/3, "volume": 1/3, "breakout": 1/3}

    wick  = wick_rejection_trap(df)
    vol   = volume_anomaly(df)
    fb    = fake_breakout(df)

    score = (
        w.get("wick",     1/3) * wick
        + w.get("volume", 1/3) * vol
        + w.get("breakout",1/3) * fb
    )

    parts = []
    if wick  > 0.3: parts.append(f"WickRejection={wick:.2f}")
    if vol   > 0.3: parts.append(f"VolumeAnomaly={vol:.2f}")
    if fb    > 0.3: parts.append(f"FakeBreakout={fb:.2f}")

    desc = ", ".join(parts) if parts else "Clean"
    return float(score), desc


# ──────────────────────────────────────────────────────────────────────────────
# Module 3: Three specific pre-trade manipulation filter functions
# (wick_rejection_trap, volume_anomaly, fake_breakout, manipulation_score)
# ──────────────────────────────────────────────────────────────────────────────

def wick_rejection_trap(df: pd.DataFrame, window: int = 5) -> float:
    """
    Detect high wick/body ratio (>3x) with close in opposite direction.
    A large wick followed by close back inside = fake breakout / stop hunt.
    Returns score 0-1 (1 = strong trap detected).
    """
    if len(df) < 3:
        return 0.0
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    scores = []
    for i in range(max(0, len(df) - window), len(df)):
        body  = abs(c[i] - o[i]) + 1e-9
        upper = h[i] - max(c[i], o[i])
        lower = min(c[i], o[i]) - l[i]
        if upper / body > 3 and c[i] < o[i]:
            scores.append(min(1.0, (upper / body) / 6.0))
        if lower / body > 3 and c[i] > o[i]:
            scores.append(min(1.0, (lower / body) / 6.0))
    return float(max(scores)) if scores else 0.0


def volume_anomaly(df: pd.DataFrame, window: int = 20, spike_factor: float = 3.0) -> float:
    """
    Volume spike >3x rolling average but price change <0.05%.
    High volume + almost no price movement = possible quote stuffing.
    Returns score 0-1.
    """
    if len(df) < window + 1:
        return 0.0
    v = df["volume"].values.astype(float)
    c = df["close"].values.astype(float)
    if v.sum() < 1e-6:
        return 0.0
    avg_vol = v[-window - 1:-1].mean() + 1e-9
    ratio   = v[-1] / avg_vol
    if ratio < spike_factor:
        return 0.0
    price_chg_pct = abs(c[-1] - c[-2]) / (c[-2] + 1e-9) * 100
    if price_chg_pct >= 0.05:
        return 0.0
    return float(min(1.0, (ratio - spike_factor) / spike_factor))


def fake_breakout(df: pd.DataFrame, resistance_window: int = 20) -> float:
    """
    Close above recent 20-bar high, then next candle closes back inside range.
    Same logic inverted for bearish fake breakout.
    Returns score 0-1.
    """
    if len(df) < resistance_window + 2:
        return 0.0
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    prior_high = h[-(resistance_window + 2):-2].max()
    prior_low  = l[-(resistance_window + 2):-2].min()
    prev, last = c[-2], c[-1]
    if prev > prior_high and last < prior_high:
        ext = (prev - prior_high) / (prior_high + 1e-9)
        return float(min(1.0, 0.5 + ext * 100))
    if prev < prior_low and last > prior_low:
        ext = (prior_low - prev) / (prior_low + 1e-9)
        return float(min(1.0, 0.5 + ext * 100))
    return 0.0


def manipulation_score(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, str]:
    """
    Weighted average of wick_rejection_trap, volume_anomaly, fake_breakout.

    Returns (score 0-1, human-readable description).
    Caller compares score to MANIPULATION_THRESHOLD (default 0.7).
    """
    w = weights or {"wick": 1/3, "volume": 1/3, "breakout": 1/3}
    wick = wick_rejection_trap(df)
    vol  = volume_anomaly(df)
    fb   = fake_breakout(df)
    score = (w.get("wick", 1/3) * wick
             + w.get("volume", 1/3) * vol
             + w.get("breakout", 1/3) * fb)
    parts = []
    if wick > 0.3: parts.append(f"WickTrap={wick:.2f}")
    if vol  > 0.3: parts.append(f"VolumeAnomaly={vol:.2f}")
    if fb   > 0.3: parts.append(f"FakeBreakout={fb:.2f}")
    return float(score), (", ".join(parts) or "Clean")
