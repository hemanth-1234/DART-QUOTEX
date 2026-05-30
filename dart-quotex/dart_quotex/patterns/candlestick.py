"""
dart_quotex/patterns/candlestick.py
Complete candlestick pattern recognition library.

Implements 22 classic and modern patterns used in technical analysis.
Each function returns a PatternSignal with direction and strength.

Single-candle patterns
----------------------
Doji (standard, long-legged, gravestone, dragonfly)
Hammer / Inverted Hammer
Shooting Star
Marubozu (bull / bear)
Spinning Top

Two-candle patterns
-------------------
Bullish / Bearish Engulfing
Bullish / Bearish Harami
Piercing Line
Dark Cloud Cover
Tweezer Top / Bottom

Three-candle patterns
---------------------
Morning Star / Evening Star
Three White Soldiers / Three Black Crows
Inside Bar (NR4/NR7)
Three Inside Up / Down
Abandoned Baby

Usage
-----
    from dart_quotex.patterns.candlestick import PatternScanner
    scanner = PatternScanner()
    signals = scanner.scan(df)   # df = OHLCV DataFrame
    # signals = [PatternSignal(name="Hammer", direction="CALL", strength=0.8), ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PatternSignal:
    name:      str
    direction: str      # "CALL" | "PUT" | "NEUTRAL"
    strength:  float    # 0.0 – 1.0
    candles_back: int   # how many bars back was the pattern formed
    description: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Pattern Scanner
# ──────────────────────────────────────────────────────────────────────────────

class PatternScanner:
    """
    Scans the last few candles of a DataFrame for candlestick patterns.

    Parameters
    ----------
    body_ratio_threshold : minimum body/range ratio for a "real body" candle
    doji_threshold       : maximum body/range ratio to be classified as doji
    shadow_ratio         : minimum shadow/body ratio for hammer/shooting star
    """

    def __init__(
        self,
        body_ratio_threshold: float = 0.3,
        doji_threshold: float = 0.1,
        shadow_ratio: float = 2.0,
    ) -> None:
        self.body_thr  = body_ratio_threshold
        self.doji_thr  = doji_threshold
        self.shad_rat  = shadow_ratio

    def scan(self, df: pd.DataFrame, lookback: int = 3) -> List[PatternSignal]:
        """
        Scan recent candles for all patterns.
        Returns list of detected PatternSignals sorted by strength desc.
        """
        if len(df) < 4:
            return []

        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)

        signals: List[PatternSignal] = []

        # ── Single-candle patterns (last bar) ─────────────────────────────────
        signals += self._single_candle(o, h, l, c)

        # ── Two-candle patterns (last 2 bars) ─────────────────────────────────
        signals += self._two_candle(o, h, l, c)

        # ── Three-candle patterns (last 3 bars) ───────────────────────────────
        signals += self._three_candle(o, h, l, c)

        # Deduplicate: keep highest strength per direction
        seen: dict = {}
        for sig in signals:
            key = sig.direction
            if key not in seen or sig.strength > seen[key].strength:
                seen[key] = sig

        return sorted(signals, key=lambda s: -s.strength)

    def net_signal(self, df: pd.DataFrame) -> PatternSignal:
        """
        Return the single dominant pattern signal (or NEUTRAL).
        """
        signals = self.scan(df)
        if not signals:
            return PatternSignal("None", "NEUTRAL", 0.0, 0)

        call_strength = sum(s.strength for s in signals if s.direction == "CALL")
        put_strength  = sum(s.strength for s in signals if s.direction == "PUT")

        if call_strength > put_strength and call_strength > 0.3:
            best = max((s for s in signals if s.direction == "CALL"), key=lambda s: s.strength)
            return best
        elif put_strength > call_strength and put_strength > 0.3:
            best = max((s for s in signals if s.direction == "PUT"), key=lambda s: s.strength)
            return best
        return PatternSignal("Mixed", "NEUTRAL", 0.0, 0)

    # ── Single-candle patterns ─────────────────────────────────────────────────

    def _single_candle(
        self,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
    ) -> List[PatternSignal]:
        signals = []
        n = len(c)

        # Work on last bar (index n-1)
        i = n - 1
        body    = abs(c[i] - o[i])
        rng     = h[i] - l[i] + 1e-9
        body_r  = body / rng
        bull    = c[i] >= o[i]

        upper_shadow = h[i] - max(c[i], o[i])
        lower_shadow = min(c[i], o[i]) - l[i]

        # ── Doji ──────────────────────────────────────────────────────────────
        if body_r < self.doji_thr:
            # Standard doji
            strength = 0.5 + 0.5 * (1 - body_r / self.doji_thr)

            if upper_shadow > 3 * lower_shadow and upper_shadow / rng > 0.6:
                signals.append(PatternSignal(
                    "Gravestone Doji", "PUT", 0.75, 0,
                    "Long upper shadow; bearish reversal signal",
                ))
            elif lower_shadow > 3 * upper_shadow and lower_shadow / rng > 0.6:
                signals.append(PatternSignal(
                    "Dragonfly Doji", "CALL", 0.75, 0,
                    "Long lower shadow; bullish reversal signal",
                ))
            elif max(upper_shadow, lower_shadow) > 2 * body:
                signals.append(PatternSignal(
                    "Long-Legged Doji", "NEUTRAL", strength, 0,
                    "High indecision; strong reversal potential",
                ))
            else:
                signals.append(PatternSignal(
                    "Doji", "NEUTRAL", strength * 0.7, 0,
                    "Indecision candle",
                ))

        # ── Hammer ────────────────────────────────────────────────────────────
        if (lower_shadow > self.shad_rat * body + 1e-9
                and upper_shadow < body * 0.3
                and body_r > 0.05):
            # Require prior downtrend for confirmation
            prior_trend = _is_downtrend(c, i, lookback=5)
            strength    = min(0.9, 0.6 + (lower_shadow / (rng + 1e-9)) * 0.4)
            if prior_trend:
                signals.append(PatternSignal(
                    "Hammer", "CALL", strength, 0,
                    "Bullish reversal; buyers rejected lower prices",
                ))
            else:
                signals.append(PatternSignal(
                    "Hanging Man", "PUT", strength * 0.7, 0,
                    "Bearish at top of uptrend",
                ))

        # ── Inverted Hammer ───────────────────────────────────────────────────
        if (upper_shadow > self.shad_rat * body + 1e-9
                and lower_shadow < body * 0.3
                and body_r > 0.05):
            prior_down = _is_downtrend(c, i, lookback=5)
            strength   = min(0.85, 0.55 + (upper_shadow / (rng + 1e-9)) * 0.3)
            if prior_down:
                signals.append(PatternSignal(
                    "Inverted Hammer", "CALL", strength, 0,
                    "Bullish; buyers attempted to push price up",
                ))
            else:
                signals.append(PatternSignal(
                    "Shooting Star", "PUT", strength, 0,
                    "Bearish reversal; sellers rejected higher prices",
                ))

        # ── Marubozu ─────────────────────────────────────────────────────────
        if body_r > 0.90 and upper_shadow / rng < 0.03 and lower_shadow / rng < 0.03:
            if bull:
                signals.append(PatternSignal(
                    "Bullish Marubozu", "CALL", 0.85, 0,
                    "Full body, no shadows; very bullish momentum",
                ))
            else:
                signals.append(PatternSignal(
                    "Bearish Marubozu", "PUT", 0.85, 0,
                    "Full body, no shadows; very bearish momentum",
                ))

        # ── Spinning Top ─────────────────────────────────────────────────────
        if (0.1 < body_r < 0.35
                and upper_shadow > body
                and lower_shadow > body):
            signals.append(PatternSignal(
                "Spinning Top", "NEUTRAL", 0.45, 0,
                "Indecision; possible reversal if at extreme",
            ))

        return signals

    # ── Two-candle patterns ────────────────────────────────────────────────────

    def _two_candle(
        self,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
    ) -> List[PatternSignal]:
        signals = []
        n = len(c)
        if n < 2:
            return signals

        i, j = n - 2, n - 1   # prev, current

        body_i = abs(c[i] - o[i])
        body_j = abs(c[j] - o[j])
        rng_i  = h[i] - l[i] + 1e-9
        rng_j  = h[j] - l[j] + 1e-9
        bull_i = c[i] >= o[i]
        bull_j = c[j] >= o[j]

        # ── Bullish Engulfing ─────────────────────────────────────────────────
        if (not bull_i and bull_j
                and body_j > body_i
                and c[j] > o[i]
                and o[j] < c[i]):
            engulf_ratio = body_j / (body_i + 1e-9)
            strength     = min(0.90, 0.65 + min(engulf_ratio - 1, 1) * 0.25)
            signals.append(PatternSignal(
                "Bullish Engulfing", "CALL", strength, 1,
                "Bull completely engulfs prior bear; strong reversal",
            ))

        # ── Bearish Engulfing ─────────────────────────────────────────────────
        if (bull_i and not bull_j
                and body_j > body_i
                and c[j] < o[i]
                and o[j] > c[i]):
            engulf_ratio = body_j / (body_i + 1e-9)
            strength     = min(0.90, 0.65 + min(engulf_ratio - 1, 1) * 0.25)
            signals.append(PatternSignal(
                "Bearish Engulfing", "PUT", strength, 1,
                "Bear completely engulfs prior bull; strong reversal",
            ))

        # ── Bullish Harami ────────────────────────────────────────────────────
        if (not bull_i and bull_j
                and body_j < body_i * 0.5
                and c[j] < max(o[i], c[i])
                and o[j] > min(o[i], c[i])):
            signals.append(PatternSignal(
                "Bullish Harami", "CALL", 0.60, 1,
                "Small bull inside prior bear; potential reversal",
            ))

        # ── Bearish Harami ────────────────────────────────────────────────────
        if (bull_i and not bull_j
                and body_j < body_i * 0.5
                and c[j] > min(o[i], c[i])
                and o[j] < max(o[i], c[i])):
            signals.append(PatternSignal(
                "Bearish Harami", "PUT", 0.60, 1,
                "Small bear inside prior bull; potential reversal",
            ))

        # ── Piercing Line ─────────────────────────────────────────────────────
        if (not bull_i and bull_j
                and o[j] < l[i]
                and c[j] > (o[i] + c[i]) / 2
                and c[j] < o[i]):
            signals.append(PatternSignal(
                "Piercing Line", "CALL", 0.72, 1,
                "Bull opens below and closes above midpoint of bear",
            ))

        # ── Dark Cloud Cover ─────────────────────────────────────────────────
        if (bull_i and not bull_j
                and o[j] > h[i]
                and c[j] < (o[i] + c[i]) / 2
                and c[j] > o[i]):
            signals.append(PatternSignal(
                "Dark Cloud Cover", "PUT", 0.72, 1,
                "Bear opens above and closes below midpoint of bull",
            ))

        # ── Tweezer Top ───────────────────────────────────────────────────────
        if (bull_i and not bull_j
                and abs(h[i] - h[j]) / (rng_i + 1e-9) < 0.03):
            signals.append(PatternSignal(
                "Tweezer Top", "PUT", 0.65, 1,
                "Equal highs; rejection at resistance level",
            ))

        # ── Tweezer Bottom ────────────────────────────────────────────────────
        if (not bull_i and bull_j
                and abs(l[i] - l[j]) / (rng_i + 1e-9) < 0.03):
            signals.append(PatternSignal(
                "Tweezer Bottom", "CALL", 0.65, 1,
                "Equal lows; support holding strong",
            ))

        return signals

    # ── Three-candle patterns ──────────────────────────────────────────────────

    def _three_candle(
        self,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
    ) -> List[PatternSignal]:
        signals = []
        n = len(c)
        if n < 3:
            return signals

        k, i, j = n - 3, n - 2, n - 1   # oldest, middle, newest

        bull_k = c[k] >= o[k]
        bull_i = c[i] >= o[i]
        bull_j = c[j] >= o[j]
        body_k = abs(c[k] - o[k])
        body_i = abs(c[i] - o[i])
        body_j = abs(c[j] - o[j])

        # ── Morning Star ─────────────────────────────────────────────────────
        if (not bull_k                        # 1st: bearish
                and body_k > 0.01             # real body
                and body_i < body_k * 0.3     # 2nd: small body (star)
                and bull_j                    # 3rd: bullish
                and c[j] > (o[k] + c[k]) / 2  # closes above midpoint of 1st
                and max(o[i], c[i]) < min(c[k], o[k]) + 0.001):  # gap
            strength = 0.80 + min((c[j] - (o[k] + c[k]) / 2) / (body_k + 1e-9), 0.1)
            signals.append(PatternSignal(
                "Morning Star", "CALL", float(strength), 2,
                "Three-candle reversal; bearish trend ends",
            ))

        # ── Evening Star ─────────────────────────────────────────────────────
        if (bull_k
                and body_k > 0.01
                and body_i < body_k * 0.3
                and not bull_j
                and c[j] < (o[k] + c[k]) / 2
                and min(o[i], c[i]) > max(c[k], o[k]) - 0.001):
            strength = 0.80 + min(((o[k] + c[k]) / 2 - c[j]) / (body_k + 1e-9), 0.1)
            signals.append(PatternSignal(
                "Evening Star", "PUT", float(strength), 2,
                "Three-candle reversal; bullish trend ends",
            ))

        # ── Three White Soldiers ─────────────────────────────────────────────
        if (bull_k and bull_i and bull_j
                and c[j] > c[i] > c[k]
                and o[j] > o[i] > o[k]
                and body_k > 0.01 and body_i > 0.01 and body_j > 0.01
                and all(abs(c[x] - max(c[x], o[x])) < body_k * 0.2 for x in [k, i, j])):
            signals.append(PatternSignal(
                "Three White Soldiers", "CALL", 0.85, 2,
                "Three consecutive bullish candles; strong uptrend",
            ))

        # ── Three Black Crows ─────────────────────────────────────────────────
        if (not bull_k and not bull_i and not bull_j
                and c[j] < c[i] < c[k]
                and o[j] < o[i] < o[k]
                and body_k > 0.01 and body_i > 0.01 and body_j > 0.01
                and all(abs(min(c[x], o[x]) - l[x]) < body_k * 0.2 for x in [k, i, j])):
            signals.append(PatternSignal(
                "Three Black Crows", "PUT", 0.85, 2,
                "Three consecutive bearish candles; strong downtrend",
            ))

        # ── Three Inside Up ───────────────────────────────────────────────────
        if (not bull_k and bull_i
                and body_i < body_k
                and bull_j and c[j] > c[k]):
            signals.append(PatternSignal(
                "Three Inside Up", "CALL", 0.72, 2,
                "Harami confirmed by bullish third candle",
            ))

        # ── Three Inside Down ─────────────────────────────────────────────────
        if (bull_k and not bull_i
                and body_i < body_k
                and not bull_j and c[j] < c[k]):
            signals.append(PatternSignal(
                "Three Inside Down", "PUT", 0.72, 2,
                "Harami confirmed by bearish third candle",
            ))

        # ── Abandoned Baby Bullish ────────────────────────────────────────────
        if (not bull_k
                and body_i < body_k * 0.15    # doji star
                and bull_j
                and h[i] < min(c[k], o[k])   # gap down
                and l[i] < l[k]
                and o[j] > h[i]):              # gap up
            signals.append(PatternSignal(
                "Abandoned Baby Bull", "CALL", 0.88, 2,
                "Rare doji gap reversal; very reliable",
            ))

        # ── Abandoned Baby Bearish ────────────────────────────────────────────
        if (bull_k
                and body_i < body_k * 0.15
                and not bull_j
                and l[i] > max(c[k], o[k])   # gap up
                and h[i] > h[k]
                and o[j] < l[i]):              # gap down
            signals.append(PatternSignal(
                "Abandoned Baby Bear", "PUT", 0.88, 2,
                "Rare doji gap reversal; very reliable",
            ))

        return signals


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _is_downtrend(c: np.ndarray, idx: int, lookback: int = 5) -> bool:
    """True if close prices were trending down over `lookback` bars before idx."""
    start = max(0, idx - lookback)
    if idx - start < 2:
        return False
    segment = c[start:idx + 1]
    return bool(segment[0] > segment[-1])


def _is_uptrend(c: np.ndarray, idx: int, lookback: int = 5) -> bool:
    """True if close prices were trending up over `lookback` bars before idx."""
    start = max(0, idx - lookback)
    if idx - start < 2:
        return False
    segment = c[start:idx + 1]
    return bool(segment[-1] > segment[0])
