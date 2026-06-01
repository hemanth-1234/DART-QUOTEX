"""
dart_quotex/smc/indicators.py
Smart Money Concepts (SMC / ICT) indicator suite.

Detects:
  · Unmitigated Order Blocks (bullish + bearish)
  · Fair Value Gaps (FVG)
  · Liquidity Sweeps (equal highs/lows grabbed)
  · Break of Structure (BOS) / Change of Character (CHoCH)
  · Premium / Discount zones

All functions accept a pandas DataFrame with columns:
    open, high, low, close  (index = DatetimeIndex or RangeIndex)

Returns are always new columns added to a copy of the input DataFrame,
or scalar / list values depending on the function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderBlock:
    idx: int              # index in DataFrame
    direction: str        # "bull" | "bear"
    top: float
    bottom: float
    mitigated: bool = False

@dataclass
class FVG:
    idx: int              # middle candle index
    direction: str        # "bull" | "bear"
    top: float
    bottom: float
    filled: bool = False

@dataclass
class LiquiditySweep:
    idx: int
    direction: str        # "buy_side" (sweep above) | "sell_side" (sweep below)
    level: float


# ──────────────────────────────────────────────────────────────────────────────
# Order Blocks
# ──────────────────────────────────────────────────────────────────────────────

def find_order_blocks(
    df: pd.DataFrame,
    lookback: int = 5,
    keep_unmitigated: bool = True,
) -> List[OrderBlock]:
    """
    Identify the most recent unmitigated order blocks.

    A **bullish OB** is the last bearish candle before a strong bullish impulse.
    A **bearish OB** is the last bullish candle before a strong bearish impulse.
    """
    obs: List[OrderBlock] = []
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    for i in range(1, n - lookback):
        # Bullish OB: bearish candle followed by bullish displacement
        if closes[i] < opens[i]:    # bearish candle at i
            # Check if subsequent candles break upward
            impulse_high = max(highs[i + 1 : i + lookback + 1])
            if impulse_high > highs[i] * 1.0005:   # small threshold
                ob = OrderBlock(
                    idx=i,
                    direction="bull",
                    top=max(opens[i], closes[i]),
                    bottom=min(opens[i], closes[i]),
                )
                # Check mitigation: price revisited and closed inside
                if keep_unmitigated:
                    mitigated = _check_ob_mitigated(ob, df, i)
                    ob.mitigated = mitigated
                    if not mitigated:
                        obs.append(ob)
                else:
                    obs.append(ob)

        # Bearish OB: bullish candle followed by bearish displacement
        elif closes[i] > opens[i]:  # bullish candle at i
            impulse_low = min(lows[i + 1 : i + lookback + 1])
            if impulse_low < lows[i] * 0.9995:
                ob = OrderBlock(
                    idx=i,
                    direction="bear",
                    top=max(opens[i], closes[i]),
                    bottom=min(opens[i], closes[i]),
                )
                if keep_unmitigated:
                    mitigated = _check_ob_mitigated(ob, df, i)
                    ob.mitigated = mitigated
                    if not mitigated:
                        obs.append(ob)
                else:
                    obs.append(ob)

    return obs[-10:]   # return last 10 to avoid memory blow-up


def _check_ob_mitigated(ob: OrderBlock, df: pd.DataFrame, ob_idx: int) -> bool:
    """True if price re-entered the OB zone after it was formed."""
    subsequent = df.iloc[ob_idx + 1 :]
    if ob.direction == "bull":
        return bool((subsequent["low"] <= ob.top).any())
    else:
        return bool((subsequent["high"] >= ob.bottom).any())


# ──────────────────────────────────────────────────────────────────────────────
# Fair Value Gaps
# ──────────────────────────────────────────────────────────────────────────────

def find_fvg(df: pd.DataFrame) -> List[FVG]:
    """
    Detect Fair Value Gaps (imbalances between candle i-1 and i+1).

    Bullish FVG: low[i+1] > high[i-1]  (gap upward)
    Bearish FVG: high[i+1] < low[i-1]  (gap downward)
    """
    fvgs: List[FVG] = []
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    for i in range(1, n - 1):
        # Bullish FVG
        if lows[i + 1] > highs[i - 1]:
            fvgs.append(FVG(
                idx=i,
                direction="bull",
                top=lows[i + 1],
                bottom=highs[i - 1],
            ))
        # Bearish FVG
        elif highs[i + 1] < lows[i - 1]:
            fvgs.append(FVG(
                idx=i,
                direction="bear",
                top=lows[i - 1],
                bottom=highs[i + 1],
            ))

    # Mark filled FVGs
    closes = df["close"].values
    for fvg in fvgs:
        sub = closes[fvg.idx + 2 :]
        if fvg.direction == "bull":
            fvg.filled = bool((sub <= fvg.top).any())
        else:
            fvg.filled = bool((sub >= fvg.bottom).any())

    return [f for f in fvgs if not f.filled][-8:]


# ──────────────────────────────────────────────────────────────────────────────
# Liquidity Sweeps
# ──────────────────────────────────────────────────────────────────────────────

def find_liquidity_sweeps(
    df: pd.DataFrame,
    swing_lookback: int = 10,
    tolerance: float = 0.0003,
) -> List[LiquiditySweep]:
    """
    Detect when price briefly exceeds a swing high/low before reversing
    (classic stop-hunt / liquidity grab pattern).
    """
    sweeps: List[LiquiditySweep] = []
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    for i in range(swing_lookback, n - 1):
        window_high = max(highs[i - swing_lookback : i])
        window_low = min(lows[i - swing_lookback : i])

        # Buy-side liquidity sweep: wick above swing high, close back below
        if highs[i] > window_high * (1 + tolerance) and closes[i] < window_high:
            sweeps.append(LiquiditySweep(idx=i, direction="buy_side", level=window_high))

        # Sell-side liquidity sweep: wick below swing low, close back above
        elif lows[i] < window_low * (1 - tolerance) and closes[i] > window_low:
            sweeps.append(LiquiditySweep(idx=i, direction="sell_side", level=window_low))

    return sweeps[-5:]


# ──────────────────────────────────────────────────────────────────────────────
# Break of Structure / Change of Character
# ──────────────────────────────────────────────────────────────────────────────

def detect_bos_choch(df: pd.DataFrame, swing_n: int = 5) -> pd.Series:
    """
    Returns a Series with values:
      +1 = BOS bullish (price broke last swing high — trend continuation)
      -1 = BOS bearish
      +2 = CHoCH bullish (reversal from down to up trend)
      -2 = CHoCH bearish
       0 = nothing
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)
    result = np.zeros(n, dtype=int)

    last_swing_high = highs[:swing_n].max()
    last_swing_low = lows[:swing_n].min()
    trend = 0   # +1 = up, -1 = down

    for i in range(swing_n, n):
        c = closes[i]
        if c > last_swing_high:
            if trend == -1:
                result[i] = 2    # CHoCH bullish
            else:
                result[i] = 1    # BOS bullish
            trend = 1
            last_swing_high = max(highs[max(0, i - swing_n) : i + 1])
        elif c < last_swing_low:
            if trend == 1:
                result[i] = -2   # CHoCH bearish
            else:
                result[i] = -1   # BOS bearish
            trend = -1
            last_swing_low = min(lows[max(0, i - swing_n) : i + 1])

    return pd.Series(result, index=df.index, name="bos_choch")


# ──────────────────────────────────────────────────────────────────────────────
# Premium / Discount
# ──────────────────────────────────────────────────────────────────────────────

def premium_discount(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    Returns position within the recent range: >0.5 = premium, <0.5 = discount.
    """
    rng_high = df["high"].rolling(lookback).max()
    rng_low = df["low"].rolling(lookback).min()
    mid = (rng_high + rng_low) / 2
    rng = rng_high - rng_low
    pos = (df["close"] - rng_low) / rng.replace(0, np.nan)
    return pos.fillna(0.5).rename("pd_zone")


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: add all SMC columns to a DataFrame
# ──────────────────────────────────────────────────────────────────────────────

def add_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add scalar SMC feature columns to df (in-place copy).
    Suitable for passing to the ML feature pipeline.
    """
    df = df.copy()

    # BOS/CHoCH
    df["bos_choch"] = detect_bos_choch(df)

    # Premium/discount zone
    df["pd_zone"] = premium_discount(df)

    # Nearest active OB distance (as fraction of ATR)
    atr = (df["high"] - df["low"]).rolling(14).mean().fillna(0.001)
    obs = find_order_blocks(df)
    if obs:
        last_close = df["close"].iloc[-1]
        bull_obs = [o for o in obs if o.direction == "bull"]
        bear_obs = [o for o in obs if o.direction == "bear"]

        if bull_obs:
            nearest_bull = min(bull_obs, key=lambda o: abs(last_close - o.top))
            dist_bull = (last_close - nearest_bull.top) / atr.iloc[-1]
        else:
            dist_bull = 999.0

        if bear_obs:
            nearest_bear = min(bear_obs, key=lambda o: abs(last_close - o.bottom))
            dist_bear = (nearest_bear.bottom - last_close) / atr.iloc[-1]
        else:
            dist_bear = 999.0

        df["ob_bull_dist"] = dist_bull
        df["ob_bear_dist"] = dist_bear
    else:
        df["ob_bull_dist"] = 999.0
        df["ob_bear_dist"] = 999.0

    # FVG presence (1 = unfilled bull, -1 = unfilled bear, 0 = none)
    fvgs = find_fvg(df)
    if fvgs:
        last_fvg = fvgs[-1]
        df["fvg_signal"] = 1 if last_fvg.direction == "bull" else -1
    else:
        df["fvg_signal"] = 0

    # Liquidity sweep (recent)
    sweeps = find_liquidity_sweeps(df)
    if sweeps:
        last_sweep = sweeps[-1]
        df["liq_sweep"] = 1 if last_sweep.direction == "buy_side" else -1
    else:
        df["liq_sweep"] = 0

    return df
