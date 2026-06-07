"""
dart_quotex/smc/advanced_ict.py
Feature 2 — Institutional Order Flow (ICT / SMC) Signal Engine
==============================================================
Multi-step, high-confidence signal pipeline:

  Step 1  Liquidity Sweep (LS)
          Price wicks beyond a recent swing high/low and closes back inside,
          indicating stop-runs above/below equal highs/lows.

  Step 2  Market Structure Shift (MSS) / Change of Character (CHoCH)
          After the sweep, a new pivot forms on the opposite side and a
          candle closes beyond it — confirming that smart money has reversed.

  Step 3  Fair Value Gap (FVG) / Inversion FVG (IFVG) entry trigger
          Once LS + MSS are confirmed, look for a 3-candle imbalance zone.
          An IFVG is a previously-filled FVG that now acts as new S/R.

All three must align for a HIGH confidence signal.
Partial alignment (LS + MSS only, or FVG alone) produces MEDIUM confidence.

Output
------
ICTSignal(direction, confidence, components, entry_zone_top, entry_zone_bot)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

@dataclass
class OrderBlock:
    """Unmitigated order block (bullish or bearish)."""
    idx:       int
    direction: str    # "bull" | "bear"
    top:       float
    bottom:    float
    mitigated: bool = False

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ICTSignal:
    direction:       str              # "CALL" | "PUT" | "NEUTRAL"
    confidence:      float            # 0.0 – 1.0
    components:      List[str]        # which patterns confirmed
    entry_zone_top:  float = 0.0      # upper edge of FVG/IFVG entry zone
    entry_zone_bot:  float = 0.0      # lower edge
    sweep_level:     float = 0.0      # the liquidity level that was taken
    mss_level:       float = 0.0      # the MSS pivot level
    description:     str   = ""


# ──────────────────────────────────────────────────────────────────────────────
# Swing detection helper
# ──────────────────────────────────────────────────────────────────────────────

def _swing_highs(h: np.ndarray, n: int = 3) -> np.ndarray:
    """Indices where highs[i] is a swing high (local max over ±n bars)."""
    swings = []
    for i in range(n, len(h) - n):
        if h[i] == h[i - n : i + n + 1].max():
            swings.append(i)
    return np.array(swings, dtype=int)


def _swing_lows(l: np.ndarray, n: int = 3) -> np.ndarray:
    """Indices where lows[i] is a swing low (local min over ±n bars)."""
    swings = []
    for i in range(n, len(l) - n):
        if l[i] == l[i - n : i + n + 1].min():
            swings.append(i)
    return np.array(swings, dtype=int)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Liquidity Sweep Detector
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LiquiditySweep:
    direction:  str    # "buy_side" (sweep above) | "sell_side" (sweep below)
    bar_idx:    int    # candle index where sweep occurred
    level:      float  # the swing high/low that was taken out
    close:      float  # close price after the sweep


def detect_liquidity_sweeps(
    df:              pd.DataFrame,
    swing_lookback:  int   = 10,
    swing_strength:  int   = 3,
    tolerance_ticks: float = 0.0003,
) -> List[LiquiditySweep]:
    """
    Scan for liquidity sweeps (stop hunts) in recent price data.

    A buy-side LS:  high[i] > recent swing high, but close[i] < swing high
    A sell-side LS: low[i]  < recent swing low,  but close[i] > swing low

    Parameters
    ----------
    df               : OHLCV DataFrame
    swing_lookback   : how many bars back to look for swing levels
    swing_strength   : how many bars either side to qualify a swing
    tolerance_ticks  : minimum wick extension beyond the level (fraction)
    """
    if len(df) < swing_lookback + swing_strength * 2 + 5:
        return []

    h  = df["high"].values.astype(float)
    l  = df["low"].values.astype(float)
    c  = df["close"].values.astype(float)
    n  = len(df)

    sh_idx = _swing_highs(h, swing_strength)
    sl_idx = _swing_lows(l,  swing_strength)

    sweeps: List[LiquiditySweep] = []

    # Only scan recent bars (last swing_lookback bars)
    scan_start = max(swing_strength * 2, n - swing_lookback)

    for i in range(scan_start, n):
        # ── Buy-side sweep: wick above recent swing high, close back below ────
        recent_sh = sh_idx[sh_idx < i - swing_strength]
        if len(recent_sh):
            level = h[recent_sh[-1]]
            extension = h[i] - level
            if extension > level * tolerance_ticks and c[i] < level:
                sweeps.append(LiquiditySweep(
                    direction="buy_side",
                    bar_idx=i,
                    level=level,
                    close=c[i],
                ))

        # ── Sell-side sweep: wick below recent swing low, close back above ───
        recent_sl = sl_idx[sl_idx < i - swing_strength]
        if len(recent_sl):
            level = l[recent_sl[-1]]
            extension = level - l[i]
            if extension > level * tolerance_ticks and c[i] > level:
                sweeps.append(LiquiditySweep(
                    direction="sell_side",
                    bar_idx=i,
                    level=level,
                    close=c[i],
                ))

    return sweeps


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Market Structure Shift / Change of Character
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MSSEvent:
    direction:  str    # "bullish_mss" | "bearish_mss"
    bar_idx:    int    # candle that closed beyond the pivot
    pivot_level: float # the pivot high/low that was broken
    confidence:  float # 0-1


def detect_mss(
    df:           pd.DataFrame,
    after_bar:    int,
    sweep_dir:    str,
    swing_n:      int = 3,
    max_look_fwd: int = 10,
) -> Optional[MSSEvent]:
    """
    After a liquidity sweep at `after_bar`, look forward `max_look_fwd`
    candles for a Market Structure Shift (Change of Character).

    After a buy-side sweep (sweep_dir="buy_side") → look for bearish MSS:
        price breaks below the most recent pre-sweep swing low.

    After a sell-side sweep (sweep_dir="sell_side") → look for bullish MSS:
        price breaks above the most recent pre-sweep swing high.
    """
    n = len(df)
    if after_bar >= n - 1:
        return None

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)

    look_end = min(n, after_bar + max_look_fwd + 1)

    if sweep_dir == "buy_side":
        # Looking for bearish MSS: close below recent swing low
        sl_idx = _swing_lows(l, swing_n)
        pre_sweep_sl = sl_idx[sl_idx < after_bar]
        if not len(pre_sweep_sl):
            return None
        pivot = l[pre_sweep_sl[-1]]

        for j in range(after_bar + 1, look_end):
            if c[j] < pivot:
                # Measure confidence by how decisively price broke
                extension = (pivot - c[j]) / (pivot + 1e-9)
                conf = float(min(1.0, 0.5 + extension * 200))
                return MSSEvent(
                    direction="bearish_mss",
                    bar_idx=j,
                    pivot_level=pivot,
                    confidence=conf,
                )

    else:  # sell_side sweep → bullish MSS
        sh_idx = _swing_highs(h, swing_n)
        pre_sweep_sh = sh_idx[sh_idx < after_bar]
        if not len(pre_sweep_sh):
            return None
        pivot = h[pre_sweep_sh[-1]]

        for j in range(after_bar + 1, look_end):
            if c[j] > pivot:
                extension = (c[j] - pivot) / (pivot + 1e-9)
                conf = float(min(1.0, 0.5 + extension * 200))
                return MSSEvent(
                    direction="bullish_mss",
                    bar_idx=j,
                    pivot_level=pivot,
                    confidence=conf,
                )

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Fair Value Gap (FVG) and Inversion FVG (IFVG) detector
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FVGZone:
    kind:       str    # "fvg" | "ifvg"
    direction:  str    # "bullish" | "bearish"
    top:        float
    bottom:     float
    bar_idx:    int    # middle candle index
    filled:     bool   = False


def detect_fvg_zones(
    df:          pd.DataFrame,
    after_bar:   int,
    direction:   str,          # "bullish" | "bearish"
    max_look_fwd: int = 15,
    min_size_pct: float = 0.0002,
) -> List[FVGZone]:
    """
    Scan forward from `after_bar` for FVG and IFVG zones in the expected
    trade direction.

    Bullish FVG : low[i+1] > high[i-1]  → gap upward (demand zone)
    Bearish FVG : high[i+1] < low[i-1]  → gap downward (supply zone)

    IFVG: a previously-identified FVG whose gap has been retested (filled)
          but price bounced from its midpoint — now acting as S/R.
    """
    n   = len(df)
    h   = df["high"].values.astype(float)
    l   = df["low"].values.astype(float)
    c   = df["close"].values.astype(float)
    end = min(n - 1, after_bar + max_look_fwd)

    zones: List[FVGZone] = []
    atr_proxy = float(np.mean(h[max(0, after_bar - 14):after_bar + 1]
                              - l[max(0, after_bar - 14):after_bar + 1]) + 1e-9)

    for i in range(max(1, after_bar), end):
        if i + 1 >= n:
            break

        if direction == "bullish":
            gap_bot = h[i - 1]
            gap_top = l[i + 1]
            if gap_top > gap_bot and (gap_top - gap_bot) > atr_proxy * min_size_pct:
                # Check if filled (price dipped into gap)
                sub   = l[i + 2 : min(n, i + 10)]
                filled = bool(len(sub) and sub.min() <= gap_top)
                kind   = "ifvg" if filled else "fvg"
                zones.append(FVGZone(
                    kind=kind, direction="bullish",
                    top=gap_top, bottom=gap_bot,
                    bar_idx=i, filled=filled,
                ))

        else:  # bearish
            gap_top = l[i - 1]
            gap_bot = h[i + 1]
            if gap_bot < gap_top and (gap_top - gap_bot) > atr_proxy * min_size_pct:
                sub    = h[i + 2 : min(n, i + 10)]
                filled = bool(len(sub) and sub.max() >= gap_bot)
                kind   = "ifvg" if filled else "fvg"
                zones.append(FVGZone(
                    kind=kind, direction="bearish",
                    top=gap_top, bottom=gap_bot,
                    bar_idx=i, filled=filled,
                ))

    return zones


# ──────────────────────────────────────────────────────────────────────────────
# Master ICT Signal Engine
# ──────────────────────────────────────────────────────────────────────────────

class ICTSignalEngine:
    """
    Combines LS → MSS → FVG/IFVG into one high-confidence signal.

    Parameters
    ----------
    swing_lookback  : bars to scan for swing levels
    swing_strength  : bars either side for swing qualification
    mss_lookforward : bars after sweep to find MSS
    fvg_lookforward : bars after MSS to find FVG/IFVG
    tolerance       : minimum wick extension fraction for LS
    min_confidence  : minimum confidence to emit a signal (default 0.65)
    """

    def __init__(
        self,
        swing_lookback:  int   = 20,
        swing_strength:  int   = 3,
        mss_lookforward: int   = 10,
        fvg_lookforward: int   = 15,
        tolerance:       float = 0.0002,
        min_confidence:  float = 0.65,
    ) -> None:
        self.swing_lookback  = swing_lookback
        self.swing_strength  = swing_strength
        self.mss_lookforward = mss_lookforward
        self.fvg_lookforward = fvg_lookforward
        self.tolerance       = tolerance
        self.min_confidence  = min_confidence

    def scan(self, df: pd.DataFrame) -> ICTSignal:
        """
        Run the full three-step scan on `df`.
        Returns the highest-confidence ICTSignal found, or a NEUTRAL signal.
        """
        if len(df) < self.swing_lookback + self.swing_strength * 2 + 10:
            return ICTSignal("NEUTRAL", 0.0, [], description="Insufficient data")

        candidates: List[ICTSignal] = []

        # Find all recent liquidity sweeps
        sweeps = detect_liquidity_sweeps(
            df,
            swing_lookback=self.swing_lookback,
            swing_strength=self.swing_strength,
            tolerance_ticks=self.tolerance,
        )

        for sweep in sweeps[-5:]:    # consider the 5 most recent sweeps
            sig = self._evaluate_sweep(df, sweep)
            if sig and sig.confidence >= self.min_confidence:
                candidates.append(sig)

        if not candidates:
            return ICTSignal("NEUTRAL", 0.0, [], description="No ICT setup found")

        # Return highest confidence signal
        best = max(candidates, key=lambda s: s.confidence)
        log.info(
            "ICT Signal: %s conf=%.2f components=%s",
            best.direction, best.confidence, best.components,
        )
        return best

    # ── internal ─────────────────────────────────────────────────────────────

    def _evaluate_sweep(
        self, df: pd.DataFrame, sweep: LiquiditySweep
    ) -> Optional[ICTSignal]:
        """Evaluate one liquidity sweep for MSS + FVG confirmation."""
        components = [f"LS_{sweep.direction.upper()}"]
        conf       = 0.40   # base confidence for LS alone

        # Map sweep direction to trade direction
        # buy_side sweep → bearish trade (PUT) — smart money selling
        # sell_side sweep → bullish trade (CALL) — smart money buying
        if sweep.direction == "buy_side":
            trade_dir  = "PUT"
            fvg_dir    = "bearish"
        else:
            trade_dir  = "CALL"
            fvg_dir    = "bullish"

        # Step 2: MSS confirmation
        mss = detect_mss(
            df,
            after_bar=sweep.bar_idx,
            sweep_dir=sweep.direction,
            swing_n=self.swing_strength,
            max_look_fwd=self.mss_lookforward,
        )

        if mss is None:
            return ICTSignal(
                direction=trade_dir,
                confidence=conf,
                components=components,
                sweep_level=sweep.level,
                description="LS only — no MSS",
            )

        components.append(f"MSS_{mss.direction.upper()}")
        conf = 0.55 + mss.confidence * 0.15   # 0.55 – 0.70

        # Step 3: FVG / IFVG entry zone
        fvg_zones = detect_fvg_zones(
            df,
            after_bar=mss.bar_idx,
            direction=fvg_dir,
            max_look_fwd=self.fvg_lookforward,
        )

        if not fvg_zones:
            return ICTSignal(
                direction=trade_dir,
                confidence=conf,
                components=components,
                sweep_level=sweep.level,
                mss_level=mss.pivot_level,
                description="LS + MSS — no FVG trigger",
            )

        # Use most recent valid FVG/IFVG
        fvg  = fvg_zones[-1]
        kind = fvg.kind.upper()
        components.append(f"{kind}_{fvg.direction.upper()}")

        # IFVG is stronger confirmation than plain FVG
        fvg_bonus = 0.15 if fvg.kind == "ifvg" else 0.10
        conf      = min(0.95, conf + fvg_bonus)

        # Verify price is currently inside or approaching the FVG zone
        current_price = float(df["close"].iloc[-1])
        in_zone = fvg.bottom <= current_price <= fvg.top
        if in_zone:
            conf = min(0.98, conf + 0.05)
            components.append("PRICE_IN_ZONE")

        description = " + ".join(components)

        return ICTSignal(
            direction=trade_dir,
            confidence=conf,
            components=components,
            entry_zone_top=fvg.top,
            entry_zone_bot=fvg.bottom,
            sweep_level=sweep.level,
            mss_level=mss.pivot_level,
            description=description,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility aliases — used by smc/indicators.py and ml/features.py
# ──────────────────────────────────────────────────────────────────────────────

def _check_ob_mitigated(ob, df: "pd.DataFrame", ob_idx: int) -> bool:
    """True if price re-entered the order block zone after it was formed."""
    subsequent = df.iloc[ob_idx + 1:]
    if ob.direction == "bull":
        return bool((subsequent["low"] <= ob.top).any())
    else:
        return bool((subsequent["high"] >= ob.bottom).any())



def find_order_blocks(
    df: pd.DataFrame,
    lookback: int = 5,
    keep_unmitigated: bool = True,
) -> List[OrderBlock]:
    """
    Public alias — find unmitigated order blocks in `df`.
    Returns list of OrderBlock (direction, top, bottom, mitigated).
    """
    obs: List[OrderBlock] = []
    closes = df["close"].values.astype(float)
    opens  = df["open"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    n      = len(df)

    for i in range(1, n - lookback):
        # Bullish OB: bearish candle before upward impulse
        if closes[i] < opens[i]:
            if highs[i + 1 : i + lookback + 1].max() > highs[i] * 1.0005:
                ob = OrderBlock(
                    idx=i, direction="bull",
                    top=max(opens[i], closes[i]),
                    bottom=min(opens[i], closes[i]),
                )
                if keep_unmitigated:
                    ob.mitigated = _check_ob_mitigated(ob, df, i)
                    if not ob.mitigated:
                        obs.append(ob)
                else:
                    obs.append(ob)
        # Bearish OB: bullish candle before downward impulse
        elif closes[i] > opens[i]:
            if lows[i + 1 : i + lookback + 1].min() < lows[i] * 0.9995:
                ob = OrderBlock(
                    idx=i, direction="bear",
                    top=max(opens[i], closes[i]),
                    bottom=min(opens[i], closes[i]),
                )
                if keep_unmitigated:
                    ob.mitigated = _check_ob_mitigated(ob, df, i)
                    if not ob.mitigated:
                        obs.append(ob)
                else:
                    obs.append(ob)

    return obs[-10:]


def find_fvg(df: pd.DataFrame) -> List[FVGZone]:
    """
    Public alias — find unfilled Fair Value Gaps in `df`.
    Returns list of FVGZone (kind, direction, top, bottom, bar_idx, filled).
    """
    fvgs: List[FVGZone] = []
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(df)

    for i in range(1, n - 1):
        # Bullish FVG: gap upward between candle i-1 high and candle i+1 low
        if lows[i + 1] > highs[i - 1]:
            fvg = FVGZone(kind="fvg", direction="bull",
                          top=lows[i + 1], bottom=highs[i - 1], bar_idx=i)
            fvg.filled = bool((closes[i + 2:] <= fvg.top).any()) if i + 2 < n else False
            fvgs.append(fvg)
        # Bearish FVG: gap downward between candle i-1 low and candle i+1 high
        elif highs[i + 1] < lows[i - 1]:
            fvg = FVGZone(kind="fvg", direction="bear",
                          top=lows[i - 1], bottom=highs[i + 1], bar_idx=i)
            fvg.filled = bool((closes[i + 2:] >= fvg.bottom).any()) if i + 2 < n else False
            fvgs.append(fvg)

    return [f for f in fvgs if not f.filled][-8:]


def find_liquidity_sweeps(
    df: pd.DataFrame,
    swing_lookback: int = 10,
    tolerance: float = 0.0003,
) -> List[LiquiditySweep]:
    """
    Public alias — find liquidity sweeps (stop hunts) in `df`.
    Returns list of LiquiditySweep (direction, level).
    """
    return detect_liquidity_sweeps(
        df,
        swing_lookback=swing_lookback,
        tolerance_ticks=tolerance,
    )


def detect_bos_choch(df: pd.DataFrame, swing_n: int = 5) -> pd.Series:
    """
    Break of Structure (BOS) / Change of Character (CHoCH).
    +1 BOS bullish, -1 BOS bearish, +2 CHoCH bullish, -2 CHoCH bearish, 0 none.
    """
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(df)
    result = np.zeros(n, dtype=int)
    last_swing_high = highs[:swing_n].max() if n >= swing_n else highs[0]
    last_swing_low  = lows[:swing_n].min()  if n >= swing_n else lows[0]
    trend = 0
    for i in range(swing_n, n):
        c = closes[i]
        if c > last_swing_high:
            result[i]       = 2 if trend == -1 else 1
            trend           = 1
            last_swing_high = highs[max(0, i - swing_n): i + 1].max()
        elif c < last_swing_low:
            result[i]      = -2 if trend == 1 else -1
            trend          = -1
            last_swing_low = lows[max(0, i - swing_n): i + 1].min()
    return pd.Series(result, index=df.index, name="bos_choch")


def premium_discount(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    Position within the recent range: >0.5 = premium zone, <0.5 = discount zone.
    """
    rng_high = df["high"].rolling(lookback).max()
    rng_low  = df["low"].rolling(lookback).min()
    rng      = rng_high - rng_low
    pos      = (df["close"] - rng_low) / rng.replace(0, np.nan)
    return pos.fillna(0.5).rename("pd_zone")
