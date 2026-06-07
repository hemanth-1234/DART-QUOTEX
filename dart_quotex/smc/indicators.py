"""
dart_quotex/smc/indicators.py
Module 4 — Stop Hunt + FVG Signal Engine
"""
from __future__ import annotations
import logging
from typing import Tuple
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def liquidity_sweep(df: pd.DataFrame, lookback: int = 15,
                    swing_n: int = 3, tolerance: float = 0.0002) -> Tuple[str, float]:
    """Detects a stop hunt: price wicks beyond swing high/low and closes back inside."""
    if len(df) < lookback + swing_n * 2 + 3:
        return "none", 0.0
    h, l, c = df["high"].values.astype(float), df["low"].values.astype(float), df["close"].values.astype(float)
    n = len(df)
    sh_levels, sl_levels = [], []
    for i in range(max(swing_n, n - lookback - swing_n), n - swing_n - 1):
        win_h = h[max(0, i-swing_n):i+swing_n+1]
        win_l = l[max(0, i-swing_n):i+swing_n+1]
        if len(win_h) >= swing_n*2+1:
            if h[i] == win_h.max(): sh_levels.append((i, h[i]))
            if l[i] == win_l.min(): sl_levels.append((i, l[i]))
    best_dir, best_str = "none", 0.0
    for idx in range(max(0, n-2), n):
        for sh_i, sh_lv in sh_levels[-5:]:
            if idx <= sh_i: continue
            ext = h[idx] - sh_lv
            if ext > sh_lv * tolerance and c[idx] < sh_lv:
                s = min(1.0, ext / (sh_lv * tolerance * 5))
                if s > best_str: best_dir, best_str = "bearish", s
        for sl_i, sl_lv in sl_levels[-5:]:
            if idx <= sl_i: continue
            ext = sl_lv - l[idx]
            if ext > sl_lv * tolerance and c[idx] > sl_lv:
                s = min(1.0, ext / (sl_lv * tolerance * 5))
                if s > best_str: best_dir, best_str = "bullish", s
    return best_dir, best_str


def fvg_after_sweep(df: pd.DataFrame, sweep_dir: str,
                    look_forward: int = 10, min_size_pct: float = 0.0002) -> Tuple[str, float]:
    """After a sweep, finds a 3-candle FVG confirming the reversal direction."""
    if sweep_dir == "none" or len(df) < 5:
        return "none", 0.0
    h, l = df["high"].values.astype(float), df["low"].values.astype(float)
    n = len(df)
    atr = float(np.mean(h[-14:] - l[-14:]) if n >= 14 else np.mean(h - l)) + 1e-9
    fvg_dir = "bullish" if sweep_dir == "bullish" else "bearish"
    best = 0.0
    for i in range(max(1, n - look_forward), n - 1):
        if i + 1 >= n: break
        if fvg_dir == "bullish":
            gap = l[i+1] - h[i-1]
        else:
            gap = l[i-1] - h[i+1]
        if gap > 0 and gap > atr * min_size_pct:
            best = max(best, min(1.0, 0.5 + (gap / atr) * 0.5))
    return (fvg_dir, best) if best > 0 else ("none", 0.0)


def stop_hunt_signal(df: pd.DataFrame, min_sweep_strength: float = 0.70,
                     min_fvg_confidence: float = 0.60, lookback: int = 15) -> Tuple[str, float]:
    """
    Master signal: sweep + FVG => CALL/PUT with confidence.
    Returns ("CALL"|"PUT"|"HOLD", confidence).
    Use to override AI when ai_confidence < 0.6 and this returns > MIN_HUNT_CONFIDENCE.
    """
    if len(df) < lookback + 10:
        return "HOLD", 0.0
    sweep_dir, sweep_str = liquidity_sweep(df, lookback=lookback)
    if sweep_dir == "none" or sweep_str < min_sweep_strength:
        return "HOLD", 0.0
    fvg_dir, fvg_conf = fvg_after_sweep(df, sweep_dir)
    trade_dir = "CALL" if sweep_dir == "bullish" else "PUT"
    if fvg_conf < min_fvg_confidence:
        return trade_dir, round(sweep_str * 0.65, 3)
    overall = min(0.95, sweep_str * 0.55 + fvg_conf * 0.45)
    log.info("StopHunt: %s sweep=%.2f fvg=%.2f => %s conf=%.2f",
             sweep_dir, sweep_str, fvg_conf, trade_dir, overall)
    return trade_dir, round(overall, 3)


def add_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compat wrapper used by ml/features.py."""
    from dart_quotex.smc.advanced_ict import (
        find_order_blocks, find_fvg, find_liquidity_sweeps,
        detect_bos_choch, premium_discount,
    )
    df = df.copy()
    df["bos_choch"] = detect_bos_choch(df)
    df["pd_zone"]   = premium_discount(df)
    atr = (df["high"] - df["low"]).rolling(14).mean().fillna(0.001)
    last_c = float(df["close"].iloc[-1])
    obs = find_order_blocks(df)
    bull_obs = [o for o in obs if o.direction == "bull"]
    bear_obs = [o for o in obs if o.direction == "bear"]
    df["ob_bull_dist"] = (last_c - min(bull_obs, key=lambda o: abs(last_c-o.top)).top) / atr.iloc[-1] if bull_obs else 999.0
    df["ob_bear_dist"] = (min(bear_obs, key=lambda o: abs(last_c-o.bottom)).bottom - last_c) / atr.iloc[-1] if bear_obs else 999.0
    fvgs = find_fvg(df)
    df["fvg_signal"] = (1 if fvgs and fvgs[-1].direction == "bull" else -1 if fvgs and fvgs[-1].direction == "bear" else 0)
    sweeps = find_liquidity_sweeps(df)
    df["liq_sweep"] = (1 if sweeps and sweeps[-1].direction == "buy_side" else -1 if sweeps and sweeps[-1].direction == "sell_side" else 0)
    return df


# Re-export functions that tests and features.py import from this module
from dart_quotex.smc.advanced_ict import (
    detect_bos_choch,
    premium_discount,
    find_order_blocks,
    find_fvg,
    find_liquidity_sweeps,
)

__all__ = [
    "liquidity_sweep", "fvg_after_sweep", "stop_hunt_signal", "add_smc_features",
    "detect_bos_choch", "premium_discount",
    "find_order_blocks", "find_fvg", "find_liquidity_sweeps",
]
