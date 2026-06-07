"""
dart_quotex/ml/features.py  (UPDATED — full MTF integration)
Feature engineering pipeline — 34 base + 52 MTF = 86 total features.

Feature groups
--------------
1.  Price-derived        : returns, log-returns, body/wick ratios         (7)
2.  Momentum             : RSI-7/14, MACD, Stoch, ROC, CCI               (9)
3.  Volatility           : ATR, Bollinger Bands, Historical Vol           (5)
4.  Trend                : SMA/EMA cross, ADX, DI+/-                      (5)
5.  Volume               : OBV, volume z-score                            (2)
6.  SMC / ICT            : OB dist, FVG, liq sweep, BOS/CHoCH, PD zone   (6)
7.  Candlestick patterns : net signal strength + direction                (2)
                                                                  TOTAL = 34+2 = 36 base
8.  Multi-timeframe (4 TF × 13 features per TF)                          (52)
                                                           GRAND TOTAL = 88
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from dart_quotex.smc.indicators import add_smc_features
from dart_quotex.ml.multi_timeframe import MultiTimeframeAnalyzer
from dart_quotex.patterns.candlestick import PatternScanner

FEATURE_NAMES: List[str] = []   # populated on first call

_pattern_scanner = PatternScanner()


def build_features(
    df: pd.DataFrame,
    lookback: int = 100,
    mtf_analyzer: Optional[MultiTimeframeAnalyzer] = None,
) -> np.ndarray:
    """
    Build full feature vector from OHLCV DataFrame.

    Parameters
    ----------
    df            : OHLCV DataFrame (DatetimeIndex, UTC)
    lookback      : candles to use for calculation window
    mtf_analyzer  : pre-initialised MultiTimeframeAnalyzer (optional)

    Returns
    -------
    np.ndarray shape (88,), dtype float32
    """
    df = df.tail(lookback).copy()
    if len(df) < 20:
        raise ValueError(f"Need at least 20 candles, got {len(df)}")

    df = _add_technical(df)
    df = add_smc_features(df)
    df = df.ffill().bfill()

    last = df.iloc[-1]

    # ── 1. Price-derived ─────────────────────────────────────────────────────
    base = [
        last.get("ret_1",    0.0),
        last.get("ret_3",    0.0),
        last.get("ret_5",    0.0),
        last.get("log_ret",  0.0),
        last.get("body_ratio", 0.0),
        last.get("upper_wick", 0.0),
        last.get("lower_wick", 0.0),
    ]

    # ── 2. Momentum ──────────────────────────────────────────────────────────
    base += [
        last.get("rsi_14",     50) / 100.0,
        last.get("rsi_7",      50) / 100.0,
        last.get("macd",        0.0),
        last.get("macd_signal", 0.0),
        last.get("macd_hist",   0.0),
        last.get("stoch_k",    50) / 100.0,
        last.get("stoch_d",    50) / 100.0,
        last.get("roc_5",       0.0),
        last.get("cci_20",      0.0) / 200.0,
    ]

    # ── 3. Volatility ────────────────────────────────────────────────────────
    base += [
        last.get("atr_14",       0.0),
        last.get("bb_upper_dist",0.0),
        last.get("bb_lower_dist",0.0),
        last.get("bb_width",     0.0),
        last.get("hist_vol_20",  0.0),
    ]

    # ── 4. Trend ─────────────────────────────────────────────────────────────
    base += [
        last.get("sma_cross",  0.0),
        last.get("ema_cross",  0.0),
        last.get("adx_14",    25.0) / 100.0,
        last.get("di_plus",   25.0) / 100.0,
        last.get("di_minus",  25.0) / 100.0,
    ]

    # ── 5. Volume ────────────────────────────────────────────────────────────
    base += [
        last.get("obv_norm",   0.0),
        last.get("vol_zscore", 0.0),
    ]

    # ── 6. SMC / ICT ─────────────────────────────────────────────────────────
    base += [
        float(np.clip(last.get("ob_bull_dist", 10), -10, 10)) / 10.0,
        float(np.clip(last.get("ob_bear_dist", 10), -10, 10)) / 10.0,
        float(last.get("fvg_signal",  0.0)),
        float(last.get("liq_sweep",   0.0)),
        float(np.clip(last.get("bos_choch", 0.0), -2, 2)) / 2.0,
        float(last.get("pd_zone",     0.5)),
    ]

    # ── 7. Candlestick patterns ───────────────────────────────────────────────
    try:
        pat_sig = _pattern_scanner.net_signal(df.tail(5))
        pat_dir = 1.0 if pat_sig.direction == "CALL" else (-1.0 if pat_sig.direction == "PUT" else 0.0)
        pat_str = float(pat_sig.strength) * pat_dir
        base += [pat_str, float(pat_sig.strength)]
    except Exception:
        base += [0.0, 0.0]

    feats_base = np.array(base, dtype=np.float32)

    # ── 8. Multi-timeframe features ───────────────────────────────────────────
    if mtf_analyzer is not None:
        try:
            mtf_analyzer.update(df)
            mtf_feats = mtf_analyzer.features()    # (52,)
        except Exception:
            mtf_feats = np.zeros(52, dtype=np.float32)
    else:
        mtf_feats = np.zeros(52, dtype=np.float32)

    feats = np.concatenate([feats_base, mtf_feats], dtype=np.float32)

    global FEATURE_NAMES
    if not FEATURE_NAMES:
        FEATURE_NAMES = (
            ["ret_1","ret_3","ret_5","log_ret","body_ratio","upper_wick","lower_wick",
             "rsi_14_n","rsi_7_n","macd","macd_signal","macd_hist",
             "stoch_k_n","stoch_d_n","roc_5","cci_20_n",
             "atr_14","bb_upper_dist","bb_lower_dist","bb_width","hist_vol_20",
             "sma_cross","ema_cross","adx_14_n","di_plus_n","di_minus_n",
             "obv_norm","vol_zscore",
             "ob_bull_dist_n","ob_bear_dist_n","fvg_signal","liq_sweep",
             "bos_choch_n","pd_zone","pat_dir","pat_str"]
            + [f"mtf_{i}" for i in range(52)]
        )

    return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)


def build_sequence(df: pd.DataFrame, seq_len: int = 20) -> np.ndarray:
    """Build (seq_len, n_features) array for SAC agent."""
    n = len(df)
    if n < seq_len + 20:
        raise ValueError(f"Need at least {seq_len + 20} candles")
    seq = []
    for i in range(seq_len):
        window = df.iloc[: n - (seq_len - i - 1)]
        seq.append(build_features(window))
    return np.array(seq, dtype=np.float32)


def _add_technical(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"].replace(0, np.nan).fillna(1)

    df["ret_1"] = c.pct_change(1).fillna(0)
    df["ret_3"] = c.pct_change(3).fillna(0)
    df["ret_5"] = c.pct_change(5).fillna(0)
    df["log_ret"] = np.log(c / c.shift(1)).fillna(0)

    rng = (h - l).replace(0, 1e-8)
    body = (df["close"] - df["open"]).abs()
    df["body_ratio"]  = body / rng
    df["upper_wick"]  = (h - df[["open","close"]].max(axis=1)) / rng
    df["lower_wick"]  = (df[["open","close"]].min(axis=1) - l) / rng

    df["rsi_14"] = _rsi(c, 14)
    df["rsi_7"]  = _rsi(c, 7)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = (ema12 - ema26) / c
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-8)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    df["roc_5"] = c.pct_change(5).fillna(0)
    tp = (h + l + c) / 3
    df["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-8)

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(span=14, adjust=False).mean() / c

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["bb_upper_dist"] = (bb_upper - c) / (c + 1e-8)
    df["bb_lower_dist"] = (c - bb_lower) / (c + 1e-8)
    df["bb_width"]      = (bb_upper - bb_lower) / (sma20 + 1e-8)
    df["hist_vol_20"]   = df["log_ret"].rolling(20).std().fillna(0)

    sma10 = c.rolling(10).mean()
    sma30 = c.rolling(30).mean()
    df["sma_cross"] = ((sma10 - sma30) / (c + 1e-8)).fillna(0)
    ema9  = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    df["ema_cross"] = ((ema9 - ema21) / (c + 1e-8)).fillna(0)

    df["adx_14"], df["di_plus"], df["di_minus"] = _adx(h, l, c, 14)

    obv = (np.sign(c.diff()) * v).cumsum()
    obv_min = obv.rolling(50).min()
    obv_max = obv.rolling(50).max()
    df["obv_norm"]  = (2 * (obv - obv_min) / (obv_max - obv_min + 1e-8) - 1).fillna(0)
    vol_mean = v.rolling(20).mean()
    vol_std  = v.rolling(20).std()
    df["vol_zscore"] = ((v - vol_mean) / (vol_std + 1e-8)).clip(-3, 3).fillna(0)

    return df


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up   = high.diff()
    dn   = -low.diff()
    dm_p = np.where((up > dn) & (up > 0), up, 0.0)
    dm_m = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr   = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr  = tr.ewm(span=period, adjust=False).mean()
    di_p = 100 * pd.Series(dm_p, index=high.index).ewm(span=period, adjust=False).mean() / (atr + 1e-8)
    di_m = 100 * pd.Series(dm_m, index=high.index).ewm(span=period, adjust=False).mean() / (atr + 1e-8)
    dx   = (di_p - di_m).abs() / (di_p + di_m + 1e-8) * 100
    adx  = dx.ewm(span=period, adjust=False).mean()
    return adx.fillna(25), di_p.fillna(25), di_m.fillna(25)
