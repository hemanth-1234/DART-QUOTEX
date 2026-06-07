"""
dart_quotex/signals/temporal_patterns.py
Temporal Pattern Scanner
========================
Finds recurring, statistically significant price patterns at specific
times that are invisible to human observation.

Examples of what this catches
------------------------------
·  "Every hour at minute 23 there is a bullish candle"  (minute-of-hour bias)
·  "Between 14:00 and 15:00 UTC the market trends down"  (hour-of-day bias)
·  "On Mondays GBPUSD_OTC opens with a bearish gap"     (day-of-week bias)
·  "Every 15 minutes there is a volatility spike"       (period cycle)
·  "The 5th candle of every hour is always the largest" (intra-hour rhythm)

Why OTC markets have these patterns
-------------------------------------
OTC broker pricing engines are deterministic software.  They use algorithms
to generate synthetic prices derived from real feeds but with their own
smoothing, interpolation, and spread adjustment cycles.  These cycles can
create repeating patterns at fixed intervals.

How the scanner works
---------------------
1.  Load all historical candles from the DB
2.  For each time-bucket type (minute-of-hour, hour-of-day, etc.):
      a.  Group candles into buckets
      b.  Compute directional bias (% bullish), avg return, avg range
      c.  Run a binomial significance test vs. the null hypothesis (50/50)
      d.  Record patterns where p < 0.05 and n >= MIN_SAMPLES
3.  Store patterns in a JSON file
4.  At live-trading time, check if the current candle's time bucket has
    a significant pattern and emit a signal

Usage
-----
    # Offline — scan for patterns once per week
    scanner = TemporalPatternScanner()
    scanner.scan(db, asset="EURUSD_OTC", granularity=60)
    scanner.save("models/temporal_patterns.json")

    # Live — check current candle for a known pattern
    signal = scanner.check_now(asset="EURUSD_OTC")
    if signal.has_pattern:
        print(signal)   # PatternSignal(direction="CALL", confidence=0.72, ...)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

MIN_SAMPLES    = 30      # minimum candles in a bucket to trust the stats
P_VALUE_CUTOFF = 0.05    # maximum p-value to call a pattern significant
MIN_BIAS       = 0.60    # minimum directional bias (60%+ bull/bear) to trade
MIN_CONFIDENCE = 0.55    # minimum signal confidence emitted to the main bot


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TemporalPattern:
    """A single statistically-significant recurring price pattern."""
    bucket_type:  str       # "minute_of_hour" | "hour_of_day" | "minute_abs" | "period"
    bucket_key:   str       # e.g. "23" (23rd minute), "14" (14:xx UTC)
    direction:    str       # "CALL" | "PUT"
    bull_rate:    float     # fraction of candles that are bullish (0-1)
    avg_return:   float     # mean candle return in this bucket
    p_value:      float     # binomial test p-value (lower = more significant)
    n_samples:    int       # number of historical observations
    avg_range:    float     # average candle range (volatility proxy)
    confidence:   float     # derived signal confidence (0-1)
    description:  str = ""


@dataclass
class PatternSignal:
    """Live signal from the temporal pattern scanner."""
    has_pattern:  bool
    direction:    str    # "CALL" | "PUT" | "NEUTRAL"
    confidence:   float  # 0-1
    pattern:      Optional[TemporalPattern] = None
    bucket_type:  str  = ""
    bucket_key:   str  = ""
    description:  str  = ""


# ──────────────────────────────────────────────────────────────────────────────
# Core scanner
# ──────────────────────────────────────────────────────────────────────────────

class TemporalPatternScanner:
    """
    Scans historical OHLCV data for statistically significant time-based
    patterns and emits live signals during trading.

    Parameters
    ----------
    min_samples    : minimum candles per bucket to qualify
    p_threshold    : maximum p-value (significance threshold)
    min_bias       : minimum directional bias fraction
    """

    BUCKET_TYPES = [
        "minute_of_hour",   # 0-59 : which minute within the hour
        "hour_of_day",      # 0-23 : which hour of the day (UTC)
        "minute_of_day",    # 0-1439 : absolute minute of day
        "day_of_week",      # 0-6  : Monday=0 ... Sunday=6
        "candle_sequence",  # 0-N  : position within each period (e.g. 5-min block)
    ]

    def __init__(
        self,
        min_samples:   int   = MIN_SAMPLES,
        p_threshold:   float = P_VALUE_CUTOFF,
        min_bias:      float = MIN_BIAS,
        granularity:   int   = 60,
    ) -> None:
        self.min_samples = min_samples
        self.p_threshold = p_threshold
        self.min_bias    = min_bias
        self.granularity = granularity
        self._patterns:  List[TemporalPattern] = []
        self._pattern_index: Dict[str, List[TemporalPattern]] = {}

    # ── offline scan ──────────────────────────────────────────────────────────

    def scan(
        self,
        df:         pd.DataFrame,
        asset:      str = "",
        verbose:    bool = True,
    ) -> List[TemporalPattern]:
        """
        Scan a historical OHLCV DataFrame for significant temporal patterns.

        Parameters
        ----------
        df      : DataFrame with DatetimeIndex (UTC) and OHLCV columns
        asset   : asset name (informational)
        verbose : log all discovered patterns

        Returns list of TemporalPattern objects (also stored internally).
        """
        if len(df) < self.min_samples * 5:
            log.warning(
                "Insufficient data for pattern scan (%d rows, need %d)",
                len(df), self.min_samples * 5,
            )
            return []

        df = self._prepare(df)
        all_patterns: List[TemporalPattern] = []

        for bucket_type in self.BUCKET_TYPES:
            patterns = self._scan_bucket_type(df, bucket_type)
            all_patterns.extend(patterns)

        # Sort by significance (lowest p first)
        all_patterns.sort(key=lambda p: p.p_value)

        self._patterns = all_patterns
        self._build_index()

        if verbose:
            log.info(
                "Temporal scan complete for %s: %d significant patterns found",
                asset, len(all_patterns),
            )
            for p in all_patterns[:10]:
                log.info(
                    "  %s bucket=%s %s bias=%.1f%% p=%.4f n=%d conf=%.2f",
                    p.bucket_type, p.bucket_key, p.direction,
                    p.bull_rate * 100, p.p_value, p.n_samples, p.confidence,
                )

        return all_patterns

    def scan_from_db(
        self,
        db,
        asset:       str = "EURUSD_OTC",
        granularity: int = 60,
        limit:       int = 10_000,
    ) -> List[TemporalPattern]:
        """Convenience: load candles from Database and scan."""
        df = db.get_candles(asset, granularity, limit=limit)
        if df.empty:
            log.warning("No candles in DB for %s — run harvest first", asset)
            return []
        return self.scan(df, asset=asset)

    # ── live signal ───────────────────────────────────────────────────────────

    def check_now(
        self,
        ts: Optional[datetime] = None,
    ) -> PatternSignal:
        """
        Check if the current time matches any known significant pattern.

        Parameters
        ----------
        ts : timestamp to check (defaults to now UTC)

        Returns PatternSignal — integrate into your signal pipeline.
        """
        if not self._patterns:
            return PatternSignal(False, "NEUTRAL", 0.0,
                                 description="No patterns loaded")

        now = ts or datetime.now(timezone.utc)
        keys = self._time_keys(now)

        best: Optional[TemporalPattern] = None
        for bucket_type, key in keys.items():
            lookup = f"{bucket_type}:{key}"
            matches = self._pattern_index.get(lookup, [])
            for p in matches:
                if best is None or p.confidence > best.confidence:
                    best = p

        if best is None:
            return PatternSignal(False, "NEUTRAL", 0.0,
                                 description="No pattern for current time")

        return PatternSignal(
            has_pattern=True,
            direction=best.direction,
            confidence=best.confidence,
            pattern=best,
            bucket_type=best.bucket_type,
            bucket_key=best.bucket_key,
            description=(
                f"{best.bucket_type}={best.bucket_key} "
                f"{best.direction} {best.bull_rate:.0%} of the time "
                f"(n={best.n_samples}, p={best.p_value:.4f})"
            ),
        )

    def check_df(self, df: pd.DataFrame) -> PatternSignal:
        """Check using the timestamp of the last candle in `df`."""
        if df.empty:
            return PatternSignal(False, "NEUTRAL", 0.0)
        ts = df.index[-1]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        return self.check_now(ts)

    # ── statistics helpers ────────────────────────────────────────────────────

    def top_patterns(self, n: int = 10) -> List[TemporalPattern]:
        """Return the `n` highest-confidence patterns."""
        return sorted(self._patterns, key=lambda p: -p.confidence)[:n]

    def heatmap_data(
        self, bucket_type: str = "minute_of_hour"
    ) -> pd.DataFrame:
        """
        Return a DataFrame suitable for plotting a bias heatmap.
        Rows = bucket keys, columns = direction_bias, p_value, n_samples.
        """
        rows = []
        for p in self._patterns:
            if p.bucket_type == bucket_type:
                rows.append({
                    "bucket": int(p.bucket_key),
                    "bull_rate": p.bull_rate,
                    "direction": p.direction,
                    "p_value": p.p_value,
                    "n_samples": p.n_samples,
                    "confidence": p.confidence,
                })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("bucket")

    def summary_report(self) -> str:
        """Human-readable summary of all discovered patterns."""
        if not self._patterns:
            return "No patterns found. Run scan() first."

        lines = [
            "=" * 65,
            "  TEMPORAL PATTERN SCANNER — DISCOVERED PATTERNS",
            "=" * 65,
            f"  Total significant patterns: {len(self._patterns)}",
            "",
        ]

        for btype in self.BUCKET_TYPES:
            bucket_patterns = [p for p in self._patterns if p.bucket_type == btype]
            if not bucket_patterns:
                continue
            lines.append(f"  {btype.upper().replace('_', ' ')}")
            lines.append("  " + "─" * 60)
            for p in sorted(bucket_patterns, key=lambda x: -x.confidence)[:5]:
                marker = "★" if p.confidence > 0.70 else "·"
                lines.append(
                    f"  {marker} {p.bucket_key:>4}  {p.direction:4}  "
                    f"bias={p.bull_rate:.0%}  "
                    f"n={p.n_samples:>4}  "
                    f"p={p.p_value:.4f}  "
                    f"conf={p.confidence:.2f}"
                )
            lines.append("")

        lines.append("=" * 65)
        return "\n".join(lines)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save discovered patterns to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "bucket_type": p.bucket_type,
                "bucket_key":  p.bucket_key,
                "direction":   p.direction,
                "bull_rate":   p.bull_rate,
                "avg_return":  p.avg_return,
                "p_value":     p.p_value,
                "n_samples":   p.n_samples,
                "avg_range":   p.avg_range,
                "confidence":  p.confidence,
                "description": p.description,
            }
            for p in self._patterns
        ]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("Temporal patterns saved to %s (%d patterns)", path, len(data))

    def load(self, path: str | Path) -> bool:
        """Load patterns from a saved JSON file."""
        path = Path(path)
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        self._patterns = [TemporalPattern(**d) for d in data]
        self._build_index()
        log.info(
            "Temporal patterns loaded from %s (%d patterns)",
            path, len(self._patterns),
        )
        return True

    # ── internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-feature columns to the DataFrame."""
        df = df.copy()
        # Ensure UTC DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")

        # Direction: 1 = bullish, 0 = bearish
        df["bullish"] = (df["close"] >= df["open"]).astype(int)
        df["return"]  = (df["close"] - df["open"]) / (df["open"] + 1e-9)
        df["range"]   = (df["high"] - df["low"])   / (df["open"] + 1e-9)

        # Time buckets
        df["minute_of_hour"] = df.index.minute
        df["hour_of_day"]    = df.index.hour
        df["minute_of_day"]  = df.index.hour * 60 + df.index.minute
        df["day_of_week"]    = df.index.dayofweek
        # Candle sequence within each hour (e.g. granularity=60 → 0, 1m→0-59)
        df["candle_sequence"] = df.index.minute

        return df

    def _scan_bucket_type(
        self, df: pd.DataFrame, bucket_type: str
    ) -> List[TemporalPattern]:
        """Find significant patterns for one bucket type."""
        patterns: List[TemporalPattern] = []
        grouped = df.groupby(bucket_type)

        for key, group in grouped:
            n = len(group)
            if n < self.min_samples:
                continue

            bull_count = int(group["bullish"].sum())
            bull_rate  = bull_count / n

            # Binomial test: is this bias significantly different from 50/50?
            result  = scipy_stats.binomtest(bull_count, n, p=0.5)
            p_value = float(result.pvalue)

            if p_value > self.p_threshold:
                continue
            if abs(bull_rate - 0.5) < (self.min_bias - 0.5):
                continue

            direction  = "CALL" if bull_rate > 0.5 else "PUT"
            avg_return = float(group["return"].mean())
            avg_range  = float(group["range"].mean())

            # Confidence: combination of effect size and statistical significance
            effect_size = abs(bull_rate - 0.5) * 2   # 0→0 when 50/50, 1→1 when 100/0
            sig_score   = 1.0 - min(1.0, p_value / self.p_threshold)
            sample_score = min(1.0, n / (self.min_samples * 5))
            confidence  = float(0.45 * effect_size + 0.35 * sig_score + 0.20 * sample_score)

            if confidence < MIN_CONFIDENCE:
                continue

            patterns.append(TemporalPattern(
                bucket_type=bucket_type,
                bucket_key=str(key),
                direction=direction,
                bull_rate=round(bull_rate, 4),
                avg_return=round(avg_return, 6),
                p_value=round(p_value, 6),
                n_samples=n,
                avg_range=round(avg_range, 6),
                confidence=round(confidence, 4),
                description=(
                    f"{bucket_type}={key}: {direction} "
                    f"{bull_rate:.0%} of {n} candles, p={p_value:.4f}"
                ),
            ))

        return patterns

    def _build_index(self) -> None:
        """Build lookup dict: 'bucket_type:key' → [patterns]."""
        self._pattern_index = {}
        for p in self._patterns:
            key = f"{p.bucket_type}:{p.bucket_key}"
            self._pattern_index.setdefault(key, []).append(p)

    @staticmethod
    def _time_keys(ts: datetime) -> Dict[str, str]:
        """Extract all bucket keys from a timestamp."""
        return {
            "minute_of_hour": str(ts.minute),
            "hour_of_day":    str(ts.hour),
            "minute_of_day":  str(ts.hour * 60 + ts.minute),
            "day_of_week":    str(ts.weekday()),
            "candle_sequence": str(ts.minute),
        }


# ──────────────────────────────────────────────────────────────────────────────
# FFT cycle detector (finds repeating periods, e.g. every 15 min)
# ──────────────────────────────────────────────────────────────────────────────

def detect_dominant_periods(
    df:         pd.DataFrame,
    top_n:      int = 5,
    min_period: int = 3,
    max_period: int = 240,
) -> List[dict]:
    """
    Apply Fast Fourier Transform to the return series to find dominant
    periodic cycles (e.g. every 15 candles, every 60 candles).

    Parameters
    ----------
    df         : OHLCV DataFrame
    top_n      : how many dominant periods to return
    min_period : minimum period length (in candles)
    max_period : maximum period length (in candles)

    Returns list of dicts: {period_candles, frequency, power, significance}
    """
    if len(df) < max_period * 2:
        return []

    returns = (df["close"] - df["open"]).values.astype(float)
    n       = len(returns)

    # Detrend and normalise
    returns = returns - returns.mean()
    returns = returns / (returns.std() + 1e-9)

    # FFT
    fft_vals = np.fft.rfft(returns)
    power    = np.abs(fft_vals) ** 2
    freqs    = np.fft.rfftfreq(n)

    # Convert frequency to period (candles)
    results = []
    for i, (freq, pwr) in enumerate(zip(freqs, power)):
        if freq < 1e-9:
            continue
        period = int(round(1.0 / freq))
        if not (min_period <= period <= max_period):
            continue

        # Significance: compare power to median of all powers
        sig = float(pwr / (np.median(power) + 1e-9))
        results.append({
            "period_candles": period,
            "frequency":      round(float(freq), 6),
            "power":          round(float(pwr), 4),
            "significance":   round(sig, 2),
        })

    # Sort by significance, deduplicate similar periods
    results.sort(key=lambda x: -x["significance"])
    seen_periods: set = set()
    deduped = []
    for r in results:
        p = r["period_candles"]
        if not any(abs(p - seen) <= 2 for seen in seen_periods):
            deduped.append(r)
            seen_periods.add(p)
        if len(deduped) >= top_n:
            break

    return deduped


# ──────────────────────────────────────────────────────────────────────────────
# Autocorrelation pattern finder
# ──────────────────────────────────────────────────────────────────────────────

def autocorrelation_patterns(
    df:        pd.DataFrame,
    max_lag:   int = 120,
    threshold: float = 0.08,
) -> List[dict]:
    """
    Find lags where the return autocorrelation is unusually high.
    A high autocorrelation at lag=60 means 'the direction 60 candles ago
    predicts today's direction'.

    Returns list of {lag, autocorr, p_value}.
    """
    if len(df) < max_lag + 20:
        return []

    rets = pd.Series(
        (df["close"].values - df["open"].values) / (df["open"].values + 1e-9)
    )
    n = len(rets)

    significant = []
    for lag in range(1, min(max_lag + 1, n // 3)):
        ac = float(rets.autocorr(lag=lag))
        if abs(ac) < threshold:
            continue
        # Approximate significance test
        se     = 1.0 / np.sqrt(n - lag)
        z      = ac / (se + 1e-9)
        p_val  = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
        if p_val < 0.05:
            significant.append({
                "lag":      lag,
                "autocorr": round(ac, 4),
                "p_value":  round(p_val, 4),
                "direction": "same" if ac > 0 else "opposite",
            })

    return sorted(significant, key=lambda x: abs(x["autocorr"]), reverse=True)
