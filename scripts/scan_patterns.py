#!/usr/bin/env python3
"""
scripts/scan_patterns.py
Offline algorithmic pattern scanner.

Run this once per week (or whenever you harvest new data) to discover
all statistically significant patterns in your stored candle history.

The script runs BOTH scanners:
  · AlgoPatternScanner   — 8 mechanical pattern detectors
  · TemporalPatternScanner — time-bucket pattern scanner
  · FFT cycle detector    — dominant periodicity (e.g. "every 15 min")
  · Autocorrelation scan  — lag-based predictive relationships

Usage
-----
    python scripts/scan_patterns.py --asset EURUSD_OTC --candles 10000

Output
------
  · Console: full pattern report
  · models/algo_patterns.json       — saved AlgoPatternScanner state
  · models/temporal_patterns.json   — saved TemporalPatternScanner state
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scan_patterns")

SEPARATOR = "═" * 68


def _header(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


async def main(args: argparse.Namespace) -> None:
    from dart_quotex.config import cfg
    from dart_quotex.data.database import Database
    from dart_quotex.signals.algo_patterns import AlgoPatternScanner
    from dart_quotex.signals.temporal_patterns import (
        TemporalPatternScanner,
        detect_dominant_periods,
        autocorrelation_patterns,
    )

    # ── Load candles ──────────────────────────────────────────────────────────
    db = Database(args.db or cfg.data.db_path)
    df = db.get_candles(args.asset, args.granularity, limit=args.candles)

    if len(df) < 200:
        print(f"\n  ERROR: Only {len(df)} candles in DB for {args.asset}.")
        print("  Run harvest first:  python main.py harvest --asset " + args.asset)
        sys.exit(1)

    print(f"\n  Asset       : {args.asset}")
    print(f"  Candles     : {len(df):,}  (gran={args.granularity}s)")
    print(f"  From        : {df.index[0]}")
    print(f"  To          : {df.index[-1]}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 1.  Algorithmic Pattern Scan (8 mechanical detectors)
    # ══════════════════════════════════════════════════════════════════════════
    _header("ALGORITHMIC PATTERN SCAN  (8 detectors)")
    print("  Running... this may take 10-30 seconds.\n")

    algo = AlgoPatternScanner(pip_size=args.pip_size)
    algo.fit(df, verbose=False)
    algo.save(model_dir / "algo_patterns.json")

    recent = df.tail(50)
    active = algo.all_signals(recent)

    if active:
        print(algo.report(recent))
    else:
        print("  No active algorithmic patterns for the most recent candles.")

    print(f"\n  Saved → {model_dir}/algo_patterns.json")

    # ══════════════════════════════════════════════════════════════════════════
    # 2.  Temporal Pattern Scan (time-bucket analysis)
    # ══════════════════════════════════════════════════════════════════════════
    _header("TEMPORAL PATTERN SCAN  (time-bucket analysis)")
    print("  Scanning minute-of-hour, hour-of-day, day-of-week …\n")

    temporal = TemporalPatternScanner(
        min_samples=args.min_samples,
        p_threshold=args.p_threshold,
    )
    patterns = temporal.scan(df, asset=args.asset, verbose=False)
    temporal.save(model_dir / "temporal_patterns.json")

    if patterns:
        print(temporal.summary_report())

        # Heatmap data for the most insightful bucket type
        hm = temporal.heatmap_data("minute_of_hour")
        if not hm.empty:
            print("\n  Minute-of-Hour directional bias (top 10 buckets):")
            top = hm.nlargest(10, "confidence")
            for _, row in top.iterrows():
                bar  = "█" * int(row["confidence"] * 20)
                bias = row["bull_rate"] - 0.5
                sym  = "▲ CALL" if bias > 0 else "▼ PUT "
                print(
                    f"  Minute {int(row['bucket']):>2}   "
                    f"{sym}  {abs(row['bull_rate'] - 0.5):.0%} bias  "
                    f"p={row['p_value']:.4f}  n={int(row['n_samples'])}  "
                    f"{bar}"
                )
    else:
        print("  No significant temporal patterns found.")

    print(f"\n  Saved → {model_dir}/temporal_patterns.json")

    # ══════════════════════════════════════════════════════════════════════════
    # 3.  FFT Cycle Detector (dominant periodic cycles)
    # ══════════════════════════════════════════════════════════════════════════
    _header("FFT CYCLE DETECTOR  (dominant periodic cycles)")

    periods = detect_dominant_periods(df, top_n=8)
    if periods:
        print(f"  {'Period':>8}  {'Significance':>14}  {'Power':>10}")
        print("  " + "─" * 40)
        for p in periods:
            bar = "█" * min(30, int(p["significance"] / 5))
            print(
                f"  {p['period_candles']:>6}c   "
                f"{p['significance']:>12.1f}×   "
                f"{p['power']:>10.2f}   {bar}"
            )
        best = periods[0]
        gran_s = args.granularity
        minutes = best["period_candles"] * gran_s // 60
        print(
            f"\n  Dominant cycle: every {best['period_candles']} candles "
            f"= {minutes} minutes  (significance={best['significance']:.1f}×)"
        )
    else:
        print("  No dominant cycles detected.")

    # ══════════════════════════════════════════════════════════════════════════
    # 4.  Autocorrelation Scan (lag-based predictive lags)
    # ══════════════════════════════════════════════════════════════════════════
    _header("AUTOCORRELATION SCAN  (predictive lag relationships)")

    ac_patterns = autocorrelation_patterns(df, max_lag=120)
    if ac_patterns:
        print(
            f"  {'Lag':>5}  {'Autocorr':>10}  {'Direction':>12}  {'p-value':>9}"
        )
        print("  " + "─" * 45)
        for a in ac_patterns[:10]:
            gran_min = args.granularity // 60
            lag_min  = a["lag"] * gran_min
            bar      = "█" * int(abs(a["autocorr"]) * 30)
            print(
                f"  {a['lag']:>5}  {a['autocorr']:>+10.4f}  "
                f"{'same direction' if a['direction'] == 'same' else 'reversal':>12}  "
                f"{a['p_value']:>9.4f}  "
                f"({lag_min}min)  {bar}"
            )
        if ac_patterns:
            best_ac = ac_patterns[0]
            print(
                f"\n  Strongest lag: {best_ac['lag']} candles "
                f"(ac={best_ac['autocorr']:+.3f}) → "
                f"candle from {best_ac['lag']} bars ago is predictive"
            )
    else:
        print("  No significant autocorrelation patterns found.")

    # ══════════════════════════════════════════════════════════════════════════
    # 5.  Summary
    # ══════════════════════════════════════════════════════════════════════════
    _header("SUMMARY")
    print(f"  Algorithmic patterns discovered : {len(active)}")
    print(f"  Temporal patterns discovered    : {len(patterns)}")
    print(f"  FFT cycles found                : {len(periods)}")
    print(f"  Autocorrelation lags found      : {len(ac_patterns)}")
    print()
    if active or patterns:
        print("  To activate in live trading, set in .env:")
        print("    ENABLE_ALGO_PATTERNS=true")
        print("    ENABLE_TEMPORAL_PATTERNS=true")
    print()
    print(f"  Models saved to: {model_dir}/")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Offline algorithmic & temporal pattern scanner"
    )
    p.add_argument("--asset",       default="EURUSD_OTC")
    p.add_argument("--granularity", type=int,   default=60)
    p.add_argument("--candles",     type=int,   default=10_000)
    p.add_argument("--pip-size",    type=float, default=0.0001)
    p.add_argument("--min-samples", type=int,   default=25)
    p.add_argument("--p-threshold", type=float, default=0.05)
    p.add_argument("--model-dir",   default="models")
    p.add_argument("--db",          default=None,
                   help="Override DB path from .env")
    asyncio.run(main(p.parse_args()))
