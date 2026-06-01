#!/usr/bin/env python3
"""
run_pipeline.py
DART-Quotex end-to-end pipeline.

Usage
-----
    python run_pipeline.py
    python run_pipeline.py --asset GBPUSD_OTC --min-wr 0.55 --min-pf 1.2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dart_quotex.config import cfg
from dart_quotex.data.database import Database
from dart_quotex.advisor import AIAdvisor
from dart_quotex.backtester import Backtester
from dart_quotex.trader import LiveTrader

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logging.getLogger("dart_quotex.sentiment").setLevel(logging.WARNING)
logging.getLogger("dart_quotex.data.harvester").setLevel(logging.WARNING)
log = logging.getLogger("pipeline")

# ── colour helpers (no external dep) ─────────────────────────────────────────
_USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def green(t):  return _c("92", t)
def red(t):    return _c("91", t)
def yellow(t): return _c("93", t)
def cyan(t):   return _c("96", t)
def bold(t):   return _c("1",  t)

BANNER = cyan(r"""
  ██████╗  █████╗ ██████╗ ████████╗      ██████╗ ██╗  ██╗
  ██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝     ██╔═══██╗╚██╗██╔╝
  ██║  ██║███████║██████╔╝   ██║        ██║   ██║ ╚███╔╝
  ██║  ██║██╔══██║██╔══██╗   ██║        ██║▄▄ ██║ ██╔██╗
  ██████╔╝██║  ██║██║  ██║   ██║        ╚██████╔╝██╔╝ ██╗
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝         ╚══▀▀═╝ ╚═╝  ╚═╝
  DART-Quotex  ·  AI-Driven Binary Options Pipeline
""")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ask(prompt: str, valid: list[str]) -> str:
    """Read a validated answer from stdin."""
    valid_lower = [v.lower() for v in valid]
    while True:
        try:
            ans = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if ans in valid_lower:
            return ans
        print(red(f"  Please enter one of: {', '.join(valid)}"))


def _separator(char: str = "─", width: int = 60) -> None:
    print(cyan(char * width))


def _print_backtest(result) -> None:
    _separator()
    print(bold("  BACKTEST RESULTS"))
    _separator()
    wr_colour  = green if result.win_rate >= 0.55 else red
    pf_colour  = green if result.profit_factor >= 1.2 else red
    dd_colour  = red   if result.max_drawdown > 0.10 else yellow
    roi_colour = green if result.roi >= 0 else red

    print(f"  Trades         : {bold(str(result.n_trades))}")
    print(f"  Win Rate       : {wr_colour(f'{result.win_rate:.1%}')}")
    print(f"  Profit Factor  : {pf_colour(f'{result.profit_factor:.3f}')}")
    print(f"  Max Drawdown   : {dd_colour(f'{result.max_drawdown:.1%}')}")
    print(f"  Sharpe         : {result.sharpe:.3f}")
    print(f"  ROI            : {roi_colour(f'{result.roi:+.1%}')}")
    print(f"  Start Balance  : {result.start_balance:,.2f}")
    print(f"  End Balance    : {result.end_balance:,.2f}")
    _separator()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ──────────────────────────────────────────────────────────────────────────────

async def step_check_data(
    db: Database,
    advisor: AIAdvisor,
    asset: str,
    granularity: int,
    min_candles: int,
) -> int:
    """Step 1 — Verify candle count; harvest if insufficient."""
    _separator("═")
    print(bold(f"  STEP 1 · DATA CHECK  —  {asset}"))
    _separator("═")

    count = db.count_candles(asset, granularity)
    print(f"  Candles in DB  : {count:,}")
    print(f"  Required       : {min_candles:,}")

    if count >= min_candles:
        print(green(f"  ✓  Sufficient data ({count:,} candles)"))
        return count

    needed = min_candles - count
    print(yellow(f"  ⚠  Missing ~{needed:,} candles — harvesting now…"))
    print()

    await advisor.connect()
    stored = await advisor.harvest_history(asset=asset, total_candles=min_candles)
    await advisor.disconnect()

    count = db.count_candles(asset, granularity)
    print(green(f"  ✓  Harvest complete — {stored:,} new rows | total: {count:,}"))
    return count


async def step_backtest(
    db: Database,
    advisor: AIAdvisor,
    asset: str,
    granularity: int,
    start_balance: float,
    payout: float,
    min_confidence: float,
    lookback: int,
):
    """Step 2 — Walk-forward backtest on stored history."""
    _separator("═")
    print(bold("  STEP 2 · BACKTEST"))
    _separator("═")
    print(f"  Asset          : {asset}")
    print(f"  Granularity    : {granularity}s")
    print(f"  Start Balance  : {start_balance:,.2f}")
    print(f"  Payout         : {payout:.0%}")
    print(f"  Min Confidence : {min_confidence:.0%}")
    print()
    print(yellow("  Running walk-forward backtest… (this may take 10–30 s)"))

    bt     = Backtester(db=db, advisor=advisor, lookback=lookback, train_online=True)
    result = bt.run(
        asset=asset,
        granularity=granularity,
        start_balance=start_balance,
        payout=payout,
        min_confidence=min_confidence,
        limit=3_000,
    )
    _print_backtest(result)
    return result


def step_gate(result, min_wr: float, min_pf: float, min_trades: int) -> bool:
    """Step 3 — Quality gate; ask user whether to continue."""
    _separator("═")
    print(bold("  STEP 3 · QUALITY GATE"))
    _separator("═")

    checks = [
        ("Win Rate",      result.win_rate,      min_wr,     "≥"),
        ("Profit Factor", result.profit_factor,  min_pf,     "≥"),
        ("Trade Count",   result.n_trades,       min_trades, "≥"),
    ]

    all_pass = True
    for name, val, threshold, op in checks:
        ok = (val >= threshold)
        all_pass = all_pass and ok
        sym    = green("PASS ✓") if ok else red("FAIL ✗")
        colour = green if ok else red
        val_str = f"{val:.1%}" if isinstance(val, float) and val < 10 else f"{val:.3f}" if isinstance(val, float) else str(val)
        thr_str = f"{threshold:.1%}" if isinstance(threshold, float) and threshold < 10 else str(threshold)
        print(f"  {name:<18} {colour(val_str):>12}  (need {op}{thr_str})  {sym}")

    print()

    if all_pass:
        print(green("  ✓  All checks passed."))
    else:
        print(red("  ✗  One or more checks FAILED."))
        print(yellow("  The strategy may not be profitable on live data."))

    ans = _ask(
        bold("  Continue anyway? [y/n]: "),
        ["y", "n"],
    )
    if ans == "n":
        print(yellow("  Pipeline aborted by user."))
        return False
    return True


def step_choose_mode() -> str:
    """Step 4 — Select DEMO or REAL account."""
    _separator("═")
    print(bold("  STEP 4 · ACCOUNT MODE"))
    _separator("═")
    print(f"  Current .env mode : {bold(cfg.quotex.mode.upper())}")
    print()
    print("  [1]  DEMO   — practice account (safe, no real money)")
    print("  [2]  REAL   — live account (real money at risk)")
    print()

    ans = _ask(bold("  Choose mode [1/2]: "), ["1", "2"])
    mode = "demo" if ans == "1" else "real"

    if mode == "real":
        _separator()
        print(red("  ⚠  WARNING — REAL MONEY MODE"))
        _separator()
        print(red("  You are about to trade with real money."))
        print(red("  Losses can exceed your stake on each trade."))
        print(red("  Only proceed if you fully understand the risks."))
        print()
        confirm = _ask(red(bold("  Type 'yes' to confirm REAL trading: ")), ["yes", "no"])
        if confirm != "yes":
            print(yellow("  Switched back to DEMO mode."))
            mode = "demo"

    print()
    colour = green if mode == "demo" else red
    print(colour(f"  ✓  Mode selected: {mode.upper()}"))
    return mode


async def step_live_trade(
    mode: str,
    asset: str,
    session_minutes: int,
    trade_interval: int,
) -> None:
    """Step 5 — Launch live trading session."""
    _separator("═")
    print(bold("  STEP 5 · LIVE TRADING"))
    _separator("═")
    print(f"  Asset          : {asset}")
    print(f"  Mode           : {bold(mode.upper())}")
    print(f"  Session Length : {session_minutes} minutes")
    print(f"  Trade Interval : {trade_interval} seconds")
    print()

    # Override mode at runtime without touching .env
    os.environ["QUOTEX_MODE"] = mode

    ans = _ask(
        bold(f"  Start {mode.upper()} session now? [y/n]: "),
        ["y", "n"],
    )
    if ans == "n":
        print(yellow("  Session cancelled."))
        return

    _separator()
    print(green(f"  ▶  Starting {mode.upper()} session — press Ctrl-C to stop"))
    _separator()
    print()

    trader = LiveTrader(
        session_minutes=session_minutes,
        trade_interval=trade_interval,
    )
    await trader.run(asset=asset)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    print(BANNER)

    asset       = args.asset
    granularity = args.granularity
    min_candles = args.min_candles
    start_bal   = args.balance
    payout      = args.payout
    min_conf    = args.min_confidence
    lookback    = args.lookback
    min_wr      = args.min_wr
    min_pf      = args.min_pf
    min_trades  = args.min_trades
    session_min = args.session
    interval_s  = args.interval

    # Shared DB and advisor
    db      = Database(cfg.data.db_path)
    advisor = AIAdvisor(use_robust_client=True)

    # ── Step 1: Data check ────────────────────────────────────────────────────
    count = await step_check_data(db, advisor, asset, granularity, min_candles)
    if count < lookback + 10:
        print(red(f"\n  Not enough candles ({count}) to run backtest."
                  f" Need at least {lookback + 10}. Run harvest manually:\n"
                  f"  python main.py harvest --asset {asset} --total {min_candles}"))
        sys.exit(1)

    # ── Step 2: Backtest ──────────────────────────────────────────────────────
    result = await step_backtest(
        db, advisor, asset, granularity,
        start_bal, payout, min_conf, lookback,
    )

    # ── Step 3: Quality gate ──────────────────────────────────────────────────
    if not step_gate(result, min_wr, min_pf, min_trades):
        sys.exit(0)

    # ── Step 4: Account mode ──────────────────────────────────────────────────
    mode = step_choose_mode()

    # ── Step 5: Live trading ──────────────────────────────────────────────────
    await step_live_trade(mode, asset, session_min, interval_s)

    print()
    _separator("═")
    print(green("  Pipeline complete."))
    _separator("═")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="DART-Quotex end-to-end pipeline: data → backtest → live",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--asset",          default=cfg.quotex.asset,
                   help="Asset symbol")
    p.add_argument("--granularity",    type=int,   default=cfg.data.granularity,
                   help="Candle size in seconds")
    p.add_argument("--min-candles",    type=int,   default=7 * 24 * 60,
                   help="Minimum candles before trading (default = 7 days of 1m candles)")
    p.add_argument("--balance",        type=float, default=1000.0,
                   help="Simulated starting balance for backtest")
    p.add_argument("--payout",         type=float, default=0.80,
                   help="Broker net payout fraction (e.g. 0.80 = 80%%)")
    p.add_argument("--min-confidence", type=float, default=0.55,
                   help="Minimum AI confidence to trade (backtest + live)")
    p.add_argument("--lookback",       type=int,   default=cfg.ml.lookback,
                   help="Feature lookback window (candles)")
    p.add_argument("--min-wr",         type=float, default=0.55,
                   help="Minimum win rate to pass quality gate")
    p.add_argument("--min-pf",         type=float, default=1.2,
                   help="Minimum profit factor to pass quality gate")
    p.add_argument("--min-trades",     type=int,   default=10,
                   help="Minimum trade count to pass quality gate")
    p.add_argument("--session",        type=int,   default=60,
                   help="Live trading session length (minutes)")
    p.add_argument("--interval",       type=int,   default=65,
                   help="Seconds between live trade attempts")
    return p.parse_args()


if __name__ == "__main__":
    args = _build_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print(f"\n{yellow('  Interrupted.')}")
        sys.exit(0)
