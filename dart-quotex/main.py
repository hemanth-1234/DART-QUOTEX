#!/usr/bin/env python3
"""
main.py — DART-Quotex command-line entry point

Commands
--------
  trade      Start live trading session
  harvest    Download historical data into SQLite
  backtest   Run backtester on stored history
  advisor    One-shot advisor mode (print signal and exit)

Examples
--------
  python main.py trade --asset EURUSD_OTC --session 60
  python main.py harvest --asset EURUSD_OTC --total 5000
  python main.py backtest --asset EURUSD_OTC --balance 1000
  python main.py advisor --asset EURUSD_OTC
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dart_quotex.config import cfg


# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)-28s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("dart_quotex.log", encoding="utf-8"),
        ],
    )
    # Silence noisy third-party loggers
    for noisy in ("websocket", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────────────────
# Command handlers
# ──────────────────────────────────────────────────────────────────────────────

async def cmd_trade(args: argparse.Namespace) -> None:
    from dart_quotex.trader import LiveTrader

    trader = LiveTrader(
        session_minutes=args.session,
        trade_interval=args.interval,
    )
    await trader.run(asset=args.asset)


async def cmd_harvest(args: argparse.Namespace) -> None:
    from dart_quotex.advisor import AIAdvisor

    advisor = AIAdvisor()
    await advisor.connect()
    stored = await advisor.harvest_history(
        asset=args.asset,
        total_candles=args.total,
    )
    print(f"\nHarvest complete: {stored} new rows stored")
    await advisor.disconnect()


async def cmd_backtest(args: argparse.Namespace) -> None:
    from dart_quotex.advisor import AIAdvisor
    from dart_quotex.backtester import Backtester
    from dart_quotex.data.database import Database

    db = Database(cfg.data.db_path)
    advisor = AIAdvisor()

    bt = Backtester(
        db=db,
        advisor=advisor,
        lookback=cfg.ml.lookback,
        train_online=args.train_online,
    )

    result = bt.run(
        asset=args.asset,
        granularity=args.granularity,
        start_balance=args.balance,
        payout=args.payout,
        min_confidence=args.confidence,
        output_dir=Path("backtest_output") if args.save else None,
    )
    print(result.summary())

    if args.save:
        result.save_csv(Path("backtest_output") / f"trades_{args.asset}.csv")
        print("Trade log saved to backtest_output/")

    if args.crossval:
        print("\nRunning 5-fold time-series cross-validation...")
        fold_results = bt.cross_validate(
            asset=args.asset,
            granularity=args.granularity,
            start_balance=args.balance,
            payout=args.payout,
        )
        for i, r in enumerate(fold_results, 1):
            print(f"  Fold {i}: WR={r.win_rate:.1%} ROI={r.roi:+.1%} DD={r.max_drawdown:.1%}")


async def cmd_advisor(args: argparse.Namespace) -> None:
    """One-shot advisor: print signal and exit (for integration testing)."""
    from dart_quotex.advisor import AIAdvisor

    advisor = AIAdvisor()
    await advisor.connect()

    direction, confidence = await advisor.get_signal(asset=args.asset)
    print(f"\nAsset     : {args.asset}")
    print(f"Direction : {direction}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Action    : {'TRADE' if confidence >= cfg.risk.min_confidence else 'SKIP (low confidence)'}")

    await advisor.disconnect()


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dart-quotex",
        description="DART-Quotex: AI-driven binary options for Quotex OTC markets",
    )
    parser.add_argument(
        "--log-level", default=cfg.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── trade ──────────────────────────────────────────────────────────────────
    p_trade = sub.add_parser("trade", help="Start live trading session")
    p_trade.add_argument("--asset", default=cfg.quotex.asset)
    p_trade.add_argument("--session", type=int, default=60, help="Session length (minutes)")
    p_trade.add_argument("--interval", type=int, default=65, help="Seconds between signals")

    # ── harvest ───────────────────────────────────────────────────────────────
    p_harvest = sub.add_parser("harvest", help="Download historical data")
    p_harvest.add_argument("--asset", default=cfg.quotex.asset)
    p_harvest.add_argument("--total", type=int, default=cfg.data.harvest_total)

    # ── backtest ──────────────────────────────────────────────────────────────
    p_bt = sub.add_parser("backtest", help="Backtest on stored history")
    p_bt.add_argument("--asset", default=cfg.quotex.asset)
    p_bt.add_argument("--granularity", type=int, default=cfg.data.granularity)
    p_bt.add_argument("--balance", type=float, default=1000.0)
    p_bt.add_argument("--payout", type=float, default=0.80)
    p_bt.add_argument("--confidence", type=float, default=0.55)
    p_bt.add_argument("--save", action="store_true", help="Save trade log to CSV")
    p_bt.add_argument("--crossval", action="store_true", help="Run 5-fold cross-validation")
    p_bt.add_argument("--train-online", action="store_true", default=True)

    # ── advisor ───────────────────────────────────────────────────────────────
    p_adv = sub.add_parser("advisor", help="One-shot signal (for integration testing)")
    p_adv.add_argument("--asset", default=cfg.quotex.asset)

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.log_level)

    dispatch = {
        "trade":    cmd_trade,
        "harvest":  cmd_harvest,
        "backtest": cmd_backtest,
        "advisor":  cmd_advisor,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    asyncio.run(handler(args))


if __name__ == "__main__":
    main()


# ── dashboard command (add to build_parser) ───────────────────────────────────
async def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch Streamlit web dashboard."""
    import subprocess, sys
    script = Path(__file__).parent / "dart_quotex" / "gui" / "web_dashboard.py"
    port   = getattr(args, "port", 8501)
    print(f"\nLaunching dashboard at http://localhost:{port}\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(script),
         "--server.port", str(port), "--server.headless", "true"],
        check=False,
    )


async def cmd_gui(args: argparse.Namespace) -> None:
    """Launch CustomTkinter desktop GUI."""
    from dart_quotex.gui.desktop_app import launch
    launch()   # blocking
