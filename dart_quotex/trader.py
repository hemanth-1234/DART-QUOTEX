"""
dart_quotex/trader.py
Live trading session manager.

Runs a continuous loop during your trading window:
  1. Refresh latest candles
  2. Get AI signal
  3. Risk-gate
  4. Place order
  5. Wait for expiry
  6. Record outcome & update models

Designed for a 1-hour daily session — connect once, trade until
the session ends or the drawdown limit is hit, then save and disconnect.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from typing import Optional

from dart_quotex.advisor import AIAdvisor
from dart_quotex.config import cfg

logger = logging.getLogger(__name__)


class LiveTrader:
    """
    Async live trading loop.

    Parameters
    ----------
    session_minutes : max trading session length in minutes
    trade_interval  : seconds between signal checks (≥ candle duration)
    """

    def __init__(
        self,
        session_minutes: int = 60,
        trade_interval: int = 65,
    ) -> None:
        self.session_minutes = session_minutes
        self.trade_interval = trade_interval
        self._advisor: Optional[AIAdvisor] = None
        self._stop = False

        # Graceful shutdown on Ctrl-C or SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum: int, frame) -> None:  # noqa: ANN001
        logger.info("Shutdown signal received")
        self._stop = True

    async def run(self, asset: Optional[str] = None) -> None:
        asset = asset or cfg.quotex.asset

        logger.info(
            "Starting live trading session | asset=%s | duration=%dmin",
            asset,
            self.session_minutes,
        )

        self._advisor = AIAdvisor()
        await self._advisor.connect()

        session_end = time.time() + self.session_minutes * 60
        trade_count = 0

        try:
            while not self._stop and time.time() < session_end:
                cycle_start = time.time()

                result = await self._advisor.trade(asset=asset)

                if result:
                    trade_count += 1
                    status = "✓ WIN" if result["won"] else "✗ LOSS"
                    logger.info(
                        "[%d] %s %s $%.2f | conf=%.2f | bal=%.2f | %s",
                        trade_count,
                        result["direction"].upper(),
                        asset,
                        result["stake"],
                        result["confidence"],
                        result["balance"],
                        status,
                    )

                # Wait until next interval
                elapsed = time.time() - cycle_start
                wait = max(0, self.trade_interval - elapsed)
                if wait > 0 and not self._stop:
                    await asyncio.sleep(wait)

        except Exception as exc:
            logger.error("Trading loop error: %s", exc, exc_info=True)

        finally:
            logger.info("Session complete. Total trades: %d", trade_count)
            await self._advisor.disconnect()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DART-Quotex Live Trader")
    parser.add_argument("--asset", default=cfg.quotex.asset, help="Asset symbol")
    parser.add_argument("--session", type=int, default=60, help="Session length (minutes)")
    parser.add_argument("--interval", type=int, default=65, help="Trade interval (seconds)")
    args = parser.parse_args()

    trader = LiveTrader(session_minutes=args.session, trade_interval=args.interval)
    await trader.run(asset=args.asset)


if __name__ == "__main__":
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(_main())
