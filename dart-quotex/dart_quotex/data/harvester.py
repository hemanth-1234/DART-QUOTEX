"""
dart_quotex/data/harvester.py
Deep historical data harvester for Quotex OTC assets.

Quotex limits single requests to ~180 candles.  This module stitches
multiple chunks together to build an arbitrarily long history, storing
everything in the local SQLite database so live trading never has to
make heavy API calls.

Usage (CLI):
    python -m dart_quotex.data.harvester --asset EURUSD_OTC --total 5000
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional, Tuple

from dart_quotex.config import cfg
from dart_quotex.data.database import Database

logger = logging.getLogger(__name__)

# Type alias
CandleRow = Tuple[int, float, float, float, float, float]  # ts,o,h,l,c,v


class DataHarvester:
    """
    Fetches historical candles from Quotex in chunks and stores them
    in the local SQLite database.

    Parameters
    ----------
    client : QuotexClient
        Already-connected Quotex API wrapper.
    db : Database
        SQLite database instance.
    """

    def __init__(self, client, db: Database) -> None:  # noqa: ANN001
        self.client = client
        self.db = db

    # ── public API ────────────────────────────────────────────────────────────

    async def harvest(
        self,
        asset: str,
        granularity: int,
        total_candles: int = 5_000,
        chunk_size: int = 180,
    ) -> int:
        """
        Fetch `total_candles` candles for `asset` at `granularity` seconds
        and persist them.  Returns total rows stored.
        """
        logger.info(
            "Harvesting %d candles for %s (gran=%ds, chunk=%d)",
            total_candles,
            asset,
            granularity,
            chunk_size,
        )

        existing = self.db.count_candles(asset, granularity)
        oldest_ts = self.db.oldest_ts(asset, granularity)

        # We fetch going backwards in time from oldest known ts (or now)
        end_time: Optional[int] = int(oldest_ts) - 1 if oldest_ts else None
        all_rows: List[CandleRow] = []

        while len(all_rows) < (total_candles - existing):
            chunk = await self._fetch_chunk(asset, granularity, chunk_size, end_time)
            if not chunk:
                logger.info("No more data available for %s", asset)
                break

            all_rows.extend(chunk)
            # Move end_time back to fetch the next older chunk
            end_time = min(r[0] for r in chunk) - 1

            logger.debug(
                "  chunk fetched: %d rows | total buffered: %d",
                len(chunk),
                len(all_rows),
            )

            # Anti-automation delay
            await _random_delay(cfg.quotex.delay_min, cfg.quotex.delay_max)

        stored = self.db.upsert_candles(asset, granularity, all_rows)
        total_in_db = self.db.count_candles(asset, granularity)
        logger.info(
            "Harvest complete: stored %d new rows | total in DB: %d",
            stored,
            total_in_db,
        )
        return stored

    async def refresh_recent(
        self,
        asset: str,
        granularity: int,
        n: int = 10,
    ) -> List[CandleRow]:
        """
        Fetch the latest `n` candles and update the DB.
        Used during live trading to keep the DB current.
        """
        chunk = await self._fetch_chunk(asset, granularity, n, end_time=None)
        if chunk:
            self.db.upsert_candles(asset, granularity, chunk)
        return chunk

    # ── internal ──────────────────────────────────────────────────────────────

    async def _fetch_chunk(
        self,
        asset: str,
        granularity: int,
        count: int,
        end_time: Optional[int],
    ) -> List[CandleRow]:
        """
        Call QuotexClient.get_candles_deep and return normalised rows.
        Falls back to get_candles if end_time is None (most-recent fetch).
        """
        try:
            if end_time is None:
                raw = await self.client.get_candles(asset, granularity, count)
            else:
                raw = await self.client.get_candles_deep(
                    asset, granularity, count, end_time
                )
        except Exception as exc:
            logger.error("API error fetching candles: %s", exc)
            return []

        return _normalise(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalise(raw: list) -> List[CandleRow]:
    """
    Convert whatever pyquotex returns into (ts, o, h, l, c, v) tuples.
    pyquotex candle dicts can have slightly different key names depending
    on version; we handle the common variants.
    """
    rows: List[CandleRow] = []
    for c in raw:
        if isinstance(c, dict):
            ts = int(
                c.get("time") or c.get("ts") or c.get("timestamp") or c.get("from", 0)
            )
            o = float(c.get("open", 0))
            h = float(c.get("max") or c.get("high", 0))
            l = float(c.get("min") or c.get("low", 0))
            cl = float(c.get("close") or c.get("value", 0))
            v = float(c.get("volume", 0))
        elif isinstance(c, (list, tuple)) and len(c) >= 5:
            ts, o, h, l, cl = int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4])
            v = float(c[5]) if len(c) > 5 else 0.0
        else:
            continue

        if ts and o and cl:
            rows.append((ts, o, h or cl, l or cl, cl, v))

    return rows


async def _random_delay(lo: float, hi: float) -> None:
    import random
    await asyncio.sleep(random.uniform(lo, hi))


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from dart_quotex.api.quotex_client import QuotexClient

    parser = argparse.ArgumentParser(description="Harvest historical candles")
    parser.add_argument("--asset", default=cfg.quotex.asset)
    parser.add_argument("--granularity", type=int, default=cfg.data.granularity)
    parser.add_argument("--total", type=int, default=cfg.data.harvest_total)
    parser.add_argument("--chunk", type=int, default=cfg.data.harvest_chunk)
    args = parser.parse_args()

    async def _run() -> None:
        client = QuotexClient()
        await client.connect()
        db = Database(cfg.data.db_path)
        harvester = DataHarvester(client, db)
        await harvester.harvest(args.asset, args.granularity, args.total, args.chunk)
        await client.disconnect()

    asyncio.run(_run())
