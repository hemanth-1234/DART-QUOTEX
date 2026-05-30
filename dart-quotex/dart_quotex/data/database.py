"""
dart_quotex/data/database.py
SQLite-backed store for historical OHLCV candles.

Schema
------
candles(asset TEXT, granularity INT, ts INT, open REAL, high REAL,
        low REAL, close REAL, volume REAL)
  PRIMARY KEY (asset, granularity, ts)

trades(id INTEGER PK, asset, direction, stake, payout, result,
       confidence, ts_open, ts_close)
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS candles (
    asset       TEXT    NOT NULL,
    granularity INTEGER NOT NULL,
    ts          INTEGER NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL DEFAULT 0,
    PRIMARY KEY (asset, granularity, ts)
);

CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON candles (asset, granularity, ts DESC);

CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset       TEXT    NOT NULL,
    direction   TEXT    NOT NULL,   -- CALL | PUT
    stake       REAL    NOT NULL,
    payout      REAL    NOT NULL,
    result      TEXT,               -- WIN | LOSS | NULL (pending)
    confidence  REAL,
    ts_open     INTEGER NOT NULL,
    ts_close    INTEGER
);
"""


# ──────────────────────────────────────────────────────────────────────────────
# Database class
# ──────────────────────────────────────────────────────────────────────────────

class Database:
    """Thread-safe SQLite wrapper."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── internal ──────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._conn() as con:
            con.executescript(_DDL)
        logger.info("Database initialised at %s", self.db_path)

    # ── candles ───────────────────────────────────────────────────────────────

    def upsert_candles(
        self,
        asset: str,
        granularity: int,
        rows: List[Tuple[int, float, float, float, float, float]],
    ) -> int:
        """
        Upsert OHLCV rows.

        rows: list of (ts, open, high, low, close, volume)
        Returns number of rows inserted/replaced.
        """
        if not rows:
            return 0
        with self._conn() as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO candles
                    (asset, granularity, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(asset, granularity, *r) for r in rows],
            )
        logger.debug("Upserted %d candles for %s@%ds", len(rows), asset, granularity)
        return len(rows)

    def get_candles(
        self,
        asset: str,
        granularity: int,
        limit: int = 500,
        since_ts: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return candles as a DataFrame sorted by ts ascending."""
        with self._conn() as con:
            if since_ts is not None:
                rows = con.execute(
                    """
                    SELECT ts, open, high, low, close, volume
                    FROM candles
                    WHERE asset=? AND granularity=? AND ts>=?
                    ORDER BY ts ASC
                    LIMIT ?
                    """,
                    (asset, granularity, since_ts, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    """
                    SELECT ts, open, high, low, close, volume
                    FROM   candles
                    WHERE  asset=? AND granularity=?
                    ORDER  BY ts DESC
                    LIMIT  ?
                    """,
                    (asset, granularity, limit),
                ).fetchall()
                rows = list(reversed(rows))   # back to ascending

        if not rows:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df.set_index("ts", inplace=True)
        return df

    def count_candles(self, asset: str, granularity: int) -> int:
        with self._conn() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM candles WHERE asset=? AND granularity=?",
                (asset, granularity),
            ).fetchone()
        return row[0] if row else 0

    def oldest_ts(self, asset: str, granularity: int) -> Optional[int]:
        with self._conn() as con:
            row = con.execute(
                "SELECT MIN(ts) FROM candles WHERE asset=? AND granularity=?",
                (asset, granularity),
            ).fetchone()
        return row[0] if row else None

    # ── trades ────────────────────────────────────────────────────────────────

    def insert_trade(
        self,
        asset: str,
        direction: str,
        stake: float,
        payout: float,
        confidence: float,
        ts_open: int,
    ) -> int:
        with self._conn() as con:
            cur = con.execute(
                """
                INSERT INTO trades (asset, direction, stake, payout,
                                    confidence, ts_open)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (asset, direction, stake, payout, confidence, ts_open),
            )
        return cur.lastrowid  # type: ignore[return-value]

    def close_trade(self, trade_id: int, result: str, ts_close: int) -> None:
        with self._conn() as con:
            con.execute(
                "UPDATE trades SET result=?, ts_close=? WHERE id=?",
                (result, ts_close, trade_id),
            )

    def get_recent_trades(self, n: int = 100) -> pd.DataFrame:
        with self._conn() as con:
            rows = con.execute(
                """
                SELECT id, asset, direction, stake, payout, result,
                       confidence, ts_open, ts_close
                FROM   trades
                ORDER  BY ts_open DESC
                LIMIT  ?
                """,
                (n,),
            ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(
            rows,
            columns=["id", "asset", "direction", "stake", "payout",
                     "result", "confidence", "ts_open", "ts_close"],
        )

    def win_rate(self, asset: Optional[str] = None, n: int = 50) -> float:
        """Return win rate over last n completed trades."""
        with self._conn() as con:
            if asset:
                row = con.execute(
                    """
                    SELECT AVG(CASE WHEN result='WIN' THEN 1.0 ELSE 0.0 END)
                    FROM (
                        SELECT result FROM trades
                        WHERE  asset=? AND result IS NOT NULL
                        ORDER  BY ts_open DESC LIMIT ?
                    )
                    """,
                    (asset, n),
                ).fetchone()
            else:
                row = con.execute(
                    """
                    SELECT AVG(CASE WHEN result='WIN' THEN 1.0 ELSE 0.0 END)
                    FROM (
                        SELECT result FROM trades
                        WHERE  result IS NOT NULL
                        ORDER  BY ts_open DESC LIMIT ?
                    )
                    """,
                    (n,),
                ).fetchone()
        return float(row[0] or 0.5)
