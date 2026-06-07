"""
dart_quotex/api/quotex_client.py
Drop-in Quotex adapter for DART.

Public interface mirrors the original deriv_client.py so that all upstream
AI / risk modules remain unchanged.  Internally it wraps the `pyquotex`
library (https://github.com/cleitonleonel/pyquotex).

Key public methods
------------------
connect()                          → None
disconnect()                       → None
get_balance()                      → float
get_candles(asset, gran, count)    → list[dict]
get_candles_deep(asset, gran, n,   → list[dict]
                 end_time)
buy(asset, amount, direction,      → (bool, trade_id)
    duration)
check_win(trade_id)                → (bool, payout)   WIN→True, LOSS→False
subscribe_realtime(asset, cb)      → None
unsubscribe_realtime(asset)        → None
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from dart_quotex.config import cfg

logger = logging.getLogger(__name__)

_QuotexLib          = None
_PYQUOTEX_AVAILABLE = False
for _try_module in ("pyquotex.stable_api", "quotexapi.stable_api"):
    try:
        import importlib as _il
        _mod = _il.import_module(_try_module)
        _cls = getattr(_mod, "Quotex", None)
        if _cls is not None:
            _QuotexLib          = _cls
            _PYQUOTEX_AVAILABLE = True
        break
    except Exception:
        continue


# ──────────────────────────────────────────────────────────────────────────────
# Optional import guard — pyquotex might not be installed in some envs
# ──────────────────────────────────────────────────────────────────────────────
    logger.warning(
        "pyquotex not installed — QuotexClient will run in MOCK mode. "
        "Install with: pip install git+https://github.com/cleitonleonel/pyquotex"
    )


# ──────────────────────────────────────────────────────────────────────────────
# QuotexClient
# ──────────────────────────────────────────────────────────────────────────────

class QuotexClient:
    """
    Async Quotex broker adapter.

    Falls back to a realistic mock when pyquotex is not installed (useful
    for unit tests and CI environments).
    """

    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        self._email = email or cfg.quotex.email
        self._password = password or cfg.quotex.password
        self._mode = mode or cfg.quotex.mode
        self._api: Any = None
        self._connected = False
        self._mock = not _PYQUOTEX_AVAILABLE

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish authenticated WebSocket session."""
        if self._mock:
            logger.info("[MOCK] QuotexClient connected (mock mode)")
            self._connected = True
            return

        self._api = _QuotexLib(
            email=self._email,
            password=self._password,
        )

        try:
            check, reason = await self._api.connect()
            if not check:
                raise ConnectionError(f"Quotex login failed: {reason}")

            # Switch demo/real account
            await self._set_account_mode(self._mode)
            self._connected = True
            logger.info("QuotexClient connected (%s account)", self._mode)
        except Exception as exc:
            logger.error("QuotexClient.connect failed: %s", exc)
            raise

    async def disconnect(self) -> None:
        if self._api and hasattr(self._api, "close"):
            try:
                self._api.close()
            except Exception:
                pass
        self._connected = False
        logger.info("QuotexClient disconnected")

    # ── account ───────────────────────────────────────────────────────────────

    async def get_balance(self) -> float:
        """Return current account balance."""
        if self._mock:
            return _MockData.balance()

        await _delay(cfg.quotex.delay_min, cfg.quotex.delay_max)
        try:
            balance = await self._api.get_balance()
            return float(balance)
        except Exception as exc:
            logger.error("get_balance error: %s", exc)
            return 0.0

    # ── market data ───────────────────────────────────────────────────────────

    async def get_candles(
        self,
        asset: str,
        granularity: int,
        count: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch the most-recent `count` candles.

        Returns list of dicts with keys: time, open, high, low, close, volume
        """
        if self._mock:
            return _MockData.candles(asset, count)

        await _delay(cfg.quotex.delay_min, cfg.quotex.delay_max)
        try:
            # pyquotex: get_candles(asset, offset, period, count)
            # offset=0 means now; period=granularity in seconds
            candles = await self._api.get_candles(asset, 0, granularity, count)
            return _normalise_candles(candles)
        except Exception as exc:
            logger.error("get_candles error [%s]: %s", asset, exc)
            return []

    async def get_candles_deep(
        self,
        asset: str,
        granularity: int,
        count: int,
        end_time: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch `count` candles ending at (or before) `end_time` (unix ts).

        Used by DataHarvester to page backwards through history.
        Quotex allows passing an `offset` from the current time or an
        explicit epoch offset parameter (library-version-dependent).
        We calculate the offset in seconds from now.
        """
        if self._mock:
            return _MockData.candles_at(asset, count, end_time, granularity)

        await _delay(cfg.quotex.delay_min, cfg.quotex.delay_max)
        try:
            now = int(time.time())
            offset = max(0, now - end_time)   # seconds back from now
            candles = await self._api.get_candles(asset, offset, granularity, count)
            return _normalise_candles(candles)
        except Exception as exc:
            logger.error("get_candles_deep error [%s]: %s", asset, exc)
            return []

    async def subscribe_realtime(
        self,
        asset: str,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Subscribe to real-time price ticks for `asset`."""
        if self._mock:
            logger.info("[MOCK] subscribed to realtime %s", asset)
            return

        try:
            await self._api.subscribe_realtime_candle(asset, callback)
            logger.info("Subscribed to realtime candles: %s", asset)
        except Exception as exc:
            logger.error("subscribe_realtime error: %s", exc)

    async def unsubscribe_realtime(self, asset: str) -> None:
        if self._mock:
            return
        try:
            await self._api.unsubscribe_realtime_candle(asset)
        except Exception:
            pass

    # ── trading ───────────────────────────────────────────────────────────────

    async def buy(
        self,
        asset: str,
        amount: float,
        direction: str,   # "call" | "put"
        duration: int,    # seconds
    ) -> Tuple[bool, Any]:
        """
        Place a binary options trade.

        Returns (success: bool, trade_id: str | int)
        On failure returns (False, None).
        """
        if self._mock:
            trade_id = _MockData.next_trade_id()
            logger.info(
                "[MOCK] BUY %s %s $%.2f %ds → id=%s",
                direction.upper(), asset, amount, duration, trade_id,
            )
            return True, trade_id

        await _delay(cfg.quotex.delay_min, cfg.quotex.delay_max)
        try:
            success, trade_id = await self._api.buy(amount, asset, direction, duration)
            if success:
                logger.info(
                    "Trade placed: %s %s $%.2f %ds → id=%s",
                    direction.upper(), asset, amount, duration, trade_id,
                )
            else:
                logger.warning("Trade rejected by broker: %s", trade_id)
            return bool(success), trade_id
        except Exception as exc:
            logger.error("buy() error: %s", exc)
            return False, None

    async def check_win(
        self,
        trade_id: Any,
        wait: bool = True,
    ) -> Tuple[bool, float]:
        """
        Check the outcome of a trade.

        Returns (won: bool, net_payout: float)
        If wait=True, blocks until the trade expires.
        """
        if self._mock:
            won, payout = _MockData.trade_result(trade_id)
            logger.info("[MOCK] check_win id=%s → %s payout=%.2f", trade_id, "WIN" if won else "LOSS", payout)
            return won, payout

        await _delay(0.3, 0.8)
        try:
            if wait:
                # pyquotex: check_win returns (result_str, net_payout)
                result, payout = await self._api.check_win(trade_id)
            else:
                result, payout = await self._api.check_win_v2(trade_id)

            won = str(result).lower() in ("win", "true", "1", "profit")
            return won, float(payout or 0)
        except Exception as exc:
            logger.error("check_win error: %s", exc)
            return False, 0.0

    # ── utilities ─────────────────────────────────────────────────────────────

    async def get_payout(self, asset: str) -> float:
        """Return current payout % for `asset` (0-1 scale)."""
        if self._mock:
            return 0.80

        try:
            payout = await self._api.get_payout(asset)
            return float(payout or 80) / 100.0
        except Exception:
            return 0.80

    async def _set_account_mode(self, mode: str) -> None:
        mode = mode.lower()
        if mode == "real":
            await self._api.change_account("REAL")
        else:
            await self._api.change_account("PRACTICE")


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helper
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_candles(raw: Any) -> List[Dict[str, Any]]:
    """Normalise whatever pyquotex returns into a consistent dict format."""
    if not raw:
        return []

    result = []
    for c in raw:
        if isinstance(c, dict):
            ts = int(c.get("time") or c.get("ts") or c.get("from") or 0)
            result.append({
                "time":   ts,
                "open":   float(c.get("open", 0)),
                "high":   float(c.get("max") or c.get("high", 0)),
                "low":    float(c.get("min") or c.get("low", 0)),
                "close":  float(c.get("close") or c.get("value", 0)),
                "volume": float(c.get("volume", 0)),
            })
        elif isinstance(c, (list, tuple)) and len(c) >= 5:
            result.append({
                "time": int(c[0]), "open": float(c[1]),
                "high": float(c[2]), "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]) if len(c) > 5 else 0.0,
            })

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Anti-automation delay
# ──────────────────────────────────────────────────────────────────────────────

async def _delay(lo: float, hi: float) -> None:
    await asyncio.sleep(random.uniform(lo, hi))


# ──────────────────────────────────────────────────────────────────────────────
# Mock data generator (no pyquotex required)
# ──────────────────────────────────────────────────────────────────────────────

class _MockData:
    _trade_counter = 0
    _balance = 1_000.0

    @classmethod
    def balance(cls) -> float:
        return cls._balance

    @classmethod
    def candles(cls, asset: str, count: int) -> List[Dict[str, Any]]:
        now = int(time.time())
        base = 1.10000 + random.uniform(-0.005, 0.005)
        out = []
        for i in range(count):
            ts = now - (count - i) * 60
            o = base + random.uniform(-0.002, 0.002)
            h = o + random.uniform(0, 0.001)
            l = o - random.uniform(0, 0.001)
            c = l + random.uniform(0, h - l)
            base = c
            out.append({"time": ts, "open": o, "high": h, "low": l, "close": c, "volume": random.uniform(100, 500)})
        return out

    @classmethod
    def candles_at(cls, asset: str, count: int, end_time: int, gran: int) -> List[Dict[str, Any]]:
        base = 1.10000
        out = []
        for i in range(count):
            ts = end_time - (count - i) * gran
            o = base + random.uniform(-0.002, 0.002)
            h = o + random.uniform(0, 0.001)
            l = o - random.uniform(0, 0.001)
            c = l + random.uniform(0, h - l)
            base = c
            out.append({"time": ts, "open": o, "high": h, "low": l, "close": c, "volume": random.uniform(100, 500)})
        return out

    @classmethod
    def next_trade_id(cls) -> str:
        cls._trade_counter += 1
        return f"mock_{cls._trade_counter:06d}"

    @classmethod
    def trade_result(cls, trade_id: Any) -> Tuple[bool, float]:
        won = random.random() > 0.45   # slight edge for demo
        payout = 8.0 if won else -10.0
        cls._balance += payout
        return won, payout
