"""
dart_quotex/arbitrage/external_feed.py
Module 2 — External Real-Time Price Feed
=========================================
Async WebSocket connections to low-latency external price sources.
Supports Twelve Data and Finnhub; auto-falls back to REST polling.

Usage
-----
    feed = ExternalFeedFactory.create(provider="twelvedata",
                                      api_key=TWELVEDATA_API_KEY,
                                      symbol="EUR/USD")
    await feed.start()
    tick = feed.latest()   # PriceTick(price, ts_ms)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Shared data container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PriceTick:
    price:  float
    ts_ms:  float   # epoch milliseconds


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class BaseFeed(ABC):
    def __init__(self, symbol: str, buffer: int = 500) -> None:
        self.symbol   = symbol
        self._ticks:  Deque[PriceTick] = deque(maxlen=buffer)
        self._running = False

    @abstractmethod
    async def start(self) -> None: ...

    async def stop(self) -> None:
        self._running = False

    def latest(self) -> Optional[PriceTick]:
        return self._ticks[-1] if self._ticks else None

    def recent(self, n: int = 60) -> list:
        return list(self._ticks)[-n:]

    def _push(self, price: float, ts_ms: Optional[float] = None) -> None:
        if price > 0:
            self._ticks.append(PriceTick(
                price=price,
                ts_ms=ts_ms if ts_ms else time.time() * 1000,
            ))


# ──────────────────────────────────────────────────────────────────────────────
# Twelve Data feed
# ──────────────────────────────────────────────────────────────────────────────

class TwelveDataFeed(BaseFeed):
    """
    Real-time WebSocket feed from Twelve Data.
    Falls back to REST polling (~2 req/min free tier).
    """

    WS_URL   = "wss://ws.twelvedata.com/v1/quotes/price"
    REST_URL = "https://api.twelvedata.com/price"

    def __init__(self, api_key: str, symbol: str = "EUR/USD", buffer: int = 500) -> None:
        super().__init__(symbol, buffer)
        self.api_key = api_key

    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._run())
        log.info("TwelveDataFeed starting for %s", self.symbol)

    async def stop(self) -> None:
        self._running = False

    async def _run(self) -> None:
        backoff = 2.0
        while self._running:
            try:
                await self._ws_loop()
                backoff = 2.0
            except Exception as exc:
                log.warning("TwelveData WS error: %s — retry in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    async def _ws_loop(self) -> None:
        try:
            import websockets
        except ImportError:
            log.warning("websockets not installed — using REST polling")
            await self._rest_loop()
            return

        async with websockets.connect(self.WS_URL, ping_interval=20) as ws:
            await ws.send(json.dumps({
                "action": "subscribe",
                "params": {
                    "symbols": self.symbol,
                    "apikey":  self.api_key,
                },
            }))
            async for msg in ws:
                if not self._running:
                    break
                try:
                    d = json.loads(msg)
                    if d.get("event") == "price":
                        ts_ms = float(d.get("timestamp", time.time())) * 1000
                        self._push(float(d.get("price", 0)), ts_ms)
                except Exception:
                    pass

    async def _rest_loop(self) -> None:
        sym = self.symbol.replace("/", "%2F")
        url = f"{self.REST_URL}?symbol={sym}&apikey={self.api_key}"
        while self._running:
            try:
                with urllib.request.urlopen(url, timeout=4) as r:
                    d = json.loads(r.read())
                    self._push(float(d.get("price", 0)))
            except Exception as exc:
                log.debug("TwelveData REST: %s", exc)
            await asyncio.sleep(0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Finnhub feed
# ──────────────────────────────────────────────────────────────────────────────

class FinnhubFeed(BaseFeed):
    """
    Real-time WebSocket feed from Finnhub.
    Free tier: 60 messages/min, adequate for single-symbol monitoring.
    Symbol format for forex: "OANDA:EUR_USD"
    """

    WS_URL = "wss://ws.finnhub.io"

    def __init__(self, api_key: str, symbol: str = "OANDA:EUR_USD",
                 buffer: int = 500) -> None:
        super().__init__(symbol, buffer)
        self.api_key = api_key

    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._run())
        log.info("FinnhubFeed starting for %s", self.symbol)

    async def stop(self) -> None:
        self._running = False

    async def _run(self) -> None:
        backoff = 2.0
        while self._running:
            try:
                await self._ws_loop()
                backoff = 2.0
            except Exception as exc:
                log.warning("Finnhub WS error: %s — retry in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    async def _ws_loop(self) -> None:
        try:
            import websockets
        except ImportError:
            log.warning("websockets not installed — Finnhub feed unavailable")
            await asyncio.sleep(60)
            return

        url = f"{self.WS_URL}?token={self.api_key}"
        async with websockets.connect(url, ping_interval=20) as ws:
            await ws.send(json.dumps({
                "type": "subscribe",
                "symbol": self.symbol,
            }))
            async for msg in ws:
                if not self._running:
                    break
                try:
                    d = json.loads(msg)
                    if d.get("type") == "trade" and d.get("data"):
                        for trade in d["data"]:
                            price = float(trade.get("p", 0))
                            ts_ms = float(trade.get("t", time.time() * 1000))
                            self._push(price, ts_ms)
                except Exception:
                    pass


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

class ExternalFeedFactory:
    """Create the right feed based on provider name."""

    @staticmethod
    def create(
        provider: str,
        api_key:  str,
        symbol:   str,
        buffer:   int = 500,
    ) -> BaseFeed:
        p = provider.lower().strip()
        if p in ("twelvedata", "twelve_data", "12data"):
            return TwelveDataFeed(api_key=api_key, symbol=symbol, buffer=buffer)
        if p in ("finnhub",):
            return FinnhubFeed(api_key=api_key, symbol=symbol, buffer=buffer)
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Supported: 'twelvedata', 'finnhub'"
        )
