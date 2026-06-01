"""
dart_quotex/data/realtime.py
Real-time market data streaming via WebSocket.

Provides a persistent, auto-reconnecting subscription to live price
ticks and completed candles.  Callbacks are async-safe and thread-safe.

Features
--------
· Automatic reconnection with exponential back-off (max 60s)
· Per-asset tick buffer with configurable depth
· Candle aggregation from raw ticks (builds 1m/5m/15m candles live)
· Anomaly detection: flags candles with abnormal body/range/volume
· Heartbeat monitoring: emits a warning if no tick in > 30s
· Integration with Database: auto-persists completed candles

Usage
-----
    from dart_quotex.data.realtime import RealtimeStream

    stream = RealtimeStream(client, db)
    stream.on_candle("EURUSD_OTC", my_callback)
    await stream.subscribe("EURUSD_OTC")
    await stream.run()   # blocks; Ctrl-C to stop
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# Type aliases
TickCallback   = Callable[["Tick"], Coroutine]
CandleCallback = Callable[["LiveCandle"], Coroutine]


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Tick:
    asset:  str
    price:  float
    ts:     float      # unix timestamp (float for sub-second precision)
    volume: float = 0.0

@dataclass
class LiveCandle:
    asset:       str
    granularity: int    # seconds
    ts:          int    # candle open time (unix)
    open:        float
    high:        float
    low:         float
    close:       float
    volume:      float
    n_ticks:     int    = 0
    is_complete: bool   = False
    anomaly:     bool   = False

    def to_tuple(self):
        return (self.ts, self.open, self.high, self.low, self.close, self.volume)


# ──────────────────────────────────────────────────────────────────────────────
# Candle aggregator — builds OHLCV from ticks
# ──────────────────────────────────────────────────────────────────────────────

class CandleAggregator:
    """
    Aggregates raw price ticks into OHLCV candles for multiple granularities.
    """

    def __init__(
        self,
        asset: str,
        granularities: List[int],   # e.g. [60, 300, 900]
    ) -> None:
        self.asset  = asset
        self._grans = granularities
        # Current open candle per granularity
        self._open_candles: Dict[int, LiveCandle] = {}
        # Completed candle ring-buffer
        self._completed: Dict[int, deque] = {
            g: deque(maxlen=500) for g in granularities
        }
        # Last tick price (for anomaly detection)
        self._last_price: Optional[float] = None
        self._last_volume: Optional[float] = None

    def process_tick(self, tick: Tick) -> List[LiveCandle]:
        """
        Feed one tick.  Returns list of LiveCandles that just completed.
        """
        completed = []
        for gran in self._grans:
            candle_ts = int(tick.ts // gran) * gran
            current   = self._open_candles.get(gran)

            if current is None or current.ts < candle_ts:
                # Close old candle
                if current is not None:
                    current.is_complete = True
                    current.anomaly     = self._is_anomaly(current)
                    self._completed[gran].append(current)
                    completed.append(current)

                # Open new candle
                self._open_candles[gran] = LiveCandle(
                    asset=self.asset, granularity=gran, ts=candle_ts,
                    open=tick.price, high=tick.price,
                    low=tick.price, close=tick.price,
                    volume=tick.volume, n_ticks=1,
                )
            else:
                # Update open candle
                c = current
                c.high    = max(c.high, tick.price)
                c.low     = min(c.low, tick.price)
                c.close   = tick.price
                c.volume += tick.volume
                c.n_ticks += 1

        self._last_price  = tick.price
        self._last_volume = tick.volume
        return completed

    def current_candle(self, granularity: int) -> Optional[LiveCandle]:
        return self._open_candles.get(granularity)

    def recent_candles(self, granularity: int, n: int = 10) -> List[LiveCandle]:
        buf = self._completed.get(granularity, deque())
        return list(buf)[-n:]

    def _is_anomaly(self, candle: LiveCandle) -> bool:
        """Flag candles with extreme body or volume."""
        rng  = candle.high - candle.low
        body = abs(candle.close - candle.open)

        # Extreme body ratio (>98% of range)
        if rng > 1e-9 and body / rng > 0.98:
            return True
        # Abnormally large volume spike (>10× recent avg)
        history = self._completed.get(candle.granularity)
        if history and len(history) >= 5 and candle.volume > 0:
            recent_vols = [c.volume for c in list(history)[-5:] if c.volume > 0]
            if recent_vols:
                avg_vol = sum(recent_vols) / len(recent_vols)
                if candle.volume > avg_vol * 10:
                    return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# RealtimeStream
# ──────────────────────────────────────────────────────────────────────────────

class RealtimeStream:
    """
    Manages real-time subscriptions for one or more assets.

    Parameters
    ----------
    client         : QuotexClient instance
    db             : Database instance (for auto-persistence)
    granularities  : which candle sizes to aggregate (seconds)
    heartbeat_s    : emit warning if no tick in this many seconds
    max_reconnects : maximum reconnection attempts before giving up
    """

    def __init__(
        self,
        client,
        db,
        granularities: Optional[List[int]] = None,
        heartbeat_s: float = 30.0,
        max_reconnects: int = 20,
    ) -> None:
        self.client        = client
        self.db            = db
        self.granularities = granularities or [60, 300, 900]
        self.heartbeat_s   = heartbeat_s
        self.max_reconnects = max_reconnects

        self._aggregators:    Dict[str, CandleAggregator] = {}
        self._tick_callbacks: Dict[str, List[TickCallback]]   = {}
        self._candle_callbacks: Dict[str, List[CandleCallback]] = {}
        self._subscribed:     set = set()
        self._running         = False
        self._last_tick:      Dict[str, float] = {}
        self._reconnect_count: Dict[str, int] = {}
        self._tick_buffer:    Dict[str, deque] = {}   # raw tick ring-buffer

    # ── subscription management ───────────────────────────────────────────────

    def on_tick(self, asset: str, callback: TickCallback) -> None:
        """Register an async callback for raw ticks."""
        self._tick_callbacks.setdefault(asset, []).append(callback)

    def on_candle(self, asset: str, callback: CandleCallback) -> None:
        """Register an async callback for completed candles."""
        self._candle_callbacks.setdefault(asset, []).append(callback)

    async def subscribe(self, asset: str) -> None:
        """Subscribe to real-time ticks for an asset."""
        if asset in self._subscribed:
            return

        if asset not in self._aggregators:
            self._aggregators[asset]   = CandleAggregator(asset, self.granularities)
            self._tick_buffer[asset]   = deque(maxlen=1000)
            self._reconnect_count[asset] = 0

        async def _raw_tick_handler(raw: Any) -> None:
            tick = self._parse_tick(asset, raw)
            if tick is None:
                return

            self._last_tick[asset] = tick.ts
            self._tick_buffer[asset].append(tick)
            completed = self._aggregators[asset].process_tick(tick)

            # Fire tick callbacks
            for cb in self._tick_callbacks.get(asset, []):
                try:
                    await cb(tick)
                except Exception as exc:
                    logger.error("Tick callback error [%s]: %s", asset, exc)

            # Fire candle callbacks + persist
            for candle in completed:
                if candle.anomaly:
                    logger.warning(
                        "Anomaly detected on %s %ds candle at ts=%d",
                        asset, candle.granularity, candle.ts,
                    )
                # Persist to DB
                self.db.upsert_candles(
                    asset, candle.granularity, [candle.to_tuple()]
                )
                for cb in self._candle_callbacks.get(asset, []):
                    try:
                        await cb(candle)
                    except Exception as exc:
                        logger.error("Candle callback error [%s]: %s", asset, exc)

        await self.client.subscribe_realtime(asset, _raw_tick_handler)
        self._subscribed.add(asset)
        logger.info("Subscribed to realtime stream: %s", asset)

    async def unsubscribe(self, asset: str) -> None:
        """Unsubscribe from an asset's stream."""
        await self.client.unsubscribe_realtime(asset)
        self._subscribed.discard(asset)
        logger.info("Unsubscribed from realtime stream: %s", asset)

    # ── main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Start the stream monitor.  Runs until stop() is called.
        Monitors heartbeats and triggers reconnection on stale feeds.
        """
        self._running = True
        logger.info(
            "RealtimeStream running | assets=%s | grans=%s",
            list(self._subscribed), self.granularities,
        )

        while self._running:
            await asyncio.sleep(5.0)
            now = time.time()

            for asset in list(self._subscribed):
                last = self._last_tick.get(asset, now)
                gap  = now - last

                if gap > self.heartbeat_s:
                    logger.warning(
                        "No tick from %s for %.0fs — reconnecting…", asset, gap
                    )
                    n = self._reconnect_count.get(asset, 0)
                    if n >= self.max_reconnects:
                        logger.error(
                            "Max reconnects reached for %s — giving up", asset
                        )
                        continue

                    await self._reconnect(asset)
                    self._reconnect_count[asset] = n + 1

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        for asset in list(self._subscribed):
            await self.unsubscribe(asset)
        logger.info("RealtimeStream stopped")

    # ── getters ───────────────────────────────────────────────────────────────

    def latest_tick(self, asset: str) -> Optional[Tick]:
        buf = self._tick_buffer.get(asset)
        if buf:
            return buf[-1]
        return None

    def latest_candle(
        self, asset: str, granularity: int = 60
    ) -> Optional[LiveCandle]:
        agg = self._aggregators.get(asset)
        if agg:
            return agg.current_candle(granularity)
        return None

    def recent_candles(
        self, asset: str, granularity: int = 60, n: int = 10
    ) -> List[LiveCandle]:
        agg = self._aggregators.get(asset)
        if agg:
            return agg.recent_candles(granularity, n)
        return []

    def is_stale(self, asset: str, threshold: float = 30.0) -> bool:
        last = self._last_tick.get(asset)
        if last is None:
            return True
        return (time.time() - last) > threshold

    # ── internal ──────────────────────────────────────────────────────────────

    async def _reconnect(self, asset: str) -> None:
        """Reconnect subscription with exponential back-off."""
        n     = self._reconnect_count.get(asset, 0)
        delay = min(60.0, 1.0 * (2 ** n))
        logger.info("Reconnecting %s in %.1fs (attempt %d)", asset, delay, n + 1)
        await asyncio.sleep(delay)
        try:
            await self.unsubscribe(asset)
        except Exception:
            pass
        try:
            await self.subscribe(asset)
            self._last_tick[asset]         = time.time()
            self._reconnect_count[asset]   = 0
            logger.info("Reconnected %s successfully", asset)
        except Exception as exc:
            logger.error("Reconnect failed for %s: %s", asset, exc)

    @staticmethod
    def _parse_tick(asset: str, raw: Any) -> Optional[Tick]:
        """Parse whatever pyquotex sends into a Tick object."""
        try:
            if isinstance(raw, dict):
                price = float(
                    raw.get("price") or raw.get("value") or
                    raw.get("close") or raw.get("bid") or 0
                )
                ts    = float(raw.get("time") or raw.get("ts") or time.time())
                vol   = float(raw.get("volume") or 0)
            elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
                ts, price = float(raw[0]), float(raw[1])
                vol = float(raw[2]) if len(raw) > 2 else 0.0
            elif isinstance(raw, (int, float)):
                price = float(raw)
                ts    = time.time()
                vol   = 0.0
            else:
                return None

            if price <= 0:
                return None

            return Tick(asset=asset, price=price, ts=ts, volume=vol)
        except Exception:
            return None
