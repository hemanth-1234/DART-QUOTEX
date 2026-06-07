"""
dart_quotex/arbitrage/latency.py
Feature 5 — Latency Arbitrage System
=====================================
Monitors a fast external price feed (Twelve Data WebSocket or Dukascopy)
against Quotex's internal price.  When a statistically significant lag is
detected and the external price moves decisively, a trade signal is issued
before Quotex's price catches up.

Pipeline
--------
1.  ExternalFeed  — async WebSocket → rolling price / timestamp buffer
2.  LagMonitor    — computes rolling lag (ms) and correlation
3.  MispricingDetector — fires when lag > dynamic threshold AND direction clear
4.  LatencyArbitrageEngine — orchestrates all three + risk controls

Risk controls
-------------
· Per-trade max risk: ARBIT_MAX_RISK_PCT of balance (default 0.5 %)
· Daily loss cap:     ARBIT_DAILY_LOSS_PCT of balance (default 2 %)
· Min lag threshold:  ARBIT_MIN_LAG_MS (default 80 ms)
· Min correlation:    ARBIT_MIN_CORR   (default 0.85)
· Cool-down after hit: ARBIT_COOLDOWN_S (default 30 s)

Activation
----------
Set ENABLE_LATENCY_ARBITRAGE=true in .env, provide TWELVEDATA_API_KEY.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PriceTick:
    price:  float
    ts_ms:  float    # epoch milliseconds


@dataclass
class LagSnapshot:
    lag_ms:      float   # positive = external is ahead
    correlation: float   # rolling price correlation
    ext_price:   float
    qx_price:    float
    ts:          float   # epoch seconds


@dataclass
class ArbSignal:
    direction:   str     # "CALL" | "PUT"
    confidence:  float   # 0-1
    lag_ms:      float
    ext_price:   float
    qx_price:    float
    stake:       float
    reason:      str


# ──────────────────────────────────────────────────────────────────────────────
# External price feed (Twelve Data WebSocket)
# ──────────────────────────────────────────────────────────────────────────────

class ExternalFeed:
    """
    Async WebSocket connection to Twelve Data real-time feed.
    Falls back to REST polling when WebSocket is unavailable.

    Parameters
    ----------
    api_key  : Twelve Data API key
    symbol   : e.g. "EUR/USD"
    buffer   : number of recent ticks to retain
    """

    WS_URL = "wss://ws.twelvedata.com/v1/quotes/price"

    def __init__(
        self,
        api_key: str,
        symbol:  str = "EUR/USD",
        buffer:  int = 500,
    ) -> None:
        self.api_key = api_key
        self.symbol  = symbol
        self._ticks: Deque[PriceTick] = deque(maxlen=buffer)
        self._running   = False
        self._ws        = None
        self._last_ts   = 0.0
        self._last_price = 0.0

    # ── public ────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    def latest(self) -> Optional[PriceTick]:
        return self._ticks[-1] if self._ticks else None

    def recent_prices(self, n: int = 60) -> np.ndarray:
        ticks = list(self._ticks)[-n:]
        return np.array([t.price for t in ticks], dtype=float)

    def recent_timestamps(self, n: int = 60) -> np.ndarray:
        ticks = list(self._ticks)[-n:]
        return np.array([t.ts_ms for t in ticks], dtype=float)

    # ── async loop ────────────────────────────────────────────────────────────

    async def _run(self) -> None:
        """Main WebSocket loop with auto-reconnection."""
        backoff = 1.0
        while self._running:
            try:
                await self._connect()
                backoff = 1.0
            except Exception as exc:
                log.warning("ExternalFeed WS error: %s — retry in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    async def _connect(self) -> None:
        try:
            import websockets
        except ImportError:
            log.warning("websockets not installed — falling back to REST polling")
            await self._rest_poll_loop()
            return

        log.info("ExternalFeed: connecting to %s for %s", self.WS_URL, self.symbol)
        async with websockets.connect(self.WS_URL, ping_interval=20) as ws:
            self._ws = ws
            # Subscribe
            await ws.send(
                f'{{"action":"subscribe","params":{{"symbols":"{self.symbol}",'
                f'"apikey":"{self.api_key}"}}}}'
            )
            async for raw_msg in ws:
                if not self._running:
                    break
                self._handle_message(raw_msg)

    async def _rest_poll_loop(self) -> None:
        """Fallback: poll Twelve Data REST endpoint every 500 ms."""
        import urllib.request
        url = (
            f"https://api.twelvedata.com/price"
            f"?symbol={self.symbol.replace('/', '%2F')}"
            f"&apikey={self.api_key}"
        )
        while self._running:
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    import json
                    data  = json.loads(resp.read())
                    price = float(data.get("price", 0))
                    if price > 0:
                        ts_ms = time.time() * 1000
                        self._ticks.append(PriceTick(price=price, ts_ms=ts_ms))
                        self._last_price = price
                        self._last_ts    = ts_ms
            except Exception as exc:
                log.debug("REST poll error: %s", exc)
            await asyncio.sleep(0.5)

    def _handle_message(self, raw: str) -> None:
        try:
            import json
            msg = json.loads(raw)
            if msg.get("event") == "price":
                price = float(msg.get("price", 0))
                if price > 0:
                    ts_ms = float(msg.get("timestamp", time.time())) * 1000
                    self._ticks.append(PriceTick(price=price, ts_ms=ts_ms))
                    self._last_price = price
                    self._last_ts    = ts_ms
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Lag Monitor
# ──────────────────────────────────────────────────────────────────────────────

class LagMonitor:
    """
    Continuously measures the time lag (in ms) between the external price
    feed and Quotex's price using cross-correlation of recent price series.

    Parameters
    ----------
    buffer_size  : number of paired (ext, qx) price points to keep
    min_points   : minimum points before lag is considered reliable
    """

    def __init__(self, buffer_size: int = 120, min_points: int = 30) -> None:
        self._ext: Deque[Tuple[float, float]] = deque(maxlen=buffer_size)
        self._qx:  Deque[Tuple[float, float]] = deque(maxlen=buffer_size)
        self._lag_history: Deque[float] = deque(maxlen=500)
        self.min_points    = min_points

    def record(self, ext_tick: PriceTick, qx_price: float, qx_ts_ms: float) -> None:
        self._ext.append((ext_tick.ts_ms, ext_tick.price))
        self._qx.append((qx_ts_ms, qx_price))

    def snapshot(self) -> Optional[LagSnapshot]:
        """Compute current lag snapshot. Returns None if insufficient data."""
        if len(self._ext) < self.min_points:
            return None

        ext_arr = np.array(self._ext)
        qx_arr  = np.array(self._qx)

        # Align by timestamp — compute average time difference
        min_len = min(len(ext_arr), len(qx_arr))
        if min_len < self.min_points:
            return None

        # Lag = mean(ext_ts) - mean(qx_ts)  [positive = external is ahead]
        ts_lag = float(np.mean(ext_arr[-min_len:, 0]) - np.mean(qx_arr[-min_len:, 0]))

        # Price correlation
        ep = ext_arr[-min_len:, 1]
        qp = qx_arr[-min_len:, 1]
        ep_norm = (ep - ep.mean()) / (ep.std() + 1e-9)
        qp_norm = (qp - qp.mean()) / (qp.std() + 1e-9)
        corr    = float(np.dot(ep_norm, qp_norm) / min_len)

        self._lag_history.append(abs(ts_lag))

        return LagSnapshot(
            lag_ms=ts_lag,
            correlation=corr,
            ext_price=float(ext_arr[-1, 1]),
            qx_price=float(qx_arr[-1, 1]),
            ts=time.time(),
        )

    def dynamic_threshold(self, sigma: float = 2.0) -> float:
        """Return the lag threshold (mean + sigma × std of recent history)."""
        if len(self._lag_history) < 20:
            return 80.0   # conservative default
        arr = np.array(self._lag_history)
        return float(arr.mean() + sigma * arr.std())


# ──────────────────────────────────────────────────────────────────────────────
# Mispricing Detector
# ──────────────────────────────────────────────────────────────────────────────

class MispricingDetector:
    """
    Determines whether the current lag constitutes a tradeable mispricing.

    Conditions required
    -------------------
    1. lag_ms > dynamic_threshold  (statistically significant lag)
    2. correlation >= min_corr     (prices are tracking each other)
    3. External price has moved decisively in one direction (returns > epsilon)
    4. Cool-down period has elapsed since last signal
    """

    def __init__(
        self,
        min_lag_ms:      float = 80.0,
        min_corr:        float = 0.85,
        min_ext_move:    float = 0.00005,   # 0.5 pips minimum move
        cooldown_s:      float = 30.0,
    ) -> None:
        self.min_lag_ms   = min_lag_ms
        self.min_corr     = min_corr
        self.min_ext_move = min_ext_move
        self.cooldown_s   = cooldown_s
        self._last_signal = 0.0

    def evaluate(
        self,
        snap:            LagSnapshot,
        dyn_threshold:   float,
        ext_feed:        ExternalFeed,
    ) -> Optional[ArbSignal]:
        """Return an ArbSignal if conditions are met, else None."""
        now = time.time()

        # Cool-down check
        if now - self._last_signal < self.cooldown_s:
            return None

        abs_lag = abs(snap.lag_ms)

        # Condition 1: lag is statistically significant
        if abs_lag < max(self.min_lag_ms, dyn_threshold):
            return None

        # Condition 2: prices are correlated
        if snap.correlation < self.min_corr:
            return None

        # Condition 3: external price has moved decisively
        ext_prices = ext_feed.recent_prices(n=10)
        if len(ext_prices) < 3:
            return None

        ext_move = float(ext_prices[-1] - ext_prices[-3])
        if abs(ext_move) < self.min_ext_move:
            return None

        # Direction: external moved up → Quotex will follow → CALL
        #            external moved dn → Quotex will follow → PUT
        direction = "CALL" if ext_move > 0 else "PUT"

        confidence = min(0.95, (
            0.4 * min(1.0, abs_lag / 200.0)          # lag contribution
            + 0.3 * snap.correlation                   # correlation
            + 0.3 * min(1.0, abs(ext_move) / 0.0005)  # move magnitude
        ))

        self._last_signal = now

        return ArbSignal(
            direction=direction,
            confidence=confidence,
            lag_ms=abs_lag,
            ext_price=snap.ext_price,
            qx_price=snap.qx_price,
            stake=0.0,    # filled in by the engine
            reason=(
                f"lag={abs_lag:.0f}ms corr={snap.correlation:.2f} "
                f"ext_move={ext_move:+.5f}"
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Latency Arbitrage Engine — orchestrates all components
# ──────────────────────────────────────────────────────────────────────────────

class LatencyArbitrageEngine:
    """
    Full latency arbitrage system.

    Parameters
    ----------
    api_key          : Twelve Data API key (from .env TWELVEDATA_API_KEY)
    symbol           : external ticker, e.g. "EUR/USD"
    max_risk_pct     : per-trade max stake as fraction of balance (0.005 = 0.5%)
    daily_loss_pct   : daily loss cap as fraction of balance (0.02 = 2%)
    min_lag_ms       : minimum lag (ms) to trigger a trade
    min_corr         : minimum price correlation to proceed
    cooldown_s       : seconds between consecutive arb signals
    lag_sigma        : dynamic threshold sigma
    """

    def __init__(
        self,
        api_key:         str,
        symbol:          str   = "EUR/USD",
        max_risk_pct:    float = 0.005,
        daily_loss_pct:  float = 0.02,
        min_lag_ms:      float = 80.0,
        min_corr:        float = 0.85,
        cooldown_s:      float = 30.0,
        lag_sigma:       float = 2.0,
    ) -> None:
        self.max_risk_pct    = max_risk_pct
        self.daily_loss_pct  = daily_loss_pct
        self.lag_sigma       = lag_sigma

        self._ext_feed   = ExternalFeed(api_key=api_key, symbol=symbol)
        self._lag_mon    = LagMonitor()
        self._mispricing = MispricingDetector(
            min_lag_ms=min_lag_ms,
            min_corr=min_corr,
            cooldown_s=cooldown_s,
        )

        self._session_loss: float = 0.0
        self._start_balance: float = 0.0
        self._running = False

        # Callback: called when a signal fires
        # Signature: async def on_signal(signal: ArbSignal) -> None
        self.on_signal: Optional[Callable] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, balance: float) -> None:
        self._start_balance = balance
        self._running       = True
        await self._ext_feed.start()
        asyncio.create_task(self._monitor_loop())
        log.info(
            "LatencyArbitrageEngine started | balance=%.2f "
            "max_risk=%.1f%% daily_loss=%.1f%%",
            balance, self.max_risk_pct * 100, self.daily_loss_pct * 100,
        )

    async def stop(self) -> None:
        self._running = False
        await self._ext_feed.stop()
        log.info("LatencyArbitrageEngine stopped")

    def record_trade_result(self, pnl: float) -> None:
        """Call after each arb trade settles to track daily P&L."""
        self._session_loss -= pnl   # negative pnl = loss

    def feed_quotex_tick(self, price: float, ts_ms: float) -> None:
        """
        Feed the latest Quotex price tick into the lag monitor.
        Call this from your realtime subscription callback.
        """
        ext = self._ext_feed.latest()
        if ext:
            self._lag_mon.record(ext, price, ts_ms)

    # ── internal ──────────────────────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        while self._running:
            await asyncio.sleep(0.05)   # 50 ms polling interval
            snap = self._lag_mon.snapshot()
            if snap is None:
                continue

            # Daily loss circuit breaker
            daily_cap = self._start_balance * self.daily_loss_pct
            if self._session_loss >= daily_cap:
                log.warning(
                    "Arb daily loss cap hit (%.2f >= %.2f) — engine paused",
                    self._session_loss, daily_cap,
                )
                await asyncio.sleep(3600)   # pause for 1 hour
                self._session_loss = 0.0
                continue

            dyn_thr = self._lag_mon.dynamic_threshold(self.lag_sigma)
            signal  = self._mispricing.evaluate(snap, dyn_thr, self._ext_feed)

            if signal is None:
                continue

            # Size the stake
            signal.stake = round(
                min(
                    self._start_balance * self.max_risk_pct,
                    (self._start_balance * self.daily_loss_pct
                     - self._session_loss) * 0.25,   # use max 25% of remaining cap
                ),
                2,
            )
            if signal.stake < 1.0:
                continue

            log.info(
                "ARB SIGNAL  %s  conf=%.2f  stake=%.2f  %s",
                signal.direction, signal.confidence, signal.stake, signal.reason,
            )

            if self.on_signal:
                await self.on_signal(signal)
