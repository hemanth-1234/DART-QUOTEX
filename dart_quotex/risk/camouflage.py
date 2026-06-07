"""
dart_quotex/risk/camouflage.py
Module 5 — Behavioral Variation Engine
========================================
Introduces legitimate, natural-looking variance into trade execution to
avoid triggering automated rate-limiting or pattern-detection systems that
flag purely mechanical (zero-variance) bots.

What this module DOES
---------------------
· Random execution delays   (5–30 seconds between signal and order placement)
· Stake size variance       (±10% of calculated stake, within risk limits)
· Hold-time blending        (20% of trades use a longer expiry duration)

What this module does NOT do
-----------------------------
· Does NOT place intentional losing trades
· Does NOT manufacture fake losing results
· Does NOT manipulate reported win rates

Configuration (all in .env)
---------------------------
ENABLE_CAMOUFLAGE        = true
CAMOUFLAGE_INTENSITY     = 0.5        # 0.0 = off, 1.0 = maximum variance
CAMOUFLAGE_DELAY_MIN     = 5          # minimum pre-trade delay (seconds)
CAMOUFLAGE_DELAY_MAX     = 30         # maximum pre-trade delay (seconds)
CAMOUFLAGE_STAKE_VARIANCE= 0.10       # ±10% stake randomisation
CAMOUFLAGE_LONG_HOLD_PCT = 0.20       # 20% of trades use extended expiry
CAMOUFLAGE_LONG_HOLD_S   = 120        # extended expiry (seconds)
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeParams:
    stake:    float    # final stake after variance applied
    duration: int      # expiry duration in seconds
    delay:    float    # pre-trade delay applied


# ──────────────────────────────────────────────────────────────────────────────
# CamouflageEngine
# ──────────────────────────────────────────────────────────────────────────────

class CamouflageEngine:
    """
    Wraps trade execution with configurable behavioral variation.

    Parameters
    ----------
    intensity        : 0.0 = no variation, 1.0 = maximum variance applied
    delay_min        : minimum pre-trade delay in seconds
    delay_max        : maximum pre-trade delay in seconds
    stake_variance   : fraction of stake to vary  (0.10 = ±10%)
    long_hold_pct    : fraction of trades that use extended expiry  (0.20 = 20%)
    long_hold_s      : extended expiry duration in seconds
    min_stake        : floor on stake after variance is applied
    """

    def __init__(
        self,
        intensity:      float = 0.5,
        delay_min:      float = 5.0,
        delay_max:      float = 30.0,
        stake_variance: float = 0.10,
        long_hold_pct:  float = 0.20,
        long_hold_s:    int   = 120,
        min_stake:      float = 1.0,
    ) -> None:
        self.intensity      = float(max(0.0, min(1.0, intensity)))
        self.delay_min      = delay_min
        self.delay_max      = delay_max
        self.stake_variance = stake_variance
        self.long_hold_pct  = long_hold_pct
        self.long_hold_s    = long_hold_s
        self.min_stake      = min_stake

        self._trade_count   = 0
        self._total_delay_s = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    async def prepare_trade(
        self,
        base_stake:   float,
        base_duration: int,
    ) -> TradeParams:
        """
        Apply behavioral variation to a trade before execution.

        1. Waits a random pre-trade delay
        2. Adjusts stake by ±stake_variance
        3. Possibly extends the expiry duration

        Parameters
        ----------
        base_stake    : calculated stake from risk manager (currency units)
        base_duration : default expiry in seconds (e.g. 60)

        Returns
        -------
        TradeParams with final stake, duration, and the delay that was applied
        """
        self._trade_count += 1

        # ── Step 1: Random pre-trade delay ────────────────────────────────────
        delay = self._compute_delay()
        if delay > 0:
            log.debug("Camouflage: pre-trade delay %.1fs", delay)
            await asyncio.sleep(delay)
            self._total_delay_s += delay

        # ── Step 2: Stake variance ────────────────────────────────────────────
        stake = self._vary_stake(base_stake)

        # ── Step 3: Hold-time blending ────────────────────────────────────────
        duration = self._vary_duration(base_duration)

        params = TradeParams(stake=stake, duration=duration, delay=delay)
        log.debug(
            "Camouflage trade #%d: stake=%.2f (base %.2f) "
            "duration=%ds (base %ds) delay=%.1fs",
            self._trade_count, stake, base_stake, duration, base_duration, delay,
        )
        return params

    def stats(self) -> dict:
        """Return engine statistics."""
        return {
            "trades_processed": self._trade_count,
            "total_delay_s":    round(self._total_delay_s, 1),
            "avg_delay_s":      round(
                self._total_delay_s / max(1, self._trade_count), 1
            ),
            "intensity":        self.intensity,
        }

    # ── internal ──────────────────────────────────────────────────────────────

    def _compute_delay(self) -> float:
        """Return a random delay in seconds, scaled by intensity."""
        if self.intensity < 0.01:
            return 0.0
        lo = self.delay_min * self.intensity
        hi = self.delay_max * self.intensity
        return round(random.uniform(lo, hi), 1)

    def _vary_stake(self, base: float) -> float:
        """Apply ±stake_variance to the base stake."""
        if self.intensity < 0.01 or self.stake_variance < 0.001:
            return base
        variance = base * self.stake_variance * self.intensity
        adjusted = base + random.uniform(-variance, variance)
        adjusted = max(self.min_stake, round(adjusted, 2))
        return adjusted

    def _vary_duration(self, base: int) -> int:
        """
        With probability long_hold_pct (scaled by intensity), use the
        extended expiry.  Otherwise return the base duration unchanged.
        """
        if self.intensity < 0.01:
            return base
        threshold = self.long_hold_pct * self.intensity
        if random.random() < threshold:
            log.debug("Camouflage: extended hold %ds", self.long_hold_s)
            return self.long_hold_s
        return base


# ──────────────────────────────────────────────────────────────────────────────
# Factory — build from environment / config
# ──────────────────────────────────────────────────────────────────────────────

def build_from_env() -> Optional[CamouflageEngine]:
    """
    Create a CamouflageEngine from environment variables.
    Returns None if ENABLE_CAMOUFLAGE is not 'true'.

    Used in trader.py:
        engine = build_from_env()
        if engine:
            params = await engine.prepare_trade(stake, duration)
        else:
            params = TradeParams(stake=stake, duration=duration, delay=0)
    """
    import os
    if os.environ.get("ENABLE_CAMOUFLAGE", "false").lower() != "true":
        return None

    return CamouflageEngine(
        intensity=float(os.environ.get("CAMOUFLAGE_INTENSITY",     "0.5")),
        delay_min=float(os.environ.get("CAMOUFLAGE_DELAY_MIN",     "5")),
        delay_max=float(os.environ.get("CAMOUFLAGE_DELAY_MAX",     "30")),
        stake_variance=float(os.environ.get("CAMOUFLAGE_STAKE_VARIANCE", "0.10")),
        long_hold_pct=float(os.environ.get("CAMOUFLAGE_LONG_HOLD_PCT",  "0.20")),
        long_hold_s=int(os.environ.get("CAMOUFLAGE_LONG_HOLD_S",     "120")),
    )
