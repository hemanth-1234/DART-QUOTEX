"""
integration_example.py
Two integration patterns for your existing Titan Bot.

─────────────────────────────────────────────────────
MODE 1 — Complete brain.py replacement
─────────────────────────────────────────────────────
Replace your brain.py with this pattern.  The AIAdvisor handles
everything: data fetching, signal generation, risk sizing, and
order execution.

─────────────────────────────────────────────────────
MODE 2 — Confirmation filter (recommended first step)
─────────────────────────────────────────────────────
Keep your existing brain.py and Titan Bot logic.
Use AIAdvisor as a second opinion — only trade when
both your rules-based system AND the AI agree.
"""

# ══════════════════════════════════════════════════════════════════
# MODE 1: Complete replacement
# ══════════════════════════════════════════════════════════════════

MODE_1_EXAMPLE = '''
# In your main.py — replace all brain.py imports with:

from dart_quotex import AIAdvisor

async def main():
    advisor = AIAdvisor()
    await advisor.connect()

    # Run a 1-hour session
    from dart_quotex.trader import LiveTrader
    trader = LiveTrader(session_minutes=60, trade_interval=65)
    await trader.run(asset="EURUSD_OTC")

    # Models auto-save on disconnect
    await advisor.disconnect()
'''


# ══════════════════════════════════════════════════════════════════
# MODE 2: Confirmation filter
# ══════════════════════════════════════════════════════════════════

import asyncio
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TitanBotWithAIFilter:
    """
    Drop-in wrapper for your existing Titan Bot.

    Usage: replace your main trading loop with this class.
    Your existing signal generation, order execution, and win-check
    code is preserved — AI only acts as a gate.
    """

    def __init__(
        self,
        existing_bot,          # your existing bot instance
        min_ai_confidence: float = 0.62,
        require_agreement: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        existing_bot      : your Titan Bot / brain.py instance
        min_ai_confidence : minimum AI confidence to allow trade
        require_agreement : if True, AI must agree with bot direction
        """
        self.bot = existing_bot
        self.min_conf = min_ai_confidence
        self.require_agreement = require_agreement

        # Import here so the main bot doesn't need dart_quotex at startup
        from dart_quotex import AIAdvisor
        self.advisor = AIAdvisor()
        self._ai_ready = False

    async def start(self) -> None:
        """Call once before your trading loop."""
        await self.advisor.connect()
        self._ai_ready = True
        logger.info("AI filter activated (min_confidence=%.0f%%)", self.min_conf * 100)

    async def stop(self) -> None:
        """Call when your session ends."""
        await self.advisor.disconnect()

    async def should_trade(
        self,
        candles: List[Dict[str, Any]],
        bot_direction: str,          # "call" | "put" from your brain.py
        asset: str = "EURUSD_OTC",
    ) -> bool:
        """
        Returns True only if AI confirms the trade.

        Parameters
        ----------
        candles       : recent OHLCV candles (your existing data)
        bot_direction : signal from your existing brain.py
        asset         : asset being traded

        Example usage in your bot:
        --------------------------
            signal = brain.get_signal(candles)
            if await ai_filter.should_trade(candles, signal, asset):
                # proceed with order placement
                success, trade_id = await client.buy(...)
        """
        if not self._ai_ready:
            return True    # fail open if AI not initialised

        try:
            ai_direction, ai_confidence = self.advisor.assess(candles, asset)
        except Exception as exc:
            logger.warning("AI assess failed: %s — allowing trade", exc)
            return True    # fail open on error

        if ai_direction == "HOLD":
            logger.info("AI: HOLD — skipping trade")
            return False

        if ai_confidence < self.min_conf:
            logger.info(
                "AI: %s @ %.1f%% confidence — below threshold %.0f%% — skipping",
                ai_direction, ai_confidence * 100, self.min_conf * 100,
            )
            return False

        if self.require_agreement:
            bot_dir_upper = bot_direction.upper()
            if bot_dir_upper not in ("CALL", "PUT"):
                return False

            ai_dir_upper = ai_direction.upper()
            if ai_dir_upper != bot_dir_upper:
                logger.info(
                    "Disagreement: bot=%s AI=%s @ %.1f%% — skipping",
                    bot_dir_upper, ai_dir_upper, ai_confidence * 100,
                )
                return False

        logger.info(
            "AI confirmed: %s @ %.1f%% — proceeding",
            ai_direction, ai_confidence * 100,
        )
        return True

    def record_outcome(self, won: bool) -> None:
        """
        Call after each trade completes so the AI can learn.

            won = True  → trade was profitable
            won = False → trade was a loss
        """
        self.advisor.record_outcome(won=won)


# ══════════════════════════════════════════════════════════════════
# Minimal integration snippet for brain.py
# ══════════════════════════════════════════════════════════════════

BRAIN_PY_SNIPPET = '''
# Add to your existing brain.py (top of file):
from dart_quotex import AIAdvisor as _AIAdvisor
_ai_advisor = _AIAdvisor()

# Modify your get_signal() or wherever you place trades:
async def place_trade_with_ai_gate(candles, direction, asset, client):
    """Wrap your existing trade placement with AI confirmation."""
    ai_dir, ai_conf = _ai_advisor.assess(candles, asset)

    if ai_conf < 0.62:
        logger.info("AI confidence too low (%.0f%%) — skipping", ai_conf * 100)
        return None

    if ai_dir != direction.upper():
        logger.info("AI disagrees (%s vs %s) — skipping", ai_dir, direction)
        return None

    # Your existing order placement code here:
    success, trade_id = await client.buy(asset, amount, direction, duration)
    return trade_id
'''


# ══════════════════════════════════════════════════════════════════
# Standalone script: assess_once.py
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Quick one-shot test: print AI signal for the configured asset.
    Run: python integration_example.py
    """
    import asyncio
    from dart_quotex import AIAdvisor
    from dart_quotex.config import cfg

    async def assess_once():
        advisor = AIAdvisor()
        await advisor.connect()

        direction, confidence = await advisor.get_signal(asset=cfg.quotex.asset)

        print("\n" + "─" * 45)
        print(f"  Asset      : {cfg.quotex.asset}")
        print(f"  Direction  : {direction}")
        print(f"  Confidence : {confidence:.1%}")
        threshold = cfg.risk.min_confidence
        action = "✓ TRADE" if confidence >= threshold else f"✗ SKIP  (need ≥{threshold:.0%})"
        print(f"  Decision   : {action}")
        print("─" * 45 + "\n")

        await advisor.disconnect()

    asyncio.run(assess_once())
