"""
dart_quotex — AI-driven binary options trading for Quotex OTC markets.

Quick start
-----------
    from dart_quotex import AIAdvisor

    # Full async trading session:
    async with AIAdvisor() as advisor:
        direction, confidence = await advisor.get_signal("EURUSD_OTC")

    # Sync MODE 2 filter (brain.py replacement):
    advisor = AIAdvisor()
    direction, confidence = advisor.assess(candles, "EURUSD_OTC")
    advisor.record_outcome(won=True)
"""
from dart_quotex.advisor import AIAdvisor

__all__ = ["AIAdvisor"]
__version__ = "2.0.0"
