"""
dart_quotex/api/candle_patch.py
Feature 1 — Deep Candle Fetching (Unlimited Historical Data)
=============================================================
Patches the pyquotex Quotex client with get_candles_deep(), which uses
reverse-engineered WebSocket parameters to bypass the 199-candle limit.

Reverse-engineered parameters (from usmanch96/quotex-historical-data):
  offset  : 3600   (fixed offset parameter, not a Unix timestamp)
  step    : 2940   (seconds per pagination step)
  index   : 12-digit integer, not a Unix timestamp — incremented each page
  response key: "data"  (not "history" as in standard get_candles)

The method loops backwards, fetching chunks and stitching them together,
then deduplicates and sorts by timestamp.

If the patch fails, the system falls back to standard get_candles() and
logs a warning.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import types
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Reverse-engineered constants ──────────────────────────────────────────────
_DEEP_OFFSET    = 3600      # fixed offset parameter
_DEEP_STEP      = 2940      # seconds per pagination step
_INDEX_START    = 100_000_000_000  # 12-digit starting index
_CHUNK_SIZE     = 99        # candles per deep request (safe below limit)
_HISTORY_KEY    = "data"    # JSON key in deep-history responses
_FALLBACK_KEY   = "history" # JSON key in standard responses


def patch_pyquotex(api_instance: Any) -> bool:
    """
    Dynamically graft get_candles_deep() onto a live pyquotex Quotex instance.

    Returns True if patch applied successfully.
    """
    if api_instance is None:
        log.warning("patch_pyquotex: api_instance is None — patch skipped")
        return False

    if hasattr(api_instance, "_dart_patched"):
        return True

    # ── Build the deep-fetch coroutine ────────────────────────────────────────

    async def get_candles_deep(
        _self_unused,
        asset:       str,
        granularity: int  = 60,
        total:       int  = 3000,
        chunk:       int  = _CHUNK_SIZE,
        delay_lo:    float = 0.4,
        delay_hi:    float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Fetch `total` candles for `asset` at `granularity` seconds using
        reverse-engineered pagination with index-based stepping.

        Algorithm
        ---------
        1.  Start with index = _INDEX_START, offset = _DEEP_OFFSET
        2.  Call the broker WebSocket with these parameters
        3.  Parse response under JSON key "data"
        4.  Advance index by _DEEP_STEP
        5.  Repeat until we have `total` unique candles or data exhausts
        """
        all_candles: Dict[int, Dict[str, Any]] = {}
        index = _INDEX_START
        pages_without_new = 0

        log.info(
            "get_candles_deep: %s gran=%ds target=%d",
            asset, granularity, total,
        )

        while len(all_candles) < total:
            try:
                # Attempt deep fetch with reverse-engineered parameters
                raw = await _fetch_deep(
                    api_instance, asset, granularity, chunk,
                    _DEEP_OFFSET, index,
                )
            except Exception as exc:
                log.warning("Deep fetch error (idx=%d): %s", index, exc)
                # Try standard fallback
                try:
                    offset_sec = int((index - _INDEX_START) // 1)
                    raw = await api_instance.get_candles(
                        asset, offset_sec, granularity, chunk
                    )
                except Exception as exc2:
                    log.error("Fallback also failed: %s", exc2)
                    break

            if not raw:
                pages_without_new += 1
                if pages_without_new >= 3:
                    log.info(
                        "History boundary reached after %d candles", len(all_candles)
                    )
                    break
                index += _DEEP_STEP
                continue

            parsed   = _parse(raw)
            new_this_page = 0
            for c in parsed:
                ts = c.get("time", 0)
                if ts and ts not in all_candles:
                    all_candles[ts] = c
                    new_this_page  += 1

            log.debug(
                "  index=%d  parsed=%d  new=%d  total=%d",
                index, len(parsed), new_this_page, len(all_candles),
            )

            if new_this_page == 0:
                pages_without_new += 1
                if pages_without_new >= 3:
                    break
            else:
                pages_without_new = 0

            index += _DEEP_STEP
            await asyncio.sleep(random.uniform(delay_lo, delay_hi))

        candles = sorted(all_candles.values(), key=lambda c: c.get("time", 0))
        log.info(
            "get_candles_deep complete: %d unique candles for %s",
            len(candles), asset,
        )
        return candles[-total:]

    # Bind and mark
    api_instance.get_candles_deep = types.MethodType(
        get_candles_deep, api_instance
    )
    api_instance._dart_patched = True
    log.info(
        "patch_pyquotex: get_candles_deep patched onto %s",
        type(api_instance).__name__,
    )
    return True


async def _fetch_deep(
    api: Any,
    asset: str,
    granularity: int,
    count: int,
    offset: int,
    index: int,
) -> Optional[Any]:
    """
    Issue the deep-history WebSocket request using reverse-engineered params.
    Falls back gracefully if the internal send method differs across versions.
    """
    # Try the documented internal method first
    if hasattr(api, "_send_message"):
        payload = {
            "action":    "candles",
            "asset":     asset,
            "period":    granularity,
            "count":     count,
            "offset":    offset,
            "index":     index,
        }
        resp = await api._send_message(payload)
        if isinstance(resp, dict):
            return resp.get(_HISTORY_KEY) or resp.get(_FALLBACK_KEY)
        return resp

    # Try the standard get_candles with computed offset
    computed_offset = max(0, (index - _INDEX_START) * granularity // _DEEP_STEP)
    return await api.get_candles(asset, computed_offset, granularity, count)


def _parse(raw: Any) -> List[Dict[str, Any]]:
    """Normalise raw response into a list of OHLCV dicts."""
    if not raw:
        return []
    if isinstance(raw, dict):
        # Response wrapped in a dict — try known keys
        for key in (_HISTORY_KEY, _FALLBACK_KEY, "candles", "result"):
            if key in raw:
                raw = raw[key]
                break

    result = []
    for c in (raw if isinstance(raw, list) else []):
        if isinstance(c, dict):
            ts  = int(c.get("time") or c.get("ts") or c.get("from") or 0)
            result.append({
                "time":   ts,
                "open":   float(c.get("open",  0)),
                "high":   float(c.get("max")  or c.get("high",  0)),
                "low":    float(c.get("min")   or c.get("low",   0)),
                "close":  float(c.get("close") or c.get("value", 0)),
                "volume": float(c.get("volume", 0)),
            })
        elif isinstance(c, (list, tuple)) and len(c) >= 5:
            result.append({
                "time":   int(c[0]),   "open": float(c[1]),
                "high":   float(c[2]), "low":  float(c[3]),
                "close":  float(c[4]),
                "volume": float(c[5]) if len(c) > 5 else 0.0,
            })
    return [r for r in result if r["time"] > 0 and r["close"] > 0]


def apply_patch(api_instance: Any) -> None:
    """
    Convenience function — apply patch and log the result.

    Call this immediately after the Quotex API connects:

        await client.connect()
        from dart_quotex.api.candle_patch import apply_patch
        apply_patch(client._api)
    """
    if not patch_pyquotex(api_instance):
        log.warning(
            "Deep candle patch NOT applied.  "
            "Falling back to standard get_candles() — max 199 candles/request."
        )
