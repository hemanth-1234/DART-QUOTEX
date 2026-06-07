# DART-Quotex Advanced Features — Integration Guide

Complete step-by-step instructions for wiring all six modules into the
existing codebase.  Each section lists the exact file to edit, the import
to add, and the precise location to call each function.

---

## Module 1 — Deep Candle Fetching

**File: `dart_quotex/api/quotex_client.py`**

Add to `connect()`, immediately after the broker API connects:

```python
# At the top of the file, add:
from dart_quotex.api.candle_patch import apply_patch

# Inside QuotexClient.connect(), after self._api connects:
apply_patch(self._api)
```

**File: `dart_quotex/data/harvester.py`**

Replace the existing `_fetch_chunk` method body:

```python
async def _fetch_chunk(self, asset, granularity, count, end_time):
    try:
        if end_time is None:
            # Standard recent fetch
            raw = await self.client.get_candles(asset, granularity, count)
        else:
            # Use deep fetch if patched, fall back to standard
            api = getattr(self.client, '_api', None)
            if api and hasattr(api, 'get_candles_deep'):
                now = int(time.time())
                total_needed = count
                raw = await api.get_candles_deep(
                    asset,
                    granularity=granularity,
                    total=total_needed,
                )
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "get_candles_deep not available — using standard get_candles"
                )
                raw = await self.client.get_candles(asset, granularity, count)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("API error: %s", exc)
        return []
    return _normalise(raw)
```

**Activation**: Always active once `apply_patch()` is called. No `.env` flag needed.

---

## Module 2 — Latency Arbitrage

**File: `dart_quotex/trader.py`**

```python
# Top of file, add imports:
import os
from dart_quotex.arbitrage.external_feed import ExternalFeedFactory
from dart_quotex.arbitrage.latency import LatencyArbitrageEngine

# In LiveTrader.__init__(), add:
self._arb_engine = None
if os.environ.get("ENABLE_LATENCY_ARBITRAGE", "false").lower() == "true":
    api_key  = os.environ.get("EXTERNAL_API_KEY", "")
    provider = os.environ.get("EXTERNAL_FEED_PROVIDER", "twelvedata")
    symbol   = os.environ.get("EXTERNAL_SYMBOL", "EUR/USD")
    self._arb_engine = LatencyArbitrageEngine(
        api_key=api_key,
        symbol=symbol,
        max_risk_pct=float(os.environ.get("ARBIT_MAX_RISK_PCT", "0.005")),
        daily_loss_pct=float(os.environ.get("ARBIT_DAILY_LOSS_PCT", "0.02")),
        min_lag_ms=float(os.environ.get("LAG_THRESHOLD_MS", "80")),
        min_corr=float(os.environ.get("ARBIT_MIN_CORR", "0.85")),
        cooldown_s=float(os.environ.get("ARBIT_COOLDOWN_S", "30")),
        lag_sigma=float(os.environ.get("LAG_SIGMA", "2.0")),
    )

# In LiveTrader.run(), after connecting the advisor, start arb engine:
if self._arb_engine:
    balance = await self._advisor.client.get_balance()
    async def _on_arb_signal(sig):
        success, tid = await self._advisor.client.buy(
            asset, sig.stake, sig.direction.lower(), 60
        )
        if success:
            logger.info("ARB TRADE placed: %s %s", sig.direction, sig.reason)
    self._arb_engine.on_signal = _on_arb_signal
    await self._arb_engine.start(balance)

# In the realtime tick callback (wherever Quotex price is received):
if self._arb_engine:
    self._arb_engine.feed_quotex_tick(price, time.time() * 1000)

# In LiveTrader cleanup:
if self._arb_engine:
    await self._arb_engine.stop()
```

**Activation**: Set `ENABLE_LATENCY_ARBITRAGE=true` and `EXTERNAL_API_KEY=your_key` in `.env`.

---

## Module 3 — Manipulation Detection

**File: `dart_quotex/advisor.py`**

```python
# Top of file, add:
import os
from dart_quotex.signals.manipulation import manipulation_score, ManipulationDetector

# In AIAdvisor.__init__(), add:
self._manip_detector = ManipulationDetector() if os.environ.get(
    "ENABLE_MANIPULATION_DETECTOR", "true"
).lower() == "true" else None
self._manip_threshold = float(os.environ.get("MANIPULATION_THRESHOLD", "0.7"))

# In AIAdvisor.trade(), after fetching df and BEFORE calling _full_assess:
if self._manip_detector and df is not None and len(df) >= 25:
    # Quick 3-function score
    from dart_quotex.signals.manipulation import manipulation_score
    m_score, m_desc = manipulation_score(df)
    if m_score > self._manip_threshold:
        logger.warning(
            "MANIPULATION DETECTED (score=%.2f): %s — trade skipped",
            m_score, m_desc,
        )
        return None   # skip trade

    # Full AIMM-X integrity check
    result = self._manip_detector.score(df)
    if result.recommended_action == "SKIP":
        logger.warning("AIMM-X SKIP: %s", result)
        return None
    if result.recommended_action == "REDUCE_SIZE":
        logger.info("AIMM-X REDUCE_SIZE: %s", result)
        # The risk manager will use a smaller stake on next call
        # You can store result.integrity_score and pass to risk.evaluate()
```

**Activation**: `ENABLE_MANIPULATION_DETECTOR=true` (default). Set `MANIPULATION_THRESHOLD` to adjust sensitivity.

---

## Module 4 — Stop-Hunt + FVG Override

**File: `dart_quotex/advisor.py`**

```python
# Top of file, add:
from dart_quotex.smc.indicators import stop_hunt_signal

# In AIAdvisor._full_assess(), after computing combined_conf:
# (add this block near the end of _full_assess, before returning)

enable_sh = os.environ.get("ENABLE_STOP_HUNT", "true").lower() == "true"
min_hunt  = float(os.environ.get("MIN_HUNT_CONFIDENCE", "0.65"))
ai_low_thr= float(os.environ.get("AI_LOW_CONFIDENCE_THRESHOLD", "0.60"))

if enable_sh and combined_conf < ai_low_thr and df is not None and len(df) >= 30:
    sh_dir, sh_conf = stop_hunt_signal(
        df,
        min_sweep_strength=float(os.environ.get("MIN_SWEEP_STRENGTH", "0.70")),
        min_fvg_confidence=float(os.environ.get("MIN_FVG_CONFIDENCE", "0.60")),
    )
    if sh_conf >= min_hunt:
        logger.info(
            "STOP-HUNT OVERRIDE: AI conf=%.2f < %.2f, using %s conf=%.2f",
            combined_conf, ai_low_thr, sh_dir, sh_conf,
        )
        detail["stop_hunt_override"] = True
        detail["stop_hunt_conf"]     = sh_conf
        return sh_dir, sh_conf, detail
```

**Activation**: `ENABLE_STOP_HUNT=true` (default). Tune `MIN_HUNT_CONFIDENCE` to control override aggressiveness.

---

## Module 5 — Behavioral Variation

**File: `dart_quotex/advisor.py`** (in the `trade()` method)

```python
# Top of file, add:
from dart_quotex.risk.camouflage import build_from_env, TradeParams

# In AIAdvisor.__init__(), add:
self._camouflage = build_from_env()   # returns None if ENABLE_CAMOUFLAGE=false

# In AIAdvisor.trade(), AFTER risk.evaluate() approves the trade but
# BEFORE calling client.buy():

if self._camouflage:
    params = await self._camouflage.prepare_trade(
        base_stake=decision.stake,
        base_duration=duration,
    )
    final_stake    = params.stake
    final_duration = params.duration
    logger.debug("Camouflage: stake %.2f->%.2f  dur %d->%d  delay %.1fs",
                 decision.stake, final_stake, duration, final_duration, params.delay)
else:
    final_stake    = decision.stake
    final_duration = duration

# Then use final_stake and final_duration in client.buy():
success, trade_id = await self.client.buy(
    asset, final_stake, decision.direction, final_duration
)
```

**Activation**: `ENABLE_CAMOUFLAGE=true` and tune `CAMOUFLAGE_INTENSITY` (0.0–1.0).

---

## Feature 6 — TCN Spoofing Detection

### Step 1: Build pattern library (offline, run once)

```python
# Run this script once after harvesting data:
# scripts/build_tcn_library.py

import asyncio
from dart_quotex.data.database import Database
from dart_quotex.manipulation.tcn_spoofing import TCNSpoofingDetector
from dart_quotex.config import cfg

db  = Database(cfg.data.db_path)
df  = db.get_candles("EURUSD_OTC", 60, limit=5000)

det = TCNSpoofingDetector(
    seq_len=int(os.environ.get("TCN_SEQ_LEN", "30")),
    threshold=float(os.environ.get("TCN_SPOOFING_THRESHOLD", "0.80")),
)
det.build_pattern_library(df, epochs=10)
det.save("models/")
print("TCN library built and saved.")
```

### Step 2: Load and use in advisor.py

```python
# Top of file:
from dart_quotex.manipulation.tcn_spoofing import TCNSpoofingDetector

# In AIAdvisor.__init__():
self._tcn_detector = None
if os.environ.get("ENABLE_TCN_SPOOFING", "false").lower() == "true":
    self._tcn_detector = TCNSpoofingDetector(
        seq_len=int(os.environ.get("TCN_SEQ_LEN", "30")),
        threshold=float(os.environ.get("TCN_SPOOFING_THRESHOLD", "0.80")),
    )
    self._tcn_detector.load(self._model_dir)

# In AIAdvisor.trade(), before placing the order:
if self._tcn_detector and df is not None:
    spoof = self._tcn_detector.detect(df)
    if spoof.suspicious:
        logger.warning(
            "TCN SPOOFING DETECTED (score=%.2f sim=%.2f): %s — skipping",
            spoof.score, spoof.similarity, spoof.details,
        )
        return None
```

**Activation**: Set `ENABLE_TCN_SPOOFING=true`. Run `scripts/build_tcn_library.py` first.

---

## Quick-activation checklist

| Module | `.env` flag | Default | Notes |
|--------|------------|---------|-------|
| Deep candles | — | always on | `apply_patch()` in `connect()` |
| Latency arb | `ENABLE_LATENCY_ARBITRAGE` | `false` | needs `EXTERNAL_API_KEY` |
| Manipulation filter | `ENABLE_MANIPULATION_DETECTOR` | `true` | tune `MANIPULATION_THRESHOLD` |
| Stop-hunt signal | `ENABLE_STOP_HUNT` | `true` | tune `MIN_HUNT_CONFIDENCE` |
| Behavioral variation | `ENABLE_CAMOUFLAGE` | `false` | tune `CAMOUFLAGE_INTENSITY` |
| TCN spoofing | `ENABLE_TCN_SPOOFING` | `false` | run build script first |
