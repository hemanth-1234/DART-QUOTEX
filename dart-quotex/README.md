# DART-Quotex
**AI-driven binary options trading bot for Quotex OTC markets**

Built on the [DART](https://github.com/ItzSwapnil/DART) architecture with all Deriv dependencies removed and replaced by a full Quotex integration via [pyquotex](https://github.com/cleitonleonel/pyquotex).

---

## What's inside

| Component | Description |
|-----------|-------------|
| `dart_quotex/api/quotex_client.py` | Async Quotex adapter — same interface as the original `deriv_client.py` |
| `dart_quotex/ml/ensemble.py` | RandomForest + GradientBoosting + SGD ensemble with **incremental learning** (`partial_fit`) |
| `dart_quotex/ml/sac_agent.py` | Soft Actor-Critic RL agent — learns trade timing & sizing |
| `dart_quotex/ml/features.py` | 34-feature pipeline: price, momentum, volatility, trend, volume, SMC |
| `dart_quotex/smc/indicators.py` | SMC/ICT: order blocks, FVG, liquidity sweeps, BOS/CHoCH, premium/discount |
| `dart_quotex/risk/manager.py` | Kelly Criterion + Monte Carlo VaR + drawdown guard |
| `dart_quotex/data/database.py` | SQLite store for all historical OHLCV and trade history |
| `dart_quotex/data/harvester.py` | Deep candle harvester — bypasses 180-candle limit via chunked pagination |
| `dart_quotex/backtester.py` | Walk-forward backtester with cross-validation |
| `dart_quotex/advisor.py` | **`AIAdvisor`** — unified public interface, imports into your `main.py` |
| `dart_quotex/trader.py` | Live trading session loop |
| `main.py` | CLI dispatcher |

---

## Installation

### 1. Clone
```bash
git clone <this-repo-url>
cd dart-quotex
```

### 2. Install Python dependencies
```bash
# Using pip
pip install -e .

# Using uv (recommended — much faster)
uv pip install -e .

# With PyTorch for full SAC neural network
pip install -e ".[torch]"
```

### 3. Install pyquotex
```bash
pip install git+https://github.com/cleitonleonel/pyquotex
```

> **Without pyquotex**: The bot runs in **mock mode** automatically — all broker calls use realistic simulated data. Useful for testing and development without a Quotex account.

---

## Configuration

```bash
cp .env.example .env
```

Edit `.env` with your credentials and settings:

```env
# Required
QUOTEX_EMAIL=your@email.com
QUOTEX_PASSWORD=yourpassword
QUOTEX_MODE=demo              # demo | real
QUOTEX_ASSET=EURUSD_OTC

# Risk settings
RISK_MIN_CONFIDENCE=0.60      # minimum AI confidence to trade
RISK_KELLY_FRACTION=0.25      # fractional Kelly (0.25 = conservative)
RISK_MAX_DD_PCT=0.10          # stop trading at 10% session drawdown
```

See `.env.example` for all 30+ configurable parameters.

---

## Quickstart

### Step 1 — Download historical data
```bash
python main.py harvest --asset EURUSD_OTC --total 5000
```
This stores ~5,000 one-minute candles in `data/market.db`. Run once; subsequent calls append new data only.

### Step 2 — Validate with a backtest
```bash
python main.py backtest --asset EURUSD_OTC --balance 1000 --save
```
Output:
```
═══════════════════════════════════════════════════════
  BACKTEST RESULTS  —  EURUSD_OTC (60s)
═══════════════════════════════════════════════════════
  Trades         : 312
  Win Rate       : 58.3%
  Profit Factor  : 1.42
  Max Drawdown   : 7.1%
  Sharpe         : 0.847
  ROI            : +14.2%
  Start Balance  : 1000.00
  End Balance    : 1142.00
═══════════════════════════════════════════════════════
```
Add `--crossval` for 5-fold time-series cross-validation.

### Step 3 — Start live trading
```bash
python main.py trade --asset EURUSD_OTC --session 60
```
Runs for 60 minutes then saves models and exits gracefully.

### One-shot signal test
```bash
python main.py advisor --asset EURUSD_OTC
```

---

## Integrating with your existing Titan Bot

### Option A — Drop-in `brain.py` replacement

```python
# main.py — replace all brain imports with:
from dart_quotex import AIAdvisor

async def main():
    async with AIAdvisor() as advisor:
        # Get a signal from the AI
        direction, confidence = await advisor.get_signal("EURUSD_OTC")
        # direction = "CALL" | "PUT" | "HOLD"
        # confidence = 0.0 – 1.0

        # Or let it trade end-to-end:
        result = await advisor.trade(asset="EURUSD_OTC")
        # result = {"won": True, "payout": 8.0, "balance": 1008.0, ...}
```

### Option B — Confirmation filter (keep your existing bot)

```python
from dart_quotex import AIAdvisor

advisor = AIAdvisor()
await advisor.connect()

# Inside your existing signal loop:
your_signal = brain.get_signal(candles)   # your existing code

ai_direction, ai_confidence = advisor.assess(candles, asset="EURUSD_OTC")

if (
    ai_confidence >= 0.62
    and ai_direction == your_signal.upper()
):
    # Both agree — proceed with your existing order placement
    success, trade_id = await client.buy(...)

    # After trade closes, teach the AI what happened:
    advisor.record_outcome(won=True)   # or False
```

See `integration_example.py` for a full `TitanBotWithAIFilter` wrapper class.

---

## Incremental learning

The AI updates itself after **every trade** — no batch retraining needed.

### What happens automatically
After each trade closes:
1. **Ensemble** — SGD model updates immediately via `partial_fit`. RF + GB retrain every 20 new samples.
2. **SAC agent** — new transition is stored in replay buffer; 4 gradient steps taken.
3. **Meta-weights** — ensemble model weights adjust based on recent per-model accuracy.
4. **Persistence** — models are saved to disk when `advisor.disconnect()` is called.

### Manual update (Option B integration)
```python
# After your trade closes:
advisor.record_outcome(won=True)   # or won=False
```

### Offline batch training from history
```python
from dart_quotex.data.database import Database
from dart_quotex.ml.ensemble import EnsembleModel
from dart_quotex.ml.features import build_features

db = Database("data/market.db")
df = db.get_candles("EURUSD_OTC", 60, limit=3000)

# Build labelled dataset (label = 1 if next close > current close)
X, y = [], []
for i in range(100, len(df) - 1):
    features = build_features(df.iloc[:i])
    label = 1 if df["close"].iloc[i] < df["close"].iloc[i+1] else 0
    X.append(features)
    y.append(label)

model = EnsembleModel()
model.train_batch(np.array(X), np.array(y))
model.save("models/")
```

---

## Architecture

```
main.py ──► CLI dispatcher
              │
              ├── harvest  ──► DataHarvester ──► QuotexClient (pyquotex)
              │                                   └──► Database (SQLite)
              │
              ├── backtest ──► Backtester ──► AIAdvisor
              │                              ├── EnsembleModel  (ML prediction)
              │                              ├── SACAgent       (RL sizing)
              │                              ├── RiskManager    (Kelly + VaR)
              │                              └── FeaturePipeline (34 features)
              │                                      └── SMC indicators
              │
              └── trade ───► LiveTrader ──► AIAdvisor ──► QuotexClient
```

---

## Key design decisions

| Decision | Rationale |
|----------|-----------|
| SQLite offline DB | No 24/7 internet required; data survives restarts |
| 180-candle chunked harvest | Bypasses Quotex API limit; builds unlimited history |
| Random delays (0.5–2s) | Anti-automation; mimics human behaviour |
| Fractional Kelly (0.25×) | Mathematically optimal sizing, conservatively scaled |
| Monte Carlo VaR gate | Rejects trades that would blow daily risk budget |
| pyquotex mock fallback | Full test coverage without live broker connection |
| SAC entropy maximisation | Prevents action collapse; built-in uncertainty quantification |
| Unmitigated OBs only | Consistent with your existing `brain.py` OB logic |

---

## Running tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

All 25 tests run offline in mock mode — no Quotex credentials needed.

---

## File structure

```
dart-quotex/
├── .env.example              ← copy to .env, fill credentials
├── main.py                   ← CLI entry point
├── integration_example.py    ← integration patterns for Titan Bot
├── pyproject.toml
├── dart_quotex/
│   ├── __init__.py           ← exports AIAdvisor
│   ├── advisor.py            ← unified public interface
│   ├── trader.py             ← live session loop
│   ├── backtester.py         ← walk-forward backtest engine
│   ├── config.py             ← all settings from .env
│   ├── api/
│   │   └── quotex_client.py  ← Quotex adapter (replaces deriv_client.py)
│   ├── ml/
│   │   ├── features.py       ← 34-feature engineering pipeline
│   │   ├── ensemble.py       ← RF + GB + SGD with incremental learning
│   │   └── sac_agent.py      ← Soft Actor-Critic RL agent
│   ├── risk/
│   │   └── manager.py        ← Kelly + Monte Carlo VaR + drawdown guard
│   ├── data/
│   │   ├── database.py       ← SQLite OHLCV + trade store
│   │   └── harvester.py      ← deep historical data downloader
│   └── smc/
│       └── indicators.py     ← order blocks, FVG, liquidity sweeps, BOS
├── tests/
│   └── test_dart_quotex.py   ← 25-test offline test suite
└── data/                     ← auto-created; holds market.db
```

---

## Important notes

- **Start on demo** — Always validate with `python main.py backtest` before switching `QUOTEX_MODE=real`.
- **1-hour window** — The bot is designed for short daily sessions. Run `harvest` offline (any time), then `trade` during your window.
- **Model warmup** — The ensemble needs 50 trades before confidence scores are reliable. The SAC agent starts from its neural prior immediately. During warmup, the system is intentionally conservative.
- **pyquotex stability** — pyquotex is community-maintained and may occasionally break on Quotex platform updates. If `connect()` fails, check for a newer version.

---

## License

MIT
