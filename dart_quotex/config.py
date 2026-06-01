"""
dart_quotex/config.py
Centralised configuration — reads from .env / environment variables.
All other modules import from here; never read os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env from project root (or wherever the process starts)
load_dotenv(override=False)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()

def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default

def _env_bool(key: str, default: bool = False) -> bool:
    val = _env(key, str(default)).lower()
    return val in ("1", "true", "yes", "on")


# ──────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QuotexConfig:
    email: str = field(default_factory=lambda: _env("QUOTEX_EMAIL"))
    password: str = field(default_factory=lambda: _env("QUOTEX_PASSWORD"))
    mode: Literal["demo", "real"] = field(
        default_factory=lambda: _env("QUOTEX_MODE", "demo")  # type: ignore[return-value]
    )
    asset: str = field(default_factory=lambda: _env("QUOTEX_ASSET", "EURUSD_OTC"))
    duration: int = field(default_factory=lambda: _env_int("QUOTEX_DURATION", 60))
    # Anti-automation delays (seconds)
    delay_min: float = field(default_factory=lambda: _env_float("QUOTEX_DELAY_MIN", 0.5))
    delay_max: float = field(default_factory=lambda: _env_float("QUOTEX_DELAY_MAX", 2.0))


@dataclass(frozen=True)
class RiskConfig:
    # Fraction of balance to risk per trade (pre-Kelly)
    base_risk_pct: float = field(default_factory=lambda: _env_float("RISK_BASE_PCT", 0.02))
    # Kelly multiplier (0-1, fractional Kelly)
    kelly_fraction: float = field(default_factory=lambda: _env_float("RISK_KELLY_FRACTION", 0.25))
    max_risk_pct: float = field(default_factory=lambda: _env_float("RISK_MAX_PCT", 0.05))
    min_stake: float = field(default_factory=lambda: _env_float("RISK_MIN_STAKE", 1.0))
    # Monte Carlo VaR confidence level
    var_confidence: float = field(default_factory=lambda: _env_float("RISK_VAR_CONFIDENCE", 0.95))
    var_simulations: int = field(default_factory=lambda: _env_int("RISK_VAR_SIMS", 10_000))
    # Stop trading if drawdown exceeds this fraction
    max_drawdown_pct: float = field(default_factory=lambda: _env_float("RISK_MAX_DD_PCT", 0.10))
    # Minimum confidence score (0-1) to place a trade
    min_confidence: float = field(default_factory=lambda: _env_float("RISK_MIN_CONFIDENCE", 0.60))


@dataclass(frozen=True)
class MLConfig:
    # Path to persisted model artefacts
    model_dir: Path = field(
        default_factory=lambda: Path(_env("ML_MODEL_DIR", "models"))
    )
    # Lookback window (candles) for feature calculation
    lookback: int = field(default_factory=lambda: _env_int("ML_LOOKBACK", 100))
    # Minimum samples before the ensemble trains / predicts
    min_samples: int = field(default_factory=lambda: _env_int("ML_MIN_SAMPLES", 50))
    # SAC hyperparameters
    sac_lr: float = field(default_factory=lambda: _env_float("SAC_LR", 3e-4))
    sac_gamma: float = field(default_factory=lambda: _env_float("SAC_GAMMA", 0.99))
    sac_tau: float = field(default_factory=lambda: _env_float("SAC_TAU", 0.005))
    sac_alpha: float = field(default_factory=lambda: _env_float("SAC_ALPHA", 0.2))
    replay_buffer_size: int = field(default_factory=lambda: _env_int("SAC_REPLAY_SIZE", 50_000))
    batch_size: int = field(default_factory=lambda: _env_int("SAC_BATCH_SIZE", 64))


@dataclass(frozen=True)
class DataConfig:
    db_path: Path = field(
        default_factory=lambda: Path(_env("DATA_DB_PATH", "data/market.db"))
    )
    # How many candles to fetch per chunk when harvesting
    harvest_chunk: int = field(default_factory=lambda: _env_int("DATA_HARVEST_CHUNK", 180))
    # Total candles to harvest per asset/timeframe
    harvest_total: int = field(default_factory=lambda: _env_int("DATA_HARVEST_TOTAL", 5_000))
    # Candle granularity options (seconds)
    granularity: int = field(default_factory=lambda: _env_int("DATA_GRANULARITY", 60))


@dataclass(frozen=True)
class AppConfig:
    quotex: QuotexConfig = field(default_factory=QuotexConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    # "live" | "backtest" | "advisor"
    run_mode: str = field(default_factory=lambda: _env("RUN_MODE", "live"))


# Singleton — import this everywhere
cfg = AppConfig()
