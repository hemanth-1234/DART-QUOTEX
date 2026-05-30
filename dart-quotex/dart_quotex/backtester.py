"""
dart_quotex/backtester.py
Historical backtesting engine.

Replays candles from the local SQLite database through the full
AI + risk pipeline and produces a detailed performance report.

Features
--------
· Walk-forward simulation (no look-ahead)
· Commission-aware P&L (payout fraction applied per trade)
· Per-trade logging to output CSV
· Summary statistics: win rate, profit factor, max drawdown, Sharpe
· Optional incremental model training during replay (simulate online learning)

Usage
-----
    from dart_quotex.backtester import Backtester
    bt = Backtester(db, advisor)
    results = bt.run(asset="EURUSD_OTC", granularity=60,
                     start_balance=1000, payout=0.80)
    print(results.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from dart_quotex.data.database import Database
from dart_quotex.ml.features import build_features, FEATURE_NAMES
from dart_quotex.risk.manager import RiskManager, SessionStats

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    ts: int
    direction: str     # call | put
    stake: float
    payout: float      # net payout (positive = win, negative = loss)
    balance_after: float
    confidence: float
    won: bool


@dataclass
class BacktestResult:
    asset: str
    granularity: int
    start_balance: float
    end_balance: float
    trades: List[Trade] = field(default_factory=list)

    # ── aggregate metrics ─────────────────────────────────────────────────────

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.won) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.payout for t in self.trades if t.payout > 0)
        gross_loss = abs(sum(t.payout for t in self.trades if t.payout < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0.0
        balances = [self.start_balance] + [t.balance_after for t in self.trades]
        peak = self.start_balance
        dd = 0.0
        for b in balances:
            peak = max(peak, b)
            dd = max(dd, (peak - b) / peak)
        return dd

    @property
    def sharpe(self) -> float:
        """Simplified Sharpe: mean trade return / std."""
        if len(self.trades) < 2:
            return 0.0
        rets = np.array([t.payout / t.stake for t in self.trades if t.stake > 0])
        return float(rets.mean() / (rets.std() + 1e-8))

    @property
    def roi(self) -> float:
        return (self.end_balance - self.start_balance) / self.start_balance

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  BACKTEST RESULTS  —  {self.asset} ({self.granularity}s)\n"
            f"{'='*55}\n"
            f"  Trades         : {self.n_trades}\n"
            f"  Win Rate       : {self.win_rate:.1%}\n"
            f"  Profit Factor  : {self.profit_factor:.2f}\n"
            f"  Max Drawdown   : {self.max_drawdown:.1%}\n"
            f"  Sharpe         : {self.sharpe:.3f}\n"
            f"  ROI            : {self.roi:+.1%}\n"
            f"  Start Balance  : {self.start_balance:.2f}\n"
            f"  End Balance    : {self.end_balance:.2f}\n"
            f"{'='*55}\n"
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "ts": t.ts, "direction": t.direction,
                "stake": t.stake, "payout": t.payout,
                "balance": t.balance_after, "confidence": t.confidence,
                "won": t.won,
            }
            for t in self.trades
        ])

    def save_csv(self, path: Path) -> None:
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info("Backtest trades saved to %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# Backtester
# ──────────────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Walk-forward backtester.

    Parameters
    ----------
    db           : Database instance with historical candles
    advisor      : AIAdvisor (or any object with .assess())
    lookback     : candles required before first trade
    train_online : if True, call advisor.update() after each trade
    """

    def __init__(
        self,
        db: Database,
        advisor,         # AIAdvisor
        lookback: int = 100,
        train_online: bool = True,
    ) -> None:
        self.db = db
        self.advisor = advisor
        self.lookback = lookback
        self.train_online = train_online

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        asset: str,
        granularity: int = 60,
        start_balance: float = 1_000.0,
        payout: float = 0.80,
        limit: int = 2_000,
        min_confidence: float = 0.55,
        output_dir: Optional[Path] = None,
    ) -> BacktestResult:
        """
        Run backtest on stored history.

        Parameters
        ----------
        asset           : e.g. "EURUSD_OTC"
        granularity     : candle size in seconds
        start_balance   : simulated starting balance
        payout          : broker net payout fraction
        limit           : max candles to use from DB
        min_confidence  : minimum AI confidence to trade
        output_dir      : if set, save trade CSV here
        """
        df = self.db.get_candles(asset, granularity, limit=limit)

        if len(df) < self.lookback + 10:
            raise ValueError(
                f"Not enough data for backtest. "
                f"Have {len(df)} candles, need at least {self.lookback + 10}."
            )

        logger.info(
            "Backtesting %s (%ds) | %d candles | balance=%.2f",
            asset, granularity, len(df), start_balance,
        )

        risk = RiskManager(min_confidence=min_confidence)
        risk.start_session(start_balance)

        balance = start_balance
        result = BacktestResult(
            asset=asset,
            granularity=granularity,
            start_balance=start_balance,
            end_balance=start_balance,
        )

        for i in range(self.lookback, len(df)):
            window = df.iloc[:i]   # walk-forward — no future data
            ts = int(window.index[-1].timestamp())

            # ── get AI signal ─────────────────────────────────────────────
            try:
                features = build_features(window, lookback=self.lookback)
            except Exception:
                continue

            direction, confidence, _ = self.advisor.assess_features(features)

            if direction == "HOLD":
                continue

            # ── risk evaluation ───────────────────────────────────────────
            dir_int = 1 if direction == "CALL" else 0
            decision = risk.evaluate(dir_int, confidence, balance, payout)

            if not decision.allowed:
                continue

            # ── simulate trade outcome ────────────────────────────────────
            # Look at the NEXT candle's close to determine direction
            if i + 1 >= len(df):
                break
            next_close = float(df["close"].iloc[i])
            current_close = float(df["close"].iloc[i - 1])

            actual_up = next_close > current_close
            won = (direction == "CALL" and actual_up) or (direction == "PUT" and not actual_up)

            if won:
                net_payout = decision.stake * payout
            else:
                net_payout = -decision.stake

            balance += net_payout
            balance = max(balance, 0)

            risk.record_trade(decision.stake, decision.stake + net_payout, balance, won)

            result.trades.append(Trade(
                ts=ts,
                direction=decision.direction,
                stake=decision.stake,
                payout=net_payout,
                balance_after=balance,
                confidence=confidence,
                won=won,
            ))

            # ── optional online training ───────────────────────────────────
            if self.train_online:
                label = 1 if won else 0
                self.advisor.update_model(features, label)

        result.end_balance = balance
        logger.info(result.summary())

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            result.save_csv(out / f"backtest_{asset}_{granularity}.csv")

        return result

    def cross_validate(
        self,
        asset: str,
        granularity: int,
        n_folds: int = 5,
        **run_kwargs,
    ) -> List[BacktestResult]:
        """
        Time-series cross-validation: split historical data into n_folds
        and run a backtest on each fold.
        """
        df = self.db.get_candles(asset, granularity, limit=10_000)
        fold_size = len(df) // n_folds
        results = []

        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size
            fold_df = df.iloc[start:end]

            # Temporarily patch database with fold data
            logger.info("Cross-val fold %d/%d (%d candles)", fold + 1, n_folds, len(fold_df))

            # Write fold to temp table (simple approach: use subset df directly)
            # For simplicity we call _run_on_df directly
            fold_result = self._run_on_df(fold_df, asset, granularity, **run_kwargs)
            results.append(fold_result)

        avg_wr = np.mean([r.win_rate for r in results])
        avg_roi = np.mean([r.roi for r in results])
        logger.info(
            "Cross-val summary: avg WR=%.1f%% avg ROI=%.1f%%",
            avg_wr * 100, avg_roi * 100,
        )
        return results

    def _run_on_df(
        self,
        df: pd.DataFrame,
        asset: str,
        granularity: int,
        start_balance: float = 1_000.0,
        payout: float = 0.80,
        min_confidence: float = 0.55,
    ) -> BacktestResult:
        """Internal: run backtest on an already-loaded DataFrame."""
        risk = RiskManager(min_confidence=min_confidence)
        risk.start_session(start_balance)
        balance = start_balance
        result = BacktestResult(
            asset=asset, granularity=granularity,
            start_balance=start_balance, end_balance=start_balance,
        )

        for i in range(self.lookback, len(df)):
            window = df.iloc[:i]
            ts = int(window.index[-1].timestamp())
            try:
                features = build_features(window, lookback=self.lookback)
            except Exception:
                continue

            direction, confidence, _ = self.advisor.assess_features(features)
            if direction == "HOLD":
                continue

            dir_int = 1 if direction == "CALL" else 0
            decision = risk.evaluate(dir_int, confidence, balance, payout)
            if not decision.allowed:
                continue

            if i + 1 >= len(df):
                break
            next_c = float(df["close"].iloc[i])
            curr_c = float(df["close"].iloc[i - 1])
            actual_up = next_c > curr_c
            won = (direction == "CALL" and actual_up) or (direction == "PUT" and not actual_up)
            net = decision.stake * payout if won else -decision.stake
            balance = max(balance + net, 0)
            risk.record_trade(decision.stake, decision.stake + net, balance, won)
            result.trades.append(Trade(ts=ts, direction=decision.direction,
                stake=decision.stake, payout=net, balance_after=balance,
                confidence=confidence, won=won))

        result.end_balance = balance
        return result
