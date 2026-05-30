"""
dart_quotex/metrics/performance.py
Advanced performance analytics suite.

Metrics implemented
-------------------
Standard
  Win Rate, Profit Factor, Total Return, ROI
Risk-adjusted
  Sharpe Ratio (annualised)
  Sortino Ratio (downside deviation only)
  Calmar Ratio  (return / max drawdown)
  Information Ratio
  Omega Ratio
Tail-risk
  Maximum Drawdown (with duration)
  Value at Risk (VaR) — parametric + historical
  Expected Shortfall / CVaR
  Maximum Adverse Excursion (MAE)
  Maximum Favourable Excursion (MFE)
Trade quality
  Average Win / Average Loss
  Win/Loss Ratio
  Expectancy (per-trade expected value)
  Consecutive Wins/Losses (streaks)
  R-multiple distribution
  Trade Duration analysis
Market exposure
  Time in Market fraction
  Trades per day
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Trade record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    ts_open:    float        # unix timestamp open
    ts_close:   float        # unix timestamp close
    direction:  str          # "call" | "put"
    stake:      float
    payout:     float        # net payout (positive = win, negative = loss)
    confidence: float = 0.5
    asset:      str   = ""
    won:        bool  = False

    @property
    def duration(self) -> float:
        return self.ts_close - self.ts_open

    @property
    def r_multiple(self) -> float:
        """R-multiple: payout / stake (positive = win multiple)."""
        return self.payout / (self.stake + 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# PerformanceMetrics — computed from a list of TradeRecords
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    """Full performance analytics result set."""

    # ── basic ─────────────────────────────────────────────────────────────────
    n_trades:        int   = 0
    n_wins:          int   = 0
    n_losses:        int   = 0
    win_rate:        float = 0.0
    gross_profit:    float = 0.0
    gross_loss:      float = 0.0
    net_profit:      float = 0.0
    roi:             float = 0.0
    profit_factor:   float = 0.0
    avg_win:         float = 0.0
    avg_loss:        float = 0.0
    win_loss_ratio:  float = 0.0
    expectancy:      float = 0.0     # per-trade expected value in currency

    # ── risk-adjusted ─────────────────────────────────────────────────────────
    sharpe:          float = 0.0
    sortino:         float = 0.0
    calmar:          float = 0.0
    omega:           float = 0.0
    information_ratio: float = 0.0

    # ── drawdown ──────────────────────────────────────────────────────────────
    max_drawdown:    float = 0.0     # fraction
    max_dd_duration: int   = 0       # consecutive trades in drawdown
    avg_drawdown:    float = 0.0

    # ── tail risk ─────────────────────────────────────────────────────────────
    var_95:          float = 0.0     # VaR at 95% confidence
    var_99:          float = 0.0     # VaR at 99% confidence
    cvar_95:         float = 0.0     # Expected Shortfall (CVaR) at 95%
    cvar_99:         float = 0.0
    max_mae:         float = 0.0     # Maximum Adverse Excursion
    avg_mfe:         float = 0.0     # Average Maximum Favourable Excursion

    # ── streaks ───────────────────────────────────────────────────────────────
    max_consec_wins:   int = 0
    max_consec_losses: int = 0
    current_streak:    int = 0       # + = wins, - = losses

    # ── R-multiple stats ──────────────────────────────────────────────────────
    r_mean:          float = 0.0
    r_std:           float = 0.0
    r_skew:          float = 0.0

    # ── time ─────────────────────────────────────────────────────────────────
    trades_per_day:  float = 0.0
    avg_duration_s:  float = 0.0
    total_session_s: float = 0.0

    # ── asset breakdown ───────────────────────────────────────────────────────
    per_asset:       Dict[str, dict] = field(default_factory=dict)

    def summary_str(self) -> str:
        """One-page text summary."""
        sep = "═" * 58
        lines = [
            f"\n{sep}",
            "  PERFORMANCE ANALYTICS",
            sep,
            f"  Trades          : {self.n_trades}",
            f"  Win Rate        : {self.win_rate:.1%}",
            f"  Net Profit      : {self.net_profit:+.2f}",
            f"  ROI             : {self.roi:+.1%}",
            f"  Profit Factor   : {self.profit_factor:.3f}",
            f"  Expectancy      : {self.expectancy:+.4f} per trade",
            sep,
            "  RISK-ADJUSTED",
            f"  Sharpe          : {self.sharpe:.3f}",
            f"  Sortino         : {self.sortino:.3f}",
            f"  Calmar          : {self.calmar:.3f}",
            f"  Omega           : {self.omega:.3f}",
            sep,
            "  DRAWDOWN",
            f"  Max Drawdown    : {self.max_drawdown:.1%}",
            f"  Max DD Duration : {self.max_dd_duration} trades",
            sep,
            "  TAIL RISK",
            f"  VaR 95%         : {self.var_95:.4f}",
            f"  VaR 99%         : {self.var_99:.4f}",
            f"  CVaR 95%        : {self.cvar_95:.4f}",
            f"  CVaR 99%        : {self.cvar_99:.4f}",
            sep,
            "  STREAKS",
            f"  Max Consec Wins : {self.max_consec_wins}",
            f"  Max Consec Loss : {self.max_consec_losses}",
            sep,
            "  R-MULTIPLES",
            f"  Mean R          : {self.r_mean:.3f}",
            f"  Std R           : {self.r_std:.3f}",
            f"  Skew R          : {self.r_skew:.3f}",
            sep,
            "  TIME",
            f"  Trades/day      : {self.trades_per_day:.1f}",
            f"  Avg Duration    : {self.avg_duration_s:.0f}s",
            sep,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "per_asset"}


# ──────────────────────────────────────────────────────────────────────────────
# Calculator
# ──────────────────────────────────────────────────────────────────────────────

class PerformanceCalculator:
    """
    Compute all performance metrics from a list of TradeRecords.

    Parameters
    ----------
    start_balance      : initial capital
    risk_free_rate     : annualised risk-free rate (default 0)
    threshold          : minimum acceptable return (for Sortino / Omega)
    annualisation      : number of trades per year (for Sharpe etc.)
                         default 252 × 10 = ~10 trades/day
    """

    def __init__(
        self,
        start_balance: float = 1000.0,
        risk_free_rate: float = 0.0,
        threshold: float = 0.0,
        annualisation: float = 2520.0,
    ) -> None:
        self.start_balance = start_balance
        self.rfr           = risk_free_rate
        self.threshold     = threshold
        self.annual        = annualisation

    def compute(self, trades: List[TradeRecord]) -> PerformanceMetrics:
        """Compute all metrics from trade list."""
        m = PerformanceMetrics()
        if not trades:
            return m

        m.n_trades  = len(trades)
        m.n_wins    = sum(1 for t in trades if t.won)
        m.n_losses  = m.n_trades - m.n_wins
        m.win_rate  = m.n_wins / m.n_trades

        wins_pnl  = [t.payout for t in trades if t.won]
        loss_pnl  = [t.payout for t in trades if not t.won]

        m.gross_profit = sum(wins_pnl)
        m.gross_loss   = abs(sum(loss_pnl))
        m.net_profit   = m.gross_profit - m.gross_loss
        m.roi          = m.net_profit / (self.start_balance + 1e-9)

        m.profit_factor = (
            m.gross_profit / m.gross_loss if m.gross_loss > 0 else float("inf")
        )
        m.avg_win   = float(np.mean(wins_pnl)) if wins_pnl else 0.0
        m.avg_loss  = float(np.mean([abs(p) for p in loss_pnl])) if loss_pnl else 0.0
        m.win_loss_ratio = m.avg_win / (m.avg_loss + 1e-9)

        # Expectancy = WR × avg_win - LR × avg_loss (as fraction of stake)
        wr  = m.win_rate
        lr  = 1 - wr
        m.expectancy = wr * m.avg_win - lr * m.avg_loss

        # ── Returns series ────────────────────────────────────────────────────
        returns  = np.array([t.payout / (t.stake + 1e-9) for t in trades])
        m.r_mean = float(returns.mean())
        m.r_std  = float(returns.std()) + 1e-9
        if m.r_std > 1e-10 and len(returns) >= 3:
            m.r_skew = float(
                np.mean(((returns - m.r_mean) / m.r_std) ** 3)
            )

        # ── Equity curve ──────────────────────────────────────────────────────
        equity = np.zeros(m.n_trades + 1)
        equity[0] = self.start_balance
        for i, t in enumerate(trades):
            equity[i + 1] = equity[i] + t.payout
        equity = np.maximum(equity, 0)

        # ── Sharpe ────────────────────────────────────────────────────────────
        excess = returns - self.rfr / self.annual
        m.sharpe = float(
            np.sqrt(self.annual) * excess.mean() / (excess.std() + 1e-9)
        )

        # ── Sortino (downside deviation) ─────────────────────────────────────
        downside = np.where(returns < self.threshold, returns - self.threshold, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside ** 2))) + 1e-9
        m.sortino = float(
            np.sqrt(self.annual) * (returns.mean() - self.threshold) / downside_dev
        )

        # ── Drawdown ──────────────────────────────────────────────────────────
        m.max_drawdown, m.max_dd_duration, m.avg_drawdown = self._drawdown(equity)

        # ── Calmar ────────────────────────────────────────────────────────────
        annual_ret = m.roi * (self.annual / max(m.n_trades, 1))
        m.calmar   = annual_ret / (m.max_drawdown + 1e-9)

        # ── Omega ─────────────────────────────────────────────────────────────
        above = returns[returns > self.threshold] - self.threshold
        below = self.threshold - returns[returns < self.threshold]
        m.omega = float(above.sum() / (below.sum() + 1e-9))

        # ── Information Ratio ─────────────────────────────────────────────────
        # Against a zero benchmark
        m.information_ratio = float(
            returns.mean() / (returns.std() + 1e-9) * math.sqrt(self.annual)
        )

        # ── VaR / CVaR ────────────────────────────────────────────────────────
        m.var_95  = float(-np.percentile(returns, 5))
        m.var_99  = float(-np.percentile(returns, 1))
        tail_95   = returns[returns <= -m.var_95]
        tail_99   = returns[returns <= -m.var_99]
        m.cvar_95 = float(-tail_95.mean()) if len(tail_95) > 0 else m.var_95
        m.cvar_99 = float(-tail_99.mean()) if len(tail_99) > 0 else m.var_99

        # ── MAE / MFE (approximate for binary options) ────────────────────────
        m.max_mae = float(max(abs(p) for p in loss_pnl) if loss_pnl else 0.0)
        m.avg_mfe = float(np.mean(wins_pnl)) if wins_pnl else 0.0

        # ── Streaks ───────────────────────────────────────────────────────────
        m.max_consec_wins, m.max_consec_losses, m.current_streak = self._streaks(trades)

        # ── Time metrics ──────────────────────────────────────────────────────
        durations = [t.duration for t in trades if t.ts_close > t.ts_open]
        m.avg_duration_s = float(np.mean(durations)) if durations else 0.0

        if len(trades) >= 2:
            total_s = trades[-1].ts_close - trades[0].ts_open
            m.total_session_s = total_s
            days = max(1, total_s / 86400)
            m.trades_per_day = m.n_trades / days

        # ── Per-asset breakdown ───────────────────────────────────────────────
        assets = set(t.asset for t in trades if t.asset)
        for asset in assets:
            asset_trades = [t for t in trades if t.asset == asset]
            w = sum(1 for t in asset_trades if t.won)
            n = len(asset_trades)
            net = sum(t.payout for t in asset_trades)
            m.per_asset[asset] = {
                "n_trades":  n,
                "win_rate":  w / n if n else 0.0,
                "net_profit": net,
            }

        return m

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _drawdown(equity: np.ndarray) -> Tuple[float, int, float]:
        """Compute max drawdown, max drawdown duration, average drawdown."""
        peak       = equity[0]
        max_dd     = 0.0
        max_dur    = 0
        cur_dur    = 0
        total_dd   = 0.0
        dd_samples = 0

        for v in equity[1:]:
            if v >= peak:
                peak    = v
                cur_dur = 0
            else:
                cur_dur += 1
                dd = (peak - v) / (peak + 1e-9)
                max_dd   = max(max_dd, dd)
                max_dur  = max(max_dur, cur_dur)
                total_dd += dd
                dd_samples += 1

        avg_dd = total_dd / max(1, dd_samples)
        return float(max_dd), int(max_dur), float(avg_dd)

    @staticmethod
    def _streaks(trades: List[TradeRecord]) -> Tuple[int, int, int]:
        """Return (max_win_streak, max_loss_streak, current_streak)."""
        max_w = max_l = 0
        cur_w = cur_l = 0
        current = 0

        for t in trades:
            if t.won:
                cur_w  += 1
                cur_l   = 0
                current += 1
                max_w   = max(max_w, cur_w)
            else:
                cur_l  += 1
                cur_w   = 0
                current = -(cur_l)
                max_l   = max(max_l, cur_l)

        return max_w, max_l, current


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: build from backtester trades
# ──────────────────────────────────────────────────────────────────────────────

def metrics_from_backtest(
    backtest_result,          # BacktestResult from backtester.py
    start_balance: float = 1000.0,
) -> PerformanceMetrics:
    """Convert BacktestResult.trades → TradeRecords → PerformanceMetrics."""
    records = []
    for t in backtest_result.trades:
        records.append(TradeRecord(
            ts_open=float(t.ts),
            ts_close=float(t.ts) + 60.0,
            direction=t.direction,
            stake=t.stake,
            payout=t.payout,
            confidence=t.confidence,
            won=t.won,
        ))

    calc = PerformanceCalculator(start_balance=start_balance)
    return calc.compute(records)
