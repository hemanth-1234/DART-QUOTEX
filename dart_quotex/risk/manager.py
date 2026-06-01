"""
dart_quotex/risk/manager.py
Risk management layer.

Components
----------
1. Kelly Criterion        – optimal bet sizing from win-rate + payout
2. Monte Carlo VaR        – portfolio Value-at-Risk over session horizon
3. Drawdown Guard         – halts trading if session loss exceeds threshold
4. Confidence Gate        – rejects low-confidence signals

All sizing is expressed as fraction of current balance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeDecision:
    allowed: bool
    direction: str           # "call" | "put" | "hold"
    stake: float
    confidence: float
    reason: str = ""


@dataclass
class SessionStats:
    start_balance: float
    current_balance: float
    trades: int = 0
    wins: int = 0
    losses: int = 0
    peak_balance: float = field(init=False)

    def __post_init__(self) -> None:
        self.peak_balance = self.current_balance

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.5

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak as a fraction."""
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance

    @property
    def session_pnl_pct(self) -> float:
        return (self.current_balance - self.start_balance) / self.start_balance

    def update(self, balance: float, won: bool) -> None:
        self.current_balance = balance
        self.peak_balance = max(self.peak_balance, balance)
        self.trades += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1


# ──────────────────────────────────────────────────────────────────────────────
# Risk Manager
# ──────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """
    Evaluates whether a signal should be traded and how much to stake.

    Parameters
    ----------
    base_risk_pct     : fraction of balance to risk per trade (pre-Kelly)
    kelly_fraction    : fractional Kelly multiplier (0.25 = conservative)
    max_risk_pct      : hard cap on single-trade stake / balance
    min_stake         : broker minimum stake (currency units)
    min_confidence    : ML confidence threshold to allow a trade
    max_drawdown_pct  : stop-trading drawdown threshold
    var_confidence    : confidence level for Monte Carlo VaR
    var_simulations   : MC simulation count
    """

    def __init__(
        self,
        base_risk_pct: float = 0.02,
        kelly_fraction: float = 0.25,
        max_risk_pct: float = 0.05,
        min_stake: float = 1.0,
        min_confidence: float = 0.60,
        max_drawdown_pct: float = 0.10,
        var_confidence: float = 0.95,
        var_simulations: int = 10_000,
    ) -> None:
        self._base_risk = base_risk_pct
        self._kelly_frac = kelly_fraction
        self._max_risk = max_risk_pct
        self._min_stake = min_stake
        self._min_conf = min_confidence
        self._max_dd = max_drawdown_pct
        self._var_conf = var_confidence
        self._var_sims = var_simulations

        self._session: Optional[SessionStats] = None
        self._trade_returns: List[float] = []  # historical trade P&L as fractions

    # ── session lifecycle ─────────────────────────────────────────────────────

    def start_session(self, balance: float) -> None:
        self._session = SessionStats(
            start_balance=balance,
            current_balance=balance,
        )
        logger.info("Risk session started. Balance: %.2f", balance)

    def end_session(self) -> Optional[SessionStats]:
        s = self._session
        if s:
            logger.info(
                "Risk session ended. Trades: %d | WR: %.1f%% | P&L: %+.1f%%",
                s.trades,
                s.win_rate * 100,
                s.session_pnl_pct * 100,
            )
        self._session = None
        return s

    # ── core decision ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        direction: int,         # 1 = CALL, 0 = PUT
        confidence: float,      # 0-1 from ensemble
        balance: float,
        payout: float = 0.80,   # broker payout fraction (e.g. 0.80 = 80%)
    ) -> TradeDecision:
        """
        Return a TradeDecision with stake and allow/deny flag.
        """
        dir_str = "call" if direction == 1 else "put"

        # ── 1. Confidence gate ────────────────────────────────────────────────
        if confidence < self._min_conf:
            return TradeDecision(
                allowed=False,
                direction="hold",
                stake=0.0,
                confidence=confidence,
                reason=f"confidence {confidence:.2f} < threshold {self._min_conf:.2f}",
            )

        # ── 2. Drawdown guard ─────────────────────────────────────────────────
        if self._session and self._session.drawdown >= self._max_dd:
            return TradeDecision(
                allowed=False,
                direction="hold",
                stake=0.0,
                confidence=confidence,
                reason=(
                    f"drawdown {self._session.drawdown:.1%} >= "
                    f"max {self._max_dd:.1%}"
                ),
            )

        # ── 3. Kelly stake ────────────────────────────────────────────────────
        stake = self._kelly_stake(confidence, payout, balance)

        # ── 4. VaR check ─────────────────────────────────────────────────────
        risk_pct = stake / balance if balance > 0 else 0
        if self._trade_returns and risk_pct > 0.01:
            var = self._monte_carlo_var(balance, stake)
            if var > balance * self._max_risk:
                # Scale stake down to stay within VaR limit
                stake = balance * self._base_risk
                logger.debug("VaR triggered — stake reduced to %.2f", stake)

        if stake < self._min_stake:
            return TradeDecision(
                allowed=False,
                direction="hold",
                stake=0.0,
                confidence=confidence,
                reason=f"stake {stake:.2f} < minimum {self._min_stake:.2f}",
            )

        return TradeDecision(
            allowed=True,
            direction=dir_str,
            stake=round(stake, 2),
            confidence=confidence,
        )

    def record_trade(
        self,
        stake: float,
        payout_received: float,
        balance: float,
        won: bool,
    ) -> None:
        """Call after each trade settles to update internal state."""
        net = (payout_received - stake) / stake if stake > 0 else 0.0
        self._trade_returns.append(net)
        # Keep a rolling window
        if len(self._trade_returns) > 200:
            self._trade_returns.pop(0)

        if self._session:
            self._session.update(balance, won)

    # ── Kelly Criterion ───────────────────────────────────────────────────────

    def _kelly_stake(
        self,
        confidence: float,
        payout: float,
        balance: float,
    ) -> float:
        """
        Fractional Kelly Criterion.

        Kelly fraction = (p * b - q) / b
        where:
          p = estimated win probability (from ML confidence)
          q = 1 - p
          b = net payout odds (e.g. 0.80)
        """
        # Use historical win rate if available, blend with ML confidence
        if self._session and self._session.trades >= 5:
            p = 0.6 * confidence + 0.4 * self._session.win_rate
        else:
            p = confidence

        q = 1.0 - p
        b = payout

        kelly = (p * b - q) / b if b > 0 else 0.0
        kelly = max(0.0, kelly)

        # Apply fractional Kelly and cap
        fraction = kelly * self._kelly_frac
        fraction = min(fraction, self._max_risk)
        fraction = max(fraction, self._base_risk * 0.5)   # minimum floor

        stake = balance * fraction
        return stake

    # ── Monte Carlo VaR ───────────────────────────────────────────────────────

    def _monte_carlo_var(
        self,
        balance: float,
        stake: float,
        n_trades: int = 10,
    ) -> float:
        """
        Simulate `n_trades` forward using historical return distribution.
        Return the VaR (loss not exceeded with probability `var_confidence`).
        """
        if len(self._trade_returns) < 10:
            return 0.0

        returns = np.array(self._trade_returns)
        mean_r = returns.mean()
        std_r = returns.std()

        # MC simulation
        rng = np.random.default_rng()
        sim_returns = rng.normal(mean_r, std_r, (self._var_sims, n_trades))
        # Compound returns for each simulation path
        final_factors = (1 + sim_returns).prod(axis=1)
        final_balances = balance * (stake / balance) * final_factors

        losses = balance - final_balances
        var = float(np.percentile(losses, self._var_conf * 100))
        return max(0.0, var)

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def session(self) -> Optional[SessionStats]:
        return self._session

    def win_rate_estimate(self) -> float:
        if not self._trade_returns:
            return 0.5
        wins = sum(1 for r in self._trade_returns if r > 0)
        return wins / len(self._trade_returns)
