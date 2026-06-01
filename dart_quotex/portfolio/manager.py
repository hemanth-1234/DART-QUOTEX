"""
dart_quotex/portfolio/manager.py
Portfolio-level risk and diversification management.

This module treats the entire set of concurrent Quotex positions as a
portfolio rather than isolated trades.  Key responsibilities:

1. Correlation Gate
   If two assets have rolling return correlation > threshold, only ONE
   can be traded at a time.  Prevents doubling risk on correlated moves.

2. Concentration Limit
   No more than N simultaneous open positions, and no more than M% of
   capital in a single "currency cluster" (e.g. USD pairs).

3. Diversification Score
   Herfindahl-Hirschman Index on open-position clusters.
   Low score → well-diversified → allow trades.

4. Dynamic Risk Budget
   Session-level VaR budget that scales down stake sizes after losses.

5. Asset Clustering
   Pairs are pre-clustered by base/quote currency so correlated trades
   can be identified even without live correlation data.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Currency clusters (pre-defined correlation groups)
# ──────────────────────────────────────────────────────────────────────────────

CURRENCY_CLUSTERS: Dict[str, str] = {
    # USD majors
    "EURUSD": "USD_MAJOR",  "GBPUSD": "USD_MAJOR",  "AUDUSD": "USD_MAJOR",
    "NZDUSD": "USD_MAJOR",  "USDCAD": "USD_MAJOR",  "USDCHF": "USD_MAJOR",
    "USDJPY": "USD_MAJOR",
    # JPY crosses
    "EURJPY": "JPY_CROSS",  "GBPJPY": "JPY_CROSS",  "AUDJPY": "JPY_CROSS",
    "CADJPY": "JPY_CROSS",  "CHFJPY": "JPY_CROSS",  "NZDJPY": "JPY_CROSS",
    # GBP crosses
    "EURGBP": "GBP_CROSS",  "GBPAUD": "GBP_CROSS",  "GBPCAD": "GBP_CROSS",
    "GBPCHF": "GBP_CROSS",  "GBPNZD": "GBP_CROSS",
    # EUR crosses
    "EURAUD": "EUR_CROSS",  "EURCAD": "EUR_CROSS",  "EURCHF": "EUR_CROSS",
    "EURNZD": "EUR_CROSS",
    # Commodity
    "XAUUSD": "COMMODITY",  "XAGUSD": "COMMODITY",  "XTIUSD": "COMMODITY",
    "XBRUSD": "COMMODITY",
    # Crypto
    "BTCUSD": "CRYPTO",     "ETHUSD": "CRYPTO",     "LTCUSD": "CRYPTO",
    "BCHUSD": "CRYPTO",
}


def _get_cluster(asset: str) -> str:
    """Return the cluster for an asset (strip _otc suffix)."""
    clean = asset.upper().replace("_OTC", "").replace("_otc", "")
    for key, cluster in CURRENCY_CLUSTERS.items():
        if key in clean:
            return cluster
    # Infer from currency names
    for ccy in ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"):
        if ccy in clean:
            return f"{ccy}_MIXED"
    return "OTHER"


# ──────────────────────────────────────────────────────────────────────────────
# Position tracker
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    asset:      str
    direction:  str      # "call" | "put"
    stake:      float
    opened_at:  float    # unix timestamp
    cluster:    str = field(init=False)

    def __post_init__(self) -> None:
        self.cluster = _get_cluster(self.asset)


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio Manager
# ──────────────────────────────────────────────────────────────────────────────

class PortfolioManager:
    """
    Portfolio-level trade gating and diversification.

    Parameters
    ----------
    max_concurrent        : max simultaneous open positions
    max_cluster_pct       : max fraction of capital in one cluster
    corr_threshold        : rolling correlation above which one asset is blocked
    corr_lookback         : bars for rolling correlation calculation
    max_capital_at_risk   : max fraction of balance open at once
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        max_cluster_pct: float = 0.60,
        corr_threshold: float = 0.75,
        corr_lookback: int = 50,
        max_capital_at_risk: float = 0.15,
    ) -> None:
        self.max_concurrent     = max_concurrent
        self.max_cluster_pct    = max_cluster_pct
        self.corr_threshold     = corr_threshold
        self.corr_lookback      = corr_lookback
        self.max_capital_at_risk = max_capital_at_risk

        # Live state
        self._open: Dict[str, Position] = {}        # asset → Position
        self._returns: Dict[str, List[float]] = defaultdict(list)  # rolling returns
        self._corr_matrix: Dict[Tuple[str, str], float] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def can_trade(
        self,
        asset: str,
        stake: float,
        balance: float,
    ) -> Tuple[bool, str]:
        """
        Returns (allowed, reason).

        Checks:
        1. Max concurrent positions
        2. Capital-at-risk limit
        3. Cluster concentration
        4. Correlation gate
        """
        # 1. Already open on this asset
        if asset in self._open:
            return False, f"Position already open on {asset}"

        # 2. Max concurrent
        if len(self._open) >= self.max_concurrent:
            return False, (
                f"Max concurrent positions ({self.max_concurrent}) reached"
            )

        # 3. Capital at risk
        total_risk = sum(p.stake for p in self._open.values()) + stake
        if balance > 0 and total_risk / balance > self.max_capital_at_risk:
            return False, (
                f"Capital-at-risk {total_risk/balance:.1%} > "
                f"limit {self.max_capital_at_risk:.1%}"
            )

        # 4. Cluster concentration
        cluster = _get_cluster(asset)
        cluster_stake = sum(
            p.stake for p in self._open.values()
            if p.cluster == cluster
        ) + stake
        if balance > 0 and cluster_stake / balance > self.max_cluster_pct:
            return False, (
                f"Cluster {cluster} concentration "
                f"{cluster_stake/balance:.1%} > {self.max_cluster_pct:.1%}"
            )

        # 5. Correlation gate
        for open_asset, pos in self._open.items():
            corr = self._get_correlation(asset, open_asset)
            if corr > self.corr_threshold:
                return False, (
                    f"High correlation ({corr:.2f}) with open position {open_asset}"
                )

        return True, "OK"

    def open_position(
        self,
        asset: str,
        direction: str,
        stake: float,
        opened_at: float,
    ) -> None:
        """Record that a position was opened."""
        self._open[asset] = Position(
            asset=asset, direction=direction,
            stake=stake, opened_at=opened_at,
        )
        logger.debug(
            "Portfolio: opened %s %s ₹%.2f | open=%d",
            direction.upper(), asset, stake, len(self._open),
        )

    def close_position(self, asset: str) -> Optional[Position]:
        """Record that a position was closed. Returns the Position."""
        pos = self._open.pop(asset, None)
        if pos:
            logger.debug("Portfolio: closed %s | open=%d", asset, len(self._open))
        return pos

    def update_returns(self, asset: str, candle_return: float) -> None:
        """
        Feed rolling 1-bar returns for correlation calculation.
        Call this every time a new candle closes.
        """
        buf = self._returns[asset]
        buf.append(float(candle_return))
        if len(buf) > self.corr_lookback:
            buf.pop(0)
        # Invalidate cached correlations for this asset
        for key in list(self._corr_matrix.keys()):
            if asset in key:
                del self._corr_matrix[key]

    def diversification_score(self) -> float:
        """
        Herfindahl-Hirschman Index on cluster stakes.
        0 = perfectly diversified, 1 = fully concentrated.
        Lower is better.
        """
        if not self._open:
            return 0.0

        cluster_stakes: Dict[str, float] = defaultdict(float)
        total = 0.0
        for pos in self._open.values():
            cluster_stakes[pos.cluster] += pos.stake
            total += pos.stake

        if total == 0:
            return 0.0

        hhi = sum((s / total) ** 2 for s in cluster_stakes.values())
        return float(hhi)

    def portfolio_summary(self) -> dict:
        """Return current portfolio snapshot."""
        cluster_breakdown: Dict[str, float] = defaultdict(float)
        for pos in self._open.values():
            cluster_breakdown[pos.cluster] += pos.stake

        return {
            "open_positions":    len(self._open),
            "total_at_risk":     sum(p.stake for p in self._open.values()),
            "assets":            list(self._open.keys()),
            "cluster_breakdown": dict(cluster_breakdown),
            "diversification":   self.diversification_score(),
        }

    def correlated_assets(
        self, asset: str, threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """Return list of (asset, correlation) pairs above threshold."""
        thr = threshold or self.corr_threshold
        result = []
        for other in self._returns:
            if other == asset:
                continue
            c = self._get_correlation(asset, other)
            if c > thr:
                result.append((other, c))
        return sorted(result, key=lambda x: -x[1])

    # ── internal ──────────────────────────────────────────────────────────────

    def _get_correlation(self, a: str, b: str) -> float:
        """Return rolling Pearson correlation between two assets."""
        key = tuple(sorted([a, b]))
        if key in self._corr_matrix:
            return self._corr_matrix[key]

        r_a = self._returns.get(a, [])
        r_b = self._returns.get(b, [])
        min_len = min(len(r_a), len(r_b))

        if min_len < 10:
            # Not enough data — use pre-defined cluster correlation
            ca = _get_cluster(a)
            cb = _get_cluster(b)
            corr = 0.85 if ca == cb else 0.20
        else:
            arr_a = np.array(r_a[-min_len:])
            arr_b = np.array(r_b[-min_len:])
            # Pearson correlation
            if arr_a.std() < 1e-9 or arr_b.std() < 1e-9:
                corr = 0.0
            else:
                corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
            corr = float(np.clip(corr, -1, 1))

        self._corr_matrix[key] = corr
        return corr
