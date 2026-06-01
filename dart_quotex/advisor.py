"""
dart_quotex/advisor.py  (FULL REWRITE — all components integrated)

AIAdvisor — unified public interface integrating all DART-Quotex components.

Full pipeline per signal
------------------------
1.  Fetch latest candles (robust client + DB cache)
2.  Multi-timeframe analysis (1m/5m/15m/1h)
3.  Feature engineering (88 features: base + MTF + SMC + patterns)
4.  Market regime detection (VAE -> 7 regimes)
5.  Uncertainty quantification (MC Dropout + ensemble disagreement)
6.  Ensemble prediction (RF + GB + SGD)
7.  SAC agent action (with ICM intrinsic reward)
8.  Candlestick pattern confirmation
9.  News sentiment filter
10. Portfolio management gate (correlation + concentration)
11. Regime-aware strategy multipliers
12. Final confidence fusion -> risk manager
13. Persist signal state to JSON (for web dashboard)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dart_quotex.config import cfg
from dart_quotex.api.quotex_client import QuotexClient
from dart_quotex.api.robust_client import RobustQuotexClient
from dart_quotex.data.database import Database
from dart_quotex.data.harvester import DataHarvester
from dart_quotex.ml.ensemble import EnsembleModel
from dart_quotex.ml.features import build_features, FEATURE_NAMES
from dart_quotex.ml.sac_agent import SACAgent
from dart_quotex.ml.regime_detector import MarketRegimeDetector
from dart_quotex.ml.icm import ICM
from dart_quotex.ml.uncertainty import UncertaintyQuantifier
from dart_quotex.ml.multi_timeframe import MultiTimeframeAnalyzer
from dart_quotex.risk.manager import RiskManager, TradeDecision
from dart_quotex.portfolio.manager import PortfolioManager
from dart_quotex.patterns.candlestick import PatternScanner
from dart_quotex.sentiment.news import NewsSentimentEngine
from dart_quotex.metrics.performance import PerformanceCalculator, TradeRecord

logger = logging.getLogger(__name__)

Decision = Tuple[str, float]
_SIGNAL_STATE_FILE = Path("data/.signal_state.json")


class AIAdvisor:
    """Full AI trading advisor — integrates every DART-Quotex component."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        db_path:   Optional[Path] = None,
        mode:      Optional[str]  = None,
        use_robust_client: bool = True,
    ) -> None:
        self._model_dir = Path(model_dir or cfg.ml.model_dir)
        self._db_path   = Path(db_path   or cfg.data.db_path)
        self._mode      = mode or cfg.quotex.mode

        if use_robust_client:
            self.client = RobustQuotexClient(
                email=cfg.quotex.email,
                password=cfg.quotex.password,
                mode=self._mode,
            )
        else:
            self.client = QuotexClient(mode=self._mode)

        self.db        = Database(self._db_path)
        self.harvester = DataHarvester(self.client, self.db)

        n_feats = 88
        self.ensemble = EnsembleModel(min_samples=cfg.ml.min_samples)
        self.sac      = SACAgent(
            state_dim=n_feats,
            lr=cfg.ml.sac_lr, gamma=cfg.ml.sac_gamma,
            tau=cfg.ml.sac_tau, alpha=cfg.ml.sac_alpha,
            buffer_size=cfg.ml.replay_buffer_size,
            batch_size=cfg.ml.batch_size,
        )
        self.regime   = MarketRegimeDetector(seq_len=30)
        self.icm      = ICM(state_dim=n_feats, action_dim=2)
        self.uq       = UncertaintyQuantifier(input_dim=n_feats)
        self.mtf      = MultiTimeframeAnalyzer(timeframes=["1m","5m","15m","1h"])
        self.patterns = PatternScanner()
        self.news     = NewsSentimentEngine()

        self.risk = RiskManager(
            base_risk_pct=cfg.risk.base_risk_pct,
            kelly_fraction=cfg.risk.kelly_fraction,
            max_risk_pct=cfg.risk.max_risk_pct,
            min_stake=cfg.risk.min_stake,
            min_confidence=cfg.risk.min_confidence,
            max_drawdown_pct=cfg.risk.max_drawdown_pct,
            var_confidence=cfg.risk.var_confidence,
            var_simulations=cfg.risk.var_simulations,
        )
        self.portfolio = PortfolioManager(
            max_concurrent=3,
            max_cluster_pct=0.60,
            corr_threshold=0.75,
        )

        self._last_features:  Optional[np.ndarray] = None
        self._last_action:    Optional[np.ndarray] = None
        self._last_state:     Optional[np.ndarray] = None
        self._last_regime_id: int = 6
        self._session_trades: List[TradeRecord] = []
        self._connected       = False

        self._load_models()

    # -- lifecycle -------------------------------------------------------------

    async def connect(self) -> None:
        await self.client.connect()
        balance = await self.client.get_balance()
        self.risk.start_session(balance)
        self._connected = True
        logger.info("AIAdvisor connected. Balance: %.2f", balance)

    async def disconnect(self) -> None:
        self.save_models()
        session = self.risk.end_session()
        if session:
            logger.info(
                "Session: %d trades | WR=%.1f%% | ROI=%+.1f%%",
                session.trades, session.win_rate * 100,
                session.session_pnl_pct * 100,
            )
        await self.client.disconnect()
        self._connected = False

    # -- primary signal --------------------------------------------------------

    async def get_signal(
        self,
        asset:       Optional[str] = None,
        granularity: Optional[int] = None,
    ) -> Decision:
        asset       = asset       or cfg.quotex.asset
        granularity = granularity or cfg.data.granularity

        await self.harvester.refresh_recent(asset, granularity, n=5)
        df = self.db.get_candles(asset, granularity, limit=cfg.ml.lookback + 10)
        if len(df) < 20:
            return "HOLD", 0.0

        try:
            features = build_features(df, lookback=cfg.ml.lookback,
                                       mtf_analyzer=self.mtf)
        except Exception as exc:
            logger.error("Feature build failed: %s", exc)
            return "HOLD", 0.0

        try:
            news_sentiment, news_impact, news_items = await asyncio.wait_for(
                self.news.get_sentiment(asset), timeout=3.0
            )
        except Exception:
            news_sentiment, news_impact, news_items = 0.0, 0.0, []

        direction, confidence, detail = self._full_assess(
            features, df, asset, news_sentiment, news_impact
        )
        self._write_signal_state(direction, confidence, detail, asset,
                                  news_sentiment, news_items)
        return direction, confidence

    def assess(
        self,
        candles: List[Dict[str, Any]],
        asset:   str = "",
        **kwargs,
    ) -> Decision:
        """MODE 2 interface — raw candle list -> (direction, confidence)."""
        df = _candles_to_df(candles)
        if len(df) < 20:
            return "HOLD", 0.0
        try:
            features = build_features(df, lookback=cfg.ml.lookback,
                                       mtf_analyzer=self.mtf)
        except Exception as exc:
            logger.error("assess() feature error: %s", exc)
            return "HOLD", 0.0
        direction, confidence, _ = self._full_assess(features, df, asset, 0.0, 0.0)
        return direction, confidence

    def assess_features(
        self,
        features: np.ndarray,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[str, float, dict]:
        return self._full_assess(features, df, "", 0.0, 0.0)

    # -- full pipeline ---------------------------------------------------------

    def _full_assess(
        self,
        features:       np.ndarray,
        df:             Optional[pd.DataFrame],
        asset:          str,
        news_sentiment: float,
        news_impact:    float,
    ) -> Tuple[str, float, dict]:
        detail: Dict[str, Any] = {}

        # A. Regime detection
        regime_id, regime_name, regime_probs = 6, "CHOPPY", [1/7]*7
        regime_mult = 0.4
        if df is not None and len(df) >= 30:
            candle_arr = df[["open","high","low","close","volume"]].values
            regime_id, regime_name, regime_probs_arr = self.regime.detect(candle_arr)
            regime_probs = regime_probs_arr.tolist()
            strat = self.regime.regime_for_strategy(regime_id)
            regime_mult = strat["position_size_mult"]
            detail.update({
                "regime_id":   regime_id,
                "regime_name": regime_name,
                "regime_probs": regime_probs,
                "regime_skip": strat["skip"],
            })
            self._last_regime_id = regime_id
            if strat["skip"] > 0.5:
                logger.info("Regime %s -> SKIP", regime_name)
                detail.update({"sac_direction": 0.0, "sac_conviction": 0.0,
                                "icm_novelty": 0.5, "pat_dir": 0, "pat_str": 0.0,
                                "mtf_confluence": 0.0, "mtf_direction": "NEUTRAL",
                                "mtf_divergence": 0.0, "mtf_data": {},
                                "news_sentiment": news_sentiment,
                                "news_impact": news_impact,
                                "ens_conf": 0.0, "direction": "HOLD",
                                "final_confidence": 0.0})
                return "HOLD", 0.0, detail
        else:
            detail.update({"regime_name": regime_name, "regime_probs": regime_probs})

        # B. Ensemble
        ens_dir, ens_raw_conf = self.ensemble.predict(features)
        detail["ens_dir"]  = ens_dir
        detail["ens_conf"] = ens_raw_conf

        # C. Uncertainty quantification
        ens_probs = self._get_ensemble_probs(features)
        uq_conf, uq_detail = self.uq.quantify(features, ens_probs)
        detail.update({"uq_" + k: v for k, v in uq_detail.items()})

        # D. SAC
        sac_action = self.sac.act(features)
        sac_raw    = float(sac_action[0])
        sac_conv   = float(sac_action[1])
        icm_novelty = 0.5
        if self._last_state is not None:
            icm_novelty = self.icm.novelty_score(self._last_state, features)
        detail.update({
            "sac_direction":  sac_raw,
            "sac_conviction": sac_conv,
            "icm_novelty":    icm_novelty,
        })

        # E. Candlestick patterns
        pat_dir, pat_str = 0, 0.0
        if df is not None and len(df) >= 4:
            pat = self.patterns.net_signal(df.tail(5))
            if pat.direction == "CALL":
                pat_dir, pat_str = 1, float(pat.strength)
            elif pat.direction == "PUT":
                pat_dir, pat_str = -1, float(pat.strength)
        detail["pat_dir"] = pat_dir
        detail["pat_str"] = pat_str

        # F. MTF confluence
        mtf_conf, mtf_dir = self.mtf.confluence()
        mtf_div           = self.mtf.divergence()
        detail.update({
            "mtf_confluence": mtf_conf,
            "mtf_direction":  mtf_dir,
            "mtf_divergence": mtf_div,
            "mtf_data":       self.mtf.tf_summary(),
        })

        # G. News
        news_adj = news_sentiment * 0.08 if abs(news_sentiment) > 0.1 and news_impact > 0.3 else 0.0
        detail.update({"news_sentiment": news_sentiment, "news_impact": news_impact})

        # H. Direction scoring
        call_score = put_score = 0.0
        if ens_dir == 1:           call_score += 3.0 * ens_raw_conf
        else:                      put_score  += 3.0 * ens_raw_conf
        if sac_raw > 0.05:         call_score += 2.0 * sac_conv
        elif sac_raw < -0.05:      put_score  += 2.0 * sac_conv
        if mtf_dir == "CALL":      call_score += 2.0 * mtf_conf
        elif mtf_dir == "PUT":     put_score  += 2.0 * mtf_conf
        if pat_dir == 1:           call_score += 1.5 * pat_str
        elif pat_dir == -1:        put_score  += 1.5 * pat_str
        call_score += 0.5 * max(0,  news_adj)
        put_score  += 0.5 * max(0, -news_adj)
        if mtf_div > 0.5:
            call_score *= (1.0 - mtf_div * 0.4)
            put_score  *= (1.0 - mtf_div * 0.4)

        total      = call_score + put_score + 1e-9
        final_dir  = 1 if call_score >= put_score else 0
        dir_str    = "CALL" if final_dir == 1 else "PUT"
        raw_conf   = max(call_score, put_score) / total

        # I. Calibration
        if self.ensemble.is_ready():
            sac_agree = float((sac_raw > 0) == (final_dir == 1))
            combined  = (0.45 * raw_conf + 0.35 * uq_conf
                         + 0.20 * (sac_conv if sac_agree else sac_conv * 0.3))
        else:
            combined = raw_conf * 0.65

        combined *= regime_mult
        if icm_novelty > 0.75 and combined > 0.55:
            combined = min(1.0, combined * 1.05)
        combined = float(np.clip(combined, 0.0, 1.0))

        detail["final_confidence"] = combined
        detail["direction"]        = dir_str

        self._last_features = features.copy()
        self._last_state    = features.copy()
        self._last_action   = sac_action.copy()

        logger.debug(
            "Signal: %s conf=%.3f [ens=%.2f uq=%.2f sac=%.2f mtf=%s@%.0f%% pat=%+.2f nov=%.2f]",
            dir_str, combined, ens_raw_conf, uq_conf, sac_conv,
            mtf_dir, mtf_conf*100, pat_dir*pat_str, icm_novelty,
        )
        return dir_str, combined, detail

    # -- trade execution -------------------------------------------------------

    async def trade(
        self,
        asset:       Optional[str] = None,
        granularity: Optional[int] = None,
        duration:    Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._connected:
            raise RuntimeError("Call await advisor.connect() first")

        asset       = asset       or cfg.quotex.asset
        granularity = granularity or cfg.data.granularity
        duration    = duration    or cfg.quotex.duration

        balance = await self.client.get_balance()
        payout  = await self.client.get_payout(asset)

        await self.harvester.refresh_recent(asset, granularity, n=5)
        df = self.db.get_candles(asset, granularity, limit=cfg.ml.lookback + 10)
        if len(df) < 20:
            return None

        try:
            features = build_features(df, lookback=cfg.ml.lookback,
                                       mtf_analyzer=self.mtf)
        except Exception:
            return None

        try:
            news_sentiment, news_impact, news_items = await asyncio.wait_for(
                self.news.get_sentiment(asset), timeout=3.0
            )
        except Exception:
            news_sentiment, news_impact, news_items = 0.0, 0.0, []

        direction, confidence, detail = self._full_assess(
            features, df, asset, news_sentiment, news_impact
        )
        if direction == "HOLD":
            return None

        port_ok, port_reason = self.portfolio.can_trade(asset, 0, balance)
        if not port_ok:
            logger.info("Portfolio gate: %s", port_reason)
            return None

        dir_int  = 1 if direction == "CALL" else 0
        decision = self.risk.evaluate(dir_int, confidence, balance, payout)
        if not decision.allowed:
            logger.info("Risk gate: %s", decision.reason)
            return None

        ts_open = int(time.time())
        success, trade_id = await self.client.buy(
            asset, decision.stake, decision.direction, duration
        )
        if not success:
            logger.warning("Order rejected by broker")
            return None

        self.portfolio.open_position(
            asset, decision.direction, decision.stake, float(ts_open)
        )
        trade_db_id = self.db.insert_trade(
            asset, decision.direction, decision.stake,
            decision.stake * (1 + payout), confidence, ts_open,
        )

        await asyncio.sleep(duration + 2)

        won, payout_amount = await self.client.check_win(trade_id)
        result_str = "WIN" if won else "LOSS"
        self.db.close_trade(trade_db_id, result_str, int(time.time()))
        self.portfolio.close_position(asset)

        new_balance = await self.client.get_balance()
        reward = (new_balance - balance) / (balance + 1e-9)
        self.risk.record_trade(decision.stake, payout_amount + decision.stake,
                                new_balance, won)

        self._incremental_update(features, 1 if won else 0,
                                  self._last_action, won, reward)

        if len(df) >= 2:
            last_ret = float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1)
            self.portfolio.update_returns(asset, last_ret)

        self._session_trades.append(TradeRecord(
            ts_open=float(ts_open), ts_close=float(time.time()),
            direction=decision.direction, stake=decision.stake,
            payout=payout_amount, confidence=confidence,
            asset=asset, won=won,
        ))
        self._write_signal_state(direction, confidence, detail, asset,
                                  news_sentiment, news_items)

        logger.info(
            "%s %s Rs%.2f -> %s | payout=%.2f | bal=%.2f | conf=%.0f%%",
            decision.direction.upper(), asset, decision.stake,
            result_str, payout_amount, new_balance, confidence * 100,
        )
        return {
            "asset": asset, "direction": decision.direction,
            "stake": decision.stake, "confidence": confidence,
            "won": won, "payout": payout_amount, "balance": new_balance,
            "trade_id": trade_id,
            "regime": detail.get("regime_name", "UNKNOWN"),
        }

    # -- incremental learning --------------------------------------------------

    def _incremental_update(
        self,
        features: np.ndarray,
        label:    int,
        sac_action: Optional[np.ndarray],
        won:      bool,
        reward:   float,
    ) -> None:
        self.ensemble.update(features, label)
        if self._last_features is not None:
            self.uq.update(features, label,
                           predicted_confidence=self.risk.win_rate_estimate())
        if sac_action is not None and self._last_state is not None:
            intrinsic = self.icm.intrinsic_reward(
                self._last_state, sac_action, features
            )
            self.sac.store(self._last_state, sac_action,
                           reward + intrinsic, features, done=True)
            self.sac.update(n_steps=4)
            self.icm.update(
                np.array([self._last_state]),
                np.array([sac_action]),
                np.array([features]),
            )
        if len(self._session_trades) % 10 == 0 and len(self._session_trades) > 0:
            self._train_regime_online()

    def _train_regime_online(self) -> None:
        try:
            df = self.db.get_candles(cfg.quotex.asset, cfg.data.granularity, limit=300)
            if len(df) >= 50:
                candles = df[["open","high","low","close","volume"]].values
                self.regime.train_online(candles, n_epochs=2)
        except Exception as exc:
            logger.debug("Regime online training: %s", exc)

    def update_model(
        self,
        features: np.ndarray,
        label:    int,
        sac_action: Optional[np.ndarray] = None,
        won:      Optional[bool] = None,
        reward:   Optional[float] = None,
    ) -> None:
        self._incremental_update(
            features, label, sac_action or self._last_action,
            bool(won), float(reward or (1.0 if label else -1.0))
        )

    def record_outcome(
        self,
        features: Optional[np.ndarray] = None,
        won:      bool = False,
    ) -> None:
        f = features if features is not None else self._last_features
        if f is None:
            return
        self._incremental_update(f, 1 if won else 0, self._last_action,
                                  won, 1.0 if won else -1.0)

    # -- data ------------------------------------------------------------------

    async def harvest_history(
        self,
        asset: Optional[str] = None,
        total_candles: int = 5_000,
    ) -> int:
        asset = asset or cfg.quotex.asset
        return await self.harvester.harvest(
            asset, cfg.data.granularity, total_candles, cfg.data.harvest_chunk
        )

    # -- model persistence -----------------------------------------------------

    def save_models(self) -> None:
        d = self._model_dir
        self.ensemble.save(d)
        self.sac.save(d)
        self.regime.save(d)
        self.icm.save(d)
        self.uq.save(d)
        logger.info("All models saved to %s", d)

    def _load_models(self) -> None:
        d = self._model_dir
        self.ensemble.load(d)
        self.sac.load(d)
        self.regime.load(d)
        self.icm.load(d)
        self.uq.load(d)

    def session_performance(self) -> Optional[dict]:
        if not self._session_trades:
            return None
        start_bal = (self.risk.session.start_balance
                     if self.risk.session else 1000.0)
        m = PerformanceCalculator(start_balance=start_bal).compute(
            self._session_trades
        )
        return m.to_dict()

    # -- helpers ---------------------------------------------------------------

    def _get_ensemble_probs(self, features: np.ndarray) -> Optional[list]:
        try:
            if not self.ensemble.is_ready():
                return None
            x = self.ensemble._scaler.transform(features.reshape(1, -1))
            probs = []
            for key, model in [("rf", self.ensemble._rf),
                                ("gb", self.ensemble._gb),
                                ("sgd", self.ensemble._sgd)]:
                if self.ensemble._fitted.get(key):
                    probs.append(model.predict_proba(x)[0])
            return probs if probs else None
        except Exception:
            return None

    def _write_signal_state(
        self,
        direction: str,
        confidence: float,
        detail: dict,
        asset: str,
        news_sentiment: float,
        news_items: list,
    ) -> None:
        try:
            _SIGNAL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "direction":      direction,
                "confidence":     round(confidence, 4),
                "asset":          asset,
                "regime":         detail.get("regime_name", "UNKNOWN"),
                "regime_probs":   detail.get("regime_probs", [1/7]*7),
                "mtf":            detail.get("mtf_data", {}),
                "mtf_confluence": detail.get("mtf_confluence", 0.0),
                "mtf_direction":  detail.get("mtf_direction", "NEUTRAL"),
                "news_sentiment": round(news_sentiment, 4),
                "news_items": [
                    {"title": getattr(it, "title", str(it)),
                     "sentiment": getattr(it, "sentiment", 0.0)}
                    for it in news_items[:10]
                ],
                "ens_conf":    detail.get("ens_conf", 0.0),
                "uq_final":    detail.get("uq_final", 0.0),
                "sac_dir":     detail.get("sac_direction", 0.0),
                "sac_conv":    detail.get("sac_conviction", 0.0),
                "icm_novelty": detail.get("icm_novelty", 0.5),
                "ts":          int(time.time()),
            }
            _SIGNAL_STATE_FILE.write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.debug("Signal state write: %s", exc)

    async def __aenter__(self) -> "AIAdvisor":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()


# -- helpers -------------------------------------------------------------------

def _candles_to_df(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for c in candles:
        if isinstance(c, dict):
            ts = c.get("time") or c.get("ts") or c.get("timestamp") or 0
            rows.append({
                "open":   float(c.get("open", 0)),
                "high":   float(c.get("high") or c.get("max", 0)),
                "low":    float(c.get("low")  or c.get("min", 0)),
                "close":  float(c.get("close") or c.get("value", 0)),
                "volume": float(c.get("volume", 0)),
                "ts":     int(ts),
            })
    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df.set_index("ts", inplace=True)
    df.sort_index(inplace=True)
    return df
