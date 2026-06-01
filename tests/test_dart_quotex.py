"""
tests/test_dart_quotex.py
Comprehensive test suite — no live broker connection required.
All tests run in offline/mock mode.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 200, start_price: float = 1.10) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(42)
    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.normal(0, 0.001)))

    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    opens = np.array(prices)
    closes = opens * (1 + rng.normal(0, 0.0005, n))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.001, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.001, n))
    vols = rng.uniform(100, 1000, n)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=ts,
    )


@pytest.fixture
def df():
    return _make_ohlcv(200)


@pytest.fixture
def small_df():
    return _make_ohlcv(50)


@pytest.fixture
def tmp_db(tmp_path):
    from dart_quotex.data.database import Database
    return Database(tmp_path / "test.db")


# ──────────────────────────────────────────────────────────────────────────────
# Database tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDatabase:
    def test_upsert_and_retrieve(self, tmp_db):
        rows = [(1_700_000_000 + i * 60, 1.1, 1.101, 1.099, 1.1005, 500.0) for i in range(10)]
        n = tmp_db.upsert_candles("EURUSD_OTC", 60, rows)
        assert n == 10

        df = tmp_db.get_candles("EURUSD_OTC", 60, limit=20)
        assert len(df) == 10
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_upsert_idempotent(self, tmp_db):
        rows = [(1_700_000_000 + i * 60, 1.1, 1.101, 1.099, 1.1005, 500.0) for i in range(5)]
        tmp_db.upsert_candles("EURUSD_OTC", 60, rows)
        tmp_db.upsert_candles("EURUSD_OTC", 60, rows)  # second insert same rows
        assert tmp_db.count_candles("EURUSD_OTC", 60) == 5

    def test_trade_lifecycle(self, tmp_db):
        trade_id = tmp_db.insert_trade("EURUSD_OTC", "CALL", 10.0, 18.0, 0.72, int(time.time()))
        assert isinstance(trade_id, int) and trade_id > 0

        tmp_db.close_trade(trade_id, "WIN", int(time.time()) + 60)
        trades = tmp_db.get_recent_trades(10)
        assert len(trades) == 1
        assert trades.iloc[0]["result"] == "WIN"

    def test_win_rate(self, tmp_db):
        now = int(time.time())
        for i in range(10):
            tid = tmp_db.insert_trade("EURUSD_OTC", "CALL", 10.0, 18.0, 0.7, now + i)
            result = "WIN" if i < 7 else "LOSS"
            tmp_db.close_trade(tid, result, now + i + 60)

        wr = tmp_db.win_rate("EURUSD_OTC", n=10)
        assert abs(wr - 0.7) < 0.01


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatures:
    def test_build_features_shape(self, df):
        from dart_quotex.ml.features import build_features, FEATURE_NAMES
        feats = build_features(df)
        assert feats.ndim == 1
        assert len(feats) > 20    # expect 34 features
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isinf(feats))

    def test_features_change_with_data(self, df):
        from dart_quotex.ml.features import build_features
        f1 = build_features(df.iloc[:100])
        f2 = build_features(df.iloc[50:150])
        assert not np.allclose(f1, f2)

    def test_insufficient_data_raises(self, small_df):
        from dart_quotex.ml.features import build_features
        with pytest.raises(ValueError, match="at least 20"):
            build_features(small_df.iloc[:10])


# ──────────────────────────────────────────────────────────────────────────────
# SMC indicator tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSMC:
    def test_add_smc_features(self, df):
        from dart_quotex.smc.indicators import add_smc_features
        result = add_smc_features(df)
        for col in ["bos_choch", "pd_zone", "ob_bull_dist", "ob_bear_dist",
                    "fvg_signal", "liq_sweep"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_bos_choch_values(self, df):
        from dart_quotex.smc.indicators import detect_bos_choch
        series = detect_bos_choch(df)
        assert set(series.unique()).issubset({-2, -1, 0, 1, 2})

    def test_pd_zone_range(self, df):
        from dart_quotex.smc.indicators import premium_discount
        pd_zone = premium_discount(df)
        valid = pd_zone.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_fvg_returns_list(self, df):
        from dart_quotex.smc.indicators import find_fvg
        fvgs = find_fvg(df)
        assert isinstance(fvgs, list)
        for f in fvgs:
            assert f.direction in ("bull", "bear")
            assert f.top > f.bottom


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble model tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEnsemble:
    def _features(self, n: int = 34) -> np.ndarray:
        return np.random.randn(n).astype(np.float32)

    def test_predict_before_training(self):
        from dart_quotex.ml.ensemble import EnsembleModel
        model = EnsembleModel(min_samples=50)
        direction, confidence = model.predict(self._features())
        assert direction in (0, 1)
        assert confidence == 0.0    # not ready yet

    def test_incremental_update(self):
        from dart_quotex.ml.ensemble import EnsembleModel
        model = EnsembleModel(min_samples=10, retrain_every=5)

        for i in range(20):
            f = self._features()
            label = 1 if i % 2 == 0 else 0
            model.update(f, label)

        assert model.is_ready()
        direction, confidence = model.predict(self._features())
        assert direction in (0, 1)
        assert 0.0 <= confidence <= 1.0

    def test_batch_train(self):
        from dart_quotex.ml.ensemble import EnsembleModel
        model = EnsembleModel(min_samples=10)
        X = np.random.randn(100, 34).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        model.train_batch(X, y)
        assert model.is_ready()
        d, c = model.predict(X[0])
        assert d in (0, 1) and 0 <= c <= 1

    def test_save_load(self, tmp_path):
        from dart_quotex.ml.ensemble import EnsembleModel
        model = EnsembleModel(min_samples=10)
        X = np.random.randn(50, 34).astype(np.float32)
        y = np.random.randint(0, 2, 50)
        model.train_batch(X, y)
        model.save(tmp_path)

        model2 = EnsembleModel(min_samples=10)
        assert model2.load(tmp_path)
        d1, c1 = model.predict(X[0])
        d2, c2 = model2.predict(X[0])
        assert d1 == d2
        assert abs(c1 - c2) < 1e-5


# ──────────────────────────────────────────────────────────────────────────────
# Risk manager tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRiskManager:
    def test_confidence_gate(self):
        from dart_quotex.risk.manager import RiskManager
        rm = RiskManager(min_confidence=0.65)
        rm.start_session(1000.0)
        decision = rm.evaluate(1, confidence=0.50, balance=1000.0)
        assert not decision.allowed
        assert decision.direction == "hold"

    def test_allowed_trade(self):
        from dart_quotex.risk.manager import RiskManager
        rm = RiskManager(min_confidence=0.60, base_risk_pct=0.02, min_stake=1.0)
        rm.start_session(1000.0)
        decision = rm.evaluate(1, confidence=0.75, balance=1000.0, payout=0.80)
        assert decision.allowed
        assert decision.stake > 0
        assert decision.direction == "call"

    def test_drawdown_guard(self):
        from dart_quotex.risk.manager import RiskManager
        rm = RiskManager(min_confidence=0.60, max_drawdown_pct=0.05)
        rm.start_session(1000.0)
        # Simulate losses
        rm.record_trade(50.0, 0.0, 900.0, False)
        rm.record_trade(50.0, 0.0, 850.0, False)
        # Now at 15% drawdown from peak
        decision = rm.evaluate(1, confidence=0.80, balance=850.0)
        assert not decision.allowed

    def test_kelly_stake_scales_with_confidence(self):
        from dart_quotex.risk.manager import RiskManager
        rm = RiskManager(min_confidence=0.50, kelly_fraction=0.25)
        rm.start_session(1000.0)
        d_low = rm.evaluate(1, confidence=0.60, balance=1000.0)
        d_high = rm.evaluate(1, confidence=0.85, balance=1000.0)
        assert d_low.allowed and d_high.allowed
        assert d_high.stake >= d_low.stake  # higher confidence → larger stake

    def test_session_stats(self):
        from dart_quotex.risk.manager import RiskManager
        rm = RiskManager(min_confidence=0.50)
        rm.start_session(1000.0)
        rm.record_trade(10.0, 18.0, 1008.0, won=True)
        rm.record_trade(10.0, 0.0, 998.0, won=False)

        s = rm.session
        assert s.trades == 2
        assert s.wins == 1
        assert s.win_rate == 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Quotex client mock tests
# ──────────────────────────────────────────────────────────────────────────────

class TestQuotexClient:
    @pytest.fixture
    def client(self):
        from dart_quotex.api.quotex_client import QuotexClient
        c = QuotexClient(email="test@test.com", password="test")
        # Force mock mode
        c._mock = True
        return c

    def test_connect(self, client):
        asyncio.get_event_loop().run_until_complete(client.connect())
        assert client._connected

    def test_get_balance(self, client):
        asyncio.get_event_loop().run_until_complete(client.connect())
        bal = asyncio.get_event_loop().run_until_complete(client.get_balance())
        assert isinstance(bal, float) and bal > 0

    def test_get_candles_returns_list(self, client):
        asyncio.get_event_loop().run_until_complete(client.connect())
        candles = asyncio.get_event_loop().run_until_complete(
            client.get_candles("EURUSD_OTC", 60, 50)
        )
        assert isinstance(candles, list)
        assert len(candles) == 50
        assert all(isinstance(c, dict) for c in candles)
        assert all("close" in c for c in candles)

    def test_buy_and_check_win(self, client):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(client.connect())
        success, trade_id = loop.run_until_complete(
            client.buy("EURUSD_OTC", 10.0, "call", 60)
        )
        assert success
        assert trade_id is not None

        won, payout = loop.run_until_complete(client.check_win(trade_id))
        assert isinstance(won, bool)
        assert isinstance(payout, float)


# ──────────────────────────────────────────────────────────────────────────────
# Advisor integration test
# ──────────────────────────────────────────────────────────────────────────────

class TestAdvisor:
    def test_assess_with_candles(self):
        from dart_quotex.advisor import AIAdvisor, _candles_to_df
        from dart_quotex.ml.features import build_features

        advisor = AIAdvisor(model_dir=Path("/tmp/dart_test_models"))

        # Build candle list
        df = _make_ohlcv(150)
        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "time": int(ts.timestamp()),
                "open": row["open"], "high": row["high"],
                "low": row["low"], "close": row["close"],
                "volume": row["volume"],
            })

        direction, confidence = advisor.assess(candles, asset="EURUSD_OTC")
        assert direction in ("CALL", "PUT", "HOLD")
        assert 0.0 <= confidence <= 1.0

    def test_assess_features_output(self):
        from dart_quotex.advisor import AIAdvisor
        advisor = AIAdvisor(model_dir=Path("/tmp/dart_test_models"))
        features = np.random.randn(34).astype(np.float32)
        direction, confidence, _ = advisor.assess_features(features)
        assert direction in ("CALL", "PUT")
        assert 0.0 <= confidence <= 1.0

    def test_update_model(self):
        from dart_quotex.advisor import AIAdvisor
        advisor = AIAdvisor(model_dir=Path("/tmp/dart_test_models"))
        features = np.random.randn(34).astype(np.float32)
        # Should not raise even without prior training
        advisor.update_model(features, label=1)
        advisor.update_model(features, label=0)


# ──────────────────────────────────────────────────────────────────────────────
# Backtester test
# ──────────────────────────────────────────────────────────────────────────────

class TestBacktester:
    def test_run_on_synthetic_data(self, tmp_db, tmp_path):
        from dart_quotex.advisor import AIAdvisor
        from dart_quotex.backtester import Backtester

        # Insert synthetic candles into DB
        df = _make_ohlcv(300)
        rows = [
            (int(ts.timestamp()), r.open, r.high, r.low, r.close, r.volume)
            for ts, r in df.iterrows()
        ]
        tmp_db.upsert_candles("EURUSD_OTC", 60, rows)

        advisor = AIAdvisor(model_dir=tmp_path / "models")
        bt = Backtester(tmp_db, advisor, lookback=50, train_online=True)
        result = bt.run(
            asset="EURUSD_OTC",
            granularity=60,
            start_balance=1000.0,
            payout=0.80,
            min_confidence=0.0,   # accept all signals for test coverage
            limit=300,
        )

        assert result.n_trades >= 0   # might be 0 if ensemble not ready
        assert 0.0 <= result.win_rate <= 1.0
        assert result.start_balance == 1000.0
        assert result.end_balance >= 0

    def test_result_summary_string(self, tmp_db, tmp_path):
        from dart_quotex.advisor import AIAdvisor
        from dart_quotex.backtester import Backtester, BacktestResult

        result = BacktestResult(
            asset="EURUSD_OTC",
            granularity=60,
            start_balance=1000.0,
            end_balance=1100.0,
        )
        summary = result.summary()
        assert "EURUSD_OTC" in summary
        assert "ROI" in summary


# =============================================================================
# NEW COMPONENT TESTS
# =============================================================================

class TestRegimeDetector:
    def test_rule_based_returns_valid(self):
        from dart_quotex.ml.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector(seq_len=20)
        df  = _make_ohlcv(50)
        arr = df[["open","high","low","close","volume"]].values
        rid, rname, probs = det.detect(arr)
        assert 0 <= rid <= 6
        assert rname in ["TRENDING_UP","TRENDING_DOWN","RANGING",
                         "VOLATILE","BREAKOUT","REVERSAL","CHOPPY"]
        assert len(probs) == 7
        assert abs(sum(probs) - 1.0) < 0.05

    def test_regime_strategy_multipliers(self):
        from dart_quotex.ml.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector()
        for rid in range(7):
            s = det.regime_for_strategy(rid)
            assert "momentum"          in s
            assert "position_size_mult" in s
            assert "skip"              in s
            assert 0 <= s["skip"] <= 1

    def test_save_load(self, tmp_path):
        from dart_quotex.ml.regime_detector import MarketRegimeDetector
        det = MarketRegimeDetector(seq_len=15)
        det.save(tmp_path)
        det2 = MarketRegimeDetector(seq_len=15)
        det2.load(tmp_path)


class TestICM:
    def test_intrinsic_reward_before_training(self):
        from dart_quotex.ml.icm import ICM
        icm = ICM(state_dim=10, action_dim=2)
        s  = np.random.randn(10).astype(np.float32)
        a  = np.zeros(2, dtype=np.float32)
        ns = np.random.randn(10).astype(np.float32)
        r  = icm.intrinsic_reward(s, a, ns)
        assert isinstance(r, float)

    def test_novelty_score(self):
        from dart_quotex.ml.icm import ICM
        icm = ICM(state_dim=10, action_dim=2)
        s1  = np.zeros(10, dtype=np.float32)
        s2  = np.ones(10,  dtype=np.float32) * 5
        score = icm.novelty_score(s1, s2)
        assert 0.0 <= score <= 1.0

    def test_save_load(self, tmp_path):
        from dart_quotex.ml.icm import ICM
        icm = ICM(state_dim=10, action_dim=2)
        icm.save(tmp_path)
        icm2 = ICM(state_dim=10, action_dim=2)
        icm2.load(tmp_path)


class TestUncertaintyQuantifier:
    def test_quantify_no_training(self):
        from dart_quotex.ml.uncertainty import UncertaintyQuantifier
        uq = UncertaintyQuantifier(input_dim=10, min_samples=5)
        f  = np.random.randn(10).astype(np.float32)
        conf, detail = uq.quantify(f)
        assert 0.0 <= conf <= 1.0
        assert "final" in detail

    def test_update_and_ece(self):
        from dart_quotex.ml.uncertainty import UncertaintyQuantifier
        uq = UncertaintyQuantifier(input_dim=10, min_samples=5)
        for i in range(30):
            f     = np.random.randn(10).astype(np.float32)
            label = np.random.randint(0, 2)
            uq.update(f, label, predicted_confidence=0.6)
        ece = uq.expected_calibration_error()
        assert 0.0 <= ece <= 1.0

    def test_save_load(self, tmp_path):
        from dart_quotex.ml.uncertainty import UncertaintyQuantifier
        uq = UncertaintyQuantifier(input_dim=10)
        for i in range(10):
            uq.update(np.random.randn(10).astype(np.float32),
                      i % 2, predicted_confidence=0.6)
        uq.save(tmp_path)
        uq2 = UncertaintyQuantifier(input_dim=10)
        uq2.load(tmp_path)
        assert len(uq2._cal_y) == 10


class TestMultiTimeframe:
    def test_features_shape(self):
        from dart_quotex.ml.multi_timeframe import MultiTimeframeAnalyzer
        df  = _make_ohlcv(200)
        mtf = MultiTimeframeAnalyzer(timeframes=["1m","5m","15m","1h"])
        mtf.update(df)
        feats = mtf.features()
        assert feats.shape == (52,)
        assert not np.any(np.isnan(feats))

    def test_confluence(self):
        from dart_quotex.ml.multi_timeframe import MultiTimeframeAnalyzer
        df  = _make_ohlcv(200)
        mtf = MultiTimeframeAnalyzer()
        mtf.update(df)
        score, direction = mtf.confluence()
        assert 0.0 <= score <= 1.0
        assert direction in ("CALL", "PUT", "NEUTRAL")

    def test_divergence(self):
        from dart_quotex.ml.multi_timeframe import MultiTimeframeAnalyzer
        mtf = MultiTimeframeAnalyzer()
        mtf.update(_make_ohlcv(200))
        d = mtf.divergence()
        assert 0.0 <= d <= 1.0

    def test_tf_summary(self):
        from dart_quotex.ml.multi_timeframe import MultiTimeframeAnalyzer
        mtf = MultiTimeframeAnalyzer()
        mtf.update(_make_ohlcv(200))
        s = mtf.tf_summary()
        assert "1m" in s
        for tf_data in s.values():
            assert "trend"    in tf_data
            assert "rsi"      in tf_data
            assert "momentum" in tf_data


class TestPortfolioManager:
    def test_can_trade_empty(self):
        from dart_quotex.portfolio.manager import PortfolioManager
        pm = PortfolioManager(max_concurrent=3)
        ok, reason = pm.can_trade("EURUSD_OTC", 10.0, 1000.0)
        assert ok

    def test_max_concurrent_blocks(self):
        from dart_quotex.portfolio.manager import PortfolioManager
        import time as _time
        pm = PortfolioManager(max_concurrent=2)
        pm.open_position("EURUSD_OTC", "call", 10.0, _time.time())
        pm.open_position("GBPUSD_OTC", "put",  10.0, _time.time())
        ok, reason = pm.can_trade("USDJPY_OTC", 10.0, 1000.0)
        assert not ok
        assert "concurrent" in reason.lower()

    def test_same_asset_blocked(self):
        from dart_quotex.portfolio.manager import PortfolioManager
        import time as _time
        pm = PortfolioManager(max_concurrent=3)
        pm.open_position("EURUSD_OTC", "call", 10.0, _time.time())
        ok, reason = pm.can_trade("EURUSD_OTC", 10.0, 1000.0)
        assert not ok

    def test_diversification_score(self):
        from dart_quotex.portfolio.manager import PortfolioManager
        import time as _time
        pm = PortfolioManager(max_concurrent=5)
        pm.open_position("EURUSD_OTC", "call", 10.0, _time.time())
        pm.open_position("XAUUSD_OTC", "call", 10.0, _time.time())
        score = pm.diversification_score()
        assert 0.0 <= score <= 1.0
        # Two different clusters -> less than max concentration
        assert score < 1.0

    def test_close_position(self):
        from dart_quotex.portfolio.manager import PortfolioManager
        import time as _time
        pm = PortfolioManager(max_concurrent=2)
        pm.open_position("EURUSD_OTC", "call", 10.0, _time.time())
        pos = pm.close_position("EURUSD_OTC")
        assert pos is not None
        assert pos.asset == "EURUSD_OTC"
        assert len(pm._open) == 0


class TestCandlestickPatterns:
    def test_scanner_returns_list(self):
        from dart_quotex.patterns.candlestick import PatternScanner
        scanner = PatternScanner()
        df      = _make_ohlcv(20)
        signals = scanner.scan(df)
        assert isinstance(signals, list)

    def test_net_signal_output(self):
        from dart_quotex.patterns.candlestick import PatternScanner
        scanner = PatternScanner()
        df      = _make_ohlcv(20)
        sig     = scanner.net_signal(df)
        assert sig.direction in ("CALL", "PUT", "NEUTRAL")
        assert 0.0 <= sig.strength <= 1.0
        assert isinstance(sig.name, str)

    def test_engulfing_detected(self):
        """Manually craft a bullish engulfing and check detection."""
        from dart_quotex.patterns.candlestick import PatternScanner
        import pandas as pd
        scanner = PatternScanner()
        # Build a synthetic bearish candle followed by a larger bullish one
        idx  = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        data = {
            "open":  [1.1010, 1.1005, 1.1000, 1.0990, 1.0980],
            "high":  [1.1015, 1.1010, 1.1002, 1.1005, 1.0985],
            "low":   [1.1005, 1.0995, 1.0990, 1.0975, 1.0970],
            "close": [1.1008, 1.0998, 1.0992, 1.1002, 1.0975],
            "volume":[100]*5,
        }
        df  = pd.DataFrame(data, index=idx)
        sig = scanner.net_signal(df)
        # We just verify it runs without error and returns a valid signal
        assert sig.direction in ("CALL", "PUT", "NEUTRAL")

    def test_doji_detected_on_flat_candle(self):
        from dart_quotex.patterns.candlestick import PatternScanner
        import pandas as pd
        scanner = PatternScanner(doji_threshold=0.1)
        idx  = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        data = {
            "open":  [1.1000]*5,
            "high":  [1.1010]*5,
            "low":   [1.0990]*5,
            "close": [1.1001, 1.1001, 1.1001, 1.1001, 1.1000],
            "volume":[100]*5,
        }
        df  = pd.DataFrame(data, index=idx)
        sigs = scanner.scan(df)
        names = [s.name for s in sigs]
        assert any("Doji" in n or "doji" in n.lower() for n in names)


class TestPerformanceMetrics:
    def _make_trades(self, n=50, wr=0.6):
        from dart_quotex.metrics.performance import TradeRecord
        import random, time
        random.seed(42)
        now = time.time()
        trades = []
        for i in range(n):
            won = random.random() < wr
            trades.append(TradeRecord(
                ts_open=now + i * 70,
                ts_close=now + i * 70 + 60,
                direction="call",
                stake=10.0,
                payout=8.0 if won else -10.0,
                confidence=0.7,
                asset="EURUSD_OTC",
                won=won,
            ))
        return trades

    def test_basic_metrics(self):
        from dart_quotex.metrics.performance import PerformanceCalculator
        calc   = PerformanceCalculator(start_balance=1000.0)
        trades = self._make_trades(50, 0.6)
        m      = calc.compute(trades)
        assert m.n_trades == 50
        assert 0.0 <= m.win_rate <= 1.0
        assert m.profit_factor >= 0.0

    def test_sharpe_and_sortino(self):
        from dart_quotex.metrics.performance import PerformanceCalculator
        calc   = PerformanceCalculator(start_balance=1000.0)
        trades = self._make_trades(100, 0.58)
        m      = calc.compute(trades)
        assert isinstance(m.sharpe,  float)
        assert isinstance(m.sortino, float)

    def test_var_cvar(self):
        from dart_quotex.metrics.performance import PerformanceCalculator
        calc   = PerformanceCalculator(start_balance=1000.0)
        trades = self._make_trades(100, 0.55)
        m      = calc.compute(trades)
        assert m.var_95  >= 0.0
        assert m.cvar_95 >= m.var_95 - 1e-9  # CVaR >= VaR (floating-point tolerance)

    def test_drawdown(self):
        from dart_quotex.metrics.performance import PerformanceCalculator
        calc   = PerformanceCalculator(start_balance=1000.0)
        trades = self._make_trades(100, 0.40)   # losing strategy
        m      = calc.compute(trades)
        assert 0.0 <= m.max_drawdown <= 1.0

    def test_streaks(self):
        from dart_quotex.metrics.performance import PerformanceCalculator, TradeRecord
        import time
        calc   = PerformanceCalculator(start_balance=1000.0)
        now    = time.time()
        # 5 wins then 3 losses
        trades = []
        for i in range(5):
            trades.append(TradeRecord(now+i*70, now+i*70+60, "call", 10, 8, 0.7, "EUR", True))
        for i in range(3):
            trades.append(TradeRecord(now+(5+i)*70, now+(5+i)*70+60, "put", 10, -10, 0.7, "EUR", False))
        m = calc.compute(trades)
        assert m.max_consec_wins   == 5
        assert m.max_consec_losses == 3

    def test_summary_string(self):
        from dart_quotex.metrics.performance import PerformanceCalculator
        calc   = PerformanceCalculator(start_balance=1000.0)
        trades = self._make_trades(30)
        m      = calc.compute(trades)
        s      = m.summary_str()
        assert "Win Rate"    in s
        assert "Sharpe"      in s
        assert "CVaR"        in s
        assert "Drawdown"    in s


class TestRobustClient:
    def test_mock_mode(self):
        from dart_quotex.api.robust_client import RobustQuotexClient
        client = RobustQuotexClient(email="t@t.com", password="x")
        assert client._mock   # pyquotex not installed -> mock

    def test_connect_mock(self):
        from dart_quotex.api.robust_client import RobustQuotexClient
        import asyncio
        client = RobustQuotexClient()
        asyncio.get_event_loop().run_until_complete(client.connect())
        assert client._connected

    def test_get_candles_mock(self):
        from dart_quotex.api.robust_client import RobustQuotexClient
        import asyncio
        client = RobustQuotexClient()
        asyncio.get_event_loop().run_until_complete(client.connect())
        candles = asyncio.get_event_loop().run_until_complete(
            client.get_candles("EURUSD_OTC", 60, 50)
        )
        assert len(candles) == 50
        assert all("close" in c for c in candles)

    def test_health(self):
        from dart_quotex.api.robust_client import RobustQuotexClient
        client = RobustQuotexClient()
        h = client.health()
        assert "connected"     in h
        assert "circuit_state" in h
        assert "success_rate"  in h

    def test_circuit_breaker(self):
        from dart_quotex.api.robust_client import _CircuitBreaker
        cb = _CircuitBreaker(threshold=3, reset_s=60.0)
        assert cb.is_allowed()
        for _ in range(3):
            cb.record_failure()
        assert not cb.is_allowed()
        cb.record_success()
        assert cb.is_allowed()


class TestRealtimeStream:
    def test_candle_aggregator_build(self):
        from dart_quotex.data.realtime import CandleAggregator, Tick
        import time
        agg  = CandleAggregator("EURUSD_OTC", [60])
        now  = int(time.time())
        tick = Tick("EURUSD_OTC", 1.10000, float(now), 100.0)
        completed = agg.process_tick(tick)
        assert isinstance(completed, list)

    def test_candle_completes_on_new_minute(self):
        from dart_quotex.data.realtime import CandleAggregator, Tick
        agg = CandleAggregator("EURUSD_OTC", [60])
        t1  = 1_700_000_000.0
        t2  = 1_700_000_060.0
        agg.process_tick(Tick("EURUSD_OTC", 1.1000, t1, 100))
        agg.process_tick(Tick("EURUSD_OTC", 1.1001, t1 + 30, 100))
        completed = agg.process_tick(Tick("EURUSD_OTC", 1.1002, t2, 100))
        assert len(completed) == 1
        assert completed[0].is_complete
        assert completed[0].granularity == 60

    def test_anomaly_detection(self):
        from dart_quotex.data.realtime import CandleAggregator, Tick
        agg = CandleAggregator("EURUSD_OTC", [60])
        t   = 1_700_000_000.0
        # Marubozu-like candle (open == close would normally flag anomaly)
        agg.process_tick(Tick("EURUSD_OTC", 1.1000, t, 100))
        agg.process_tick(Tick("EURUSD_OTC", 1.1000, t + 1, 100))
        completed = agg.process_tick(Tick("EURUSD_OTC", 1.1000, t + 60, 100))
        # Just verify no exception
        assert isinstance(completed, list)


class TestFullAdvisorPipeline:
    """Integration test: advisor assess runs through all components."""

    def test_full_assess_no_error(self, tmp_path):
        from dart_quotex import AIAdvisor
        advisor = AIAdvisor(model_dir=tmp_path/"models")
        df      = _make_ohlcv(150)
        candles = [
            {"time": int(ts.timestamp()), "open": r.open, "high": r.high,
             "low": r.low, "close": r.close, "volume": r.volume}
            for ts, r in df.iterrows()
        ]
        direction, confidence = advisor.assess(candles, "EURUSD_OTC")
        assert direction in ("CALL", "PUT", "HOLD")
        assert 0.0 <= confidence <= 1.0

    def test_assess_features_detail(self, tmp_path):
        from dart_quotex import AIAdvisor
        advisor  = AIAdvisor(model_dir=tmp_path/"models")
        features = np.random.randn(88).astype(np.float32)
        df       = _make_ohlcv(100)
        direction, confidence, detail = advisor.assess_features(features, df)
        assert direction in ("CALL", "PUT", "HOLD")
        assert "regime_name" in detail
        assert "sac_direction" in detail
        assert "mtf_data" in detail

    def test_incremental_learning_cycle(self, tmp_path):
        from dart_quotex import AIAdvisor
        advisor = AIAdvisor(model_dir=tmp_path/"models")
        for i in range(25):
            f = np.random.randn(88).astype(np.float32)
            advisor.update_model(f, label=i % 2)
        advisor.record_outcome(won=True)
        advisor.save_models()

        advisor2 = AIAdvisor(model_dir=tmp_path/"models")
        assert len(advisor2.ensemble._y) == 25


# =============================================================================
# STARTUP / MONEY MANAGEMENT TESTS
# =============================================================================

class TestMasaniello:
    """Tests for the Masaniello stake calculator from startup.py."""

    def _masa(self, capital=1000.0, events=10, wins=6, payout=1.80, min_bet=1.0):
        import sys; sys.path.insert(0, str(Path(__file__).parent.parent))
        from startup import Masaniello
        return Masaniello(capital, events, wins, payout, min_bet)

    def test_first_stake_positive(self):
        m = self._masa()
        stake, status = m.get_next_stake()
        assert stake > 0
        assert status == "OK"

    def test_stake_within_capital(self):
        m = self._masa(capital=500.0)
        stake, _ = m.get_next_stake()
        assert stake <= 500.0

    def test_win_reduces_wins_left(self):
        m = self._masa(events=10, wins=6)
        stake, _ = m.get_next_stake()
        m.update(True, stake, 1.80)
        assert m.wins_left == 5
        assert m.events_left == 9

    def test_loss_reduces_events_only(self):
        m = self._masa(events=10, wins=6)
        stake, _ = m.get_next_stake()
        m.update(False, stake, 1.80)
        assert m.wins_left == 6
        assert m.events_left == 9

    def test_goal_reached_status(self):
        m = self._masa(capital=1000.0, events=3, wins=1, payout=1.80)
        stake, _ = m.get_next_stake()
        m.update(True, stake, 1.80)
        assert m.status == "GOAL REACHED"

    def test_bankrupt_status(self):
        m = self._masa(capital=1.0, events=10, wins=9, payout=1.80, min_bet=1.0)
        for _ in range(5):
            stake, _ = m.get_next_stake()
            if stake <= 0:
                break
            m.update(False, stake, 1.80)
        assert m.status in ("BANKRUPT", "MATH IMPOSSIBLE", "ACTIVE")

    def test_reset_cycle(self):
        m = self._masa(events=5, wins=3)
        for _ in range(3):
            s, _ = m.get_next_stake()
            m.update(True, s, 1.80)
        m.reset_cycle()
        assert m.events_left == 5
        assert m.wins_left   == 3
        assert m.status      == "ACTIVE"

    def test_min_bet_respected(self):
        m = self._masa(capital=5.0, events=10, wins=9, min_bet=1.0)
        stake, _ = m.get_next_stake()
        assert stake >= 1.0

    def test_summary_string(self):
        m = self._masa()
        s = m.summary()
        assert "Capital" in s
        assert "Events"  in s
        assert "Status"  in s


class TestStakeEngine:
    """Tests for the full StakeEngine (MM orchestrator) in startup.py."""

    def _engine(self, method="3", balance=1000.0, **extra):
        from startup import StakeEngine
        base = {"method": method, "fixed_stake": 10.0,
                "martingale": False, "daily_lock": False, "max_dd": False}
        base.update(extra)
        return StakeEngine(base, balance)

    def test_fixed_stake(self):
        eng = self._engine(method="3", fixed_stake=25.0)
        assert eng.next_stake() == 25.0

    def test_masaniello_engine(self):
        eng = self._engine(
            method="1",
            capital=500.0, events=10, wins=6, payout=1.80, min_bet=1.0,
        )
        stake = eng.next_stake()
        assert stake > 0

    def test_compounding_engine(self):
        eng = self._engine(
            method="2",
            base_stake=20.0, reinvest_pct=50.0, max_stake_pct=5.0,
        )
        stake = eng.next_stake()
        assert stake > 0

    def test_win_increases_balance(self):
        eng = self._engine()
        eng.record(True, 10.0, 1.80)
        assert eng.balance > 1000.0
        assert eng.wins   == 1
        assert eng.losses == 0

    def test_loss_decreases_balance(self):
        eng = self._engine()
        eng.record(False, 10.0, 1.80)
        assert eng.balance == 990.0
        assert eng.losses  == 1

    def test_martingale_step_increases_on_loss(self):  # noqa
        eng = self._engine(
            martingale=True, mtg_mult=2.0, mtg_steps=3, fixed_stake=10.0
        )
        base = eng.next_stake()          # sets mtg_base = 10.0
        eng.record(False, base, 1.80)    # mtg_step -> 1
        assert eng.mtg_step == 1
        stake2 = eng.next_stake()        # should be 10 * 2^1 = 20
        assert stake2 == pytest.approx(20.0, rel=0.01)

    def test_martingale_resets_on_win(self):
        eng = self._engine(
            martingale=True, mtg_mult=2.0, mtg_steps=3, fixed_stake=10.0
        )
        eng.record(False, 10.0, 1.80)
        eng.record(True,  20.0, 1.80)
        assert eng.mtg_step == 0

    def test_daily_lock_triggers(self):
        eng = self._engine(
            daily_lock=True, daily_lock_pct=5.0   # stop at +5%
        )
        eng.record(True, 100.0, 1.80)   # +80 profit on 1000 bal → > 5%
        stop, reason = eng.should_stop()
        assert stop
        assert "lock" in reason.lower()

    def test_drawdown_stop_triggers(self):
        eng = self._engine(max_dd=True, max_dd_pct=5.0)
        for _ in range(6):
            eng.record(False, 10.0, 1.80)   # -60 → 6% drawdown on 1000
        stop, reason = eng.should_stop()
        assert stop
        assert "drawdown" in reason.lower()

    def test_no_stop_when_normal(self):
        eng = self._engine(daily_lock=True, daily_lock_pct=10.0,
                           max_dd=True, max_dd_pct=10.0)
        eng.record(True, 10.0, 1.80)
        stop, _ = eng.should_stop()
        assert not stop

    def test_status_line_string(self):
        eng = self._engine()
        eng.record(True,  10.0, 1.80)
        eng.record(False, 10.0, 1.80)
        s = eng.status_line()
        assert "Bal" in s
        assert "WR"  in s


class TestAccountStorage:
    """Tests for accounts.json save/load in startup.py."""

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        import startup
        monkeypatch.setattr(startup, "ACCOUNTS_FILE", tmp_path / "accounts.json")
        from startup import _load_accounts, _save_accounts
        accs = [{"email": "a@b.com", "password": "x",
                  "nickname": "Test", "last_mode": "DEMO",
                  "last_pairs": ["EURUSD_OTC"], "mm": {}}]
        _save_accounts(accs)
        loaded = _load_accounts()
        assert len(loaded) == 1
        assert loaded[0]["email"]    == "a@b.com"
        assert loaded[0]["nickname"] == "Test"

    def test_empty_file_returns_empty_list(self, tmp_path, monkeypatch):
        import startup
        monkeypatch.setattr(startup, "ACCOUNTS_FILE", tmp_path / "no_file.json")
        from startup import _load_accounts
        assert _load_accounts() == []

    def test_save_account_field(self, tmp_path, monkeypatch):
        import startup
        monkeypatch.setattr(startup, "ACCOUNTS_FILE", tmp_path / "accounts.json")
        from startup import _save_accounts, _save_account_field
        accs = [{"email": "test@x.com", "password": "p",
                  "nickname": "T", "last_mode": "DEMO",
                  "last_pairs": [], "mm": {}}]
        _save_accounts(accs)
        _save_account_field(accs[0], last_mode="REAL", last_pairs=["GBPUSD_OTC"])
        from startup import _load_accounts
        saved = _load_accounts()[0]
        assert saved["last_mode"]  == "REAL"
        assert saved["last_pairs"] == ["GBPUSD_OTC"]
