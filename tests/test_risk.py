"""Tests for risk management."""

import pytest

from scalper.config import ScalperConfig
from scalper.models import (
    MarketRegime, Position, Side, Signal, SignalType, TradeResult,
)
from scalper.analysis.indicators import IndicatorState
from scalper.analysis.regime import RegimeState
from scalper.risk.manager import RiskManager


@pytest.fixture
def config():
    return ScalperConfig(
        max_risk_per_trade=200.0,
        daily_loss_limit=800.0,
        max_drawdown=1800.0,
        max_contracts=5,
    )


@pytest.fixture
def risk_mgr(config):
    return RiskManager(config)


@pytest.fixture
def indicators():
    return IndicatorState(
        ema_fast=20010.0,
        ema_slow=20005.0,
        ema_trend=20000.0,
        atr=5.0,
        atr_fast=4.0,
        rsi=55.0,
    )


@pytest.fixture
def regime():
    return RegimeState(
        regime=MarketRegime.TRENDING_UP,
        confidence=0.7,
        duration=5,
        volatility_percentile=0.5,
        trend_strength=0.7,
        mean_reversion_score=0.2,
    )


class TestCanTrade:
    def test_can_trade_initially(self, risk_mgr):
        can, reason = risk_mgr.can_trade()
        assert can is True

    def test_auto_unlocks_for_paper_testing(self, risk_mgr):
        """In paper mode, locks auto-clear so testing can continue."""
        risk_mgr.state.is_locked = True
        risk_mgr.state.lock_reason = "test_lock"
        can, reason = risk_mgr.can_trade()
        assert can is True  # auto-unlocked

    def test_locks_on_critical_drawdown(self, risk_mgr):
        """Only hard lock: drawdown too tight for any trade."""
        risk_mgr.state.trailing_drawdown_remaining = 10.0
        can, reason = risk_mgr.can_trade()
        assert can is False


class TestPositionSizing:
    def test_basic_sizing(self, risk_mgr, indicators, regime):
        signal = Signal(
            timestamp=0, signal_type=SignalType.LONG, confidence=0.7,
            side=Side.LONG, entry_price=20000, stop_price=19995,
            target_price=20010, regime=MarketRegime.TRENDING_UP,
        )
        size = risk_mgr.compute_position_size(signal, indicators, regime)
        assert 1 <= size <= 5

    def test_low_confidence_smaller(self, risk_mgr, indicators, regime):
        high_conf = Signal(
            timestamp=0, signal_type=SignalType.STRONG_LONG, confidence=0.85,
            side=Side.LONG, entry_price=20000, stop_price=19995,
            target_price=20015, regime=MarketRegime.TRENDING_UP,
        )
        low_conf = Signal(
            timestamp=0, signal_type=SignalType.WEAK_LONG, confidence=0.56,
            side=Side.LONG, entry_price=20000, stop_price=19995,
            target_price=20010, regime=MarketRegime.TRENDING_UP,
        )
        size_high = risk_mgr.compute_position_size(high_conf, indicators, regime)
        size_low = risk_mgr.compute_position_size(low_conf, indicators, regime)
        assert size_high >= size_low


class TestStopComputation:
    def test_long_stop_below_entry(self, risk_mgr, indicators, regime):
        stop = risk_mgr.compute_stop(Side.LONG, 20000.0, indicators, regime)
        assert stop < 20000.0

    def test_short_stop_above_entry(self, risk_mgr, indicators, regime):
        stop = risk_mgr.compute_stop(Side.SHORT, 20000.0, indicators, regime)
        assert stop > 20000.0

    def test_volatile_regime_wider_stop(self, risk_mgr, indicators):
        calm = RegimeState(
            regime=MarketRegime.LOW_VOLATILITY, confidence=0.7, duration=5,
            volatility_percentile=0.2, trend_strength=0.2, mean_reversion_score=0.5,
        )
        volatile = RegimeState(
            regime=MarketRegime.VOLATILE, confidence=0.7, duration=5,
            volatility_percentile=0.8, trend_strength=0.2, mean_reversion_score=0.3,
        )
        stop_calm = risk_mgr.compute_stop(Side.LONG, 20000.0, indicators, calm)
        stop_vol = risk_mgr.compute_stop(Side.LONG, 20000.0, indicators, volatile)
        # Volatile stop should be further from entry
        assert (20000.0 - stop_vol) > (20000.0 - stop_calm)


class TestRiskMultiplier:
    def test_reduces_after_losses(self, risk_mgr):
        initial = risk_mgr.state.risk_multiplier
        for _ in range(3):
            risk_mgr.record_trade(TradeResult(
                entry_time=0, exit_time=1, side=Side.LONG,
                entry_price=20000, exit_price=19998,
                quantity=1, pnl=-40, max_favorable=0, max_adverse=-40,
                signal_confidence=0.6, regime=MarketRegime.RANGING,
                exit_reason="stop",
            ))
        assert risk_mgr.state.risk_multiplier < initial

    def test_reset_daily(self, risk_mgr):
        risk_mgr.state.daily_pnl = -500
        risk_mgr.state.is_locked = True
        risk_mgr.reset_daily()
        assert risk_mgr.state.daily_pnl == 0
        assert risk_mgr.state.is_locked is False
        assert risk_mgr.state.risk_multiplier == 1.0
