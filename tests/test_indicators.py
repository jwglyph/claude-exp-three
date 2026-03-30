"""Tests for technical indicators."""

import numpy as np
import pytest

from scalper.analysis.indicators import (
    compute_ema,
    compute_sma,
    compute_atr,
    compute_rsi,
    compute_bollinger,
    IndicatorEngine,
)
from scalper.models import Candle


def make_candles(closes: list[float], base_vol: int = 100) -> list[Candle]:
    """Helper to create candles from close prices."""
    candles = []
    for i, c in enumerate(closes):
        o = c - 0.5 if i % 2 == 0 else c + 0.5
        h = max(o, c) + 1.0
        l = min(o, c) - 1.0
        candles.append(Candle(
            timestamp=float(i * 60),
            open=o, high=h, low=l, close=c,
            volume=base_vol + i * 10,
            tick_count=50,
            is_complete=True,
        ))
    return candles


class TestEMA:
    def test_ema_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ema = compute_ema(values, 3)
        assert len(ema) == len(values)
        # EMA should be close to but lag behind the actual values
        assert ema[-1] < values[-1]
        assert ema[-1] > values[-2]

    def test_ema_constant(self):
        values = np.full(20, 100.0)
        ema = compute_ema(values, 10)
        np.testing.assert_allclose(ema[-1], 100.0, atol=0.01)


class TestSMA:
    def test_sma_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = compute_sma(values, 3)
        assert sma[-1] == pytest.approx(4.0)  # (3+4+5)/3
        assert sma[-2] == pytest.approx(3.0)  # (2+3+4)/3


class TestATR:
    def test_atr_basic(self):
        highs = np.array([11.0, 12.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0, 15.0])
        lows = np.array([9.0, 10.0, 11.0, 10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0])
        closes = np.array([10.0, 11.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0])
        atr = compute_atr(highs, lows, closes, 5)
        assert atr[-1] > 0
        assert len(atr) == len(highs)


class TestRSI:
    def test_rsi_uptrend(self):
        # Strong uptrend should have RSI > 50
        closes = np.array([float(i) for i in range(100, 130)])
        rsi = compute_rsi(closes, 14)
        assert rsi[-1] > 70

    def test_rsi_downtrend(self):
        closes = np.array([float(i) for i in range(130, 100, -1)])
        rsi = compute_rsi(closes, 14)
        assert rsi[-1] < 30

    def test_rsi_range(self):
        # All RSI values should be between 0 and 100
        closes = np.random.normal(100, 2, 50)
        rsi = compute_rsi(closes, 14)
        assert np.all((rsi >= 0) & (rsi <= 100))


class TestBollinger:
    def test_bollinger_basic(self):
        closes = np.random.normal(100, 2, 30)
        upper, middle, lower = compute_bollinger(closes, 20, 2.0)
        # Upper should be above middle, middle above lower
        assert upper[-1] > middle[-1] > lower[-1]

    def test_bollinger_contains_price(self):
        # Most prices should be within bands
        np.random.seed(42)
        closes = np.random.normal(100, 1, 50)
        upper, middle, lower = compute_bollinger(closes, 20, 2.0)
        within = np.sum((closes[19:] >= lower[19:]) & (closes[19:] <= upper[19:]))
        total = len(closes) - 19
        assert within / total > 0.8  # ~95% expected, allow some slack


class TestIndicatorEngine:
    def test_compute_basic(self):
        candles = make_candles([100 + i * 0.5 for i in range(60)])
        engine = IndicatorEngine()
        state = engine.compute(candles)

        assert state.ema_fast > 0
        assert state.ema_slow > 0
        assert state.atr > 0
        assert 0 <= state.rsi <= 100
        assert state.bb_upper > state.bb_lower

    def test_trend_detection(self):
        # Uptrend - use floats with enough range
        candles = make_candles([100.0 + i * 2.0 for i in range(60)])
        engine = IndicatorEngine()
        state = engine.compute(candles)
        assert state.trend_direction == 1

        # Downtrend
        candles = make_candles([300.0 - i * 2.0 for i in range(60)])
        engine2 = IndicatorEngine()
        state = engine2.compute(candles)
        assert state.trend_direction == -1
