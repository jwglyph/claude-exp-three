"""Tests for regime detection."""

import pytest
import numpy as np

from scalper.analysis.indicators import IndicatorEngine, IndicatorState
from scalper.analysis.regime import RegimeDetector, RegimeState
from scalper.models import Candle, MarketRegime


def make_trending_candles(direction: str = "up", count: int = 50) -> list[Candle]:
    """Create clearly trending candles."""
    candles = []
    price = 20000.0
    for i in range(count):
        if direction == "up":
            move = np.random.uniform(1.0, 5.0)
        else:
            move = np.random.uniform(-5.0, -1.0)
        price += move
        o = price - move * 0.2
        h = max(o, price) + np.random.uniform(0, 2)
        l = min(o, price) - np.random.uniform(0, 2)
        candles.append(Candle(
            timestamp=float(i * 60), open=o, high=h, low=l, close=price,
            volume=100 + int(np.random.uniform(0, 100)),
            tick_count=50, is_complete=True,
        ))
    return candles


def make_ranging_candles(count: int = 50) -> list[Candle]:
    """Create ranging/mean-reverting candles."""
    candles = []
    price = 20000.0
    for i in range(count):
        move = np.random.normal(0, 1.5)
        # Mean revert toward 20000
        move -= (price - 20000) * 0.1
        price += move
        o = price - move * 0.3
        h = max(o, price) + np.random.uniform(0, 1)
        l = min(o, price) - np.random.uniform(0, 1)
        candles.append(Candle(
            timestamp=float(i * 60), open=o, high=h, low=l, close=price,
            volume=100, tick_count=50, is_complete=True,
        ))
    return candles


class TestRegimeDetector:
    def test_detects_uptrend(self):
        candles = make_trending_candles("up", 50)
        engine = IndicatorEngine()
        ind = engine.compute(candles)
        detector = RegimeDetector(lookback=20)
        regime = detector.detect(candles, ind)

        assert regime.regime in (MarketRegime.TRENDING_UP, MarketRegime.VOLATILE)
        assert regime.trend_strength > 0.3

    def test_detects_downtrend(self):
        candles = make_trending_candles("down", 50)
        engine = IndicatorEngine()
        ind = engine.compute(candles)
        detector = RegimeDetector(lookback=20)
        regime = detector.detect(candles, ind)

        assert regime.regime in (MarketRegime.TRENDING_DOWN, MarketRegime.VOLATILE)

    def test_detects_ranging(self):
        np.random.seed(42)
        candles = make_ranging_candles(50)
        engine = IndicatorEngine()
        ind = engine.compute(candles)
        detector = RegimeDetector(lookback=20)
        regime = detector.detect(candles, ind)

        # Ranging should have high mean reversion score
        assert regime.mean_reversion_score > 0.2

    def test_regime_has_confidence(self):
        candles = make_trending_candles("up", 50)
        engine = IndicatorEngine()
        ind = engine.compute(candles)
        detector = RegimeDetector()
        regime = detector.detect(candles, ind)

        assert 0 <= regime.confidence <= 1.0
