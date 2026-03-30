"""Multi-timeframe analysis engine.

Maintains candles across multiple timeframes (1m, 5m, 15m, 1h)
and computes indicators on each. Provides HTF confluence scoring
that the signal generator uses to filter and weight 1m signals.

HTF confluence logic:
- If 5m+15m trend agrees with 1m signal → high confluence, full size
- If 5m agrees but 15m neutral → moderate, normal size
- If HTF disagrees with 1m signal → low confluence, skip or reduce
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from scalper.models import Candle, MarketRegime, Side
from scalper.analysis.indicators import IndicatorEngine, IndicatorState, compute_ema
from scalper.analysis.regime import RegimeDetector, RegimeState
from scalper.feeds.candle_aggregator import CandleAggregator


@dataclass
class TimeframeState:
    """State of a single timeframe."""
    interval: int  # seconds
    label: str
    candles: list  # managed by aggregator
    indicators: Optional[IndicatorState] = None
    regime: Optional[RegimeState] = None
    trend: int = 0  # +1 up, -1 down, 0 neutral
    bias: float = 0.0  # -1 to +1 directional bias


@dataclass
class HTFConfluence:
    """Result of multi-timeframe confluence analysis."""
    score: float  # -1 (all bearish) to +1 (all bullish)
    agrees_with: Optional[Side]  # which side HTF supports
    strength: float  # 0-1, how strong is the agreement
    tf_5m_trend: int  # +1/-1/0
    tf_15m_trend: int
    tf_1h_trend: int
    description: str  # human-readable


class MultiTimeframeEngine:
    """Manages multiple timeframe candles and indicators."""

    def __init__(self):
        # Create aggregators for each timeframe
        self._aggregators = {
            "5m": CandleAggregator(interval_sec=300, max_candles=100),
            "15m": CandleAggregator(interval_sec=900, max_candles=50),
            "1h": CandleAggregator(interval_sec=3600, max_candles=30),
        }

        self._indicators = {
            "5m": IndicatorEngine(ema_fast=9, ema_slow=21, ema_trend=50),
            "15m": IndicatorEngine(ema_fast=9, ema_slow=21, ema_trend=50),
            "1h": IndicatorEngine(ema_fast=9, ema_slow=21, ema_trend=50),
        }

        self._regimes = {
            "5m": RegimeDetector(lookback=15),
            "15m": RegimeDetector(lookback=10),
            "1h": RegimeDetector(lookback=8),
        }

        self._states: dict[str, TimeframeState] = {}
        for label, agg in self._aggregators.items():
            self._states[label] = TimeframeState(
                interval=agg.interval_sec,
                label=label,
                candles=[],
            )

    def process_tick(self, timestamp: float, price: float, size: int = 1, side: str = "") -> None:
        """Feed a tick to all HTF aggregators."""
        from scalper.models import Tick
        tick = Tick(timestamp=timestamp, price=price, size=size, side=side)

        for label, agg in self._aggregators.items():
            completed = agg.process_tick(tick)
            if completed:
                self._update_state(label)

    def process_1m_candle(self, candle: Candle) -> None:
        """Feed a completed 1m candle to build HTF candles.

        This is more efficient than tick-by-tick for historical preload.
        Synthesizes ticks from 1m OHLC to update HTF aggregators.
        """
        from scalper.models import Tick
        # Feed 4 synthetic ticks per 1m candle (OHLC)
        for price in [candle.open, candle.high, candle.low, candle.close]:
            tick = Tick(
                timestamp=candle.timestamp,
                price=price,
                size=max(1, candle.volume // 4),
            )
            for agg in self._aggregators.values():
                agg.process_tick(tick)

        # Update states if any HTF candle completed
        for label in self._aggregators:
            candles = self._aggregators[label].get_candles()
            if candles:
                self._update_state(label)

    def _update_state(self, label: str) -> None:
        """Recompute indicators and regime for a timeframe."""
        agg = self._aggregators[label]
        candles = agg.get_candles()

        if len(candles) < 5:
            return

        state = self._states[label]
        state.candles = candles

        # Compute indicators
        state.indicators = self._indicators[label].compute(candles)

        # Detect regime
        state.regime = self._regimes[label].detect(candles, state.indicators)

        # Determine trend
        ind = state.indicators
        if ind.ema_fast > ind.ema_slow and ind.rsi > 50:
            state.trend = 1
        elif ind.ema_fast < ind.ema_slow and ind.rsi < 50:
            state.trend = -1
        else:
            state.trend = 0

        # Bias: -1 to +1
        state.bias = ind.momentum_score

    def get_confluence(self) -> HTFConfluence:
        """Compute multi-timeframe confluence."""
        s5 = self._states.get("5m")
        s15 = self._states.get("15m")
        s1h = self._states.get("1h")

        t5 = s5.trend if s5 else 0
        t15 = s15.trend if s15 else 0
        t1h = s1h.trend if s1h else 0

        b5 = s5.bias if s5 else 0
        b15 = s15.bias if s15 else 0
        b1h = s1h.bias if s1h else 0

        # Weighted score: 1h matters most, then 15m, then 5m
        score = 0.2 * b5 + 0.35 * b15 + 0.45 * b1h

        # Determine which side HTF supports
        if score > 0.15:
            agrees_with = Side.LONG
        elif score < -0.15:
            agrees_with = Side.SHORT
        else:
            agrees_with = None

        # Strength: how aligned are the timeframes?
        trends = [t5, t15, t1h]
        non_zero = [t for t in trends if t != 0]
        if non_zero:
            agreement = abs(sum(non_zero)) / len(non_zero)
        else:
            agreement = 0

        strength = float(np.clip(agreement, 0, 1))

        # Description
        parts = []
        if t5 != 0:
            parts.append(f"5m:{'↑' if t5 > 0 else '↓'}")
        if t15 != 0:
            parts.append(f"15m:{'↑' if t15 > 0 else '↓'}")
        if t1h != 0:
            parts.append(f"1h:{'↑' if t1h > 0 else '↓'}")
        desc = " ".join(parts) if parts else "neutral"

        return HTFConfluence(
            score=score,
            agrees_with=agrees_with,
            strength=strength,
            tf_5m_trend=t5,
            tf_15m_trend=t15,
            tf_1h_trend=t1h,
            description=desc,
        )

    def get_state(self, label: str) -> Optional[TimeframeState]:
        return self._states.get(label)

    def get_all_states(self) -> dict[str, TimeframeState]:
        return self._states
