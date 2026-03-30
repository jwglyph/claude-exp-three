"""Intra-candle tick-level analysis.

Detects events WITHIN a 1-minute candle that warrant immediate action,
rather than waiting for the candle to close:

1. Momentum bursts: rapid price movement (X ticks in Y seconds)
2. Level touches: price reaching key levels (VWAP, EMA, round numbers)
3. Volume spikes: sudden surge of volume mid-candle
4. Rejection wicks: price spikes to a level then reverses sharply
5. Delta shifts: buy/sell pressure changing rapidly
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from scalper.models import Tick, Side, Candle
from scalper.analysis.indicators import IndicatorState


@dataclass
class TickEvent:
    """An intra-candle event detected from tick analysis."""
    timestamp: float
    event_type: str  # "momentum_burst", "level_touch", "volume_spike", "rejection", "delta_shift"
    direction: int  # +1 bullish, -1 bearish
    magnitude: float  # 0-1, how significant
    price: float
    description: str


@dataclass
class TickAnalyzerState:
    """Running state of the tick analyzer."""
    # Recent ticks (last 30 seconds)
    recent_prices: deque = field(default_factory=lambda: deque(maxlen=500))
    recent_times: deque = field(default_factory=lambda: deque(maxlen=500))
    recent_sizes: deque = field(default_factory=lambda: deque(maxlen=500))
    recent_sides: deque = field(default_factory=lambda: deque(maxlen=500))

    # Momentum tracking
    price_5s_ago: float = 0.0
    price_10s_ago: float = 0.0
    price_30s_ago: float = 0.0

    # Volume tracking
    volume_1s: int = 0
    volume_5s: int = 0
    avg_volume_per_sec: float = 0.0

    # Delta tracking (buy - sell volume)
    delta_5s: int = 0
    delta_30s: int = 0

    # Level tracking
    last_vwap_touch: float = 0.0
    last_ema_touch: float = 0.0

    # Cooldowns (prevent rapid-fire events)
    last_event_time: float = 0.0
    events_this_candle: int = 0


class TickAnalyzer:
    """Analyzes tick stream for intra-candle trading opportunities."""

    def __init__(
        self,
        momentum_threshold_ticks: int = 8,  # 2 points in NQ
        momentum_window_sec: float = 5.0,
        volume_spike_ratio: float = 3.0,
        level_proximity_ticks: int = 2,
        event_cooldown_sec: float = 10.0,
        max_events_per_candle: int = 3,
    ):
        self.momentum_threshold = momentum_threshold_ticks * 0.25  # convert to points
        self.momentum_window = momentum_window_sec
        self.volume_spike_ratio = volume_spike_ratio
        self.level_proximity = level_proximity_ticks * 0.25
        self.event_cooldown = event_cooldown_sec
        self.max_events_per_candle = max_events_per_candle

        self.state = TickAnalyzerState()
        self._key_levels: list[float] = []

    def update_levels(self, indicators: IndicatorState) -> None:
        """Update key price levels from indicators."""
        levels = []
        if indicators.vwap > 0:
            levels.append(indicators.vwap)
            levels.append(indicators.vwap_upper)
            levels.append(indicators.vwap_lower)
        if indicators.ema_fast > 0:
            levels.append(indicators.ema_fast)
        if indicators.ema_slow > 0:
            levels.append(indicators.ema_slow)
        if indicators.bb_upper > 0:
            levels.append(indicators.bb_upper)
            levels.append(indicators.bb_lower)

        # Add round numbers near current price
        if indicators.ema_fast > 0:
            base = round(indicators.ema_fast / 25) * 25  # nearest 25
            for offset in [-50, -25, 0, 25, 50]:
                levels.append(base + offset)

        self._key_levels = [l for l in levels if l > 0]

    def process_tick(self, tick: Tick, indicators: Optional[IndicatorState] = None) -> Optional[TickEvent]:
        """Process a tick and check for intra-candle events."""
        now = tick.timestamp
        s = self.state

        # Store tick data
        s.recent_prices.append(tick.price)
        s.recent_times.append(now)
        s.recent_sizes.append(tick.size)
        s.recent_sides.append(tick.side)

        # Cooldown check
        if now - s.last_event_time < self.event_cooldown:
            return None
        if s.events_this_candle >= self.max_events_per_candle:
            return None

        # Update rolling metrics
        self._update_metrics(now)

        # Check for events (priority order)
        event = (
            self._check_momentum_burst(tick, now) or
            self._check_level_touch(tick, indicators) or
            self._check_volume_spike(tick, now) or
            self._check_rejection(tick, now) or
            self._check_delta_shift(tick, now)
        )

        if event:
            s.last_event_time = now
            s.events_this_candle += 1

        return event

    def reset_candle(self) -> None:
        """Reset per-candle state when a new candle starts."""
        self.state.events_this_candle = 0

    def _update_metrics(self, now: float) -> None:
        """Update rolling metrics from recent ticks."""
        s = self.state
        if not s.recent_prices:
            return

        # Price at various lookbacks
        for i in range(len(s.recent_times) - 1, -1, -1):
            age = now - s.recent_times[i]
            if age >= 5 and s.price_5s_ago == 0:
                s.price_5s_ago = s.recent_prices[i]
            if age >= 10 and s.price_10s_ago == 0:
                s.price_10s_ago = s.recent_prices[i]
            if age >= 30:
                s.price_30s_ago = s.recent_prices[i]
                break

        # Volume and delta in last 5 seconds
        vol_5s = 0
        delta_5s = 0
        vol_1s = 0
        count_30s = 0
        delta_30s = 0

        for i in range(len(s.recent_times) - 1, -1, -1):
            age = now - s.recent_times[i]
            if age <= 1:
                vol_1s += s.recent_sizes[i]
            if age <= 5:
                vol_5s += s.recent_sizes[i]
                if s.recent_sides[i] == "buy":
                    delta_5s += s.recent_sizes[i]
                elif s.recent_sides[i] == "sell":
                    delta_5s -= s.recent_sizes[i]
            if age <= 30:
                count_30s += 1
                if s.recent_sides[i] == "buy":
                    delta_30s += s.recent_sizes[i]
                elif s.recent_sides[i] == "sell":
                    delta_30s -= s.recent_sizes[i]
            else:
                break

        s.volume_1s = vol_1s
        s.volume_5s = vol_5s
        s.delta_5s = delta_5s
        s.delta_30s = delta_30s

        # Average volume per second (from 30s window)
        if count_30s > 0 and len(s.recent_times) > 1:
            window = min(30, now - s.recent_times[0])
            s.avg_volume_per_sec = vol_5s / max(5, window / 6)

    def _check_momentum_burst(self, tick: Tick, now: float) -> Optional[TickEvent]:
        """Detect rapid directional price movement."""
        s = self.state
        if s.price_5s_ago == 0:
            return None

        move = tick.price - s.price_5s_ago
        if abs(move) >= self.momentum_threshold:
            direction = 1 if move > 0 else -1
            magnitude = min(1.0, abs(move) / (self.momentum_threshold * 2))

            return TickEvent(
                timestamp=now,
                event_type="momentum_burst",
                direction=direction,
                magnitude=magnitude,
                price=tick.price,
                description=f"{'↑' if direction > 0 else '↓'}{abs(move):.2f}pts in 5s",
            )
        return None

    def _check_level_touch(self, tick: Tick, indicators: Optional[IndicatorState]) -> Optional[TickEvent]:
        """Detect price touching a key level."""
        if not self._key_levels:
            return None

        for level in self._key_levels:
            distance = abs(tick.price - level)
            if distance <= self.level_proximity:
                # Which direction are we approaching from?
                s = self.state
                if s.price_5s_ago > 0:
                    approach = tick.price - s.price_5s_ago
                    if abs(approach) < 0.5:
                        continue  # not really approaching, just sitting here

                    # Approaching from below = potential resistance, from above = potential support
                    if approach > 0:
                        direction = -1  # hitting resistance = bearish
                        desc = f"resistance touch {level:.2f}"
                    else:
                        direction = 1  # hitting support = bullish
                        desc = f"support touch {level:.2f}"

                    return TickEvent(
                        timestamp=tick.timestamp,
                        event_type="level_touch",
                        direction=direction,
                        magnitude=0.6,
                        price=tick.price,
                        description=desc,
                    )
        return None

    def _check_volume_spike(self, tick: Tick, now: float) -> Optional[TickEvent]:
        """Detect sudden volume surge."""
        s = self.state
        if s.avg_volume_per_sec <= 0:
            return None

        if s.volume_1s > s.avg_volume_per_sec * self.volume_spike_ratio:
            # Volume spike - which direction?
            direction = 1 if s.delta_5s > 0 else -1 if s.delta_5s < 0 else 0
            if direction == 0:
                return None

            magnitude = min(1.0, s.volume_1s / (s.avg_volume_per_sec * self.volume_spike_ratio * 2))

            return TickEvent(
                timestamp=now,
                event_type="volume_spike",
                direction=direction,
                magnitude=magnitude,
                price=tick.price,
                description=f"vol {s.volume_1s}x avg({'buy' if direction > 0 else 'sell'})",
            )
        return None

    def _check_rejection(self, tick: Tick, now: float) -> Optional[TickEvent]:
        """Detect rejection wick / failed breakout.

        Price spikes in one direction then reverses sharply.
        """
        s = self.state
        if s.price_10s_ago == 0 or s.price_5s_ago == 0:
            return None

        # First move (10s ago to 5s ago)
        first_move = s.price_5s_ago - s.price_10s_ago
        # Second move (5s ago to now)
        second_move = tick.price - s.price_5s_ago

        # Rejection: first move significant, second move reverses >60%
        if abs(first_move) >= self.momentum_threshold * 0.7:
            if first_move > 0 and second_move < -first_move * 0.6:
                return TickEvent(
                    timestamp=now,
                    event_type="rejection",
                    direction=-1,  # rejection of upward move = bearish
                    magnitude=min(1.0, abs(second_move / first_move)),
                    price=tick.price,
                    description=f"rejection high, reversed {abs(second_move):.2f}pts",
                )
            elif first_move < 0 and second_move > -first_move * 0.6:
                return TickEvent(
                    timestamp=now,
                    event_type="rejection",
                    direction=1,  # rejection of downward move = bullish
                    magnitude=min(1.0, abs(second_move / first_move)),
                    price=tick.price,
                    description=f"rejection low, reversed {abs(second_move):.2f}pts",
                )
        return None

    def _check_delta_shift(self, tick: Tick, now: float) -> Optional[TickEvent]:
        """Detect sudden shift in order flow delta."""
        s = self.state
        if s.delta_30s == 0:
            return None

        # 5s delta vs 30s delta: if they disagree strongly
        if s.delta_30s > 0 and s.delta_5s < -abs(s.delta_30s) * 0.5:
            return TickEvent(
                timestamp=now,
                event_type="delta_shift",
                direction=-1,
                magnitude=0.5,
                price=tick.price,
                description=f"sell pressure shift Δ5s={s.delta_5s} vs Δ30s={s.delta_30s}",
            )
        elif s.delta_30s < 0 and s.delta_5s > abs(s.delta_30s) * 0.5:
            return TickEvent(
                timestamp=now,
                event_type="delta_shift",
                direction=1,
                magnitude=0.5,
                price=tick.price,
                description=f"buy pressure shift Δ5s={s.delta_5s} vs Δ30s={s.delta_30s}",
            )
        return None
