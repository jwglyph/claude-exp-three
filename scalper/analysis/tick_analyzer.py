"""Intra-candle tick-level analysis with fully adaptive thresholds.

ALL thresholds are relative to current volatility (ATR) and recent
price behavior. Nothing is a fixed number of points or ticks.

Detects:
1. Momentum bursts: rapid move relative to current ATR
2. Level touches: price reaching dynamic key levels
3. Volume spikes: surge relative to rolling average
4. Rejection wicks: spike + reversal relative to recent range
5. Delta shifts: order flow pressure changes
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
    event_type: str
    direction: int  # +1 bullish, -1 bearish
    magnitude: float  # 0-1, how significant
    price: float
    description: str


class TickAnalyzer:
    """Analyzes tick stream with volatility-adaptive thresholds.

    Every threshold is expressed as a multiple of ATR or rolling metrics,
    so the analyzer automatically adjusts to fast vs slow markets.
    """

    def __init__(self):
        # Recent tick data (rolling window)
        self._prices: deque = deque(maxlen=1000)
        self._times: deque = deque(maxlen=1000)
        self._sizes: deque = deque(maxlen=1000)
        self._sides: deque = deque(maxlen=1000)

        # Key levels (updated each candle from indicators)
        self._key_levels: list[float] = []

        # Current volatility reference (from 1m ATR)
        self._atr: float = 0.0
        self._atr_per_sec: float = 0.0  # ATR normalized to per-second

        # Rolling volume baseline
        self._vol_5m_total: int = 0
        self._vol_5m_ticks: int = 0
        self._vol_per_sec: float = 1.0

        # State
        self._last_event_time: float = 0.0
        self._events_this_candle: int = 0

    def update_context(self, indicators: IndicatorState) -> None:
        """Update volatility context from latest indicators.

        Called on each candle close to recalibrate all thresholds.
        """
        self._atr = max(indicators.atr, 0.5)  # floor to prevent div/0
        # ATR is per-candle (1 min). Normalize to per-second.
        self._atr_per_sec = self._atr / 60.0

        # Update key levels
        levels = []
        if indicators.vwap > 0:
            levels.extend([indicators.vwap, indicators.vwap_upper, indicators.vwap_lower])
        if indicators.ema_fast > 0:
            levels.append(indicators.ema_fast)
        if indicators.ema_slow > 0:
            levels.append(indicators.ema_slow)
        if indicators.ema_trend > 0:
            levels.append(indicators.ema_trend)
        if indicators.bb_upper > 0:
            levels.extend([indicators.bb_upper, indicators.bb_lower, indicators.bb_middle])

        # Round number levels near current price
        if indicators.ema_fast > 0:
            base = round(indicators.ema_fast / 25) * 25
            for offset in [-75, -50, -25, 0, 25, 50, 75]:
                levels.append(base + offset)

        self._key_levels = sorted(set(l for l in levels if l > 0))

    def process_tick(self, tick: Tick, indicators: Optional[IndicatorState] = None) -> Optional[TickEvent]:
        """Process a tick. Returns a TickEvent if something notable happens."""
        now = tick.timestamp

        # Store
        self._prices.append(tick.price)
        self._times.append(now)
        self._sizes.append(tick.size)
        self._sides.append(tick.side)

        # Update rolling volume
        self._vol_5m_total += tick.size
        self._vol_5m_ticks += 1
        # Decay old volume (approximate 5-min window)
        if self._vol_5m_ticks > 0:
            window = now - self._times[0] if len(self._times) > 1 else 300
            self._vol_per_sec = self._vol_5m_total / max(window, 1)

        # Adaptive cooldown: faster in volatile markets, slower in calm
        cooldown = self._adaptive_cooldown()
        if now - self._last_event_time < cooldown:
            return None

        # Max events per candle scales with volatility
        max_events = self._adaptive_max_events()
        if self._events_this_candle >= max_events:
            return None

        # Need minimum ATR context
        if self._atr <= 0:
            return None

        # Check events (priority order)
        event = (
            self._check_momentum(tick, now) or
            self._check_level_touch(tick) or
            self._check_volume_spike(tick, now) or
            self._check_rejection(tick, now) or
            self._check_delta_shift(tick, now)
        )

        if event:
            self._last_event_time = now
            self._events_this_candle += 1

        return event

    def reset_candle(self) -> None:
        self._events_this_candle = 0

    # --- Adaptive thresholds ---

    def _adaptive_cooldown(self) -> float:
        """Cooldown between events. Shorter when vol is high."""
        if self._atr <= 0:
            return 15.0
        # Base 10s, scale down by vol (min 3s in fast markets)
        return max(3.0, 10.0 / (self._atr / 5.0))

    def _adaptive_max_events(self) -> int:
        """Max events per candle. More in volatile markets."""
        if self._atr <= 0:
            return 2
        if self._atr > 10:
            return 5  # very volatile
        elif self._atr > 6:
            return 4
        elif self._atr > 3:
            return 3
        return 2

    def _momentum_threshold(self) -> float:
        """Points that constitute a momentum burst.

        = 0.3x ATR compressed into 5 seconds (significant intra-candle move).
        """
        return self._atr * 0.3

    def _level_proximity(self) -> float:
        """How close to a level counts as a 'touch'.

        = 0.05x ATR (tighter in calm, wider in volatile).
        """
        return self._atr * 0.05

    def _volume_spike_threshold(self) -> float:
        """Volume per second that counts as a spike.

        = 3x rolling average volume per second.
        """
        return self._vol_per_sec * 3.0

    def _rejection_threshold(self) -> float:
        """Move size to qualify as a rejection.

        = 0.2x ATR initial move + 60% reversal.
        """
        return self._atr * 0.2

    # --- Event detectors ---

    def _price_n_sec_ago(self, now: float, seconds: float) -> Optional[float]:
        """Get price from N seconds ago."""
        target = now - seconds
        for i in range(len(self._times) - 1, -1, -1):
            if self._times[i] <= target:
                return self._prices[i]
        return None

    def _vol_last_n_sec(self, now: float, seconds: float) -> tuple:
        """Get (total_volume, buy_volume, sell_volume) in last N seconds."""
        total = buy = sell = 0
        for i in range(len(self._times) - 1, -1, -1):
            if now - self._times[i] > seconds:
                break
            total += self._sizes[i]
            if self._sides[i] == "buy":
                buy += self._sizes[i]
            elif self._sides[i] == "sell":
                sell += self._sizes[i]
        return total, buy, sell

    def _check_momentum(self, tick: Tick, now: float) -> Optional[TickEvent]:
        """Detect rapid price movement relative to ATR."""
        price_5s = self._price_n_sec_ago(now, 5.0)
        if price_5s is None:
            return None

        move = tick.price - price_5s
        threshold = self._momentum_threshold()

        if abs(move) >= threshold:
            direction = 1 if move > 0 else -1
            # Magnitude: how many thresholds the move covers (0.5 to 1.0)
            magnitude = min(1.0, 0.5 + abs(move) / (threshold * 3))

            return TickEvent(
                timestamp=now,
                event_type="momentum_burst",
                direction=direction,
                magnitude=magnitude,
                price=tick.price,
                description=f"{'↑' if direction > 0 else '↓'}{abs(move):.2f}pts/5s ({abs(move)/self._atr:.0%} ATR)",
            )
        return None

    def _check_level_touch(self, tick: Tick) -> Optional[TickEvent]:
        """Detect price touching a key level."""
        proximity = self._level_proximity()
        price_5s = self._price_n_sec_ago(tick.timestamp, 5.0)

        if price_5s is None or not self._key_levels:
            return None

        for level in self._key_levels:
            if abs(tick.price - level) <= proximity:
                approach = tick.price - price_5s
                if abs(approach) < proximity * 0.5:
                    continue  # not approaching, just sitting

                if approach > 0:
                    direction = -1  # hitting from below = resistance
                    desc = f"resistance {level:.2f}"
                else:
                    direction = 1  # hitting from above = support
                    desc = f"support {level:.2f}"

                return TickEvent(
                    timestamp=tick.timestamp,
                    event_type="level_touch",
                    direction=direction,
                    magnitude=0.55,
                    price=tick.price,
                    description=desc,
                )
        return None

    def _check_volume_spike(self, tick: Tick, now: float) -> Optional[TickEvent]:
        """Detect volume surge relative to rolling average."""
        vol_1s, buy_1s, sell_1s = self._vol_last_n_sec(now, 1.0)
        threshold = self._volume_spike_threshold()

        if vol_1s > threshold and threshold > 0:
            delta = buy_1s - sell_1s
            if delta == 0:
                return None
            direction = 1 if delta > 0 else -1
            ratio = vol_1s / max(self._vol_per_sec, 1)
            magnitude = min(1.0, 0.4 + ratio / 10)

            return TickEvent(
                timestamp=now,
                event_type="volume_spike",
                direction=direction,
                magnitude=magnitude,
                price=tick.price,
                description=f"{ratio:.1f}x avg vol ({'buy' if direction > 0 else 'sell'} heavy)",
            )
        return None

    def _check_rejection(self, tick: Tick, now: float) -> Optional[TickEvent]:
        """Detect rejection / failed breakout relative to ATR."""
        price_10s = self._price_n_sec_ago(now, 10.0)
        price_5s = self._price_n_sec_ago(now, 5.0)

        if price_10s is None or price_5s is None:
            return None

        first_move = price_5s - price_10s
        second_move = tick.price - price_5s
        threshold = self._rejection_threshold()

        if abs(first_move) >= threshold:
            if first_move > 0 and second_move < -first_move * 0.6:
                pct = abs(second_move / first_move)
                return TickEvent(
                    timestamp=now,
                    event_type="rejection",
                    direction=-1,
                    magnitude=min(1.0, 0.5 + pct * 0.3),
                    price=tick.price,
                    description=f"rejected high, {pct:.0%} reversal ({abs(first_move):.1f}pts)",
                )
            elif first_move < 0 and second_move > -first_move * 0.6:
                pct = abs(second_move / first_move)
                return TickEvent(
                    timestamp=now,
                    event_type="rejection",
                    direction=1,
                    magnitude=min(1.0, 0.5 + pct * 0.3),
                    price=tick.price,
                    description=f"rejected low, {pct:.0%} reversal ({abs(first_move):.1f}pts)",
                )
        return None

    def _check_delta_shift(self, tick: Tick, now: float) -> Optional[TickEvent]:
        """Detect order flow delta reversal."""
        _, buy_5s, sell_5s = self._vol_last_n_sec(now, 5.0)
        _, buy_30s, sell_30s = self._vol_last_n_sec(now, 30.0)

        delta_5s = buy_5s - sell_5s
        delta_30s = buy_30s - sell_30s

        if delta_30s == 0:
            return None

        # 5s delta strongly opposes 30s delta
        if delta_30s > 0 and delta_5s < -abs(delta_30s) * 0.4:
            return TickEvent(
                timestamp=now,
                event_type="delta_shift",
                direction=-1,
                magnitude=0.45,
                price=tick.price,
                description=f"sell shift Δ5s={delta_5s:+d} vs Δ30s={delta_30s:+d}",
            )
        elif delta_30s < 0 and delta_5s > abs(delta_30s) * 0.4:
            return TickEvent(
                timestamp=now,
                event_type="delta_shift",
                direction=1,
                magnitude=0.45,
                price=tick.price,
                description=f"buy shift Δ5s={delta_5s:+d} vs Δ30s={delta_30s:+d}",
            )
        return None
