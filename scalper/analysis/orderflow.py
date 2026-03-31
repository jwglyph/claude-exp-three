"""Order flow intelligence engine.

Reads the tape (time & sales) to detect what institutional / smart money
is doing, beyond what OHLCV candles show.

Core concepts:
- Delta: buy volume minus sell volume. Shows who's aggressive.
- Absorption: heavy volume but no price movement → smart money absorbing.
- Imbalance: lopsided buy/sell at specific price levels → directional intent.
- Exhaustion: extreme delta/volume at a high/low → potential reversal.
- Large prints: institutional-size orders hitting the tape.

All thresholds are adaptive based on rolling averages - no fixed numbers.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class OrderFlowState:
    """Current order flow readings."""
    # Delta (cumulative buy - sell volume)
    delta_1m: int = 0       # last 1 minute
    delta_5m: int = 0       # last 5 minutes
    delta_session: int = 0  # session cumulative

    # Volume
    buy_volume_1m: int = 0
    sell_volume_1m: int = 0
    total_volume_1m: int = 0

    # Imbalance (buy_vol / total_vol, 0.5 = balanced)
    imbalance_ratio: float = 0.5

    # Pressure (rate of change of delta)
    delta_acceleration: float = 0.0  # positive = buying accelerating

    # Large prints
    large_prints_buy: int = 0  # count in last 5 min
    large_prints_sell: int = 0
    large_print_net: int = 0  # buy - sell count

    # Absorption score (-1 to +1)
    # Positive = buy absorption (selling absorbed, bullish)
    # Negative = sell absorption (buying absorbed, bearish)
    absorption: float = 0.0

    # Exhaustion score (-1 to +1)
    # Positive = buy exhaustion (too much buying, likely reversal down)
    # Negative = sell exhaustion (too much selling, likely reversal up)
    exhaustion: float = 0.0

    # Stacked imbalances
    stacked_buy_levels: int = 0   # consecutive price levels with buy imbalance
    stacked_sell_levels: int = 0

    # Overall bias (-1 to +1)
    flow_bias: float = 0.0


@dataclass
class TradeEntry:
    """A single trade from the tape."""
    timestamp: float
    price: float
    size: int
    side: str  # "buy" or "sell"


@dataclass
class PriceLevel:
    """Volume at a specific price level (footprint data)."""
    price: float
    buy_volume: int = 0
    sell_volume: int = 0
    trade_count: int = 0

    @property
    def delta(self) -> int:
        return self.buy_volume - self.sell_volume

    @property
    def total(self) -> int:
        return self.buy_volume + self.sell_volume

    @property
    def imbalance_ratio(self) -> float:
        """Buy imbalance ratio. >0.5 = more buying, <0.5 = more selling."""
        total = self.total
        return self.buy_volume / total if total > 0 else 0.5


class OrderFlowEngine:
    """Analyzes order flow from the trade stream.

    Feed every trade print into process_trade().
    Query get_state() for current order flow readings.
    """

    def __init__(self):
        # Raw trade tape (rolling window)
        self._tape: deque[TradeEntry] = deque(maxlen=10000)

        # Volume at price (footprint) - reset each session
        self._price_levels: dict[float, PriceLevel] = defaultdict(
            lambda: PriceLevel(price=0)
        )

        # Rolling delta windows
        self._delta_history: deque[tuple[float, int]] = deque(maxlen=300)  # (time, delta_1s)

        # Session cumulative
        self._session_delta: int = 0
        self._session_buy_vol: int = 0
        self._session_sell_vol: int = 0

        # Large print tracking
        self._avg_trade_size: float = 1.0
        self._trade_sizes: deque[int] = deque(maxlen=5000)

        # Absorption detection
        self._price_at_delta: deque[tuple[float, float, int]] = deque(maxlen=60)  # (time, price, cum_delta)

        # State
        self._state = OrderFlowState()
        self._last_price = 0.0
        self._tick_size = 0.25  # NQ

    def set_tick_size(self, tick_size: float) -> None:
        self._tick_size = tick_size

    def process_trade(self, timestamp: float, price: float, size: int, side: str) -> None:
        """Process a single trade print from the tape."""
        entry = TradeEntry(timestamp=timestamp, price=price, size=size, side=side)
        self._tape.append(entry)
        self._trade_sizes.append(size)

        # Update session totals
        if side == "buy":
            self._session_buy_vol += size
            self._session_delta += size
        elif side == "sell":
            self._session_sell_vol += size
            self._session_delta -= size

        # Update volume at price (quantize to tick)
        level_price = round(price / self._tick_size) * self._tick_size
        level = self._price_levels[level_price]
        level.price = level_price
        if side == "buy":
            level.buy_volume += size
        elif side == "sell":
            level.sell_volume += size
        level.trade_count += 1

        # Update average trade size (exponential)
        if self._avg_trade_size == 0:
            self._avg_trade_size = size
        else:
            self._avg_trade_size = 0.995 * self._avg_trade_size + 0.005 * size

        # Track price vs delta for absorption detection
        self._price_at_delta.append((timestamp, price, self._session_delta))

        self._last_price = price

    def get_state(self) -> OrderFlowState:
        """Compute and return current order flow state."""
        now = time.time()
        self._compute_state(now)
        return self._state

    def _compute_state(self, now: float) -> None:
        """Recompute all order flow metrics."""
        s = self._state

        # --- Time-windowed delta ---
        buy_1m = sell_1m = buy_5m = sell_5m = 0
        for t in reversed(self._tape):
            age = now - t.timestamp
            if age > 300:
                break
            if age <= 60:
                if t.side == "buy":
                    buy_1m += t.size
                elif t.side == "sell":
                    sell_1m += t.size
            if t.side == "buy":
                buy_5m += t.size
            elif t.side == "sell":
                sell_5m += t.size

        s.delta_1m = buy_1m - sell_1m
        s.delta_5m = buy_5m - sell_5m
        s.delta_session = self._session_delta
        s.buy_volume_1m = buy_1m
        s.sell_volume_1m = sell_1m
        s.total_volume_1m = buy_1m + sell_1m

        # --- Imbalance ratio ---
        total = buy_1m + sell_1m
        s.imbalance_ratio = buy_1m / total if total > 0 else 0.5

        # --- Delta acceleration ---
        # Compare last 30s delta vs prior 30s delta
        buy_30 = sell_30 = buy_60 = sell_60 = 0
        for t in reversed(self._tape):
            age = now - t.timestamp
            if age > 60:
                break
            if age <= 30:
                if t.side == "buy":
                    buy_30 += t.size
                else:
                    sell_30 += t.size
            else:
                if t.side == "buy":
                    buy_60 += t.size
                else:
                    sell_60 += t.size

        delta_recent = buy_30 - sell_30
        delta_prior = buy_60 - sell_60
        # Normalize by average volume
        avg_vol = s.total_volume_1m / 60 if s.total_volume_1m > 0 else 1
        s.delta_acceleration = (delta_recent - delta_prior) / max(avg_vol * 30, 1)

        # --- Large print detection ---
        large_threshold = self._avg_trade_size * 5  # 5x average = large
        large_buy = large_sell = 0
        for t in reversed(self._tape):
            if now - t.timestamp > 300:
                break
            if t.size >= large_threshold:
                if t.side == "buy":
                    large_buy += 1
                elif t.side == "sell":
                    large_sell += 1

        s.large_prints_buy = large_buy
        s.large_prints_sell = large_sell
        s.large_print_net = large_buy - large_sell

        # --- Absorption detection ---
        s.absorption = self._detect_absorption(now)

        # --- Exhaustion detection ---
        s.exhaustion = self._detect_exhaustion(now)

        # --- Stacked imbalances ---
        s.stacked_buy_levels, s.stacked_sell_levels = self._detect_stacked_imbalances()

        # --- Overall flow bias ---
        # Weighted composite of all signals
        bias = 0.0

        # Delta direction (normalized)
        if s.total_volume_1m > 0:
            delta_norm = s.delta_1m / s.total_volume_1m  # -1 to +1
            bias += delta_norm * 0.25

        # Imbalance
        bias += (s.imbalance_ratio - 0.5) * 2 * 0.2  # -1 to +1, weight 0.2

        # Delta acceleration
        bias += np.clip(s.delta_acceleration, -1, 1) * 0.15

        # Large prints
        if large_buy + large_sell > 0:
            large_bias = (large_buy - large_sell) / (large_buy + large_sell)
            bias += large_bias * 0.15

        # Absorption (inverted: buy absorption = bullish)
        bias += s.absorption * 0.15

        # Exhaustion (inverted: buy exhaustion = bearish signal)
        bias -= s.exhaustion * 0.1

        s.flow_bias = float(np.clip(bias, -1, 1))

    def _detect_absorption(self, now: float) -> float:
        """Detect absorption: heavy volume but price not moving.

        If there's heavy selling but price holds → buy absorption (bullish)
        If there's heavy buying but price doesn't rise → sell absorption (bearish)

        Returns: -1 (sell absorption/bearish) to +1 (buy absorption/bullish)
        """
        if len(self._price_at_delta) < 10:
            return 0.0

        # Compare price movement vs delta movement over last 60s
        entries_60s = [(t, p, d) for t, p, d in self._price_at_delta if now - t <= 60]
        if len(entries_60s) < 5:
            return 0.0

        first = entries_60s[0]
        last = entries_60s[-1]

        price_change = last[1] - first[1]
        delta_change = last[2] - first[2]

        if abs(delta_change) < 10:  # not enough volume to judge
            return 0.0

        # Normalize price change by recent ATR-like measure
        price_range = max(e[1] for e in entries_60s) - min(e[1] for e in entries_60s)
        if price_range < 0.5:
            price_range = 0.5

        # Absorption = delta moving but price not
        # Heavy selling (negative delta) but price holds = buy absorption
        if delta_change < 0 and abs(price_change) < price_range * 0.3:
            return min(1.0, abs(delta_change) / max(self._state.total_volume_1m * 0.5, 100))

        # Heavy buying (positive delta) but price doesn't rise = sell absorption
        if delta_change > 0 and abs(price_change) < price_range * 0.3:
            return -min(1.0, abs(delta_change) / max(self._state.total_volume_1m * 0.5, 100))

        return 0.0

    def _detect_exhaustion(self, now: float) -> float:
        """Detect exhaustion: extreme delta at a high/low.

        Buy exhaustion: price at session high + extreme positive delta → likely to reverse down
        Sell exhaustion: price at session low + extreme negative delta → likely to reverse up

        Returns: -1 (sell exhaustion/bullish) to +1 (buy exhaustion/bearish)
        """
        if len(self._tape) < 50:
            return 0.0

        # Recent trades
        recent = [t for t in self._tape if now - t.timestamp <= 120]
        if not recent:
            return 0.0

        prices = [t.price for t in recent]
        current = prices[-1]
        high = max(prices)
        low = min(prices)
        rng = high - low
        if rng < 1:
            return 0.0

        # Where are we in the range?
        pos_in_range = (current - low) / rng

        # Volume intensity (is current volume extreme?)
        recent_vol = sum(t.size for t in recent if now - t.timestamp <= 10)
        avg_10s = sum(t.size for t in recent) / max(len(recent), 1) * 10
        vol_intensity = recent_vol / max(avg_10s, 1)

        # Buy exhaustion: at the highs with extreme volume
        if pos_in_range > 0.85 and vol_intensity > 2.0:
            return min(1.0, (vol_intensity - 1) * 0.3)

        # Sell exhaustion: at the lows with extreme volume
        if pos_in_range < 0.15 and vol_intensity > 2.0:
            return -min(1.0, (vol_intensity - 1) * 0.3)

        return 0.0

    def _detect_stacked_imbalances(self) -> tuple[int, int]:
        """Count consecutive price levels with buy or sell imbalance.

        ONLY uses recent data (last 5 min) to avoid stale accumulation.
        Requires minimum volume per level to filter noise in thin markets.

        Returns: (stacked_buy_levels, stacked_sell_levels)
        """
        now = time.time()

        # Build fresh volume-at-price from RECENT trades only (last 5 min)
        recent_levels: dict[float, list] = defaultdict(lambda: [0, 0])  # {price: [buy, sell]}
        for t in reversed(self._tape):
            if now - t.timestamp > 300:
                break
            level_price = round(t.price / self._tick_size) * self._tick_size
            if t.side == "buy":
                recent_levels[level_price][0] += t.size
            elif t.side == "sell":
                recent_levels[level_price][1] += t.size

        if self._last_price <= 0 or len(recent_levels) < 3:
            return 0, 0

        # Filter: only levels with SIGNIFICANT volume
        # Need at least 5 trades at a level AND volume > 10% of 5-min average per level
        # This filters noise in thin overnight markets
        total_recent_vol = sum(b + s for b, s in recent_levels.values())
        avg_vol_per_level = total_recent_vol / max(len(recent_levels), 1)
        min_vol = max(10, avg_vol_per_level * 0.5, self._avg_trade_size * 5)

        nearby = sorted([
            (price, buy, sell)
            for price, (buy, sell) in recent_levels.items()
            if abs(price - self._last_price) <= self._tick_size * 15
            and (buy + sell) >= min_vol
        ])

        if len(nearby) < 3:
            return 0, 0

        # Count consecutive levels with >65% buy or sell imbalance
        imbalance_threshold = 0.65
        max_buy_stack = 0
        max_sell_stack = 0
        current_buy = 0
        current_sell = 0

        for price, buy_v, sell_v in nearby:
            total = buy_v + sell_v
            ratio = buy_v / total if total > 0 else 0.5

            if ratio > imbalance_threshold:
                current_buy += 1
                current_sell = 0
                max_buy_stack = max(max_buy_stack, current_buy)
            elif ratio < (1 - imbalance_threshold):
                current_sell += 1
                current_buy = 0
                max_sell_stack = max(max_sell_stack, current_sell)
            else:
                current_buy = 0
                current_sell = 0

        return max_buy_stack, max_sell_stack

    def get_delta_divergence(self, price_direction: int) -> float:
        """Check if delta diverges from price direction.

        If price is making new highs (+1) but delta is declining → bearish divergence
        If price is making new lows (-1) but delta is rising → bullish divergence

        Returns: 0 (no divergence) to 1 (strong divergence)
        """
        s = self._state

        if price_direction == 1:
            # Price going up - is delta confirming?
            if s.delta_1m < 0:
                return min(1.0, abs(s.delta_1m) / max(s.total_volume_1m * 0.3, 1))
        elif price_direction == -1:
            # Price going down - is delta confirming?
            if s.delta_1m > 0:
                return min(1.0, abs(s.delta_1m) / max(s.total_volume_1m * 0.3, 1))

        return 0.0

    def reset_session(self) -> None:
        """Reset for new session."""
        self._session_delta = 0
        self._session_buy_vol = 0
        self._session_sell_vol = 0
        self._price_levels.clear()
        self._tape.clear()
        self._state = OrderFlowState()
