"""Aggregates raw ticks into 1-minute OHLCV candles with volume profile data."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Optional

from scalper.models import Candle, Tick


class CandleAggregator:
    """Builds 1-minute candles from a tick stream.

    Tracks OHLCV plus buy/sell volume split and tick count for
    order flow analysis. Emits completed candles via callback.
    """

    def __init__(
        self,
        interval_sec: int = 60,
        max_candles: int = 200,
        on_candle: Optional[Callable[[Candle], None]] = None,
    ):
        self.interval_sec = interval_sec
        self.max_candles = max_candles
        self.on_candle = on_candle

        self.candles: deque[Candle] = deque(maxlen=max_candles)
        self._current: Optional[Candle] = None
        self._current_period_start: float = 0.0
        self._cumulative_vwap_num: float = 0.0
        self._cumulative_vwap_den: float = 0.0

    def _period_start(self, timestamp: float) -> float:
        """Align timestamp to candle boundary."""
        return (int(timestamp) // self.interval_sec) * self.interval_sec

    def process_tick(self, tick: Tick) -> Optional[Candle]:
        """Process a single tick. Returns a completed candle if one was closed."""
        period = self._period_start(tick.timestamp)
        completed = None

        # New candle period - close current and start new
        if self._current is not None and period > self._current_period_start:
            self._current.is_complete = True
            self.candles.append(self._current)
            completed = self._current
            if self.on_candle:
                self.on_candle(completed)
            self._current = None

        # Start new candle
        if self._current is None:
            self._current_period_start = period
            self._current = Candle(
                timestamp=period,
                open=tick.price,
                high=tick.price,
                low=tick.price,
                close=tick.price,
                volume=tick.size,
                tick_count=1,
                buy_volume=tick.size if tick.side == "buy" else 0,
                sell_volume=tick.size if tick.side == "sell" else 0,
            )
            self._cumulative_vwap_num = tick.price * tick.size
            self._cumulative_vwap_den = tick.size
        else:
            # Update current candle
            self._current.high = max(self._current.high, tick.price)
            self._current.low = min(self._current.low, tick.price)
            self._current.close = tick.price
            self._current.volume += tick.size
            self._current.tick_count += 1
            if tick.side == "buy":
                self._current.buy_volume += tick.size
            elif tick.side == "sell":
                self._current.sell_volume += tick.size

            # Running VWAP for this candle
            self._cumulative_vwap_num += tick.price * tick.size
            self._cumulative_vwap_den += tick.size

        if self._cumulative_vwap_den > 0:
            self._current.vwap = self._cumulative_vwap_num / self._cumulative_vwap_den

        return completed

    def process_candle(self, candle: Candle) -> None:
        """Directly ingest a pre-formed candle (for backtest or candle-based feeds)."""
        candle.is_complete = True
        self.candles.append(candle)
        if self.on_candle:
            self.on_candle(candle)

    def get_candles(self, count: Optional[int] = None) -> list[Candle]:
        """Return recent completed candles."""
        candles = list(self.candles)
        if count is not None:
            return candles[-count:]
        return candles

    @property
    def current_candle(self) -> Optional[Candle]:
        return self._current

    @property
    def last_candle(self) -> Optional[Candle]:
        return self.candles[-1] if self.candles else None

    def force_close(self) -> Optional[Candle]:
        """Force-close the current candle (e.g., at session end)."""
        if self._current is not None:
            self._current.is_complete = True
            self.candles.append(self._current)
            completed = self._current
            if self.on_candle:
                self.on_candle(completed)
            self._current = None
            return completed
        return None
