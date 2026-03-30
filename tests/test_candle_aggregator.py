"""Tests for candle aggregation."""

import pytest

from scalper.feeds.candle_aggregator import CandleAggregator
from scalper.models import Tick


class TestCandleAggregator:
    def test_builds_candle_from_ticks(self):
        agg = CandleAggregator(interval_sec=60)
        completed = []
        agg.on_candle = lambda c: completed.append(c)

        # Send ticks within one candle period
        base_time = 1000 * 60  # aligned to minute boundary
        ticks = [
            Tick(timestamp=base_time + 0, price=100.0, size=10, side="buy"),
            Tick(timestamp=base_time + 10, price=102.0, size=5, side="buy"),
            Tick(timestamp=base_time + 30, price=99.0, size=8, side="sell"),
            Tick(timestamp=base_time + 50, price=101.0, size=3, side="buy"),
        ]
        for t in ticks:
            agg.process_tick(t)

        # Not yet completed (same period)
        assert len(completed) == 0
        assert agg.current_candle is not None
        assert agg.current_candle.open == 100.0
        assert agg.current_candle.high == 102.0
        assert agg.current_candle.low == 99.0
        assert agg.current_candle.close == 101.0
        assert agg.current_candle.volume == 26

    def test_closes_candle_on_new_period(self):
        agg = CandleAggregator(interval_sec=60)
        completed = []
        agg.on_candle = lambda c: completed.append(c)

        base = 60000.0  # aligned
        agg.process_tick(Tick(timestamp=base, price=100.0, size=10))
        agg.process_tick(Tick(timestamp=base + 30, price=102.0, size=5))

        # New period tick closes previous candle
        agg.process_tick(Tick(timestamp=base + 60, price=101.0, size=3))

        assert len(completed) == 1
        assert completed[0].is_complete
        assert completed[0].open == 100.0
        assert completed[0].close == 102.0

    def test_tracks_buy_sell_volume(self):
        agg = CandleAggregator(interval_sec=60)
        base = 60000.0
        agg.process_tick(Tick(timestamp=base, price=100.0, size=10, side="buy"))
        agg.process_tick(Tick(timestamp=base + 1, price=100.0, size=5, side="sell"))
        agg.process_tick(Tick(timestamp=base + 2, price=100.0, size=3, side="buy"))

        assert agg.current_candle.buy_volume == 13
        assert agg.current_candle.sell_volume == 5
        assert agg.current_candle.delta == 8

    def test_force_close(self):
        agg = CandleAggregator(interval_sec=60)
        agg.process_tick(Tick(timestamp=60000.0, price=100.0, size=10))
        candle = agg.force_close()
        assert candle is not None
        assert candle.is_complete
        assert agg.current_candle is None

    def test_max_candles_limit(self):
        agg = CandleAggregator(interval_sec=60, max_candles=5)
        # Generate 10 candles
        for i in range(10):
            base = (i + 1) * 60.0
            agg.process_tick(Tick(timestamp=base, price=100.0 + i, size=1))
            # Force new period
            agg.process_tick(Tick(timestamp=base + 60, price=100.0 + i + 1, size=1))

        assert len(agg.get_candles()) <= 5
