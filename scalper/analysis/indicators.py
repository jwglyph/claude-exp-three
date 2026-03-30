"""Technical indicators computed on candle series.

All indicators operate on numpy arrays for performance.
They are designed for online/streaming computation where possible.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from scalper.models import Candle


@dataclass
class IndicatorState:
    """Snapshot of all computed indicators for current bar."""
    # EMAs
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_trend: float = 0.0

    # ATR
    atr: float = 0.0
    atr_fast: float = 0.0
    atr_percent: float = 0.0  # ATR as % of price

    # RSI
    rsi: float = 50.0

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_percent_b: float = 0.5  # where price is within bands (0=lower, 1=upper)

    # VWAP
    vwap: float = 0.0
    vwap_upper: float = 0.0  # +1 std dev
    vwap_lower: float = 0.0  # -1 std dev

    # Volume
    volume_sma: float = 0.0
    volume_ratio: float = 1.0  # current volume / avg volume
    delta: int = 0  # buy_vol - sell_vol
    cumulative_delta: int = 0

    # Price action
    candle_body_ratio: float = 0.0
    consecutive_direction: int = 0  # +N for N bullish, -N for N bearish

    # Derived
    trend_direction: int = 0  # +1 up, -1 down, 0 neutral
    momentum_score: float = 0.0  # -1 to +1


def compute_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    if len(values) < period:
        return np.full_like(values, np.nan)
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema


def compute_sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    values = values.astype(float)
    if len(values) < period:
        return np.full_like(values, np.nan)
    cumsum = np.cumsum(values)
    sma = np.empty_like(values, dtype=float)
    sma[:period - 1] = np.nan
    sma[period - 1] = cumsum[period - 1] / period
    for i in range(period, len(values)):
        sma[i] = (cumsum[i] - cumsum[i - period]) / period
    return sma


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """Average True Range."""
    n = len(highs)
    if n < 2:
        return np.zeros(n)

    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    return compute_ema(tr, period)


def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index using Wilder's smoothing."""
    n = len(closes)
    if n < period + 1:
        return np.full(n, 50.0)

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(n, 50.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def compute_bollinger(closes: np.ndarray, period: int = 20, num_std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands. Returns (upper, middle, lower)."""
    closes = closes.astype(float)
    middle = compute_sma(closes, period)
    std = np.empty_like(closes, dtype=float)
    std[:] = np.nan
    for i in range(period - 1, len(closes)):
        std[i] = np.std(closes[i - period + 1:i + 1])

    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def compute_vwap(candles: list[Candle]) -> tuple[float, float, float]:
    """Session VWAP with standard deviation bands.

    Returns (vwap, upper_band, lower_band).
    """
    if not candles:
        return 0.0, 0.0, 0.0

    cum_vol = 0.0
    cum_pv = 0.0
    cum_pv2 = 0.0

    for c in candles:
        typical = (c.high + c.low + c.close) / 3.0
        vol = max(c.volume, 1)
        cum_vol += vol
        cum_pv += typical * vol
        cum_pv2 += typical * typical * vol

    vwap = cum_pv / cum_vol if cum_vol > 0 else 0.0
    variance = (cum_pv2 / cum_vol - vwap * vwap) if cum_vol > 0 else 0.0
    std = np.sqrt(max(variance, 0))

    return vwap, vwap + std, vwap - std


class IndicatorEngine:
    """Computes and maintains all technical indicators on a candle series."""

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        ema_trend: int = 50,
        atr_period: int = 14,
        atr_fast_period: int = 5,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        volume_sma_period: int = 20,
    ):
        self.ema_fast_period = ema_fast
        self.ema_slow_period = ema_slow
        self.ema_trend_period = ema_trend
        self.atr_period = atr_period
        self.atr_fast_period = atr_fast_period
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_sma_period = volume_sma_period
        self._cumulative_delta = 0

    def compute(self, candles: list[Candle]) -> IndicatorState:
        """Compute all indicators from candle history. Returns state for latest bar."""
        if len(candles) < 2:
            return IndicatorState()

        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles], dtype=float)

        state = IndicatorState()
        n = len(closes)

        # EMAs
        ema_f = compute_ema(closes, self.ema_fast_period)
        ema_s = compute_ema(closes, self.ema_slow_period)
        ema_t = compute_ema(closes, self.ema_trend_period)
        state.ema_fast = ema_f[-1]
        state.ema_slow = ema_s[-1]
        state.ema_trend = ema_t[-1] if not np.isnan(ema_t[-1]) else closes[-1]

        # ATR
        atr_arr = compute_atr(highs, lows, closes, self.atr_period)
        atr_fast = compute_atr(highs, lows, closes, self.atr_fast_period)
        state.atr = atr_arr[-1] if not np.isnan(atr_arr[-1]) else 0.0
        state.atr_fast = atr_fast[-1] if not np.isnan(atr_fast[-1]) else 0.0
        state.atr_percent = state.atr / closes[-1] * 100 if closes[-1] > 0 else 0.0

        # RSI
        rsi_arr = compute_rsi(closes, self.rsi_period)
        state.rsi = rsi_arr[-1]

        # Bollinger Bands
        bb_u, bb_m, bb_l = compute_bollinger(closes, self.bb_period, self.bb_std)
        state.bb_upper = bb_u[-1] if not np.isnan(bb_u[-1]) else closes[-1] + state.atr
        state.bb_middle = bb_m[-1] if not np.isnan(bb_m[-1]) else closes[-1]
        state.bb_lower = bb_l[-1] if not np.isnan(bb_l[-1]) else closes[-1] - state.atr
        state.bb_width = (state.bb_upper - state.bb_lower) / state.bb_middle if state.bb_middle > 0 else 0.0
        bb_range = state.bb_upper - state.bb_lower
        state.bb_percent_b = (closes[-1] - state.bb_lower) / bb_range if bb_range > 0 else 0.5

        # VWAP
        state.vwap, state.vwap_upper, state.vwap_lower = compute_vwap(candles)

        # Volume
        vol_sma = compute_sma(volumes, self.volume_sma_period)
        state.volume_sma = vol_sma[-1] if not np.isnan(vol_sma[-1]) else volumes[-1]
        state.volume_ratio = volumes[-1] / state.volume_sma if state.volume_sma > 0 else 1.0

        # Delta (order flow)
        last = candles[-1]
        state.delta = last.delta
        self._cumulative_delta += last.delta
        state.cumulative_delta = self._cumulative_delta

        # Price action
        state.candle_body_ratio = last.body_ratio

        # Consecutive direction
        consec = 0
        for i in range(n - 1, -1, -1):
            if candles[i].is_bullish:
                if consec <= 0 and i < n - 1:
                    break
                consec += 1
            elif candles[i].is_bearish:
                if consec >= 0 and i < n - 1:
                    break
                consec -= 1
            else:
                break
        state.consecutive_direction = consec

        # Trend direction from EMAs
        if state.ema_fast > state.ema_slow > state.ema_trend:
            state.trend_direction = 1
        elif state.ema_fast < state.ema_slow < state.ema_trend:
            state.trend_direction = -1
        else:
            state.trend_direction = 0

        # Momentum score: composite of RSI, EMA alignment, delta
        rsi_score = (state.rsi - 50) / 50  # -1 to +1
        ema_score = 1.0 if state.trend_direction == 1 else -1.0 if state.trend_direction == -1 else 0.0
        delta_score = np.clip(state.delta / max(state.volume_sma, 1), -1, 1)
        state.momentum_score = 0.4 * rsi_score + 0.35 * ema_score + 0.25 * float(delta_score)

        return state

    def reset_session(self) -> None:
        """Reset session-specific data (VWAP, cumulative delta)."""
        self._cumulative_delta = 0
