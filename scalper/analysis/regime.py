"""Adaptive market regime detection.

Classifies the current market state into one of:
- TRENDING_UP: clear uptrend with momentum
- TRENDING_DOWN: clear downtrend with momentum
- RANGING: sideways, mean-reverting price action
- VOLATILE: high volatility, erratic moves
- LOW_VOLATILITY: tight range, potential breakout building

The regime determines which strategy parameters the agent uses.
NQ typically shifts between these regimes multiple times per session.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass

from scalper.models import Candle, MarketRegime
from scalper.analysis.indicators import IndicatorState, compute_atr, compute_ema


@dataclass
class RegimeState:
    """Current regime classification with confidence."""
    regime: MarketRegime
    confidence: float  # 0-1, how sure we are about this regime
    duration: int  # how many candles in this regime
    volatility_percentile: float  # current vol vs recent history
    trend_strength: float  # 0-1
    mean_reversion_score: float  # 0-1, how mean-reverting recent action is


class RegimeDetector:
    """Detects market regime using multiple signals.

    Uses a Bayesian-inspired approach:
    1. ATR ratio (fast/slow) for volatility regime shifts
    2. ADX-like directional measurement for trend detection
    3. Price oscillation around VWAP/EMA for mean-reversion detection
    4. Bollinger Band width for volatility compression/expansion
    5. Consecutive candle direction for momentum confirmation
    """

    def __init__(
        self,
        lookback: int = 20,
        atr_fast: int = 5,
        atr_slow: int = 20,
        vol_history: int = 100,
    ):
        self.lookback = lookback
        self.atr_fast = atr_fast
        self.atr_slow = atr_slow
        self.vol_history = vol_history

        self._current_regime = MarketRegime.RANGING
        self._regime_duration = 0
        self._atr_history: deque[float] = deque(maxlen=vol_history)
        self._regime_history: deque[MarketRegime] = deque(maxlen=vol_history)

    def detect(self, candles: list[Candle], indicators: IndicatorState) -> RegimeState:
        """Classify current market regime."""
        if len(candles) < self.lookback:
            return RegimeState(
                regime=MarketRegime.RANGING,
                confidence=0.3,
                duration=0,
                volatility_percentile=0.5,
                trend_strength=0.0,
                mean_reversion_score=0.5,
            )

        recent = candles[-self.lookback:]
        closes = np.array([c.close for c in recent])
        highs = np.array([c.high for c in recent])
        lows = np.array([c.low for c in recent])

        # --- Volatility analysis ---
        atr_ratio = indicators.atr_fast / indicators.atr if indicators.atr > 0 else 1.0
        self._atr_history.append(indicators.atr)
        vol_percentile = self._volatility_percentile(indicators.atr)

        # --- Trend analysis ---
        trend_strength = self._compute_trend_strength(closes, indicators)

        # --- Mean reversion analysis ---
        mr_score = self._compute_mean_reversion(closes, indicators)

        # --- Directional analysis (simplified ADX) ---
        direction = self._compute_direction(highs, lows, closes)

        # --- Regime classification ---
        regime, confidence = self._classify(
            atr_ratio=atr_ratio,
            vol_percentile=vol_percentile,
            trend_strength=trend_strength,
            mr_score=mr_score,
            direction=direction,
            indicators=indicators,
        )

        # Track regime duration
        if regime == self._current_regime:
            self._regime_duration += 1
        else:
            self._current_regime = regime
            self._regime_duration = 1

        self._regime_history.append(regime)

        return RegimeState(
            regime=regime,
            confidence=confidence,
            duration=self._regime_duration,
            volatility_percentile=vol_percentile,
            trend_strength=trend_strength,
            mean_reversion_score=mr_score,
        )

    def _volatility_percentile(self, current_atr: float) -> float:
        """Where current ATR sits in recent history."""
        if len(self._atr_history) < 10:
            return 0.5
        arr = np.array(self._atr_history)
        return float(np.sum(arr < current_atr) / len(arr))

    def _compute_trend_strength(self, closes: np.ndarray, indicators: IndicatorState) -> float:
        """Measure trend strength 0-1."""
        n = len(closes)
        if n < 5:
            return 0.0

        # Linear regression R-squared
        x = np.arange(n)
        corr = np.corrcoef(x, closes)[0, 1]
        r_squared = corr ** 2 if not np.isnan(corr) else 0.0

        # EMA alignment score
        ema_aligned = 0.0
        if indicators.ema_fast > indicators.ema_slow > indicators.ema_trend:
            ema_aligned = 1.0
        elif indicators.ema_fast < indicators.ema_slow < indicators.ema_trend:
            ema_aligned = 1.0
        elif indicators.ema_fast > indicators.ema_slow or indicators.ema_fast < indicators.ema_slow:
            ema_aligned = 0.5

        # Price making higher highs/lows or lower highs/lows
        hh_ll = self._count_hh_ll(closes)

        return float(np.clip(0.4 * r_squared + 0.3 * ema_aligned + 0.3 * hh_ll, 0, 1))

    def _count_hh_ll(self, closes: np.ndarray) -> float:
        """Count higher-highs/lower-lows pattern strength."""
        if len(closes) < 4:
            return 0.0

        # Use 5-bar swing points
        window = min(5, len(closes) // 3)
        highs_count = 0
        lows_count = 0
        total = 0

        for i in range(window, len(closes) - window):
            total += 1
            if closes[i] == max(closes[i - window:i + window + 1]):
                highs_count += 1
            if closes[i] == min(closes[i - window:i + window + 1]):
                lows_count += 1

        # Strong trend = swings consistently in one direction
        swing_ratio = max(highs_count, lows_count) / max(total, 1)
        return float(np.clip(swing_ratio * 2, 0, 1))

    def _compute_mean_reversion(self, closes: np.ndarray, indicators: IndicatorState) -> float:
        """Score how mean-reverting the recent action is (0=trending, 1=mean-reverting)."""
        if len(closes) < 10:
            return 0.5

        # Count zero-crossings around the mean (more crossings = more mean-reverting)
        mean = np.mean(closes)
        centered = closes - mean
        crossings = np.sum(np.diff(np.sign(centered)) != 0)
        max_crossings = len(closes) - 1
        crossing_ratio = crossings / max_crossings

        # Price staying within BB bands
        bb_containment = 1.0 - abs(indicators.bb_percent_b - 0.5) * 2

        # Hurst exponent approximation (H < 0.5 = mean reverting)
        hurst = self._hurst_approx(closes)
        hurst_score = max(0, 1.0 - 2 * hurst)  # H=0.5 -> 0, H=0 -> 1

        return float(np.clip(0.4 * crossing_ratio + 0.3 * bb_containment + 0.3 * hurst_score, 0, 1))

    def _hurst_approx(self, series: np.ndarray) -> float:
        """Quick Hurst exponent approximation using rescaled range."""
        n = len(series)
        if n < 20:
            return 0.5

        # Simple R/S analysis on one window
        returns = np.diff(np.log(np.maximum(series, 1)))
        mean_r = np.mean(returns)
        deviate = np.cumsum(returns - mean_r)
        r = np.max(deviate) - np.min(deviate)
        s = np.std(returns)
        if s == 0 or r == 0:
            return 0.5

        rs = r / s
        hurst = np.log(rs) / np.log(n)
        return float(np.clip(hurst, 0, 1))

    def _compute_direction(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Directional movement score: -1 (strong down) to +1 (strong up).

        Weights recent candles MORE heavily so the regime flips faster
        when price reverses.
        """
        if len(closes) < 3:
            return 0.0

        # Short-term direction (last 5 candles) - most important
        if len(closes) >= 5:
            short_change = closes[-1] - closes[-5]
            short_range = np.sum(highs[-5:] - lows[-5:])
            short_eff = short_change / short_range if short_range > 0 else 0
        else:
            short_eff = 0

        # Medium-term direction (full lookback)
        net_change = closes[-1] - closes[0]
        total_range = np.sum(highs - lows)
        med_eff = net_change / total_range if total_range > 0 else 0

        # Weight: 70% short-term, 30% medium-term
        combined = 0.7 * short_eff + 0.3 * med_eff
        return float(np.clip(combined * 3, -1, 1))

    def _classify(
        self,
        atr_ratio: float,
        vol_percentile: float,
        trend_strength: float,
        mr_score: float,
        direction: float,
        indicators: IndicatorState,
    ) -> tuple[MarketRegime, float]:
        """Classify regime and assign confidence."""

        # Score each regime
        scores = {}

        # TRENDING_UP: strong upward direction + trend strength
        scores[MarketRegime.TRENDING_UP] = (
            max(0, direction) * 0.6 +
            trend_strength * 0.2 +
            (1 if indicators.trend_direction == 1 else 0) * 0.2
        )

        # TRENDING_DOWN: strong downward direction + trend strength
        scores[MarketRegime.TRENDING_DOWN] = (
            max(0, -direction) * 0.6 +
            trend_strength * 0.2 +
            (1 if indicators.trend_direction == -1 else 0) * 0.2
        )

        # RANGING: high mean reversion + moderate volatility + low trend
        scores[MarketRegime.RANGING] = (
            mr_score * 0.5 +
            (1 - trend_strength) * 0.3 +
            (1 - abs(vol_percentile - 0.5) * 2) * 0.2  # prefer middle volatility
        )

        # VOLATILE: high volatility + low trend (erratic)
        scores[MarketRegime.VOLATILE] = (
            vol_percentile * 0.4 +
            max(0, atr_ratio - 1) * 0.3 +  # fast ATR > slow ATR
            (1 - trend_strength) * 0.3
        )

        # LOW_VOLATILITY: very low volatility, compression
        scores[MarketRegime.LOW_VOLATILITY] = (
            (1 - vol_percentile) * 0.5 +
            (1 if indicators.bb_width < 0.005 else 0) * 0.3 +
            (1 - trend_strength) * 0.2
        )

        # Pick highest scoring regime
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.3

        # Reduced hysteresis: allow faster regime switches
        # Only hold current regime if the new one is very weakly indicated
        if best_regime != self._current_regime and confidence < 0.25:
            return self._current_regime, confidence

        return best_regime, float(np.clip(confidence, 0, 1))

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    @property
    def regime_history(self) -> list[MarketRegime]:
        return list(self._regime_history)
