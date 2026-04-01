"""Adaptive signal generator for 1-minute NQ scalping.

Generates trading signals by combining multiple confluence factors.
Adapts its behavior based on detected market regime:

TRENDING: Trade pullbacks to EMA in trend direction, breakout continuations
RANGING: Fade moves at BB/support/resistance, mean revert to VWAP
VOLATILE: Only high-confidence setups, wider stops, reduced size
LOW_VOL: Watch for breakout, trade compression breakouts

Each signal gets a confidence score (0-1) based on how many factors align.
"""

from __future__ import annotations

import numpy as np
import structlog

from scalper.config import ScalperConfig
from scalper.models import Candle, MarketRegime, Side, Signal, SignalType
from scalper.analysis.indicators import IndicatorState
from scalper.analysis.regime import RegimeState

logger = structlog.get_logger()


class SignalGenerator:
    """Generates entry/exit signals with confidence scoring."""

    def __init__(self, config: ScalperConfig):
        self.config = config
        self._signal_count = 0

    def generate(
        self,
        candles: list[Candle],
        indicators: IndicatorState,
        regime: RegimeState,
    ) -> Signal | None:
        """Evaluate current bar and generate a signal if conditions met."""
        if len(candles) < self.config.warmup_candles:
            return None

        last = candles[-1]
        prev = candles[-2]
        price = last.close

        # Collect bullish and bearish factors with weights
        bull_factors: list[tuple[str, float]] = []
        bear_factors: list[tuple[str, float]] = []

        # --- EMA alignment ---
        self._score_ema(indicators, bull_factors, bear_factors)

        # --- RSI ---
        self._score_rsi(indicators, regime, bull_factors, bear_factors)

        # --- Bollinger Band position ---
        self._score_bollinger(indicators, price, regime, bull_factors, bear_factors)

        # --- VWAP relationship ---
        self._score_vwap(indicators, price, bull_factors, bear_factors)

        # --- Candle patterns ---
        self._score_candle_patterns(candles, bull_factors, bear_factors)

        # --- Volume confirmation ---
        self._score_volume(indicators, last, bull_factors, bear_factors)

        # --- Momentum ---
        self._score_momentum(indicators, bull_factors, bear_factors)

        # --- Regime-specific signals ---
        self._score_regime_specific(candles, indicators, regime, price, bull_factors, bear_factors)

        # --- Compute net signal ---
        bull_score = sum(w for _, w in bull_factors)
        bear_score = sum(w for _, w in bear_factors)

        bull_reasons = [r for r, _ in bull_factors]
        bear_reasons = [r for r, _ in bear_factors]

        # Need some factors on at least one side
        total = bull_score + bear_score
        if total == 0:
            return None

        # Confidence: how dominant is the winning side?
        # Use the winning side's score directly, scaled by separation
        # This allows high confidence when one side has multiple confirming factors
        net = bull_score - bear_score
        dominant = max(bull_score, bear_score)
        separation = abs(net) / total  # 0=tied, 1=one-sided

        # Confidence = dominant score * separation boost
        # If bull=0.5, bear=0.1: dominant=0.5, separation=0.67 → conf=0.5*0.67+0.5*0.5=0.58
        # If bull=0.7, bear=0.1: dominant=0.7, separation=0.75 → conf=0.7*0.75+0.5*0.7=0.88 (capped)
        confidence = separation * 0.5 + dominant * 0.5
        confidence = self._adjust_confidence(confidence, regime, indicators)

        # Determine signal type and side
        if net > 0 and confidence >= self.config.min_confidence:
            side = Side.LONG
            reasons = bull_reasons
            if confidence >= 0.75:
                sig_type = SignalType.STRONG_LONG
            elif confidence >= 0.6:
                sig_type = SignalType.LONG
            else:
                sig_type = SignalType.WEAK_LONG
        elif net < 0 and confidence >= self.config.min_confidence:
            side = Side.SHORT
            reasons = bear_reasons
            if confidence >= 0.75:
                sig_type = SignalType.STRONG_SHORT
            elif confidence >= 0.6:
                sig_type = SignalType.SHORT
            else:
                sig_type = SignalType.WEAK_SHORT
        else:
            return None

        self._signal_count += 1

        # Entry at current close (will execute at next bar open in live)
        entry = price

        # Stop and target will be computed by risk manager, but provide defaults
        atr = max(indicators.atr, 1.0)
        if side == Side.LONG:
            stop = entry - atr * 1.5
            target = entry + atr * 2.5
        else:
            stop = entry + atr * 1.5
            target = entry - atr * 2.5

        signal = Signal(
            timestamp=last.timestamp,
            signal_type=sig_type,
            confidence=confidence,
            side=side,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            regime=regime.regime,
            reasons=reasons,
        )

        logger.info(
            "signal_generated",
            type=sig_type.value,
            confidence=round(confidence, 3),
            side=side.value,
            entry=entry,
            regime=regime.regime.value,
            reasons=reasons[:5],
        )

        return signal

    def _score_ema(
        self,
        ind: IndicatorState,
        bull: list[tuple[str, float]],
        bear: list[tuple[str, float]],
    ) -> None:
        """Score EMA alignment."""
        if ind.ema_fast > ind.ema_slow > ind.ema_trend:
            bull.append(("ema_bullish_stack", 0.15))
        elif ind.ema_fast < ind.ema_slow < ind.ema_trend:
            bear.append(("ema_bearish_stack", 0.15))

        # Price relative to fast EMA
        if ind.ema_fast > 0:
            ema_dist = (ind.ema_fast - ind.ema_slow) / ind.ema_fast
            if ema_dist > 0.001:
                bull.append(("ema_spread_bullish", min(0.1, ema_dist * 20)))
            elif ema_dist < -0.001:
                bear.append(("ema_spread_bearish", min(0.1, abs(ema_dist) * 20)))

    def _score_rsi(
        self,
        ind: IndicatorState,
        regime: RegimeState,
        bull: list[tuple[str, float]],
        bear: list[tuple[str, float]],
    ) -> None:
        """Score RSI - context-dependent on regime."""
        if regime.regime == MarketRegime.RANGING:
            # In range: RSI extremes are mean-reversion signals
            if ind.rsi < 30:
                bull.append(("rsi_oversold_range", 0.2))
            elif ind.rsi > 70:
                bear.append(("rsi_overbought_range", 0.2))
            elif ind.rsi < 40:
                bull.append(("rsi_low_range", 0.1))
            elif ind.rsi > 60:
                bear.append(("rsi_high_range", 0.1))
        else:
            # In trend: RSI confirms momentum
            if 50 < ind.rsi < 70:
                bull.append(("rsi_bullish_momentum", 0.1))
            elif 30 < ind.rsi < 50:
                bear.append(("rsi_bearish_momentum", 0.1))
            # Divergence: RSI extreme opposite to trend is warning
            if ind.rsi > 80 and regime.regime == MarketRegime.TRENDING_UP:
                bear.append(("rsi_overbought_warning", 0.05))
            elif ind.rsi < 20 and regime.regime == MarketRegime.TRENDING_DOWN:
                bull.append(("rsi_oversold_warning", 0.05))

    def _score_bollinger(
        self,
        ind: IndicatorState,
        price: float,
        regime: RegimeState,
        bull: list[tuple[str, float]],
        bear: list[tuple[str, float]],
    ) -> None:
        """Score Bollinger Band signals."""
        if regime.regime == MarketRegime.RANGING:
            # Fade BB touches in ranges
            if ind.bb_percent_b < 0.05:
                bull.append(("bb_lower_touch_range", 0.2))
            elif ind.bb_percent_b > 0.95:
                bear.append(("bb_upper_touch_range", 0.2))
            elif ind.bb_percent_b < 0.2:
                bull.append(("bb_lower_zone_range", 0.1))
            elif ind.bb_percent_b > 0.8:
                bear.append(("bb_upper_zone_range", 0.1))
        else:
            # In trends: BB breakouts confirm
            if ind.bb_percent_b > 1.0:
                bull.append(("bb_upper_breakout", 0.1))
            elif ind.bb_percent_b < 0.0:
                bear.append(("bb_lower_breakout", 0.1))

        # BB squeeze (low volatility compression)
        if ind.bb_width < 0.003:
            # Don't trade the squeeze, but note it
            pass

    def _score_vwap(
        self,
        ind: IndicatorState,
        price: float,
        bull: list[tuple[str, float]],
        bear: list[tuple[str, float]],
    ) -> None:
        """Score VWAP relationship."""
        if ind.vwap <= 0:
            return

        vwap_dist = (price - ind.vwap) / ind.vwap

        # Price above VWAP = bullish bias
        if vwap_dist > 0.001:
            bull.append(("above_vwap", 0.1))
        elif vwap_dist < -0.001:
            bear.append(("below_vwap", 0.1))

        # Bounce off VWAP bands
        if price <= ind.vwap_lower * 1.001 and price > ind.vwap_lower * 0.999:
            bull.append(("vwap_lower_bounce", 0.15))
        elif price >= ind.vwap_upper * 0.999 and price < ind.vwap_upper * 1.001:
            bear.append(("vwap_upper_bounce", 0.15))

    def _score_candle_patterns(
        self,
        candles: list[Candle],
        bull: list[tuple[str, float]],
        bear: list[tuple[str, float]],
    ) -> None:
        """Score candle pattern signals."""
        if len(candles) < 3:
            return

        last = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3]

        # Engulfing
        if (last.is_bullish and prev.is_bearish and
                last.body > prev.body and last.close > prev.open and last.open < prev.close):
            bull.append(("bullish_engulfing", 0.2))
        elif (last.is_bearish and prev.is_bullish and
                last.body > prev.body and last.close < prev.open and last.open > prev.close):
            bear.append(("bearish_engulfing", 0.2))

        # Pin bar / hammer
        if last.lower_wick > last.body * 2 and last.upper_wick < last.body * 0.5:
            bull.append(("hammer", 0.15))
        elif last.upper_wick > last.body * 2 and last.lower_wick < last.body * 0.5:
            bear.append(("shooting_star", 0.15))

        # Strong directional candle (large body, small wicks)
        if last.body_ratio > 0.7 and last.range > 0:
            if last.is_bullish:
                bull.append(("strong_bull_candle", 0.1))
            else:
                bear.append(("strong_bear_candle", 0.1))

        # Three consecutive candles
        if all(c.is_bullish for c in [prev2, prev, last]):
            bull.append(("three_bull_candles", 0.1))
        elif all(c.is_bearish for c in [prev2, prev, last]):
            bear.append(("three_bear_candles", 0.1))

        # Reversal after extended move
        if abs(candles[-1].close - candles[-1].open) > 0 and len(candles) >= 5:
            recent_dir = sum(1 if c.is_bullish else -1 for c in candles[-5:-1])
            if recent_dir >= 3 and last.is_bearish and last.body_ratio > 0.5:
                bear.append(("reversal_after_bull_run", 0.15))
            elif recent_dir <= -3 and last.is_bullish and last.body_ratio > 0.5:
                bull.append(("reversal_after_bear_run", 0.15))

    def _score_volume(
        self,
        ind: IndicatorState,
        last: Candle,
        bull: list[tuple[str, float]],
        bear: list[tuple[str, float]],
    ) -> None:
        """Score volume confirmation."""
        # High volume confirms direction
        if ind.volume_ratio > 1.5:
            if last.is_bullish:
                bull.append(("high_vol_bullish", 0.1))
            elif last.is_bearish:
                bear.append(("high_vol_bearish", 0.1))

        # Delta (order flow)
        if ind.delta > 0:
            bull.append(("positive_delta", min(0.1, ind.delta / max(ind.volume_sma, 1) * 0.5)))
        elif ind.delta < 0:
            bear.append(("negative_delta", min(0.1, abs(ind.delta) / max(ind.volume_sma, 1) * 0.5)))

        # Volume dry-up (potential reversal)
        if ind.volume_ratio < 0.5:
            if last.is_bullish:
                bear.append(("low_vol_bull_suspect", 0.05))
            elif last.is_bearish:
                bull.append(("low_vol_bear_suspect", 0.05))

    def _score_momentum(
        self,
        ind: IndicatorState,
        bull: list[tuple[str, float]],
        bear: list[tuple[str, float]],
    ) -> None:
        """Score overall momentum."""
        if ind.momentum_score > 0.3:
            bull.append(("momentum_bullish", min(0.15, ind.momentum_score * 0.2)))
        elif ind.momentum_score < -0.3:
            bear.append(("momentum_bearish", min(0.15, abs(ind.momentum_score) * 0.2)))

    def _score_regime_specific(
        self,
        candles: list[Candle],
        ind: IndicatorState,
        regime: RegimeState,
        price: float,
        bull: list[tuple[str, float]],
        bear: list[tuple[str, float]],
    ) -> None:
        """Generate signals based on what price is ACTUALLY doing.

        NOT biased by regime label. Both long and short signals are
        always considered. The regime just adjusts weights slightly.
        """
        last = candles[-1]

        # --- Pullback to EMA (works in any regime) ---
        # Price pulling back to fast EMA from above = potential long
        if price <= ind.ema_fast * 1.001 and price >= ind.ema_slow * 0.999:
            if last.is_bullish:
                weight = 0.2 if regime.regime == MarketRegime.TRENDING_UP else 0.12
                bull.append(("pullback_to_ema_buy", weight))

        # Price rallying to fast EMA from below = potential short
        if price >= ind.ema_fast * 0.999 and price <= ind.ema_slow * 1.001:
            if last.is_bearish:
                weight = 0.2 if regime.regime == MarketRegime.TRENDING_DOWN else 0.12
                bear.append(("pullback_to_ema_sell", weight))

        # --- Range extremes (works in any regime) ---
        recent_high = max(c.high for c in candles[-20:])
        recent_low = min(c.low for c in candles[-20:])
        range_size = recent_high - recent_low
        if range_size > 0:
            pos_in_range = (price - recent_low) / range_size
            if pos_in_range < 0.15:
                bull.append(("near_range_low", 0.15))
            elif pos_in_range > 0.85:
                bear.append(("near_range_high", 0.15))

        # --- Price below BOTH EMAs = bearish structure ---
        if price < ind.ema_fast and price < ind.ema_slow:
            bear.append(("below_both_emas", 0.12))
        elif price > ind.ema_fast and price > ind.ema_slow:
            bull.append(("above_both_emas", 0.12))

        # --- Recent candle direction (last 3 candles) ---
        if len(candles) >= 3:
            recent_3 = candles[-3:]
            bearish_count = sum(1 for c in recent_3 if c.is_bearish)
            bullish_count = sum(1 for c in recent_3 if c.is_bullish)
            if bearish_count >= 3:
                bear.append(("three_bearish", 0.1))
            elif bullish_count >= 3:
                bull.append(("three_bullish", 0.1))

        # --- Compression breakout ---
        if ind.bb_width < 0.003:
            if last.is_bullish and last.close > ind.bb_upper:
                bull.append(("compression_breakout_up", 0.2))
            elif last.is_bearish and last.close < ind.bb_lower:
                    bear.append(("compression_breakout_down", 0.25))

    def _adjust_confidence(
        self,
        raw_confidence: float,
        regime: RegimeState,
        indicators: IndicatorState,
    ) -> float:
        """Adjust confidence. Minimal adjustment - let the factors speak."""
        conf = raw_confidence

        # Slight boost when regime confidence is high (market is clear)
        conf *= (0.9 + 0.2 * regime.confidence)

        # Slight reduction in volatile regime (higher noise)
        if regime.regime == MarketRegime.VOLATILE:
            conf *= 0.9

        # Reduce confidence if price is right at BB middle (no man's land)
        if abs(indicators.bb_percent_b - 0.5) < 0.1:
            conf *= 0.9

        return float(np.clip(conf, 0, 1))
