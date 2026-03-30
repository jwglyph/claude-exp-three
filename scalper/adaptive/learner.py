"""Online adaptive learning module.

Continuously adjusts strategy parameters based on recent trade performance.
Uses an exponentially-weighted approach so recent results matter more.

Key adaptations:
1. Confidence threshold: raise if too many losing trades, lower if missing winners
2. Regime-specific win rates: track which regimes we perform best in
3. Session performance: learn which time sessions are most profitable
4. Stop/target ratios: adjust based on MAE/MFE analysis
5. Signal weight tuning: strengthen signals that lead to winners
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import structlog

from scalper.config import ScalperConfig
from scalper.models import MarketRegime, Side, TradeResult

logger = structlog.get_logger()


@dataclass
class PerformanceStats:
    """Performance tracking for a specific context (regime, session, etc.)."""
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    total_win_pnl: float = 0.0
    total_loss_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    avg_mfe: float = 0.0  # average max favorable excursion
    avg_mae: float = 0.0  # average max adverse excursion
    avg_holding_time: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        return abs(self.total_win_pnl / self.total_loss_pnl) if self.total_loss_pnl != 0 else 0.0

    @property
    def expectancy(self) -> float:
        """Expected $ per trade."""
        return self.total_pnl / self.trades if self.trades > 0 else 0.0


@dataclass
class AdaptiveParameters:
    """Current adapted parameters."""
    confidence_threshold: float = 0.55
    regime_weights: dict[str, float] = field(default_factory=lambda: {
        r.value: 1.0 for r in MarketRegime
    })
    stop_multiplier: float = 1.0  # scale on computed stop distance
    target_multiplier: float = 1.0  # scale on computed target distance
    signal_weights: dict[str, float] = field(default_factory=dict)
    session_weights: dict[str, float] = field(default_factory=lambda: {
        "asian": 0.5,
        "london": 0.8,
        "ny_open": 1.0,
        "ny_afternoon": 0.9,
        "ny_close": 0.5,
    })


class AdaptiveLearner:
    """Learns from trade results to adapt parameters in real-time."""

    def __init__(self, config: ScalperConfig):
        self.config = config
        self.params = AdaptiveParameters(
            confidence_threshold=config.min_confidence,
        )
        self._alpha = config.adapt_rate  # learning rate
        self._window = config.adapt_window
        self._recent_trades: list[TradeResult] = []
        self._regime_stats: dict[str, PerformanceStats] = defaultdict(PerformanceStats)
        self._signal_outcomes: dict[str, list[float]] = defaultdict(list)
        self._total_trades = 0

    def record_trade(self, trade: TradeResult, signal_reasons: list[str] | None = None) -> None:
        """Record a trade outcome and update adaptive parameters."""
        self._recent_trades.append(trade)
        if len(self._recent_trades) > self._window:
            self._recent_trades = self._recent_trades[-self._window:]

        self._total_trades += 1

        # Update regime stats
        regime_key = trade.regime.value
        stats = self._regime_stats[regime_key]
        stats.trades += 1
        stats.total_pnl += trade.pnl
        if trade.pnl >= 0:
            stats.wins += 1
            stats.total_win_pnl += trade.pnl
        else:
            stats.total_loss_pnl += trade.pnl
        stats.avg_mfe = stats.avg_mfe * 0.9 + trade.max_favorable * 0.1
        stats.avg_mae = stats.avg_mae * 0.9 + trade.max_adverse * 0.1
        stats.avg_holding_time = stats.avg_holding_time * 0.9 + (trade.exit_time - trade.entry_time) * 0.1

        # Update signal-level outcomes
        if signal_reasons:
            for reason in signal_reasons:
                self._signal_outcomes[reason].append(trade.pnl)
                # Keep only recent outcomes
                if len(self._signal_outcomes[reason]) > self._window:
                    self._signal_outcomes[reason] = self._signal_outcomes[reason][-self._window:]

        # Adapt parameters
        self._adapt()

    def _adapt(self) -> None:
        """Run adaptation logic."""
        if len(self._recent_trades) < 5:
            return

        recent = self._recent_trades[-self._window:]

        # --- Adapt confidence threshold ---
        self._adapt_confidence(recent)

        # --- Adapt regime weights ---
        self._adapt_regime_weights()

        # --- Adapt stop/target multipliers ---
        self._adapt_stop_target(recent)

        # --- Adapt signal weights ---
        self._adapt_signal_weights()

        logger.debug(
            "parameters_adapted",
            confidence=round(self.params.confidence_threshold, 3),
            stop_mult=round(self.params.stop_multiplier, 3),
            target_mult=round(self.params.target_multiplier, 3),
        )

    def _adapt_confidence(self, recent: list[TradeResult]) -> None:
        """Adjust confidence threshold based on recent win rate."""
        win_rate = sum(1 for t in recent if t.pnl >= 0) / len(recent)

        if win_rate < 0.4:
            # Too many losses: raise the bar
            self.params.confidence_threshold = min(
                0.8,
                self.params.confidence_threshold + self._alpha * 0.05,
            )
        elif win_rate > 0.6:
            # Winning well: can be slightly more permissive
            self.params.confidence_threshold = max(
                0.45,
                self.params.confidence_threshold - self._alpha * 0.02,
            )

    def _adapt_regime_weights(self) -> None:
        """Weight regimes by their profitability."""
        for regime_key, stats in self._regime_stats.items():
            if stats.trades < 3:
                continue

            if stats.win_rate > 0.55 and stats.expectancy > 0:
                # Performing well in this regime: increase weight
                self.params.regime_weights[regime_key] = min(
                    1.5,
                    self.params.regime_weights.get(regime_key, 1.0) + self._alpha * 0.1,
                )
            elif stats.win_rate < 0.4 or stats.expectancy < 0:
                # Poor performance: reduce weight
                self.params.regime_weights[regime_key] = max(
                    0.3,
                    self.params.regime_weights.get(regime_key, 1.0) - self._alpha * 0.1,
                )

    def _adapt_stop_target(self, recent: list[TradeResult]) -> None:
        """Adjust stop/target based on MAE/MFE analysis.

        If we're consistently getting stopped out before price reaches target:
        -> widen stops slightly
        If we're leaving money on the table (MFE >> actual profit):
        -> widen targets or use trailing stops more aggressively
        """
        if len(recent) < 10:
            return

        # Analyze stop-outs
        stop_outs = [t for t in recent if t.exit_reason in ("stop_hit", "stop_or_target") and t.pnl < 0]
        if stop_outs:
            # Average how far past our stop did price eventually go favorably
            avg_mfe_of_losers = np.mean([t.max_favorable for t in stop_outs])
            if avg_mfe_of_losers > 50:  # $50+ favorable before stopping out
                # Stops too tight
                self.params.stop_multiplier = min(
                    1.5,
                    self.params.stop_multiplier + self._alpha * 0.05,
                )

        # Analyze winners
        winners = [t for t in recent if t.pnl > 0]
        if winners:
            avg_mfe = np.mean([t.max_favorable for t in winners])
            avg_pnl = np.mean([t.pnl for t in winners])
            if avg_mfe > avg_pnl * 2:
                # Leaving a lot on the table
                self.params.target_multiplier = min(
                    1.5,
                    self.params.target_multiplier + self._alpha * 0.03,
                )
            elif avg_mfe < avg_pnl * 1.2:
                # Targets are about right
                self.params.target_multiplier = max(
                    0.8,
                    self.params.target_multiplier - self._alpha * 0.01,
                )

    def _adapt_signal_weights(self) -> None:
        """Adjust signal weights based on which signal reasons lead to winners."""
        for reason, outcomes in self._signal_outcomes.items():
            if len(outcomes) < 5:
                continue

            avg_pnl = np.mean(outcomes)
            win_rate = sum(1 for o in outcomes if o >= 0) / len(outcomes)

            if win_rate > 0.6 and avg_pnl > 0:
                self.params.signal_weights[reason] = min(
                    1.5,
                    self.params.signal_weights.get(reason, 1.0) + self._alpha * 0.05,
                )
            elif win_rate < 0.35:
                self.params.signal_weights[reason] = max(
                    0.3,
                    self.params.signal_weights.get(reason, 1.0) - self._alpha * 0.05,
                )

    def get_regime_weight(self, regime: MarketRegime) -> float:
        """Get current adaptive weight for a regime."""
        return self.params.regime_weights.get(regime.value, 1.0)

    def get_confidence_threshold(self) -> float:
        return self.params.confidence_threshold

    def get_stats_summary(self) -> dict:
        """Get summary of learning state."""
        return {
            "total_trades": self._total_trades,
            "recent_window": len(self._recent_trades),
            "confidence_threshold": self.params.confidence_threshold,
            "stop_multiplier": self.params.stop_multiplier,
            "target_multiplier": self.params.target_multiplier,
            "regime_weights": dict(self.params.regime_weights),
            "regime_stats": {
                k: {
                    "trades": v.trades,
                    "win_rate": round(v.win_rate, 3),
                    "expectancy": round(v.expectancy, 2),
                    "profit_factor": round(v.profit_factor, 2),
                }
                for k, v in self._regime_stats.items()
                if v.trades > 0
            },
        }
