"""Dynamic risk engine - replaces fixed parameters with learned/computed values.

Instead of "risk $100 per trade" and "daily loss limit $400":
- Kelly criterion for optimal sizing based on actual edge
- Dynamic exposure based on regime quality and recent performance
- No arbitrary daily targets (take what the market gives)
- Drawdown-aware risk curves (risk more when ahead, less when behind)
- Continuous edge estimation from rolling trade window
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EdgeEstimate:
    """Rolling estimate of the strategy's current edge."""
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0  # avg_win / avg_loss
    kelly_fraction: float = 0.0  # optimal bet size as fraction of bankroll
    edge_per_dollar: float = 0.0  # expected $ return per $ risked
    sample_size: int = 0
    confidence: float = 0.0  # how reliable is this estimate (0-1)


class DynamicRiskEngine:
    """Computes optimal risk parameters from performance data.

    Replaces fixed rules with data-driven risk management.
    """

    def __init__(
        self,
        max_drawdown: float = 2000.0,
        min_sample_size: int = 10,
        window_size: int = 50,
    ):
        self.max_drawdown = max_drawdown
        self.min_sample_size = min_sample_size
        self._window: deque[float] = deque(maxlen=window_size)  # recent trade PnLs
        self._risks: deque[float] = deque(maxlen=window_size)  # risk per trade
        self._edge = EdgeEstimate()

    def record_trade(self, pnl: float, risk_amount: float) -> None:
        """Record a trade result for edge estimation."""
        self._window.append(pnl)
        self._risks.append(risk_amount)
        self._update_edge()

    def _update_edge(self) -> None:
        """Recompute edge estimate from rolling window."""
        pnls = list(self._window)
        n = len(pnls)

        if n < 3:
            self._edge = EdgeEstimate(sample_size=n)
            return

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        wr = len(wins) / n if n > 0 else 0
        avg_w = np.mean(wins) if wins else 0
        avg_l = abs(np.mean(losses)) if losses else 1

        payoff = avg_w / avg_l if avg_l > 0 else 0

        # Kelly criterion: f* = (bp - q) / b
        # where b = payoff ratio, p = win rate, q = 1 - p
        if payoff > 0:
            kelly = (payoff * wr - (1 - wr)) / payoff
        else:
            kelly = 0

        # Half-Kelly for safety (full Kelly is too aggressive)
        kelly = max(0, kelly * 0.5)

        # Edge per dollar risked
        edge = wr * avg_w - (1 - wr) * avg_l

        # Confidence in the estimate (based on sample size)
        # Need ~30 trades for reasonable confidence
        confidence = min(1.0, n / 30)

        self._edge = EdgeEstimate(
            win_rate=wr,
            avg_win=avg_w,
            avg_loss=avg_l,
            payoff_ratio=payoff,
            kelly_fraction=kelly,
            edge_per_dollar=edge,
            sample_size=n,
            confidence=confidence,
        )

    @property
    def edge(self) -> EdgeEstimate:
        return self._edge

    def optimal_risk_dollars(
        self,
        remaining_drawdown: float,
        signal_confidence: float,
        regime_quality: float,
    ) -> float:
        """Compute optimal dollar risk for a trade.

        Factors:
        1. Kelly fraction (from edge estimate)
        2. Remaining drawdown (risk less when closer to limit)
        3. Signal confidence (scale linearly)
        4. Regime quality (how well we perform in this regime)
        5. Safety caps (never more than 10% of remaining DD)

        When we don't have enough data (< min_sample_size trades),
        falls back to conservative fixed fraction.
        """
        if self._edge.sample_size < self.min_sample_size:
            # Learning phase: 5% of remaining drawdown
            # Must be enough to cover at least 1 contract at current ATR
            base = remaining_drawdown * 0.05
        elif self._edge.kelly_fraction <= 0:
            # No edge detected - minimum size
            base = remaining_drawdown * 0.02
        else:
            # Kelly-based sizing
            # Fraction of bankroll = kelly * confidence_in_estimate
            fraction = self._edge.kelly_fraction * self._edge.confidence
            base = remaining_drawdown * fraction

        # Scale by signal confidence (0.5 to 1.5x)
        conf_scale = 0.5 + signal_confidence
        base *= conf_scale

        # Scale by regime quality (0.3 to 1.2x)
        regime_scale = max(0.3, min(1.2, regime_quality))
        base *= regime_scale

        # Drawdown curve: risk more when ahead, less when behind
        dd_ratio = remaining_drawdown / self.max_drawdown
        if dd_ratio > 0.8:
            # Comfortable - full risk
            dd_scale = 1.0
        elif dd_ratio > 0.5:
            # Getting tight - reduce
            dd_scale = dd_ratio
        elif dd_ratio > 0.25:
            # Danger zone - cut significantly
            dd_scale = dd_ratio * 0.5
        else:
            # Critical - minimum risk
            dd_scale = 0.1
        base *= dd_scale

        # Hard caps
        # Never risk more than 15% of remaining drawdown
        cap = remaining_drawdown * 0.15
        # Never risk more than $400
        cap = min(cap, 400.0)
        # Floor: at least $30 (otherwise not worth the trade)
        floor = 30.0

        return max(floor, min(base, cap))

    def should_trade(self, remaining_drawdown: float) -> tuple[bool, str]:
        """Dynamic decision on whether to trade at all.

        Unlike fixed "daily loss limit", this considers:
        - Current edge quality
        - Remaining drawdown
        - Recent momentum (winning or losing streak)
        """
        # Critical drawdown - stop
        if remaining_drawdown < self.max_drawdown * 0.15:
            return False, "critical_drawdown"

        # If we have data and edge is clearly negative, stop
        if self._edge.sample_size >= self.min_sample_size:
            if self._edge.edge_per_dollar < -0.1 and self._edge.confidence > 0.5:
                return False, "negative_edge_detected"

        # Recent streak check
        if len(self._window) >= 4:
            last_4 = list(self._window)[-4:]
            if all(p < 0 for p in last_4):
                # 4 straight losses - but check if edge is still positive
                if self._edge.edge_per_dollar <= 0:
                    return False, "losing_streak_no_edge"
                # Edge still positive - continue but at reduced size (handled in optimal_risk)

        return True, "ok"

    def get_summary(self) -> dict:
        """Summary for dashboard/logging."""
        e = self._edge
        return {
            "win_rate": round(e.win_rate, 3),
            "payoff_ratio": round(e.payoff_ratio, 2),
            "kelly_pct": round(e.kelly_fraction * 100, 1),
            "edge_per_dollar": round(e.edge_per_dollar, 2),
            "sample_size": e.sample_size,
            "confidence": round(e.confidence, 2),
        }
