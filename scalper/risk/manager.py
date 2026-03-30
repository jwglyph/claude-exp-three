"""Risk manager enforcing TopstepX rules and adaptive position sizing.

Key responsibilities:
1. Enforce daily loss limit (hard stop)
2. Track trailing drawdown
3. Dynamic position sizing based on confidence + volatility
4. ATR-based stop placement
5. Trailing stop management
6. Scale risk down after losses, up after consistent wins
7. Session-aware risk (reduce size in thin markets)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

from scalper.config import ScalperConfig, TOPSTEP_RULES
from scalper.models import (
    Candle, MarketRegime, Position, Signal, Side, TradeResult,
)
from scalper.analysis.indicators import IndicatorState
from scalper.analysis.regime import RegimeState

logger = structlog.get_logger()


@dataclass
class RiskState:
    """Current risk accounting."""
    daily_pnl: float = 0.0
    open_pnl: float = 0.0
    total_pnl: float = 0.0
    peak_pnl: float = 0.0
    trailing_drawdown_remaining: float = 0.0
    daily_loss_remaining: float = 0.0
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    risk_multiplier: float = 1.0  # adaptive scaling
    is_locked: bool = False  # true = no more trading today
    lock_reason: str = ""


class RiskManager:
    """Manages all risk for the trading agent."""

    def __init__(self, config: ScalperConfig):
        self.config = config
        self.state = RiskState(
            trailing_drawdown_remaining=config.max_drawdown,
            daily_loss_remaining=config.daily_loss_limit,
        )
        self._trade_history: list[TradeResult] = []
        self._risk_events: list[dict] = []

    def can_trade(self) -> tuple[bool, str]:
        """Check if we're allowed to take a new trade."""
        if self.state.is_locked:
            return False, self.state.lock_reason

        if self.state.daily_pnl <= -self.config.daily_loss_limit * 0.95:
            self._lock("approaching_daily_loss_limit")
            return False, "Daily loss limit approaching"

        if self.state.trailing_drawdown_remaining < self.config.max_risk_per_trade:
            self._lock("trailing_drawdown_critical")
            return False, "Trailing drawdown too tight for any trade"

        # Reduce aggression after consecutive losses
        if self.state.consecutive_losses >= 3:
            if self.state.consecutive_losses >= 5:
                self._lock("consecutive_loss_cooldown")
                return False, "5 consecutive losses - cooling down"
            # Still allowed but will have reduced size

        return True, "ok"

    def compute_position_size(
        self,
        signal: Signal,
        indicators: IndicatorState,
        regime: RegimeState,
    ) -> int:
        """Compute number of contracts based on risk budget.

        Position sizing formula:
        1. Start with risk budget (max_risk_per_trade * risk_multiplier)
        2. Divide by per-contract risk (stop distance * point value)
        3. Cap by max_contracts and available margin
        4. Scale by confidence and regime
        """
        can, reason = self.can_trade()
        if not can:
            return 0

        # Base risk budget
        risk_budget = self.config.max_risk_per_trade * self.state.risk_multiplier

        # Scale by confidence (linear from min_confidence to 1.0)
        confidence_scale = np.clip(
            (signal.confidence - self.config.min_confidence) /
            (1.0 - self.config.min_confidence),
            0.3, 1.0
        )
        risk_budget *= confidence_scale

        # Scale by regime appropriateness
        regime_scale = self._regime_risk_scale(regime.regime, signal.side)
        risk_budget *= regime_scale

        # Scale down if daily profit target reached (protect gains)
        if self.state.daily_pnl > self.config.profit_target_daily:
            risk_budget *= 0.5

        # Per-contract risk
        stop_distance = abs(signal.entry_price - signal.stop_price)
        per_contract_risk = stop_distance * self.config.point_value
        if per_contract_risk <= 0:
            return 0

        # Calculate size
        size = int(risk_budget / per_contract_risk)
        size = max(1, min(size, self.config.max_contracts))

        # Ensure we don't exceed remaining drawdown
        max_from_drawdown = int(self.state.trailing_drawdown_remaining * 0.25 / per_contract_risk)
        size = min(size, max(1, max_from_drawdown))

        # Ensure we don't exceed daily loss remaining
        max_from_daily = int(self.state.daily_loss_remaining * 0.5 / per_contract_risk)
        size = min(size, max(1, max_from_daily))

        return size

    def compute_stop(
        self,
        side: Side,
        entry_price: float,
        indicators: IndicatorState,
        regime: RegimeState,
    ) -> float:
        """Compute adaptive stop loss price.

        Uses ATR-based stops that widen in volatile regimes and tighten in calm ones.
        Also considers recent swing structure.
        """
        atr = indicators.atr
        if atr <= 0:
            atr = 2.0  # fallback for NQ

        # Base stop distance: 1.5x ATR, adjusted by regime
        multiplier = 1.5

        if regime.regime == MarketRegime.VOLATILE:
            multiplier = 2.0  # wider stops in volatile markets
        elif regime.regime == MarketRegime.LOW_VOLATILITY:
            multiplier = 1.0  # tighter stops in calm markets
        elif regime.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            multiplier = 1.5  # standard for trends
        elif regime.regime == MarketRegime.RANGING:
            multiplier = 1.2  # tighter in ranges (expect mean reversion)

        stop_distance = atr * multiplier

        # Enforce min/max stop in ticks
        stop_ticks = stop_distance / self.config.tick_size
        stop_ticks = np.clip(stop_ticks, 8, self.config.max_stop_ticks)  # min 2 points
        stop_distance = stop_ticks * self.config.tick_size

        if side == Side.LONG:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def compute_target(
        self,
        side: Side,
        entry_price: float,
        stop_price: float,
        indicators: IndicatorState,
        regime: RegimeState,
    ) -> float:
        """Compute profit target based on risk:reward ratio and regime."""
        risk = abs(entry_price - stop_price)

        # Minimum RR, but adjust by regime
        rr = self.config.min_rr_ratio

        if regime.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            rr = max(rr, 2.0)  # let winners run in trends
        elif regime.regime == MarketRegime.RANGING:
            rr = max(rr, 1.2)  # tighter targets in ranges
        elif regime.regime == MarketRegime.VOLATILE:
            rr = max(rr, 1.5)

        target_distance = risk * rr

        if side == Side.LONG:
            return entry_price + target_distance
        else:
            return entry_price - target_distance

    def compute_trailing_stop(
        self,
        position: Position,
        current_price: float,
        indicators: IndicatorState,
        regime: RegimeState,
    ) -> float:
        """Update trailing stop for an open position.

        Trail tightens as position becomes profitable:
        - At entry: stop at initial stop
        - At 1R profit: move to breakeven
        - At 2R profit: trail at 1R behind
        - In strong trend: use ATR-based trail
        """
        risk = abs(position.entry_price - position.stop_price)
        if risk <= 0:
            return position.stop_price

        if position.side == Side.LONG:
            unrealized = current_price - position.entry_price
        else:
            unrealized = position.entry_price - current_price

        r_multiple = unrealized / risk if risk > 0 else 0

        # Don't trail until we have some profit
        if r_multiple < 0.5:
            return position.stop_price

        # Move to breakeven at 1R
        if r_multiple >= 1.0:
            be_stop = position.entry_price + (self.config.tick_size * 2 if position.side == Side.LONG else -self.config.tick_size * 2)
        else:
            be_stop = position.stop_price

        # ATR trail
        atr_trail_distance = indicators.atr * 1.2
        if position.side == Side.LONG:
            atr_stop = current_price - atr_trail_distance
            # In strong trend, use EMA as support
            if regime.regime == MarketRegime.TRENDING_UP:
                ema_stop = indicators.ema_fast - self.config.tick_size * 4
                atr_stop = max(atr_stop, ema_stop)
            new_stop = max(be_stop, atr_stop, position.trailing_stop)
        else:
            atr_stop = current_price + atr_trail_distance
            if regime.regime == MarketRegime.TRENDING_DOWN:
                ema_stop = indicators.ema_fast + self.config.tick_size * 4
                atr_stop = min(atr_stop, ema_stop)
            new_stop = min(be_stop, atr_stop, position.trailing_stop) if position.trailing_stop > 0 else min(be_stop, atr_stop)

        return new_stop

    def should_exit(
        self,
        position: Position,
        current_price: float,
        indicators: IndicatorState,
        regime: RegimeState,
    ) -> tuple[bool, str]:
        """Check if position should be exited immediately."""
        # Check stop hit
        if position.side == Side.LONG:
            effective_stop = max(position.stop_price, position.trailing_stop)
            if current_price <= effective_stop:
                return True, "stop_hit"
            if current_price >= position.target_price:
                return True, "target_hit"
        else:
            effective_stop = min(position.stop_price, position.trailing_stop) if position.trailing_stop > 0 else position.stop_price
            if current_price >= effective_stop:
                return True, "stop_hit"
            if current_price <= position.target_price:
                return True, "target_hit"

        # Check if regime flipped against us
        if position.side == Side.LONG and regime.regime == MarketRegime.TRENDING_DOWN and regime.confidence > 0.6:
            if position.unrealized_pnl > 0:
                return True, "regime_flip_with_profit"

        if position.side == Side.SHORT and regime.regime == MarketRegime.TRENDING_UP and regime.confidence > 0.6:
            if position.unrealized_pnl > 0:
                return True, "regime_flip_with_profit"

        # Emergency: daily loss limit about to be hit
        projected_loss = self.state.daily_pnl + position.unrealized_pnl
        if projected_loss <= -self.config.daily_loss_limit * 0.9:
            return True, "risk_limit_emergency"

        return False, ""

    def record_trade(self, trade: TradeResult) -> None:
        """Record a completed trade and update risk state."""
        self._trade_history.append(trade)
        self.state.daily_pnl += trade.pnl
        self.state.total_pnl += trade.pnl
        self.state.trades_today += 1

        # Track peak for trailing drawdown
        if self.state.total_pnl > self.state.peak_pnl:
            self.state.peak_pnl = self.state.total_pnl
        self.state.trailing_drawdown_remaining = self.config.max_drawdown - (self.state.peak_pnl - self.state.total_pnl)

        self.state.daily_loss_remaining = self.config.daily_loss_limit + self.state.daily_pnl

        if trade.pnl >= 0:
            self.state.wins_today += 1
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.losses_today += 1
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

        # Adaptive risk multiplier
        self._update_risk_multiplier()

        logger.info(
            "trade_recorded",
            pnl=trade.pnl,
            daily_pnl=self.state.daily_pnl,
            drawdown_remaining=self.state.trailing_drawdown_remaining,
            risk_multiplier=self.state.risk_multiplier,
        )

    def _update_risk_multiplier(self) -> None:
        """Adapt risk size based on recent performance.

        - After losses: reduce risk (anti-martingale)
        - After wins: gradually increase back to normal
        - After hitting daily target: reduce to protect profits
        """
        base = 1.0

        # Consecutive loss penalty
        if self.state.consecutive_losses >= 2:
            base *= max(0.25, 1.0 - 0.2 * self.state.consecutive_losses)

        # Win streak bonus (gentle)
        if self.state.consecutive_wins >= 3:
            base *= min(1.3, 1.0 + 0.05 * self.state.consecutive_wins)

        # Daily PnL scaling
        if self.state.daily_pnl > self.config.profit_target_daily:
            base *= 0.5  # protect profits
        elif self.state.daily_pnl < -self.config.daily_loss_limit * 0.5:
            base *= 0.5  # half way to daily limit, reduce

        # Drawdown scaling
        dd_used = self.config.max_drawdown - self.state.trailing_drawdown_remaining
        dd_pct = dd_used / self.config.max_drawdown if self.config.max_drawdown > 0 else 0
        if dd_pct > 0.5:
            base *= max(0.25, 1.0 - dd_pct)

        self.state.risk_multiplier = float(np.clip(base, 0.1, 1.5))

    def _regime_risk_scale(self, regime: MarketRegime, side: Optional[Side]) -> float:
        """Scale risk based on how tradeable this regime is."""
        scales = {
            MarketRegime.TRENDING_UP: 1.0 if side == Side.LONG else 0.5,
            MarketRegime.TRENDING_DOWN: 1.0 if side == Side.SHORT else 0.5,
            MarketRegime.RANGING: 0.8,
            MarketRegime.VOLATILE: 0.5,  # reduce in chaos
            MarketRegime.LOW_VOLATILITY: 0.6,  # thin, careful
        }
        return scales.get(regime, 0.7)

    def _lock(self, reason: str) -> None:
        """Lock trading for the rest of the session."""
        self.state.is_locked = True
        self.state.lock_reason = reason
        logger.warning("trading_locked", reason=reason, daily_pnl=self.state.daily_pnl)

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of each trading day)."""
        self.state.daily_pnl = 0.0
        self.state.trades_today = 0
        self.state.wins_today = 0
        self.state.losses_today = 0
        self.state.consecutive_losses = 0
        self.state.consecutive_wins = 0
        self.state.risk_multiplier = 1.0
        self.state.is_locked = False
        self.state.lock_reason = ""
        self.state.daily_loss_remaining = self.config.daily_loss_limit

    @property
    def trade_history(self) -> list[TradeResult]:
        return self._trade_history

    @property
    def win_rate(self) -> float:
        if not self._trade_history:
            return 0.0
        wins = sum(1 for t in self._trade_history if t.pnl >= 0)
        return wins / len(self._trade_history)
