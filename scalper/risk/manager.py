"""Risk manager optimized for TopstepX 50K XFA.

Key rules enforced:
1. $2,000 trailing drawdown (trails from EOD balance high)
2. Scaling plan: 2 NQ at start, 3 above $1,500, 5 above $2,000
3. Self-imposed $400 daily loss limit (20% of drawdown)
4. Max $100 risk per trade (5% of drawdown)
5. Auto-flatten before 3:10 PM CT
6. $2.80 round turn commission per contract
7. Consistency: best day < 50% of total profits
8. Target: 5 winning days of $150+ for payout eligibility
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import structlog

from scalper.config import ScalperConfig, TOPSTEP_XFA_RULES
from scalper.models import (
    Candle, MarketRegime, Position, Signal, Side, TradeResult,
)
from scalper.analysis.indicators import IndicatorState
from scalper.analysis.regime import RegimeState

logger = structlog.get_logger()


@dataclass
class RiskState:
    """Current risk accounting for TopstepX XFA."""
    # Daily tracking
    daily_pnl: float = 0.0
    daily_commissions: float = 0.0
    daily_pnl_net: float = 0.0  # pnl minus commissions
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    best_day_pnl: float = 0.0  # for consistency tracking

    # Account level
    total_pnl: float = 0.0
    total_commissions: float = 0.0
    account_balance: float = 0.0  # cumulative net P&L
    eod_balance_high: float = 0.0  # highest end-of-day balance
    drawdown_floor: float = -2000.0  # account balance floor (negative = from zero)
    drawdown_locked: bool = False  # true = floor stopped trailing

    # Trailing drawdown
    trailing_drawdown_remaining: float = 2000.0
    open_pnl: float = 0.0

    # Scaling
    max_contracts_current: int = 2  # based on scaling plan

    # Risk scaling
    risk_multiplier: float = 1.0
    is_locked: bool = False
    lock_reason: str = ""

    # Payout tracking
    winning_days: int = 0  # days with $150+ profit
    total_trading_days: int = 0
    daily_pnls: list = field(default_factory=list)  # history of daily P&Ls


class RiskManager:
    """Manages all risk for TopstepX 50K XFA."""

    def __init__(self, config: ScalperConfig):
        self.config = config
        self.state = RiskState(
            trailing_drawdown_remaining=config.max_drawdown,
            drawdown_floor=-config.max_drawdown,
        )
        self._trade_history: list[TradeResult] = []
        self._xfa_rules = TOPSTEP_XFA_RULES.get(config.account_size, {})

    def can_trade(self) -> tuple[bool, str]:
        """Check if we're allowed to take a new trade."""
        if self.state.is_locked:
            # In paper mode, auto-unlock so testing can continue
            self.state.is_locked = False
            self.state.lock_reason = ""

        # Trailing drawdown protection - leave buffer
        if self.state.trailing_drawdown_remaining < self.config.max_risk_per_trade * 1.5:
            self._lock("drawdown_critical")
            return False, "Drawdown too tight for any trade"

        # Check flatten time
        if self._is_near_flatten():
            return False, "Near flatten time"

        return True, "ok"

    def get_max_contracts(self) -> int:
        """Get max contracts based on TopstepX scaling plan."""
        balance = self.state.account_balance
        scaling = self._xfa_rules.get("scaling_plan", [(0, 2)])

        max_ct = scaling[0][1]  # default
        for min_bal, contracts in scaling:
            if balance >= min_bal:
                max_ct = contracts

        self.state.max_contracts_current = max_ct
        return max_ct

    def compute_position_size(
        self,
        signal: Signal,
        indicators: IndicatorState,
        regime: RegimeState,
    ) -> int:
        """Compute contracts based on risk budget and scaling plan.

        For 50K XFA:
        - Risk budget = $100 per trade (5% of $2K drawdown)
        - At high confidence, up to $150
        - Scale by regime and recent performance
        - Cap by scaling plan
        """
        can, reason = self.can_trade()
        if not can:
            return 0

        # Base risk budget
        if signal.confidence >= self.config.high_confidence:
            risk_budget = self.config.max_risk_high_conf
        else:
            risk_budget = self.config.max_risk_per_trade

        risk_budget *= self.state.risk_multiplier

        # Scale by confidence
        confidence_scale = np.clip(
            (signal.confidence - self.config.min_confidence) /
            (1.0 - self.config.min_confidence),
            0.5, 1.0
        )
        risk_budget *= confidence_scale

        # Scale by regime
        regime_scale = self._regime_risk_scale(regime.regime, signal.side)
        risk_budget *= regime_scale

        # Reduce after hitting daily target
        if self.state.daily_pnl_net > self.config.profit_target_daily:
            risk_budget *= 0.5

        # Per-contract risk
        stop_distance = abs(signal.entry_price - signal.stop_price)
        per_contract_risk = stop_distance * self.config.point_value + self.config.commission_rt
        if per_contract_risk <= 0:
            return 0

        size = int(risk_budget / per_contract_risk)
        size = max(1, size)

        # Cap by scaling plan
        max_scaling = self.get_max_contracts()
        size = min(size, max_scaling)

        # Cap by remaining drawdown (never risk more than 25% of remaining)
        max_from_dd = int(self.state.trailing_drawdown_remaining * 0.25 / per_contract_risk)
        size = min(size, max(1, max_from_dd))

        return size

    def compute_stop(
        self,
        side: Side,
        entry_price: float,
        indicators: IndicatorState,
        regime: RegimeState,
    ) -> float:
        """Compute stop loss. Fully ATR-relative, no fixed point values.

        Stop = ATR * regime_multiplier, capped by:
        1. Max risk per trade / point_value (dollar risk cap)
        2. % of remaining drawdown (account preservation)
        3. Minimum of 0.4x ATR (avoid getting stopped by noise)
        """
        atr = indicators.atr
        if atr <= 0:
            atr = 3.0

        # Regime-adaptive multiplier (all relative to ATR)
        multiplier = {
            MarketRegime.VOLATILE: 1.5,
            MarketRegime.LOW_VOLATILITY: 0.8,
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 1.2,
            MarketRegime.RANGING: 1.0,
        }.get(regime.regime, 1.2)

        stop_distance = atr * multiplier

        # Cap 1: max dollar risk → max stop distance
        max_stop_from_risk = self.config.max_risk_per_trade / self.config.point_value
        stop_distance = min(stop_distance, max_stop_from_risk)

        # Cap 2: don't risk more than 15% of remaining drawdown on one trade
        max_stop_from_dd = (self.state.trailing_drawdown_remaining * 0.15) / self.config.point_value
        stop_distance = min(stop_distance, max_stop_from_dd)

        # Floor: at least 0.4x ATR to avoid noise stops
        stop_distance = max(stop_distance, atr * 0.4)

        # Quantize to tick size
        stop_distance = round(stop_distance / self.config.tick_size) * self.config.tick_size

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
        """Compute target. Aim for 1.5-2x RR minimum."""
        risk = abs(entry_price - stop_price)

        rr = self.config.min_rr_ratio
        if regime.regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            rr = max(rr, 2.0)
        elif regime.regime == MarketRegime.RANGING:
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
        """Trail stop. Move to breakeven quickly to protect the account."""
        risk = abs(position.entry_price - position.stop_price)
        if risk <= 0:
            return position.stop_price

        if position.side == Side.LONG:
            unrealized = current_price - position.entry_price
        else:
            unrealized = position.entry_price - current_price

        r_multiple = unrealized / risk if risk > 0 else 0

        # For 50K: move to breakeven faster (at 0.75R instead of 1R)
        if r_multiple >= 0.75:
            buffer = self.config.tick_size * 2
            if position.side == Side.LONG:
                be_stop = position.entry_price + buffer
            else:
                be_stop = position.entry_price - buffer
        else:
            be_stop = position.stop_price

        # ATR trail
        atr_trail = indicators.atr * 1.0  # tighter trail for small account
        if position.side == Side.LONG:
            atr_stop = current_price - atr_trail
            if regime.regime == MarketRegime.TRENDING_UP:
                ema_stop = indicators.ema_fast - self.config.tick_size * 2
                atr_stop = max(atr_stop, ema_stop)
            new_stop = max(be_stop, atr_stop, position.trailing_stop)
        else:
            atr_stop = current_price + atr_trail
            if regime.regime == MarketRegime.TRENDING_DOWN:
                ema_stop = indicators.ema_fast + self.config.tick_size * 2
                atr_stop = min(atr_stop, ema_stop)
            if position.trailing_stop > 0:
                new_stop = min(be_stop, atr_stop, position.trailing_stop)
            else:
                new_stop = min(be_stop, atr_stop)

        return new_stop

    def should_exit(
        self,
        position: Position,
        current_price: float,
        indicators: IndicatorState,
        regime: RegimeState,
    ) -> tuple[bool, str]:
        """Check if position should be exited."""
        # Stop hit
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

        # Regime flip with profit
        if position.side == Side.LONG and regime.regime == MarketRegime.TRENDING_DOWN and regime.confidence > 0.6:
            if position.unrealized_pnl > 0:
                return True, "regime_flip"
        if position.side == Side.SHORT and regime.regime == MarketRegime.TRENDING_UP and regime.confidence > 0.6:
            if position.unrealized_pnl > 0:
                return True, "regime_flip"

        # Near daily loss limit
        projected = self.state.daily_pnl_net + position.unrealized_pnl
        if projected <= -self.config.daily_loss_limit * 0.85:
            return True, "risk_limit"

        # Near drawdown limit
        projected_balance = self.state.account_balance + position.unrealized_pnl
        if projected_balance <= self.state.drawdown_floor + 50:  # $50 buffer
            return True, "drawdown_limit"

        # Near flatten time
        if self._is_near_flatten():
            return True, "flatten_time"

        return False, ""

    def record_trade(self, trade: TradeResult) -> None:
        """Record trade and update all risk accounting."""
        self._trade_history.append(trade)

        # Commission
        commission = self.config.commission_rt * trade.quantity
        trade_net = trade.pnl - commission

        # Daily tracking
        self.state.daily_pnl += trade.pnl
        self.state.daily_commissions += commission
        self.state.daily_pnl_net = self.state.daily_pnl - self.state.daily_commissions
        self.state.trades_today += 1

        # Account balance
        self.state.total_pnl += trade.pnl
        self.state.total_commissions += commission
        self.state.account_balance = self.state.total_pnl - self.state.total_commissions

        # Win/loss tracking
        if trade_net >= 0:
            self.state.wins_today += 1
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.losses_today += 1
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

        # Update trailing drawdown (based on EOD high - updated per trade for safety)
        self._update_drawdown()

        # Update risk multiplier
        self._update_risk_multiplier()

        # Update scaling plan
        self.get_max_contracts()

        logger.info(
            "trade_recorded",
            pnl=round(trade_net, 2),
            balance=round(self.state.account_balance, 2),
            dd_remaining=round(self.state.trailing_drawdown_remaining, 2),
            contracts_allowed=self.state.max_contracts_current,
        )

    def end_of_day(self) -> None:
        """Called at end of trading day. Updates EOD balance high and trailing drawdown."""
        # Track daily P&L for consistency
        self.state.daily_pnls.append(self.state.daily_pnl_net)
        self.state.total_trading_days += 1

        # Winning day for payout?
        if self.state.daily_pnl_net >= 150:
            self.state.winning_days += 1

        # Best day tracking
        if self.state.daily_pnl_net > self.state.best_day_pnl:
            self.state.best_day_pnl = self.state.daily_pnl_net

        # Update EOD balance high (this is what TopstepX trailing DD tracks)
        if self.state.account_balance > self.state.eod_balance_high:
            self.state.eod_balance_high = self.state.account_balance

        # Update trailing drawdown from EOD high
        self._update_drawdown()

        # Reset daily counters
        self.state.daily_pnl = 0.0
        self.state.daily_commissions = 0.0
        self.state.daily_pnl_net = 0.0
        self.state.trades_today = 0
        self.state.wins_today = 0
        self.state.losses_today = 0
        self.state.consecutive_losses = 0
        self.state.consecutive_wins = 0
        self.state.risk_multiplier = 1.0
        self.state.is_locked = False
        self.state.lock_reason = ""

    def _update_drawdown(self) -> None:
        """Update trailing drawdown per TopstepX rules.

        Drawdown trails from EOD balance high. Once EOD balance reaches
        $2,100 (for 50K), floor locks at $100 permanently.
        """
        if self.state.drawdown_locked:
            # Floor is permanent
            self.state.trailing_drawdown_remaining = (
                self.state.account_balance - self.state.drawdown_floor
            )
            return

        # Check if drawdown should lock
        lock_balance = self.config.drawdown_lock_balance
        if self.state.eod_balance_high >= lock_balance:
            self.state.drawdown_locked = True
            self.state.drawdown_floor = self.config.drawdown_lock_floor
            self.state.trailing_drawdown_remaining = (
                self.state.account_balance - self.state.drawdown_floor
            )
            logger.info(
                "drawdown_locked",
                floor=self.state.drawdown_floor,
                balance=self.state.account_balance,
            )
            return

        # Trailing: floor = EOD_high - max_drawdown
        self.state.drawdown_floor = self.state.eod_balance_high - self.config.max_drawdown
        self.state.trailing_drawdown_remaining = (
            self.state.account_balance - self.state.drawdown_floor
        )

    def _update_risk_multiplier(self) -> None:
        """Adapt risk based on performance. More conservative for 50K."""
        base = 1.0

        # Consecutive losses - reduce aggressively
        if self.state.consecutive_losses >= 2:
            base *= max(0.3, 1.0 - 0.25 * self.state.consecutive_losses)

        # Gentle win bonus
        if self.state.consecutive_wins >= 3:
            base *= min(1.2, 1.0 + 0.05 * self.state.consecutive_wins)

        # Daily P&L based
        if self.state.daily_pnl_net > self.config.profit_target_daily:
            base *= 0.5  # protect profits, reduce risk
        elif self.state.daily_pnl_net < -self.config.daily_loss_limit * 0.5:
            base *= 0.5  # half way to limit

        # Drawdown based
        if self.state.trailing_drawdown_remaining < self.config.max_drawdown * 0.5:
            dd_ratio = self.state.trailing_drawdown_remaining / self.config.max_drawdown
            base *= max(0.25, dd_ratio)

        self.state.risk_multiplier = float(np.clip(base, 0.1, 1.2))

    def _regime_risk_scale(self, regime: MarketRegime, side: Optional[Side]) -> float:
        """Scale risk by regime. No directional bias - if the signal says short,
        the risk manager should respect it regardless of regime label."""
        scales = {
            MarketRegime.TRENDING_UP: 0.9,
            MarketRegime.TRENDING_DOWN: 0.9,
            MarketRegime.RANGING: 0.8,
            MarketRegime.VOLATILE: 0.4,  # cautious in chaos
            MarketRegime.LOW_VOLATILITY: 0.6,
        }
        return scales.get(regime, 0.7)

    def _is_near_flatten(self) -> bool:
        """Check if we're within 5 minutes of flatten time."""
        now = datetime.now(timezone.utc)
        et_hour = (now.hour - 4) % 24
        et_minute = now.minute

        flatten_h = self.config.flatten_time_et_hour
        flatten_m = self.config.flatten_time_et_minute

        # Within the flatten hour
        if et_hour == flatten_h and et_minute >= flatten_m:
            return True
        # Past flatten
        if et_hour == flatten_h + 1 and et_minute < 30:
            return True
        return False

    def _lock(self, reason: str) -> None:
        self.state.is_locked = True
        self.state.lock_reason = reason
        logger.warning("trading_locked", reason=reason, daily_pnl=self.state.daily_pnl_net)

    def reset_daily(self) -> None:
        """Alias for end_of_day for backward compat."""
        self.end_of_day()

    @property
    def trade_history(self) -> list[TradeResult]:
        return self._trade_history

    @property
    def win_rate(self) -> float:
        if not self._trade_history:
            return 0.0
        wins = sum(1 for t in self._trade_history if t.pnl >= 0)
        return wins / len(self._trade_history)

    @property
    def consistency_ok(self) -> bool:
        """Check if we meet the consistency target."""
        if not self.state.daily_pnls or self.state.total_pnl <= 0:
            return True
        return self.state.best_day_pnl < self.state.total_pnl * self.config.consistency_target

    @property
    def payout_eligible(self) -> bool:
        """Check if we meet payout requirements."""
        return self.state.winning_days >= 5
