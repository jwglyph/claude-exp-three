"""Core data models for the scalper."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


class SignalType(str, Enum):
    STRONG_LONG = "strong_long"
    LONG = "long"
    WEAK_LONG = "weak_long"
    NEUTRAL = "neutral"
    WEAK_SHORT = "weak_short"
    SHORT = "short"
    STRONG_SHORT = "strong_short"


@dataclass
class Tick:
    """Raw market tick."""
    timestamp: float
    price: float
    size: int
    side: str = ""  # "buy" or "sell" if available


@dataclass
class Candle:
    """OHLCV candle."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    tick_count: int = 0
    vwap: float = 0.0
    buy_volume: int = 0
    sell_volume: int = 0
    is_complete: bool = False

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def body_ratio(self) -> float:
        """Ratio of body to total range. High = strong directional candle."""
        return self.body / self.range if self.range > 0 else 0.0

    @property
    def delta(self) -> int:
        """Buy volume - sell volume (order flow delta)."""
        return self.buy_volume - self.sell_volume


@dataclass
class Signal:
    """Trading signal with confidence."""
    timestamp: float
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    side: Optional[Side]
    entry_price: float
    stop_price: float
    target_price: float
    regime: MarketRegime
    reasons: list[str] = field(default_factory=list)

    @property
    def risk_ticks(self) -> float:
        if self.side == Side.LONG:
            return (self.entry_price - self.stop_price) / 0.25
        elif self.side == Side.SHORT:
            return (self.stop_price - self.entry_price) / 0.25
        return 0

    @property
    def reward_ticks(self) -> float:
        if self.side == Side.LONG:
            return (self.target_price - self.entry_price) / 0.25
        elif self.side == Side.SHORT:
            return (self.entry_price - self.target_price) / 0.25
        return 0

    @property
    def rr_ratio(self) -> float:
        risk = self.risk_ticks
        return self.reward_ticks / risk if risk > 0 else 0.0


@dataclass
class Order:
    """Order to be sent to broker."""
    id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: int
    price: float = 0.0
    stop_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    fill_time: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class Position:
    """Current open position."""
    symbol: str
    side: Side
    quantity: int
    entry_price: float
    entry_time: float
    stop_price: float
    target_price: float
    trailing_stop: float = 0.0
    unrealized_pnl: float = 0.0
    max_favorable: float = 0.0  # MAE/MFE tracking
    max_adverse: float = 0.0
    signal_confidence: float = 0.0

    def update_pnl(self, current_price: float, point_value: float) -> None:
        if self.side == Side.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * point_value * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * point_value * self.quantity
        self.max_favorable = max(self.max_favorable, self.unrealized_pnl)
        self.max_adverse = min(self.max_adverse, self.unrealized_pnl)


@dataclass
class TradeResult:
    """Completed trade record."""
    entry_time: float
    exit_time: float
    side: Side
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    max_favorable: float
    max_adverse: float
    signal_confidence: float
    regime: MarketRegime
    exit_reason: str  # "target", "stop", "trailing_stop", "signal_exit", "risk_limit"
    holding_candles: int = 0
