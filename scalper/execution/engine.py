"""Order execution engine with broker abstraction.

Handles:
- Order submission (market, limit, stop)
- Order lifecycle management
- Fill tracking
- Position state management
- Simulated execution for paper trading / backtesting
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional

import structlog

from scalper.models import (
    Order, OrderStatus, OrderType, Position, Side, TradeResult, MarketRegime,
)
from scalper.config import ScalperConfig

logger = structlog.get_logger()


class ExecutionEngine(ABC):
    """Abstract execution engine."""

    @abstractmethod
    async def submit_order(self, order: Order) -> Order: ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    async def get_position(self) -> Optional[Position]: ...

    @abstractmethod
    async def flatten(self) -> Optional[TradeResult]: ...


class SimulatedExecution(ExecutionEngine):
    """Paper trading / backtest execution engine.

    Simulates fills with configurable slippage and latency.
    Tracks position P&L accurately for NQ contract specs.
    """

    def __init__(self, config: ScalperConfig, slippage_ticks: int = 1):
        self.config = config
        self.slippage_ticks = slippage_ticks
        self._position: Optional[Position] = None
        self._orders: dict[str, Order] = {}
        self._current_price: float = 0.0
        self._current_regime: MarketRegime = MarketRegime.RANGING

    def update_price(self, price: float) -> None:
        """Update current market price for fill simulation."""
        self._current_price = price
        if self._position:
            self._position.update_pnl(price, self.config.point_value)

    def set_regime(self, regime: MarketRegime) -> None:
        self._current_regime = regime

    async def submit_order(self, order: Order) -> Order:
        """Simulate order fill with slippage."""
        slippage = self.slippage_ticks * self.config.tick_size

        if order.order_type == OrderType.MARKET:
            if order.side == Side.LONG:
                fill_price = self._current_price + slippage
            else:
                fill_price = self._current_price - slippage

            order.fill_price = fill_price
            order.fill_time = time.time()
            order.status = OrderStatus.FILLED

            # Create or update position
            if self._position is None:
                self._position = Position(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    entry_price=fill_price,
                    entry_time=time.time(),
                    stop_price=order.stop_price,
                    target_price=order.price,  # target stored in price field for market orders
                    signal_confidence=0.0,
                )
                logger.info(
                    "position_opened",
                    side=order.side.value,
                    price=fill_price,
                    qty=order.quantity,
                )
            else:
                # Closing position
                trade = self._close_position(fill_price, "order")
                if trade:
                    return order

        elif order.order_type in (OrderType.STOP, OrderType.LIMIT):
            # Store as pending - will be checked on price updates
            order.status = OrderStatus.SUBMITTED
            self._orders[order.id] = order

        self._orders[order.id] = order
        return order

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            del self._orders[order_id]
            return True
        return False

    async def get_position(self) -> Optional[Position]:
        return self._position

    async def flatten(self) -> Optional[TradeResult]:
        """Close any open position at current price."""
        if self._position is None:
            return None
        return self._close_position(self._current_price, "flatten")

    def check_stops(self, price: float) -> Optional[TradeResult]:
        """Check if any stop/limit orders are triggered."""
        self._current_price = price
        if self._position:
            self._position.update_pnl(price, self.config.point_value)

        triggered = []
        for oid, order in self._orders.items():
            if order.status != OrderStatus.SUBMITTED:
                continue

            if order.order_type == OrderType.STOP:
                if order.side == Side.LONG and price >= order.stop_price:
                    triggered.append(oid)
                elif order.side == Side.SHORT and price <= order.stop_price:
                    triggered.append(oid)

            elif order.order_type == OrderType.LIMIT:
                if order.side == Side.LONG and price <= order.price:
                    triggered.append(oid)
                elif order.side == Side.SHORT and price >= order.price:
                    triggered.append(oid)

        result = None
        for oid in triggered:
            order = self._orders.pop(oid)
            order.status = OrderStatus.FILLED
            order.fill_price = price
            order.fill_time = time.time()

            if self._position:
                result = self._close_position(price, "stop_or_target")

        return result

    def _close_position(self, exit_price: float, reason: str) -> Optional[TradeResult]:
        """Close the current position and return trade result."""
        if self._position is None:
            return None

        pos = self._position
        if pos.side == Side.LONG:
            pnl = (exit_price - pos.entry_price) * self.config.point_value * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * self.config.point_value * pos.quantity

        trade = TradeResult(
            entry_time=pos.entry_time,
            exit_time=time.time(),
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            max_favorable=pos.max_favorable,
            max_adverse=pos.max_adverse,
            signal_confidence=pos.signal_confidence,
            regime=self._current_regime,
            exit_reason=reason,
        )

        logger.info(
            "position_closed",
            side=pos.side.value,
            entry=pos.entry_price,
            exit=exit_price,
            pnl=round(pnl, 2),
            reason=reason,
        )

        self._position = None
        # Clear any remaining stop/target orders
        self._orders.clear()

        return trade

    @property
    def has_position(self) -> bool:
        return self._position is not None


class LiveExecution(ExecutionEngine):
    """Live execution via broker API (Rithmic/Tradovate).

    This is a framework for connecting to real brokers.
    Actual implementation depends on the broker's API.
    """

    def __init__(self, config: ScalperConfig):
        self.config = config
        self._position: Optional[Position] = None
        self._session = None

    async def connect(self, url: str, api_key: str, api_secret: str) -> None:
        """Connect to broker API."""
        import aiohttp
        self._session = aiohttp.ClientSession()
        logger.info("broker_connected", url=url)

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()

    async def submit_order(self, order: Order) -> Order:
        """Submit order to broker.

        In a real implementation, this would:
        1. Send order via FIX/REST/WebSocket to broker
        2. Wait for acknowledgment
        3. Update order status
        """
        if self._session is None:
            raise RuntimeError("Not connected to broker")

        # Placeholder for actual broker integration
        # The specific implementation depends on whether using:
        # - Rithmic R | Protocol API
        # - Tradovate REST + WebSocket
        # - ProjectX / TopstepX API
        logger.info(
            "order_submitted_live",
            side=order.side.value,
            type=order.order_type.value,
            qty=order.quantity,
            price=order.price,
        )

        order.status = OrderStatus.SUBMITTED
        return order

    async def cancel_order(self, order_id: str) -> bool:
        logger.info("order_cancelled_live", order_id=order_id)
        return True

    async def get_position(self) -> Optional[Position]:
        return self._position

    async def flatten(self) -> Optional[TradeResult]:
        """Flatten all positions - emergency exit."""
        logger.warning("flatten_all_positions")
        # In live: send market order to close
        return None


def create_order(
    symbol: str,
    side: Side,
    quantity: int,
    order_type: OrderType = OrderType.MARKET,
    price: float = 0.0,
    stop_price: float = 0.0,
) -> Order:
    """Factory function for creating orders."""
    return Order(
        id=str(uuid.uuid4())[:8],
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        stop_price=stop_price,
    )
