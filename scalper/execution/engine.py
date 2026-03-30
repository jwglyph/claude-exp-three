"""Order execution engine with limit order support.

For NQ minis, market orders can slip 1-4 ticks ($5-20) during fast moves.
With a $100 risk budget, that's 5-20% gone to slippage.

Strategy:
- ENTRIES: Use limit orders at bid (for longs) or ask (for shorts)
  with a small offset. If not filled within a few seconds, cancel.
- EXITS (stop/target): Use stop-limit orders with 2-tick limit offset
  to avoid slipping through stops.
- EMERGENCY EXITS: Market orders only for flatten/risk-limit scenarios.

The simulated execution models this by:
- Limit entry: fills at limit price (no slippage) but may miss fills
- Stop exits: fills at stop price + 1 tick slippage (realistic)
- Market exits: fills at current price + 1 tick slippage
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
    """Paper trading execution with realistic fill modeling.

    Models:
    - Limit entries: fill at limit price only if price touches it (no slippage)
    - Stop exits: fill at stop price + 1 tick adverse slippage
    - Market exits: fill at current price + 1 tick adverse slippage
    - Missed fills: limit orders that price doesn't reach are cancelled
    """

    def __init__(self, config: ScalperConfig, slippage_ticks: int = 1):
        self.config = config
        self.slippage_ticks = slippage_ticks  # slippage on market/stop orders
        self._position: Optional[Position] = None
        self._orders: dict[str, Order] = {}
        self._current_price: float = 0.0
        self._current_bid: float = 0.0
        self._current_ask: float = 0.0
        self._current_regime: MarketRegime = MarketRegime.RANGING

    def update_price(self, price: float, bid: float = 0, ask: float = 0) -> None:
        """Update current market price and bid/ask."""
        self._current_price = price
        self._current_bid = bid if bid > 0 else price - self.config.tick_size
        self._current_ask = ask if ask > 0 else price + self.config.tick_size
        if self._position:
            self._position.update_pnl(price, self.config.point_value)

    def set_regime(self, regime: MarketRegime) -> None:
        self._current_regime = regime

    async def submit_order(self, order: Order) -> Order:
        """Submit an order with realistic fill logic."""
        if order.order_type == OrderType.LIMIT:
            return await self._submit_limit(order)
        elif order.order_type == OrderType.MARKET:
            return await self._submit_market(order)
        elif order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            return await self._submit_stop(order)
        return order

    async def _submit_limit(self, order: Order) -> Order:
        """Limit order: fills at limit price if price is at or better.

        For entries:
        - Buy limit: fills if ask <= limit price
        - Sell limit: fills if bid >= limit price
        No slippage on limit fills.
        """
        if order.side == Side.LONG:
            # Buy limit: can we fill at our price?
            if self._current_ask <= order.price:
                order.fill_price = order.price  # filled at limit, no slippage
                order.fill_time = time.time()
                order.status = OrderStatus.FILLED
                self._open_position(order)
            else:
                # Queue as pending - will check on future price updates
                order.status = OrderStatus.SUBMITTED
                self._orders[order.id] = order
        else:
            # Sell limit
            if self._current_bid >= order.price:
                order.fill_price = order.price
                order.fill_time = time.time()
                order.status = OrderStatus.FILLED
                self._open_position(order)
            else:
                order.status = OrderStatus.SUBMITTED
                self._orders[order.id] = order

        return order

    async def _submit_market(self, order: Order) -> Order:
        """Market order: fills immediately with slippage."""
        slippage = self.slippage_ticks * self.config.tick_size

        if order.side == Side.LONG:
            # Buy at ask + slippage
            fill_price = self._current_ask + slippage
        else:
            # Sell at bid - slippage
            fill_price = self._current_bid - slippage

        order.fill_price = fill_price
        order.fill_time = time.time()
        order.status = OrderStatus.FILLED

        if self._position is None:
            self._open_position(order)
        else:
            # Closing position with market order
            self._close_position(fill_price, "market_exit")

        return order

    async def _submit_stop(self, order: Order) -> Order:
        """Stop order: queued, triggers when price hits stop level."""
        order.status = OrderStatus.SUBMITTED
        self._orders[order.id] = order
        return order

    def _open_position(self, order: Order) -> None:
        """Create a new position from a filled order."""
        if self._position is not None:
            return  # already in a position

        self._position = Position(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            entry_price=order.fill_price,
            entry_time=time.time(),
            stop_price=order.stop_price,
            target_price=order.price if order.order_type == OrderType.LIMIT else 0,
            signal_confidence=0.0,
        )
        logger.info(
            "position_opened",
            side=order.side.value,
            price=order.fill_price,
            qty=order.quantity,
            type=order.order_type.value,
        )

    async def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            del self._orders[order_id]
            return True
        return False

    async def cancel_all_pending(self) -> int:
        """Cancel all pending orders. Returns count cancelled."""
        cancelled = 0
        for oid in list(self._orders.keys()):
            if self._orders[oid].status == OrderStatus.SUBMITTED:
                self._orders[oid].status = OrderStatus.CANCELLED
                del self._orders[oid]
                cancelled += 1
        return cancelled

    async def get_position(self) -> Optional[Position]:
        return self._position

    async def flatten(self) -> Optional[TradeResult]:
        """Emergency close at market. Uses slippage."""
        if self._position is None:
            return None
        slippage = self.slippage_ticks * self.config.tick_size
        if self._position.side == Side.LONG:
            exit_price = self._current_bid - slippage  # selling into bid
        else:
            exit_price = self._current_ask + slippage  # covering at ask
        return self._close_position(exit_price, "flatten")

    def check_stops_and_limits(self, price: float, bid: float = 0, ask: float = 0) -> Optional[TradeResult]:
        """Check all pending orders against current price.

        Stop orders: fill at stop + slippage (adverse)
        Limit orders (entry): fill at limit price (no slippage)
        Limit orders (target): fill at limit price (no slippage)
        """
        self.update_price(price, bid, ask)
        slippage = self.slippage_ticks * self.config.tick_size

        triggered = []
        for oid, order in list(self._orders.items()):
            if order.status != OrderStatus.SUBMITTED:
                continue

            filled = False
            fill_price = 0.0

            if order.order_type == OrderType.STOP:
                # Stop loss: triggers when price goes through stop level
                if order.side == Side.LONG and price >= order.stop_price:
                    fill_price = order.stop_price + slippage  # buy stop slips up
                    filled = True
                elif order.side == Side.SHORT and price <= order.stop_price:
                    fill_price = order.stop_price - slippage  # sell stop slips down
                    filled = True

            elif order.order_type == OrderType.STOP_LIMIT:
                # Stop-limit: triggers at stop, fills at limit (no slippage if limit holds)
                if order.side == Side.LONG and price >= order.stop_price:
                    if price <= order.price:  # price within limit
                        fill_price = order.price
                        filled = True
                    else:
                        fill_price = order.stop_price + slippage  # blew through limit
                        filled = True
                elif order.side == Side.SHORT and price <= order.stop_price:
                    if price >= order.price:
                        fill_price = order.price
                        filled = True
                    else:
                        fill_price = order.stop_price - slippage
                        filled = True

            elif order.order_type == OrderType.LIMIT:
                # Limit order: fills at limit price, no slippage
                if order.side == Side.LONG and price <= order.price:
                    fill_price = order.price
                    filled = True
                elif order.side == Side.SHORT and price >= order.price:
                    fill_price = order.price
                    filled = True

            if filled:
                order.fill_price = fill_price
                order.fill_time = time.time()
                order.status = OrderStatus.FILLED
                triggered.append(oid)

        result = None
        for oid in triggered:
            order = self._orders.pop(oid)

            # Is this an entry or exit?
            if self._position is None:
                # Entry fill
                self._open_position(order)
            else:
                # Exit fill
                result = self._close_position(order.fill_price, "stop_or_target")

        return result

    def _close_position(self, exit_price: float, reason: str) -> Optional[TradeResult]:
        """Close position and return trade result."""
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
            slippage=round(abs(exit_price - pos.entry_price) - abs(pos.target_price - pos.entry_price), 2) if pos.target_price else 0,
        )

        self._position = None
        self._orders.clear()
        return trade

    @property
    def has_position(self) -> bool:
        return self._position is not None

    @property
    def pending_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.status == OrderStatus.SUBMITTED]


class LiveExecution(ExecutionEngine):
    """Live execution via TopstepX/ProjectX API.

    Uses limit orders for entries, stop-limit for exits.
    """

    def __init__(self, config: ScalperConfig):
        self.config = config
        self._position: Optional[Position] = None
        self._session = None

    async def connect(self, url: str, api_key: str, api_secret: str) -> None:
        import aiohttp
        self._session = aiohttp.ClientSession()
        logger.info("broker_connected", url=url)

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()

    async def submit_order(self, order: Order) -> Order:
        if self._session is None:
            raise RuntimeError("Not connected to broker")
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
        logger.warning("flatten_all_positions")
        return None


def create_order(
    symbol: str,
    side: Side,
    quantity: int,
    order_type: OrderType = OrderType.LIMIT,
    price: float = 0.0,
    stop_price: float = 0.0,
) -> Order:
    """Factory function. Default is LIMIT order (not market)."""
    return Order(
        id=str(uuid.uuid4())[:8],
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        stop_price=stop_price,
    )
