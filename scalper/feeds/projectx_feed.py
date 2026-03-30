"""Real-time market data feed via ProjectX/TopstepX SignalR hubs.

Connects to the ProjectX Market Hub via SignalR (WebSocket transport)
and streams live quotes and trades for NQ futures.

The Market Hub supports:
- SubscribeContractQuotes(contractId) -> GatewayQuote events
- SubscribeContractTrades(contractId) -> GatewayTrade events
- SubscribeContractMarketDepth(contractId) -> GatewayDepth events
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
from typing import AsyncIterator, Callable, Optional

import structlog
from signalrcore.hub_connection_builder import HubConnectionBuilder

from scalper.models import Tick
from scalper.feeds.price_feed import PriceFeed
from scalper.feeds.projectx_client import ProjectXClient, ProjectXConfig

logger = structlog.get_logger()


class ProjectXFeed(PriceFeed):
    """Live market data feed from TopstepX/ProjectX via SignalR.

    Streams real-time quotes and trades for a given contract.
    Converts SignalR events into Tick objects for the candle aggregator.
    """

    def __init__(
        self,
        client: ProjectXClient,
        contract_id: str,
        subscribe_quotes: bool = True,
        subscribe_trades: bool = True,
        subscribe_depth: bool = False,
    ):
        super().__init__(symbol=contract_id)
        self.client = client
        self.contract_id = contract_id
        self._subscribe_quotes = subscribe_quotes
        self._subscribe_trades = subscribe_trades
        self._subscribe_depth = subscribe_depth

        self._hub: Optional[object] = None
        self._tick_queue: asyncio.Queue[Tick] = asyncio.Queue(maxsize=10000)
        self._connected = False
        self._reconnect_delay = 2.0
        self._last_price = 0.0
        self._quote_count = 0
        self._trade_count = 0

    async def connect(self) -> None:
        """Authenticate and connect to SignalR Market Hub."""
        # Ensure we have a valid token
        token = await self.client.ensure_authenticated()

        # Build SignalR connection to Market Hub
        market_hub_url = self.client.config.market_hub_url
        hub_url_with_token = f"{market_hub_url}?access_token={token}"

        logger.info(
            "connecting_market_hub",
            hub=market_hub_url,
            contract=self.contract_id,
        )

        # signalrcore uses sync callbacks, we bridge to async via queue
        self._hub = (
            HubConnectionBuilder()
            .with_url(hub_url_with_token, options={
                "skip_negotiation": True,
                "headers": {"Authorization": f"Bearer {token}"},
            })
            .configure_logging(logging_level=30)  # WARNING
            .with_automatic_reconnect({
                "type": "interval",
                "intervals": [1, 2, 5, 10, 30],
            })
            .build()
        )

        # Register event handlers
        self._hub.on("GatewayQuote", self._on_quote)
        self._hub.on("GatewayTrade", self._on_trade)
        if self._subscribe_depth:
            self._hub.on("GatewayDepth", self._on_depth)

        # Connection lifecycle handlers
        self._hub.on_open(self._on_connected)
        self._hub.on_close(self._on_disconnected)
        self._hub.on_error(self._on_error)
        self._hub.on_reconnect(self._on_reconnect)

        # Start connection (signalrcore runs its own thread)
        self._hub.start()
        self._running = True

        # Wait for connection
        for _ in range(50):  # 5 second timeout
            if self._connected:
                break
            await asyncio.sleep(0.1)

        if not self._connected:
            logger.warning("hub_connection_timeout", contract=self.contract_id)

    async def disconnect(self) -> None:
        """Disconnect from SignalR hub."""
        self._running = False
        if self._hub:
            try:
                self._hub.stop()
            except Exception as e:
                logger.warning("hub_stop_error", error=str(e))
        self._connected = False
        logger.info("market_hub_disconnected")

    async def stream(self) -> AsyncIterator[Tick]:
        """Stream ticks from the SignalR event queue."""
        while self._running:
            try:
                # Wait for ticks with timeout to allow checking _running flag
                tick = await asyncio.wait_for(
                    self._tick_queue.get(), timeout=1.0
                )
                self._emit(tick)
                yield tick
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning("stream_error", error=str(e))
                if not self._running:
                    break
                await asyncio.sleep(0.5)

    def _subscribe_all(self) -> None:
        """Subscribe to market data for the contract."""
        if not self._hub:
            return

        if self._subscribe_quotes:
            logger.info("subscribing_quotes", contract=self.contract_id)
            self._hub.send("SubscribeContractQuotes", [self.contract_id])

        if self._subscribe_trades:
            logger.info("subscribing_trades", contract=self.contract_id)
            self._hub.send("SubscribeContractTrades", [self.contract_id])

        if self._subscribe_depth:
            logger.info("subscribing_depth", contract=self.contract_id)
            self._hub.send("SubscribeContractMarketDepth", [self.contract_id])

    # --- SignalR event handlers (called from signalrcore thread) ---

    def _on_connected(self) -> None:
        """Called when SignalR connection opens."""
        self._connected = True
        logger.info("market_hub_connected")
        self._subscribe_all()

    def _on_disconnected(self) -> None:
        """Called when connection closes."""
        self._connected = False
        logger.warning("market_hub_disconnected")

    def _on_reconnect(self) -> None:
        """Called on reconnection - must resubscribe."""
        logger.info("market_hub_reconnected")
        self._connected = True
        self._subscribe_all()

    def _on_error(self, error) -> None:
        """Called on connection error."""
        logger.error("market_hub_error", error=str(error))

    def _on_quote(self, args) -> None:
        """Handle GatewayQuote event.

        Quote data typically contains:
        - lastPrice, bestBid, bestAsk
        - change, changePercent
        - open, high, low
        - session data
        """
        try:
            data = args[0] if isinstance(args, list) else args
            if isinstance(data, str):
                data = json.loads(data)

            price = float(
                data.get("lastPrice")
                or data.get("bestBid")
                or data.get("price")
                or 0
            )

            if price <= 0:
                return

            self._last_price = price
            self._quote_count += 1

            # Get bid/ask for side inference
            bid = float(data.get("bestBid", 0) or 0)
            ask = float(data.get("bestAsk", 0) or 0)

            side = ""
            if bid > 0 and ask > 0:
                if price >= ask:
                    side = "buy"
                elif price <= bid:
                    side = "sell"

            tick = Tick(
                timestamp=self._parse_timestamp(data),
                price=price,
                size=int(data.get("lastSize", data.get("size", 1)) or 1),
                side=side,
            )

            # Non-blocking put to queue
            try:
                self._tick_queue.put_nowait(tick)
            except asyncio.QueueFull:
                # Drop oldest if queue full (shouldn't happen in practice)
                try:
                    self._tick_queue.get_nowait()
                    self._tick_queue.put_nowait(tick)
                except Exception:
                    pass

        except Exception as e:
            logger.debug("quote_parse_error", error=str(e), raw=str(args)[:200])

    def _on_trade(self, args) -> None:
        """Handle GatewayTrade event.

        Trade data contains:
        - symbolId, price, timestamp, type, volume
        """
        try:
            data = args[0] if isinstance(args, list) else args
            if isinstance(data, str):
                data = json.loads(data)

            price = float(data.get("price", 0))
            if price <= 0:
                return

            self._last_price = price
            self._trade_count += 1

            # Determine aggressor side from trade type
            trade_type = data.get("type", "")
            if isinstance(trade_type, int):
                side = "buy" if trade_type == 1 else "sell" if trade_type == 2 else ""
            else:
                side = "buy" if "buy" in str(trade_type).lower() else "sell" if "sell" in str(trade_type).lower() else ""

            tick = Tick(
                timestamp=self._parse_timestamp(data),
                price=price,
                size=int(data.get("volume", data.get("size", 1)) or 1),
                side=side,
            )

            try:
                self._tick_queue.put_nowait(tick)
            except asyncio.QueueFull:
                try:
                    self._tick_queue.get_nowait()
                    self._tick_queue.put_nowait(tick)
                except Exception:
                    pass

        except Exception as e:
            logger.debug("trade_parse_error", error=str(e), raw=str(args)[:200])

    def _on_depth(self, args) -> None:
        """Handle GatewayDepth event (DOM updates). Currently just logged."""
        pass  # Can be extended for order flow analysis

    def _parse_timestamp(self, data: dict) -> float:
        """Parse timestamp from various formats."""
        ts = data.get("timestamp", data.get("t", ""))
        if isinstance(ts, (int, float)):
            # Could be epoch seconds or milliseconds
            return ts / 1000 if ts > 1e12 else ts
        if isinstance(ts, str) and ts:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.timestamp()
            except (ValueError, TypeError):
                pass
        return time.time()

    async def update_token(self) -> None:
        """Refresh the auth token and reconnect if needed.

        Call this periodically for long-running sessions (tokens expire in 24h).
        """
        try:
            new_token = await self.client.authenticate()
            logger.info("token_refreshed")

            # Reconnect with new token
            if self._hub:
                await self.disconnect()
                await asyncio.sleep(1)
                await self.connect()

        except Exception as e:
            logger.error("token_refresh_failed", error=str(e))

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "quotes": self._quote_count,
            "trades": self._trade_count,
            "last_price": self._last_price,
            "queue_size": self._tick_queue.qsize(),
        }
