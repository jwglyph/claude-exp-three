"""Real-time market data feed via ProjectX/TopstepX SignalR hubs.

Connects to the ProjectX Market Hub via SignalR (WebSocket transport)
and streams live quotes and trades for NQ futures.

The Market Hub supports:
- SubscribeContractQuotes(contractId) -> GatewayQuote events
- SubscribeContractTrades(contractId) -> GatewayTrade events
- SubscribeContractMarketDepth(contractId) -> GatewayDepth events

SignalR events arrive as: [contractId, {data_dict}]
"""

from __future__ import annotations

import asyncio
import json
import queue
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
        # Use thread-safe queue since SignalR callbacks run in a separate thread
        self._tick_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._connected = False
        self._reconnect_delay = 2.0
        self._last_price = 0.0
        self._current_bid = 0.0
        self._current_ask = 0.0
        self._quote_count = 0
        self._trade_count = 0

    async def connect(self) -> None:
        """Authenticate and connect to SignalR Market Hub."""
        token = await self.client.ensure_authenticated()

        market_hub_url = self.client.config.market_hub_url
        hub_url_with_token = f"{market_hub_url}?access_token={token}"

        logger.info(
            "connecting_market_hub",
            hub=market_hub_url,
            contract=self.contract_id,
        )

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
        """Stream ticks from the thread-safe queue."""
        while self._running:
            try:
                # Poll the thread-safe queue from the async loop
                tick = self._tick_queue.get_nowait()
                self._emit(tick)
                yield tick
            except queue.Empty:
                # No ticks available, yield control briefly
                await asyncio.sleep(0.01)
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

    def _extract_data(self, args) -> Optional[dict]:
        """Extract the data dict from SignalR event args.

        SignalR events arrive as: [contractId_string, {data_dict}]
        e.g.: ['CON.F.US.ENQ.M26', {'bestBid': 23121.0, 'bestAsk': 23121.75, ...}]
        """
        if isinstance(args, list):
            # Find the dict in the args list (skip the contract ID string)
            for item in args:
                if isinstance(item, dict):
                    return item
            # If args is a list of one dict
            if len(args) == 1 and isinstance(args[0], dict):
                return args[0]
            # Maybe it's a list of lists
            for item in args:
                if isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict):
                            return sub
        elif isinstance(args, dict):
            return args
        elif isinstance(args, str):
            try:
                return json.loads(args)
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _on_connected(self) -> None:
        self._connected = True
        logger.info("market_hub_connected")
        self._subscribe_all()

    def _on_disconnected(self) -> None:
        self._connected = False
        logger.warning("market_hub_disconnected")

    def _on_reconnect(self) -> None:
        logger.info("market_hub_reconnected")
        self._connected = True
        self._subscribe_all()

    def _on_error(self, error) -> None:
        logger.error("market_hub_error", error=str(error))

    def _on_quote(self, args) -> None:
        """Handle GatewayQuote event.

        Data contains: bestBid, bestAsk, lastPrice, change, changePercent,
        timestamp, lastUpdated, symbol, contract
        """
        try:
            data = self._extract_data(args)
            if data is None:
                logger.debug("quote_no_data", raw=str(args)[:200])
                return

            # Get price - prefer lastPrice, fall back to mid of bid/ask
            bid = float(data.get("bestBid", 0) or 0)
            ask = float(data.get("bestAsk", 0) or 0)
            last = float(data.get("lastPrice", 0) or 0)

            price = last if last > 0 else (bid + ask) / 2 if (bid > 0 and ask > 0) else bid or ask
            if price <= 0:
                return

            self._last_price = price
            if bid > 0:
                self._current_bid = bid
            if ask > 0:
                self._current_ask = ask
            self._quote_count += 1

            # Infer side from price vs bid/ask
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

            # Thread-safe put
            try:
                self._tick_queue.put_nowait(tick)
            except queue.Full:
                try:
                    self._tick_queue.get_nowait()  # drop oldest
                    self._tick_queue.put_nowait(tick)
                except Exception:
                    pass

            if self._quote_count % 100 == 1:
                logger.info(
                    "quote_received",
                    price=price,
                    bid=bid,
                    ask=ask,
                    count=self._quote_count,
                )

        except Exception as e:
            logger.debug("quote_parse_error", error=str(e), raw=str(args)[:300])

    def _on_trade(self, args) -> None:
        """Handle GatewayTrade event.

        Data contains: symbolId, price, timestamp, type, volume, contractId
        """
        try:
            data = self._extract_data(args)
            if data is None:
                logger.debug("trade_no_data", raw=str(args)[:200])
                return

            price = float(data.get("price", 0))
            if price <= 0:
                return

            self._last_price = price
            self._trade_count += 1

            # Log raw trade data for debugging side inference
            if self._trade_count <= 5:
                logger.info("trade_raw_debug", data=str(data)[:500])
            _raw_type = data.get("type", "?")

            # Determine aggressor side
            # Method 1: from trade type field
            trade_type = data.get("type", data.get("aggressorSide", data.get("aggressor", "")))
            side = ""
            if isinstance(trade_type, int):
                # Common conventions: 0=unknown, 1=buy, 2=sell (or vice versa)
                # Also possible: 1=ask hit (buy), 2=bid hit (sell)
                side = "buy" if trade_type in (1,) else "sell" if trade_type in (2,) else ""
            elif isinstance(trade_type, str):
                tl = trade_type.lower()
                if "buy" in tl or "ask" in tl:
                    side = "buy"
                elif "sell" in tl or "bid" in tl:
                    side = "sell"

            # Method 2: infer from price vs last known bid/ask
            if not side and self._current_bid > 0 and self._current_ask > 0:
                if price >= self._current_ask:
                    side = "buy"  # trade at or above ask = buyer lifted
                elif price <= self._current_bid:
                    side = "sell"  # trade at or below bid = seller hit
                else:
                    # Between bid and ask - classify by proximity
                    mid = (self._current_bid + self._current_ask) / 2
                    side = "buy" if price >= mid else "sell"

            # Log side inference for first 20 trades to verify correctness
            if self._trade_count <= 20:
                logger.info("trade_side",
                            price=price, bid=self._current_bid, ask=self._current_ask,
                            raw_type=_raw_type, side=side)

            tick = Tick(
                timestamp=self._parse_timestamp(data),
                price=price,
                size=int(data.get("volume", data.get("size", 1)) or 1),
                side=side,
            )

            try:
                self._tick_queue.put_nowait(tick)
            except queue.Full:
                try:
                    self._tick_queue.get_nowait()
                    self._tick_queue.put_nowait(tick)
                except Exception:
                    pass

            if self._trade_count % 100 == 1:
                logger.info(
                    "trade_received",
                    price=price,
                    volume=tick.size,
                    side=side,
                    count=self._trade_count,
                )

        except Exception as e:
            logger.debug("trade_parse_error", error=str(e), raw=str(args)[:300])

    def _on_depth(self, args) -> None:
        """Handle GatewayDepth event (DOM updates)."""
        pass

    def _parse_timestamp(self, data: dict) -> float:
        """Parse timestamp from various formats."""
        ts = data.get("timestamp", data.get("t", ""))
        if isinstance(ts, (int, float)):
            return ts / 1000 if ts > 1e12 else ts
        if isinstance(ts, str) and ts:
            try:
                from datetime import datetime
                # Handle ISO format with timezone
                clean = ts.replace("Z", "+00:00")
                # Handle +00:00 already present
                dt = datetime.fromisoformat(clean)
                return dt.timestamp()
            except (ValueError, TypeError):
                pass
        return time.time()

    async def update_token(self) -> None:
        """Refresh the auth token and reconnect if needed."""
        try:
            new_token = await self.client.authenticate()
            logger.info("token_refreshed")
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
