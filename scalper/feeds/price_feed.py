"""Live price feed abstraction with WebSocket support.

Supports multiple feed providers:
- Rithmic (via R | Protocol API or third-party wrapper)
- Tradovate (WebSocket API)
- Simulation (replay or random walk for testing)
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Optional

import structlog

from scalper.models import Tick

logger = structlog.get_logger()


class PriceFeed(ABC):
    """Abstract base for price feeds."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._running = False
        self._callbacks: list[Callable[[Tick], None]] = []

    def on_tick(self, callback: Callable[[Tick], None]) -> None:
        self._callbacks.append(callback)

    def _emit(self, tick: Tick) -> None:
        for cb in self._callbacks:
            cb(tick)

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def stream(self) -> AsyncIterator[Tick]: ...


class WebSocketFeed(PriceFeed):
    """Generic WebSocket-based price feed.

    Expects JSON messages with at minimum: price, size, timestamp.
    Adapts to different provider message formats.
    """

    def __init__(
        self,
        symbol: str,
        url: str,
        api_key: str = "",
        api_secret: str = "",
        provider: str = "generic",
    ):
        super().__init__(symbol)
        self.url = url
        self.api_key = api_key
        self.api_secret = api_secret
        self.provider = provider
        self._ws = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0

    async def connect(self) -> None:
        import websockets
        self._running = True
        logger.info("connecting_to_feed", url=self.url, symbol=self.symbol)
        self._ws = await websockets.connect(self.url)

        # Send subscription message
        sub_msg = self._build_subscribe_message()
        if sub_msg:
            await self._ws.send(json.dumps(sub_msg))
            logger.info("subscribed", symbol=self.symbol)

    async def disconnect(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    def _build_subscribe_message(self) -> Optional[dict]:
        """Build provider-specific subscription message."""
        if self.provider == "tradovate":
            return {
                "op": "subscribe",
                "args": [f"md/subscribeTick/{self.symbol}"],
            }
        elif self.provider == "rithmic":
            return {
                "type": "subscribe",
                "symbol": self.symbol,
                "data_type": "tick",
                "api_key": self.api_key,
            }
        return {"subscribe": self.symbol}

    def _parse_tick(self, raw: str) -> Optional[Tick]:
        """Parse provider-specific tick data."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if self.provider == "tradovate":
            return Tick(
                timestamp=data.get("timestamp", time.time()),
                price=float(data.get("price", data.get("p", 0))),
                size=int(data.get("size", data.get("s", 1))),
                side=data.get("side", ""),
            )
        elif self.provider == "rithmic":
            return Tick(
                timestamp=data.get("ssboe", time.time()) + data.get("usecs", 0) / 1e6,
                price=float(data.get("trade_price", 0)),
                size=int(data.get("trade_size", 1)),
                side="buy" if data.get("aggressor_side") == 1 else "sell",
            )
        else:
            # Generic format
            return Tick(
                timestamp=data.get("timestamp", data.get("t", time.time())),
                price=float(data.get("price", data.get("p", 0))),
                size=int(data.get("size", data.get("s", 1))),
                side=data.get("side", ""),
            )

    async def stream(self) -> AsyncIterator[Tick]:
        """Stream ticks with automatic reconnection."""
        while self._running:
            try:
                if self._ws is None:
                    await self.connect()

                async for message in self._ws:
                    tick = self._parse_tick(message)
                    if tick and tick.price > 0:
                        self._emit(tick)
                        yield tick

            except Exception as e:
                logger.warning("feed_error", error=str(e), reconnect_in=self._reconnect_delay)
                if self._ws:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    self._ws = None

                if not self._running:
                    break

                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

            else:
                self._reconnect_delay = 1.0


class SimulatedFeed(PriceFeed):
    """Simulated feed for testing - replays candle data or generates random walk."""

    def __init__(
        self,
        symbol: str = "NQ",
        start_price: float = 20000.0,
        volatility: float = 0.5,
        tick_rate: float = 0.05,  # seconds between ticks
    ):
        super().__init__(symbol)
        self.start_price = start_price
        self.volatility = volatility
        self.tick_rate = tick_rate

    async def connect(self) -> None:
        self._running = True
        logger.info("sim_feed_connected", symbol=self.symbol)

    async def disconnect(self) -> None:
        self._running = False

    async def stream(self) -> AsyncIterator[Tick]:
        """Generate random walk ticks."""
        import numpy as np

        price = self.start_price
        while self._running:
            # NQ-like random walk: mean-reverting with occasional momentum bursts
            move = np.random.normal(0, self.volatility)
            # Quantize to tick size (0.25)
            move = round(move / 0.25) * 0.25
            price += move
            size = max(1, int(np.random.exponential(3)))
            side = "buy" if move > 0 else "sell" if move < 0 else ""

            tick = Tick(
                timestamp=time.time(),
                price=round(price, 2),
                size=size,
                side=side,
            )
            self._emit(tick)
            yield tick
            await asyncio.sleep(self.tick_rate)


class CandleReplayFeed(PriceFeed):
    """Replay historical candles as ticks for backtesting."""

    def __init__(self, symbol: str, candles: list[dict], speed: float = 0.0):
        super().__init__(symbol)
        self._candles = candles
        self._speed = speed  # 0 = instant, >0 = seconds between candles

    async def connect(self) -> None:
        self._running = True

    async def disconnect(self) -> None:
        self._running = False

    async def stream(self) -> AsyncIterator[Tick]:
        """Convert candles to synthetic ticks (OHLC order)."""
        for c in self._candles:
            if not self._running:
                break
            # Emit 4 ticks per candle representing OHLC
            for price_key in ["open", "high", "low", "close"]:
                tick = Tick(
                    timestamp=c.get("timestamp", time.time()),
                    price=float(c[price_key]),
                    size=max(1, c.get("volume", 100) // 4),
                    side="",
                )
                self._emit(tick)
                yield tick

            if self._speed > 0:
                await asyncio.sleep(self._speed)
