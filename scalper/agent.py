"""Main trading agent orchestrator.

This is the central brain that coordinates all components:
1. Receives ticks from the price feed
2. Aggregates into 1-minute candles
3. Computes indicators on each completed candle
4. Detects market regime
5. Generates signals with confidence scoring
6. Applies risk management rules
7. Executes trades
8. Manages open positions (trailing stops, exits)
9. Records results and feeds them to the adaptive learner
10. Logs everything for analysis

The agent runs an async event loop processing ticks in real-time.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

import structlog

from scalper.config import ScalperConfig, TradingSession
from scalper.models import (
    Candle, MarketRegime, Order, OrderType, Position,
    Side, Signal, Tick, TradeResult,
)
from scalper.feeds.candle_aggregator import CandleAggregator
from scalper.feeds.price_feed import PriceFeed
from scalper.analysis.indicators import IndicatorEngine, IndicatorState
from scalper.analysis.regime import RegimeDetector, RegimeState
from scalper.strategy.signal_generator import SignalGenerator
from scalper.risk.manager import RiskManager
from scalper.execution.engine import (
    ExecutionEngine, SimulatedExecution, create_order,
)
from scalper.adaptive.learner import AdaptiveLearner

logger = structlog.get_logger()


class TradingAgent:
    """The adaptive NQ scalping agent."""

    def __init__(self, config: ScalperConfig, feed: PriceFeed, execution: ExecutionEngine):
        self.config = config
        self.feed = feed
        self.execution = execution

        # Core components
        self.aggregator = CandleAggregator(
            interval_sec=config.candle_interval_sec,
            max_candles=config.lookback_candles,
            on_candle=self._on_candle_complete,
        )
        self.indicators = IndicatorEngine(
            ema_fast=config.ema_fast,
            ema_slow=config.ema_slow,
            ema_trend=config.ema_trend,
            atr_period=14,
            atr_fast_period=config.regime_atr_fast,
            rsi_period=config.rsi_period,
            bb_period=config.bb_period,
            bb_std=config.bb_std,
        )
        self.regime_detector = RegimeDetector(
            lookback=config.regime_lookback,
            atr_fast=config.regime_atr_fast,
            atr_slow=config.regime_atr_slow,
        )
        self.signal_gen = SignalGenerator(config)
        self.risk_mgr = RiskManager(config)
        self.learner = AdaptiveLearner(config)

        # State
        self._running = False
        self._current_indicators: Optional[IndicatorState] = None
        self._current_regime: Optional[RegimeState] = None
        self._current_signal: Optional[Signal] = None
        self._position: Optional[Position] = None
        self._last_signal_reasons: list[str] = []
        self._candles_since_entry: int = 0

        # Stats
        self._tick_count = 0
        self._candle_count = 0
        self._signal_count = 0
        self._trade_count = 0
        self._start_time = 0.0

    def preload_candles(self, candles: list[Candle]) -> None:
        """Preload historical candles for instant warmup (no 50-min wait)."""
        for c in candles:
            self.aggregator.process_candle(c)
        self._candle_count = len(candles)

        # Compute indicators on preloaded data so dashboard shows values immediately
        all_candles = self.aggregator.get_candles()
        if len(all_candles) >= 2:
            self._current_indicators = self.indicators.compute(all_candles)
            self._current_regime = self.regime_detector.detect(all_candles, self._current_indicators)

        logger.info("preloaded_candles", count=len(candles))

    async def run(self) -> None:
        """Main run loop - connect to feed and process ticks."""
        self._running = True
        self._start_time = time.time()

        await self.feed.connect()

        try:
            async for tick in self.feed.stream():
                if not self._running:
                    break

                await self._process_tick(tick)

        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        except Exception as e:
            logger.error("agent_error", error=str(e))
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        """Gracefully stop the agent."""
        logger.info("agent_stopping")
        self._running = False

    async def _process_tick(self, tick: Tick) -> None:
        """Process a single tick through the pipeline."""
        self._tick_count += 1

        # Update execution engine with price + bid/ask
        if isinstance(self.execution, SimulatedExecution):
            self.execution.update_price(tick.price)

        # Aggregate into candles
        completed = self.aggregator.process_tick(tick)

        # Check stops AND pending limit orders on every tick
        if isinstance(self.execution, SimulatedExecution):
            trade = self.execution.check_stops_and_limits(tick.price)
            if trade:
                await self._on_trade_closed(trade, trade.exit_reason or "stop_or_target")
            # Also check if we got filled on a pending entry limit order
            if self._position is None and self.execution.has_position:
                pos = await self.execution.get_position()
                if pos:
                    self._position = pos
                    self._candles_since_entry = 0
                    self._trade_count += 1

        # Update position P&L on every tick
        if self._position:
            self._position.update_pnl(tick.price, self.config.point_value)

            # Check risk manager exits
            if self._current_indicators and self._current_regime:
                should_exit, reason = self.risk_mgr.should_exit(
                    self._position, tick.price, self._current_indicators, self._current_regime,
                )
                if should_exit:
                    await self._exit_position(reason)

    def _on_candle_complete(self, candle: Candle) -> None:
        """Called when a 1-minute candle completes. This drives all analysis."""
        self._candle_count += 1

        candles = self.aggregator.get_candles()
        if len(candles) < self.config.warmup_candles:
            logger.debug("warming_up", candles=len(candles), needed=self.config.warmup_candles)
            return

        # Check session filter
        if not self._is_tradeable_session():
            return

        # 1. Compute indicators
        self._current_indicators = self.indicators.compute(candles)

        # 2. Detect regime
        self._current_regime = self.regime_detector.detect(candles, self._current_indicators)

        if isinstance(self.execution, SimulatedExecution):
            self.execution.set_regime(self._current_regime.regime)

        # 3. Manage open position
        if self._position:
            self._candles_since_entry += 1
            self._manage_position(candles, self._current_indicators, self._current_regime)
        else:
            # 4. Generate signal (only when flat)
            self._evaluate_entry(candles, self._current_indicators, self._current_regime)

        # Log state periodically
        if self._candle_count % 10 == 0:
            self._log_state()

    def _evaluate_entry(
        self,
        candles: list[Candle],
        indicators: IndicatorState,
        regime: RegimeState,
    ) -> None:
        """Evaluate whether to enter a new trade."""
        # Check if we can trade
        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            return

        # Generate signal
        signal = self.signal_gen.generate(candles, indicators, regime)
        if signal is None:
            return

        # Apply adaptive confidence threshold
        adaptive_threshold = self.learner.get_confidence_threshold()
        if signal.confidence < adaptive_threshold:
            return

        # Apply adaptive regime weight
        regime_weight = self.learner.get_regime_weight(regime.regime)
        if regime_weight < 0.5:
            return  # Skip regimes we're performing poorly in

        # Compute risk-managed stop and target
        stop = self.risk_mgr.compute_stop(
            signal.side, signal.entry_price, indicators, regime,
        )
        # Apply adaptive stop multiplier
        stop_dist = abs(signal.entry_price - stop)
        stop_dist *= self.learner.params.stop_multiplier
        if signal.side == Side.LONG:
            stop = signal.entry_price - stop_dist
        else:
            stop = signal.entry_price + stop_dist

        signal.stop_price = stop

        target = self.risk_mgr.compute_target(
            signal.side, signal.entry_price, stop, indicators, regime,
        )
        # Apply adaptive target multiplier
        target_dist = abs(signal.entry_price - target)
        target_dist *= self.learner.params.target_multiplier
        if signal.side == Side.LONG:
            target = signal.entry_price + target_dist
        else:
            target = signal.entry_price - target_dist

        signal.target_price = target

        # Check R:R ratio
        if signal.rr_ratio < self.config.min_rr_ratio:
            return

        # Compute position size
        size = self.risk_mgr.compute_position_size(signal, indicators, regime)
        if size <= 0:
            return

        # Execute entry
        self._signal_count += 1
        self._last_signal_reasons = signal.reasons
        asyncio.get_event_loop().create_task(
            self._enter_position(signal, size)
        )

    async def _enter_position(self, signal: Signal, size: int) -> None:
        """Enter a new position using LIMIT orders to minimize slippage.

        For longs: place limit at current ask (or signal entry price)
        For shorts: place limit at current bid (or signal entry price)
        Limit orders fill at our price or better = zero slippage.
        """
        # Cancel any existing pending orders first
        if isinstance(self.execution, SimulatedExecution):
            await self.execution.cancel_all_pending()

        # Use limit order at the signal's entry price (which is the candle close)
        # Add 1 tick buffer to increase fill probability
        tick = self.config.tick_size
        if signal.side == Side.LONG:
            limit_price = signal.entry_price + tick  # willing to pay 1 tick above close
        else:
            limit_price = signal.entry_price - tick  # willing to sell 1 tick below close

        # Entry order: LIMIT
        entry_order = create_order(
            symbol=self.config.symbol,
            side=signal.side,
            quantity=size,
            order_type=OrderType.LIMIT,
            price=limit_price,
            stop_price=signal.stop_price,
        )

        filled = await self.execution.submit_order(entry_order)

        if filled.status.value == "filled":
            # Immediate fill
            self._position = Position(
                symbol=self.config.symbol,
                side=signal.side,
                quantity=size,
                entry_price=filled.fill_price,
                entry_time=time.time(),
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                signal_confidence=signal.confidence,
            )
            self._candles_since_entry = 0
            self._trade_count += 1
            self._place_exit_orders(signal, size)

            logger.info(
                "trade_entered",
                side=signal.side.value,
                price=self._position.entry_price,
                stop=signal.stop_price,
                target=signal.target_price,
                size=size,
                order_type="LIMIT",
                confidence=round(signal.confidence, 3),
                regime=signal.regime.value,
            )
        else:
            # Pending - will be checked on future ticks
            # Store signal info for when it fills
            self._pending_signal = signal
            self._pending_size = size
            logger.info(
                "limit_order_pending",
                side=signal.side.value,
                limit=limit_price,
                stop=signal.stop_price,
                target=signal.target_price,
            )

    def _place_exit_orders(self, signal: Signal, size: int) -> None:
        """Place stop-loss and take-profit orders after entry."""
        # Stop loss: use STOP order (will have 1 tick slippage in sim)
        stop_side = Side.SHORT if signal.side == Side.LONG else Side.LONG
        stop_order = create_order(
            symbol=self.config.symbol,
            side=stop_side,
            quantity=size,
            order_type=OrderType.STOP,
            stop_price=signal.stop_price,
        )

        # Take profit: use LIMIT order (fills at target, no slippage)
        target_order = create_order(
            symbol=self.config.symbol,
            side=stop_side,
            quantity=size,
            order_type=OrderType.LIMIT,
            price=signal.target_price,
        )

        # Submit both (fire and forget in sim)
        asyncio.get_event_loop().create_task(
            self.execution.submit_order(stop_order)
        )
        asyncio.get_event_loop().create_task(
            self.execution.submit_order(target_order)
        )

    def _manage_position(
        self,
        candles: list[Candle],
        indicators: IndicatorState,
        regime: RegimeState,
    ) -> None:
        """Manage open position: trail stops, check for exit signals."""
        if not self._position:
            return

        price = candles[-1].close

        # Update trailing stop
        new_trail = self.risk_mgr.compute_trailing_stop(
            self._position, price, indicators, regime,
        )
        if self._position.side == Side.LONG:
            if new_trail > self._position.trailing_stop:
                self._position.trailing_stop = new_trail
        else:
            if self._position.trailing_stop == 0 or new_trail < self._position.trailing_stop:
                self._position.trailing_stop = new_trail

        # Check for signal-based exit (counter-signal)
        counter_signal = self.signal_gen.generate(candles, indicators, regime)
        if counter_signal and counter_signal.side != self._position.side:
            if counter_signal.confidence > 0.65:
                asyncio.get_event_loop().create_task(
                    self._exit_position("counter_signal")
                )
                return

        # Time-based exit: if holding too long in a scalp
        if self._candles_since_entry > 15:
            if self._position.unrealized_pnl > 0:
                asyncio.get_event_loop().create_task(
                    self._exit_position("time_exit_with_profit")
                )
            elif self._candles_since_entry > 30:
                asyncio.get_event_loop().create_task(
                    self._exit_position("max_hold_time")
                )

    async def _exit_position(self, reason: str) -> None:
        """Exit current position.

        Uses market order for emergency exits (risk limit, flatten time).
        Uses flatten (market) for other exits since we need to get out NOW.
        Stop/target exits are already handled by pending orders.
        """
        if not self._position:
            return

        if isinstance(self.execution, SimulatedExecution):
            # Cancel any pending exit orders first (we're overriding them)
            await self.execution.cancel_all_pending()
            trade = await self.execution.flatten()
            if trade:
                trade.exit_reason = reason
                trade.holding_candles = self._candles_since_entry
                await self._on_trade_closed(trade, reason)
        else:
            exit_side = Side.SHORT if self._position.side == Side.LONG else Side.LONG
            order = create_order(
                symbol=self.config.symbol,
                side=exit_side,
                quantity=self._position.quantity,
                order_type=OrderType.MARKET,  # emergency = market
            )
            await self.execution.submit_order(order)

    async def _on_trade_closed(self, trade: TradeResult, reason: str) -> None:
        """Handle trade closure - record and learn."""
        trade.exit_reason = reason
        trade.holding_candles = self._candles_since_entry

        # Record in risk manager
        self.risk_mgr.record_trade(trade)

        # Record in learner
        self.learner.record_trade(trade, self._last_signal_reasons)

        self._position = None
        self._candles_since_entry = 0

        logger.info(
            "trade_closed",
            pnl=round(trade.pnl, 2),
            side=trade.side.value,
            entry=trade.entry_price,
            exit=trade.exit_price,
            reason=reason,
            daily_pnl=round(self.risk_mgr.state.daily_pnl, 2),
            holding_candles=trade.holding_candles,
        )

    def _is_tradeable_session(self) -> bool:
        """Check if current time is in an allowed trading session."""
        now = datetime.now(timezone.utc)
        # Convert to ET (UTC-4 or UTC-5 depending on DST)
        # Simplified: use UTC-4 for EDT
        et_hour = (now.hour - 4) % 24

        if 18 <= et_hour or et_hour < 2:
            return self.config.trade_asian
        elif 2 <= et_hour < 8:
            return self.config.trade_london
        elif 8 <= et_hour < 12:
            return self.config.trade_ny_open
        elif 12 <= et_hour < 16:
            return self.config.trade_ny_afternoon
        elif 16 <= et_hour < 17:
            return self.config.trade_ny_close
        return False

    def _log_state(self) -> None:
        """Log current agent state."""
        regime = self._current_regime.regime.value if self._current_regime else "unknown"
        logger.info(
            "agent_state",
            ticks=self._tick_count,
            candles=self._candle_count,
            signals=self._signal_count,
            trades=self._trade_count,
            daily_pnl=round(self.risk_mgr.state.daily_pnl, 2),
            regime=regime,
            risk_mult=round(self.risk_mgr.state.risk_multiplier, 2),
            position="flat" if not self._position else self._position.side.value,
            adaptive=self.learner.get_stats_summary(),
        )

    async def _shutdown(self) -> None:
        """Clean shutdown: flatten positions, disconnect."""
        logger.info("agent_shutting_down")

        # Flatten any open position
        if self._position:
            await self._exit_position("shutdown")

        # Force close any partial candle
        self.aggregator.force_close()

        # Disconnect feed
        await self.feed.disconnect()

        # Final stats
        self._log_final_stats()

    def _log_final_stats(self) -> None:
        """Log final session statistics."""
        stats = self.learner.get_stats_summary()
        risk = self.risk_mgr.state

        logger.info(
            "session_complete",
            duration_min=round((time.time() - self._start_time) / 60, 1),
            total_ticks=self._tick_count,
            total_candles=self._candle_count,
            total_signals=self._signal_count,
            total_trades=self._trade_count,
            daily_pnl=round(risk.daily_pnl, 2),
            total_pnl=round(risk.total_pnl, 2),
            wins=risk.wins_today,
            losses=risk.losses_today,
            win_rate=round(self.risk_mgr.win_rate, 3),
            drawdown_remaining=round(risk.trailing_drawdown_remaining, 2),
            adaptive_stats=stats,
        )

    def get_status(self) -> dict:
        """Get current agent status for dashboard."""
        return {
            "running": self._running,
            "ticks": self._tick_count,
            "candles": self._candle_count,
            "signals": self._signal_count,
            "trades": self._trade_count,
            "position": {
                "side": self._position.side.value if self._position else None,
                "entry": self._position.entry_price if self._position else None,
                "pnl": round(self._position.unrealized_pnl, 2) if self._position else 0,
                "stop": self._position.stop_price if self._position else None,
                "target": self._position.target_price if self._position else None,
                "trail": self._position.trailing_stop if self._position else None,
            },
            "risk": {
                "daily_pnl": round(self.risk_mgr.state.daily_pnl, 2),
                "drawdown_remaining": round(self.risk_mgr.state.trailing_drawdown_remaining, 2),
                "risk_multiplier": round(self.risk_mgr.state.risk_multiplier, 2),
                "is_locked": self.risk_mgr.state.is_locked,
                "consecutive_losses": self.risk_mgr.state.consecutive_losses,
            },
            "regime": self._current_regime.regime.value if self._current_regime else "unknown",
            "indicators": {
                "ema_fast": round(self._current_indicators.ema_fast, 2) if self._current_indicators else 0,
                "ema_slow": round(self._current_indicators.ema_slow, 2) if self._current_indicators else 0,
                "rsi": round(self._current_indicators.rsi, 1) if self._current_indicators else 50,
                "atr": round(self._current_indicators.atr, 2) if self._current_indicators else 0,
                "vwap": round(self._current_indicators.vwap, 2) if self._current_indicators else 0,
            },
            "adaptive": self.learner.get_stats_summary(),
        }
