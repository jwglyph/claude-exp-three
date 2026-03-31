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
    Side, Signal, SignalType, Tick, TradeResult,
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
from scalper.analysis.multi_tf import MultiTimeframeEngine, HTFConfluence
from scalper.analysis.tick_analyzer import TickAnalyzer, TickEvent
from scalper.analysis.orderflow import OrderFlowEngine, OrderFlowState
from scalper.journal import TradeJournal
from scalper.risk.dynamic import DynamicRiskEngine

logger = structlog.get_logger()


class TradingAgent:
    """The adaptive NQ scalping agent with multi-timeframe and intra-candle analysis."""

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

        # Multi-timeframe engine (5m, 15m, 1h)
        self.mtf = MultiTimeframeEngine()

        # Intra-candle tick analyzer (all thresholds are ATR-adaptive)
        self.tick_analyzer = TickAnalyzer()

        # Order flow engine (tape reading, delta, absorption, imbalances)
        self.orderflow = OrderFlowEngine()
        self.orderflow.set_tick_size(config.tick_size)
        self._current_flow: Optional[OrderFlowState] = None

        # Dynamic risk engine (Kelly criterion, replaces fixed risk params)
        self.dynamic_risk = DynamicRiskEngine(max_drawdown=config.max_drawdown)

        # Trade journal (logs everything for iteration)
        self.journal = TradeJournal()

        # State
        self._running = False
        self._current_indicators: Optional[IndicatorState] = None
        self._current_regime: Optional[RegimeState] = None
        self._current_confluence: Optional[HTFConfluence] = None
        self._current_signal: Optional[Signal] = None
        self._position: Optional[Position] = None
        self._last_signal_reasons: list[str] = []
        self._last_tick_event: Optional[TickEvent] = None
        self._candles_since_entry: int = 0
        self._pending_signal: Optional[Signal] = None
        self._pending_size: int = 0

        # Stats
        self._tick_count = 0
        self._candle_count = 0
        self._signal_count = 0
        self._trade_count = 0
        self._intra_candle_trades = 0
        self._start_time = 0.0

    def preload_candles(self, candles: list[Candle]) -> None:
        """Preload historical candles for instant warmup."""
        for c in candles:
            self.aggregator.process_candle(c)
            # Also feed to multi-timeframe engine
            self.mtf.process_1m_candle(c)
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
        """Process a single tick through the full pipeline.

        On EVERY tick:
        1. Update execution engine (check fills, stops)
        2. Feed to candle aggregator
        3. Feed to multi-timeframe engine
        4. Feed to order flow engine
        5. Run intra-candle tick analysis
        6. If tick event detected + HTF/orderflow confluence → consider immediate entry
        7. Update position P&L and check risk exits
        """
        self._tick_count += 1

        # Update execution engine with price
        if isinstance(self.execution, SimulatedExecution):
            self.execution.update_price(tick.price)

        # Aggregate into 1m candles
        completed = self.aggregator.process_tick(tick)

        # Feed to multi-timeframe engine
        self.mtf.process_tick(tick.timestamp, tick.price, tick.size, tick.side)

        # Feed to order flow engine - all ticks with a side (buy/sell)
        if tick.side:
            self.orderflow.process_trade(tick.timestamp, tick.price, tick.size, tick.side)

        # Update flow state periodically (every 50 ticks to avoid overhead)
        if self._tick_count % 50 == 0:
            self._current_flow = self.orderflow.get_state()

        # Check stops AND pending limit orders on every tick
        if isinstance(self.execution, SimulatedExecution):
            trade = self.execution.check_stops_and_limits(tick.price)
            if trade:
                await self._on_trade_closed(trade, trade.exit_reason or "stop_or_target")
            # Check if pending entry limit got filled
            if self._position is None and self.execution.has_position:
                pos = await self.execution.get_position()
                if pos:
                    self._position = pos
                    self._candles_since_entry = 0
                    self._trade_count += 1
                    if self._pending_signal:
                        self._place_exit_orders(self._pending_signal, self._pending_size)
                        self._pending_signal = None

        # Intra-candle tick analysis (only when we have indicators and no position)
        if self._current_indicators and self._current_regime and self._position is None:
            tick_event = self.tick_analyzer.process_tick(tick, self._current_indicators)
            if tick_event:
                self._last_tick_event = tick_event
                self._evaluate_intra_candle_entry(tick_event, tick)

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

    def _evaluate_intra_candle_entry(self, event: TickEvent, tick: Tick) -> None:
        """Evaluate an intra-candle tick event for immediate entry.

        Requires confluence across:
        1. Tick event direction
        2. HTF agreement
        3. Order flow agreement (delta, absorption, imbalances)
        4. 1m indicators support
        5. Risk manager allows
        """
        if not self._is_tradeable_session():
            return

        can_trade, _ = self.risk_mgr.can_trade()
        if not can_trade:
            return

        # HTF confluence
        confluence = self.mtf.get_confluence()
        self._current_confluence = confluence

        # Order flow state
        flow = self.orderflow.get_state()
        self._current_flow = flow

        # Determine side from event
        if event.direction > 0:
            side = Side.LONG
        elif event.direction < 0:
            side = Side.SHORT
        else:
            return

        # Check HTF agreement
        if confluence.agrees_with is not None and confluence.agrees_with != side:
            self.journal.log_tick_event(event.event_type, event.direction, event.magnitude,
                                        event.price, event.description, taken=False)
            return

        # Check order flow agreement
        flow_agrees = (
            (side == Side.LONG and flow.flow_bias > 0.1) or
            (side == Side.SHORT and flow.flow_bias < -0.1) or
            flow.flow_bias == 0  # neutral flow = don't block
        )
        if not flow_agrees:
            self.journal.log_tick_event(event.event_type, event.direction, event.magnitude,
                                        event.price, f"flow_disagrees:{flow.flow_bias:.2f}", taken=False)
            return

        # Check 1m indicators
        ind = self._current_indicators
        if side == Side.LONG and ind.rsi > 75:
            return
        if side == Side.SHORT and ind.rsi < 25:
            return  # too oversold

        # Build confidence from event + confluence + order flow + indicators
        base_conf = 0.40 + event.magnitude * 0.15  # 0.40-0.55 from event alone
        if confluence.strength > 0.5:
            base_conf += 0.08  # HTF boost
        if ind.trend_direction == (1 if side == Side.LONG else -1):
            base_conf += 0.05  # 1m trend aligned

        # Order flow confidence boost (the real edge)
        flow_strength = abs(flow.flow_bias)
        base_conf += flow_strength * 0.15  # up to 0.15 from flow

        # Strong order flow signals get extra weight
        if flow.absorption != 0:
            # Absorption in our direction = smart money supporting
            if (side == Side.LONG and flow.absorption > 0.3) or \
               (side == Side.SHORT and flow.absorption < -0.3):
                base_conf += 0.08
        if flow.stacked_buy_levels >= 3 and side == Side.LONG:
            base_conf += 0.07  # institutional buying stacked
        if flow.stacked_sell_levels >= 3 and side == Side.SHORT:
            base_conf += 0.07
        if flow.large_print_net > 0 and side == Side.LONG:
            base_conf += 0.05
        elif flow.large_print_net < 0 and side == Side.SHORT:
            base_conf += 0.05

        # Delta divergence = warning, reduce confidence
        price_dir = 1 if side == Side.LONG else -1
        divergence = self.orderflow.get_delta_divergence(price_dir)
        if divergence > 0.3:
            base_conf -= divergence * 0.15

        # Must meet minimum threshold
        adaptive_threshold = self.learner.get_confidence_threshold()
        if base_conf < adaptive_threshold:
            return

        # Build signal
        regime = self._current_regime
        stop = self.risk_mgr.compute_stop(side, tick.price, ind, regime)
        target = self.risk_mgr.compute_target(side, tick.price, stop, ind, regime)

        signal = Signal(
            timestamp=tick.timestamp,
            signal_type=SignalType.LONG if side == Side.LONG else SignalType.SHORT,
            confidence=base_conf,
            side=side,
            entry_price=tick.price,
            stop_price=stop,
            target_price=target,
            regime=regime.regime,
            reasons=[f"intra:{event.event_type}", event.description, confluence.description],
        )

        # Check R:R
        if signal.rr_ratio < self.config.min_rr_ratio:
            return

        # Size and execute
        size = self.risk_mgr.compute_position_size(signal, ind, regime)
        if size <= 0:
            return

        self._signal_count += 1
        self._intra_candle_trades += 1
        self._last_signal_reasons = signal.reasons
        asyncio.get_event_loop().create_task(self._enter_position(signal, size))

    def _on_candle_complete(self, candle: Candle) -> None:
        """Called when a 1-minute candle completes."""
        self._candle_count += 1

        # Feed to multi-timeframe engine
        self.mtf.process_1m_candle(candle)

        # Reset intra-candle analyzer for new candle
        self.tick_analyzer.reset_candle()

        candles = self.aggregator.get_candles()
        if len(candles) < self.config.warmup_candles:
            return

        if not self._is_tradeable_session():
            return

        # 1. Compute 1m indicators
        self._current_indicators = self.indicators.compute(candles)

        # 2. Detect 1m regime
        self._current_regime = self.regime_detector.detect(candles, self._current_indicators)

        # 3. Update HTF confluence
        self._current_confluence = self.mtf.get_confluence()

        # 4. Update tick analyzer with current levels
        self.tick_analyzer.update_context(self._current_indicators)

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
        """Evaluate whether to enter on candle close (with HTF confluence)."""
        can_trade, reason = self.risk_mgr.can_trade()
        if not can_trade:
            return

        # Generate 1m signal
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
            return

        # HTF confluence check
        confluence = self._current_confluence
        if confluence:
            if confluence.agrees_with is not None and confluence.agrees_with != signal.side:
                # HTF disagrees - need much higher confidence to override
                if signal.confidence < 0.80:
                    return
            elif confluence.agrees_with == signal.side:
                # HTF agrees - boost confidence
                signal.confidence = min(1.0, signal.confidence + confluence.strength * 0.1)
                signal.reasons.append(f"htf:{confluence.description}")

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
            self.journal.log_signal(signal, taken=False, skip_reason="rr_too_low")
            return

        # Dynamic risk: should we trade at all?
        should, reason = self.dynamic_risk.should_trade(self.risk_mgr.state.trailing_drawdown_remaining)
        if not should:
            self.journal.log_signal(signal, taken=False, skip_reason=f"dynamic:{reason}")
            return

        # Dynamic position sizing (Kelly-based when enough data, conservative otherwise)
        regime_quality = self.learner.get_regime_weight(regime.regime)
        optimal_risk = self.dynamic_risk.optimal_risk_dollars(
            remaining_drawdown=self.risk_mgr.state.trailing_drawdown_remaining,
            signal_confidence=signal.confidence,
            regime_quality=regime_quality,
        )

        # Convert optimal risk $ to contracts
        stop_distance = abs(signal.entry_price - signal.stop_price)
        per_contract_risk = stop_distance * self.config.point_value + self.config.commission_rt
        if per_contract_risk <= 0:
            return

        size = max(1, int(optimal_risk / per_contract_risk))

        # Cap by scaling plan
        size = min(size, self.risk_mgr.get_max_contracts())

        # Log signal
        htf_info = self._current_confluence.__dict__ if self._current_confluence else None
        self.journal.log_signal(signal, taken=True, size=size, htf_confluence=htf_info)

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

        # Record in dynamic risk engine
        risk_amount = abs(trade.entry_price - trade.exit_price) * self.config.point_value * trade.quantity
        self.dynamic_risk.record_trade(trade.pnl, risk_amount)

        # Record in learner
        self.learner.record_trade(trade, self._last_signal_reasons)

        # Journal
        ind_snapshot = {
            "atr": round(self._current_indicators.atr, 2) if self._current_indicators else 0,
            "rsi": round(self._current_indicators.rsi, 1) if self._current_indicators else 0,
            "regime": self._current_regime.regime.value if self._current_regime else "unknown",
        }
        self.journal.log_trade_exit(trade, indicators=ind_snapshot)

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
        # HTF info
        confluence = self._current_confluence
        htf = {}
        if confluence:
            htf = {
                "score": round(confluence.score, 2),
                "strength": round(confluence.strength, 2),
                "agrees": confluence.agrees_with.value if confluence.agrees_with else "neutral",
                "5m": confluence.tf_5m_trend,
                "15m": confluence.tf_15m_trend,
                "1h": confluence.tf_1h_trend,
                "desc": confluence.description,
            }

        # Last tick event
        tick_evt = None
        if self._last_tick_event:
            e = self._last_tick_event
            tick_evt = {
                "type": e.event_type,
                "dir": e.direction,
                "mag": round(e.magnitude, 2),
                "desc": e.description,
                "age": round(time.time() - e.timestamp, 1),
            }

        return {
            "running": self._running,
            "ticks": self._tick_count,
            "candles": self._candle_count,
            "signals": self._signal_count,
            "trades": self._trade_count,
            "intra_candle_trades": self._intra_candle_trades,
            "position": {
                "side": self._position.side.value if self._position else None,
                "entry": self._position.entry_price if self._position else None,
                "pnl": round(self._position.unrealized_pnl, 2) if self._position else 0,
                "stop": self._position.stop_price if self._position else None,
                "target": self._position.target_price if self._position else None,
                "trail": self._position.trailing_stop if self._position else None,
            },
            "risk": {
                "daily_pnl": round(self.risk_mgr.state.daily_pnl_net, 2),
                "drawdown_remaining": round(self.risk_mgr.state.trailing_drawdown_remaining, 2),
                "risk_multiplier": round(self.risk_mgr.state.risk_multiplier, 2),
                "is_locked": self.risk_mgr.state.is_locked,
                "consecutive_losses": self.risk_mgr.state.consecutive_losses,
                "balance": round(self.risk_mgr.state.account_balance, 2),
                "max_contracts": self.risk_mgr.state.max_contracts_current,
            },
            "regime": self._current_regime.regime.value if self._current_regime else "unknown",
            "indicators": {
                "ema_fast": round(self._current_indicators.ema_fast, 2) if self._current_indicators else 0,
                "ema_slow": round(self._current_indicators.ema_slow, 2) if self._current_indicators else 0,
                "rsi": round(self._current_indicators.rsi, 1) if self._current_indicators else 50,
                "atr": round(self._current_indicators.atr, 2) if self._current_indicators else 0,
                "vwap": round(self._current_indicators.vwap, 2) if self._current_indicators else 0,
            },
            "htf": htf,
            "tick_event": tick_evt,
            "adaptive": self.learner.get_stats_summary(),
            "dynamic_risk": self.dynamic_risk.get_summary(),
            "orderflow": {
                "delta_1m": self._current_flow.delta_1m if self._current_flow else 0,
                "delta_5m": self._current_flow.delta_5m if self._current_flow else 0,
                "imbalance": round(self._current_flow.imbalance_ratio, 2) if self._current_flow else 0.5,
                "flow_bias": round(self._current_flow.flow_bias, 2) if self._current_flow else 0,
                "absorption": round(self._current_flow.absorption, 2) if self._current_flow else 0,
                "exhaustion": round(self._current_flow.exhaustion, 2) if self._current_flow else 0,
                "large_buy": self._current_flow.large_prints_buy if self._current_flow else 0,
                "large_sell": self._current_flow.large_prints_sell if self._current_flow else 0,
                "stacked_buy": self._current_flow.stacked_buy_levels if self._current_flow else 0,
                "stacked_sell": self._current_flow.stacked_sell_levels if self._current_flow else 0,
            },
        }
