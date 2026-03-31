"""NQ Adaptive 1-Minute Candle Scalper for TopstepX."""

__version__ = "0.16.0"

# Version changelog:
# 0.16.0 - CRITICAL FIX: type=1 is SELL (bid hit), type=0 is BUY (ask lift). Was inverted!
# 0.15.1 - Debug: dump raw trade data to logs/trade_debug.txt to diagnose side bias
# 0.14.0 - Adaptive trailing: tightens at 3R (0.7x ATR) and 5R+ (0.5x ATR)
# 0.13.0 - Realistic targets: capped at 1.0-1.5x ATR, trailing stop handles runners
# 0.12.0 - Version tracking: all logs/journal tagged with version
# 0.11.0 - Fix stop distance: pure ATR stops, no dollar cap crushing to 5pts
# 0.10.0 - Remove trading locks for paper testing
# 0.9.0  - Fix long-only bias: regime faster flip, bidirectional signals, no side penalty
# 0.8.0  - Replace session filter with market quality filter
# 0.7.0  - Fix always-positive flow bias, add buy/sell diagnostics
# 0.6.0  - Order flow intelligence engine (delta, absorption, stacked, large prints)
# 0.5.0  - Dynamic Kelly-based risk, trade journal logging
# 0.4.0  - ATR-relative thresholds everywhere, no fixed points
# 0.3.0  - Multi-timeframe confluence (5m/15m/1h), intra-candle tick execution
# 0.2.0  - TopstepX 50K XFA risk optimization, limit orders
# 0.1.0  - Initial: TopstepX API, SignalR feed, candle aggregation, regime detection
