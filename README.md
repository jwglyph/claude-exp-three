# NQ Adaptive 1-Minute Candle Scalper

An adaptive trading agent for scalping NQ (Nasdaq 100 E-mini futures) on platforms like TopstepX. The agent processes a live tick feed, builds 1-minute candles, and uses multi-factor analysis with adaptive regime detection to generate high-confidence entry/exit signals.

## Architecture

```
Live Feed (ticks) --> Candle Aggregator (1m OHLCV + delta)
                          |
                    Indicator Engine (EMA, RSI, ATR, BB, VWAP, Volume)
                          |
                    Regime Detector (trending/ranging/volatile/low-vol)
                          |
                    Signal Generator (multi-factor confluence scoring)
                          |
                    Risk Manager (sizing, stops, TopstepX limits)
                          |
                    Execution Engine (paper/live)
                          |
                    Adaptive Learner (online parameter tuning)
```

## Key Features

- **Adaptive Regime Detection**: Automatically classifies market state (trending up/down, ranging, volatile, low-volatility) and adjusts strategy accordingly
- **Multi-Factor Signal Generation**: Combines EMA alignment, RSI, Bollinger Bands, VWAP, candlestick patterns, volume/delta, and regime-specific signals into a confidence score
- **Intelligent Risk Management**: ATR-based dynamic stops, trailing stops, position sizing based on confidence + volatility, daily loss limits, trailing drawdown tracking
- **TopstepX Compliant**: Enforces prop firm rules (daily loss limits, trailing drawdown, max contracts)
- **Online Learning**: Adapts confidence thresholds, regime weights, stop/target multipliers, and signal weights based on recent trade performance
- **Session Awareness**: Filters trades by market session (Asian, London, NY Open, NY Afternoon, NY Close)

## Quick Start

```bash
# Install
pip install -e .

# Run with simulated feed (paper trading)
nq-scalper trade --feed-type sim --paper

# Run with live feed
nq-scalper trade --feed-type rithmic --feed-url ws://your-feed:8080 --paper

# Backtest on historical data
nq-scalper backtest historical_candles.csv --output results.json

# Show config
nq-scalper status
```

## Configuration

All settings can be configured via environment variables (prefix `NQ_SCALPER_`) or a `.env` file. See `.env.example` for all options.

Key parameters:
- `NQ_SCALPER_ACCOUNT_SIZE`: TopstepX account tier (50k/100k/150k)
- `NQ_SCALPER_MAX_CONTRACTS`: Maximum simultaneous contracts
- `NQ_SCALPER_DAILY_LOSS_LIMIT`: Hard daily loss limit ($)
- `NQ_SCALPER_MIN_CONFIDENCE`: Minimum confidence score to enter (0-1)
- `NQ_SCALPER_MIN_RR_RATIO`: Minimum reward:risk ratio

## How It Works

### 1. Price Feed
Connects to a WebSocket feed (Rithmic, Tradovate, or simulated) and receives raw tick data with price, size, and aggressor side.

### 2. Candle Building
Aggregates ticks into 1-minute OHLCV candles with buy/sell volume split for order flow analysis (delta).

### 3. Regime Detection
Classifies market into one of 5 regimes using:
- ATR ratio (fast/slow) for volatility shifts
- Price efficiency ratio for trend strength
- Mean-crossing frequency for mean reversion
- Hurst exponent approximation
- Bollinger Band width for compression

### 4. Signal Generation
Scores bullish vs bearish factors across:
- **EMA stack alignment** (fast > slow > trend = bullish)
- **RSI** (context-dependent: mean-reversion in ranges, momentum in trends)
- **Bollinger Bands** (fade in ranges, confirm breakouts in trends)
- **VWAP** (institutional reference level)
- **Candle patterns** (engulfing, hammer, shooting star, strong directional)
- **Volume** (confirmation via high volume + delta)
- **Regime-specific** (pullback-to-EMA in trends, range fading, compression breakouts)

### 5. Risk Management
- ATR-based stops that widen in volatile regimes, tighten in calm
- Trailing stops that move to breakeven at 1R, then trail ATR
- Position sizing: risk budget / (stop distance * point value), scaled by confidence
- Anti-martingale: reduces size after losses, gradually restores after wins
- Emergency exits: daily loss limit, trailing drawdown protection

### 6. Adaptive Learning
After each trade, the learner updates:
- **Confidence threshold**: raises after losing streaks, lowers when winning
- **Regime weights**: trades more in regimes with positive expectancy
- **Stop/target multipliers**: adjusts based on MAE/MFE analysis
- **Signal weights**: strengthens signals that lead to winners

## Testing

```bash
pip install -e ".[dev]"
pytest -v
```

## Disclaimer

This software is for educational and research purposes. Trading futures involves substantial risk of loss. Past performance does not guarantee future results. Always paper trade extensively before risking real capital.
