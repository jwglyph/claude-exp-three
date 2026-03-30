"""Configuration for the adaptive NQ scalper."""

from __future__ import annotations

from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings


class TradingSession(str, Enum):
    ASIAN = "asian"          # 18:00-02:00 ET (Sun-Thu)
    LONDON = "london"        # 02:00-08:00 ET
    NY_OPEN = "ny_open"      # 08:00-12:00 ET
    NY_AFTERNOON = "ny_afternoon"  # 12:00-16:00 ET
    NY_CLOSE = "ny_close"    # 16:00-17:00 ET


class AccountSize(str, Enum):
    """TopstepX account tiers."""
    K50 = "50k"
    K100 = "100k"
    K150 = "150k"


# TopstepX rules per account size
TOPSTEP_RULES = {
    AccountSize.K50: {
        "max_contracts": 5,
        "daily_loss_limit": 1000,
        "trailing_drawdown": 2000,
        "profit_target": 3000,  # evaluation phase
    },
    AccountSize.K100: {
        "max_contracts": 10,
        "daily_loss_limit": 2000,
        "trailing_drawdown": 3000,
        "profit_target": 6000,
    },
    AccountSize.K150: {
        "max_contracts": 15,
        "daily_loss_limit": 3000,
        "trailing_drawdown": 4500,
        "profit_target": 9000,
    },
}


class ScalperConfig(BaseSettings):
    """Master configuration for the scalper agent."""

    model_config = {"env_prefix": "NQ_SCALPER_"}

    # --- Connection ---
    feed_url: str = Field(default="ws://localhost:8080/feed", description="WebSocket URL for live price feed")
    feed_type: str = Field(default="rithmic", description="Feed provider: rithmic, tradovate, sim")
    api_key: str = Field(default="", description="API key for feed/broker")
    api_secret: str = Field(default="", description="API secret for feed/broker")

    # --- Instrument ---
    symbol: str = Field(default="NQ", description="Futures symbol")
    tick_size: float = Field(default=0.25, description="NQ tick size = 0.25 points")
    tick_value: float = Field(default=5.0, description="NQ tick value = $5.00 per tick")
    point_value: float = Field(default=20.0, description="NQ point value = $20.00 per point")

    # --- Account / TopstepX ---
    account_size: AccountSize = Field(default=AccountSize.K50)
    max_contracts: int = Field(default=2, description="Max simultaneous contracts")
    daily_loss_limit: float = Field(default=800.0, description="Hard stop - daily loss limit in $")
    max_drawdown: float = Field(default=1800.0, description="Trailing max drawdown in $")
    profit_target_daily: float = Field(default=500.0, description="Daily profit target to reduce risk")

    # --- Risk per trade ---
    max_risk_per_trade: float = Field(default=200.0, description="Max dollar risk per trade")
    default_stop_ticks: int = Field(default=20, description="Default stop loss in ticks (5 points)")
    max_stop_ticks: int = Field(default=40, description="Maximum allowed stop in ticks (10 points)")
    min_rr_ratio: float = Field(default=1.5, description="Minimum reward:risk ratio")

    # --- Scalping parameters ---
    candle_interval_sec: int = Field(default=60, description="Candle interval (60 = 1 min)")
    lookback_candles: int = Field(default=200, description="Number of candles to keep in memory")
    warmup_candles: int = Field(default=50, description="Candles needed before trading")

    # --- Adaptive parameters ---
    regime_lookback: int = Field(default=20, description="Candles for regime detection")
    regime_atr_fast: int = Field(default=5, description="Fast ATR period for regime")
    regime_atr_slow: int = Field(default=20, description="Slow ATR period for regime")
    ema_fast: int = Field(default=9, description="Fast EMA period")
    ema_slow: int = Field(default=21, description="Slow EMA period")
    ema_trend: int = Field(default=50, description="Trend EMA period")
    rsi_period: int = Field(default=14, description="RSI period")
    rsi_overbought: float = Field(default=70.0, description="RSI overbought threshold")
    rsi_oversold: float = Field(default=30.0, description="RSI oversold threshold")
    bb_period: int = Field(default=20, description="Bollinger Band period")
    bb_std: float = Field(default=2.0, description="Bollinger Band std devs")
    vwap_enabled: bool = Field(default=True, description="Use VWAP")
    volume_profile_enabled: bool = Field(default=True, description="Use volume profile")

    # --- Confidence ---
    min_confidence: float = Field(default=0.55, description="Minimum confidence to enter trade")
    high_confidence: float = Field(default=0.75, description="High confidence threshold (allows larger size)")

    # --- Session filters ---
    trade_asian: bool = Field(default=False, description="Trade during Asian session")
    trade_london: bool = Field(default=True, description="Trade during London session")
    trade_ny_open: bool = Field(default=True, description="Trade during NY open")
    trade_ny_afternoon: bool = Field(default=True, description="Trade during NY afternoon")
    trade_ny_close: bool = Field(default=False, description="Trade during NY close (thin liquidity)")

    # --- Adaptive learning ---
    adapt_window: int = Field(default=50, description="Window of recent trades for adaptation")
    adapt_rate: float = Field(default=0.1, description="Learning rate for parameter adaptation")

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_trades: bool = Field(default=True)
    log_signals: bool = Field(default=True)
