"""Configuration for the adaptive NQ scalper.

Optimized for TopstepX 50K Express Funded Account (XFA) rules:
- $2,000 trailing drawdown (trails from EOD balance high, locks at $2,100)
- Scaling plan: 2 contracts at start, 3 above $1,500, 5 above $2,000
- Must flatten by 3:10 PM CT (no overnight holds)
- $2.80 round turn commission per NQ contract
- Payout: 5 winning days of $150+, 90/10 split
"""

from __future__ import annotations

from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings


class TradingSession(str, Enum):
    ASIAN = "asian"              # 18:00-02:00 ET (Sun-Thu)
    LONDON = "london"            # 02:00-08:00 ET
    NY_PREMARKET = "ny_premarket" # 08:00-09:30 ET
    NY_OPEN = "ny_open"          # 09:30-12:00 ET (best session)
    NY_AFTERNOON = "ny_afternoon" # 12:00-15:00 ET
    NY_CLOSE = "ny_close"        # 15:00-16:10 ET (must flatten by 16:10 ET / 15:10 CT)


class AccountSize(str, Enum):
    """TopstepX account tiers."""
    K50 = "50k"
    K100 = "100k"
    K150 = "150k"


# TopstepX XFA rules per account size (as of 2026)
TOPSTEP_XFA_RULES = {
    AccountSize.K50: {
        "trailing_drawdown": 2000,       # trails from EOD balance high
        "drawdown_lock_balance": 2100,   # when EOD hits this, floor locks at $100
        "drawdown_lock_floor": 100,      # permanent floor after lock
        "commission_rt": 2.80,           # round turn per NQ contract
        "scaling_plan": [                # (min_balance, max_contracts)
            (0, 2),                      # start: max 2 NQ
            (1500, 3),                   # above $1,500: max 3 NQ
            (2000, 5),                   # above $2,000: max 5 NQ
        ],
        "combine_profit_target": 3000,
        "combine_consistency": 0.50,     # best day < 50% of total
        "payout_min_winning_days": 5,
        "payout_min_per_day": 150,
        "payout_split": 0.90,
        "flatten_time_ct": "15:10",      # 3:10 PM Central Time
    },
    AccountSize.K100: {
        "trailing_drawdown": 3000,
        "drawdown_lock_balance": 3100,
        "drawdown_lock_floor": 100,
        "commission_rt": 2.80,
        "scaling_plan": [
            (0, 4),
            (1500, 7),
            (2000, 10),
        ],
        "combine_profit_target": 6000,
        "combine_consistency": 0.50,
        "payout_min_winning_days": 5,
        "payout_min_per_day": 150,
        "payout_split": 0.90,
        "flatten_time_ct": "15:10",
    },
    AccountSize.K150: {
        "trailing_drawdown": 4500,
        "drawdown_lock_balance": 4600,
        "drawdown_lock_floor": 100,
        "commission_rt": 2.80,
        "scaling_plan": [
            (0, 6),
            (1500, 10),
            (2000, 15),
        ],
        "combine_profit_target": 9000,
        "combine_consistency": 0.50,
        "payout_min_winning_days": 5,
        "payout_min_per_day": 150,
        "payout_split": 0.90,
        "flatten_time_ct": "15:10",
    },
}


class ScalperConfig(BaseSettings):
    """Master configuration for the scalper agent.

    Risk math for 50K XFA:
    - Total budget: $2,000 (trailing drawdown)
    - Target: preserve capital, grow steadily to $2,100+ (locks floor)
    - Max risk per trade: $100 (5% of drawdown) = 5 points on 1 NQ
    - Daily target: $150-300 (1 good trade per day is enough)
    - Daily stop: -$400 (20% of drawdown, leaves room for recovery)
    - With 2 contracts max, a 5-point stop = $200 risk
    """

    model_config = {"env_prefix": "NQ_SCALPER_"}

    # --- Connection ---
    feed_url: str = Field(default="ws://localhost:8080/feed")
    feed_type: str = Field(default="topstepx")
    api_key: str = Field(default="")
    api_secret: str = Field(default="")

    # --- Instrument ---
    symbol: str = Field(default="NQ")
    tick_size: float = Field(default=0.25, description="NQ tick size")
    tick_value: float = Field(default=5.0, description="$5.00 per tick")
    point_value: float = Field(default=20.0, description="$20.00 per point")
    commission_rt: float = Field(default=2.80, description="Round turn commission per contract")

    # --- Account / TopstepX 50K XFA ---
    account_size: AccountSize = Field(default=AccountSize.K50)

    # Trailing drawdown (THE critical number)
    max_drawdown: float = Field(default=2000.0, description="TopstepX trailing drawdown")
    drawdown_lock_balance: float = Field(default=2100.0, description="EOD balance that locks the floor")
    drawdown_lock_floor: float = Field(default=100.0, description="Permanent floor once locked")

    # Daily risk limits (self-imposed, TopstepX doesn't enforce on XFA)
    daily_loss_limit: float = Field(default=400.0, description="Self-imposed daily loss limit (20% of drawdown)")
    profit_target_daily: float = Field(default=300.0, description="Daily target - reduce risk after hitting")
    max_trades_per_day: int = Field(default=10, description="Max trades per day to prevent overtrading")

    # Risk per trade (conservative: 5% of drawdown per trade)
    max_risk_per_trade: float = Field(default=100.0, description="Max $ risk per trade (5% of $2K DD)")
    max_risk_high_conf: float = Field(default=150.0, description="Max risk on high-confidence trades")
    default_stop_ticks: int = Field(default=16, description="Default stop loss in ticks (4 points = $80/contract)")
    max_stop_ticks: int = Field(default=28, description="Max stop in ticks (7 points = $140/contract)")
    min_rr_ratio: float = Field(default=1.5, description="Minimum reward:risk ratio")

    # Scaling plan (contracts allowed based on account balance)
    max_contracts: int = Field(default=2, description="Starting max contracts (scaling plan)")

    # --- Scalping parameters ---
    candle_interval_sec: int = Field(default=60, description="1 minute candles")
    lookback_candles: int = Field(default=200, description="Candles in memory")
    warmup_candles: int = Field(default=50, description="Candles before trading")

    # --- Indicators ---
    regime_lookback: int = Field(default=20)
    regime_atr_fast: int = Field(default=5)
    regime_atr_slow: int = Field(default=20)
    ema_fast: int = Field(default=9)
    ema_slow: int = Field(default=21)
    ema_trend: int = Field(default=50)
    rsi_period: int = Field(default=14)
    rsi_overbought: float = Field(default=70.0)
    rsi_oversold: float = Field(default=30.0)
    bb_period: int = Field(default=20)
    bb_std: float = Field(default=2.0)
    vwap_enabled: bool = Field(default=True)
    volume_profile_enabled: bool = Field(default=True)

    # --- Confidence ---
    min_confidence: float = Field(default=0.60, description="Higher threshold for capital preservation")
    high_confidence: float = Field(default=0.75, description="Allows larger size / risk")

    # --- Session filters (optimized for 50K) ---
    trade_asian: bool = Field(default=False, description="OFF: thin, not worth the risk")
    trade_london: bool = Field(default=False, description="OFF for 50K: save bullets for NY")
    trade_ny_open: bool = Field(default=True, description="ON: best session")
    trade_ny_afternoon: bool = Field(default=True, description="ON: decent setups")
    trade_ny_close: bool = Field(default=False, description="OFF: must flatten, thin")

    # Flatten time: 3:10 PM CT = 4:10 PM ET = 16:10 ET
    flatten_time_et_hour: int = Field(default=16, description="Auto-flatten hour (ET)")
    flatten_time_et_minute: int = Field(default=5, description="Auto-flatten minute (5 min buffer before 16:10)")

    # --- Adaptive learning ---
    adapt_window: int = Field(default=50)
    adapt_rate: float = Field(default=0.1)

    # --- Consistency tracking ---
    consistency_target: float = Field(default=0.50, description="Best day must be < 50% of total profits")

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_trades: bool = Field(default=True)
    log_signals: bool = Field(default=True)
