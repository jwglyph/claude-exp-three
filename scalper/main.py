"""CLI entry point for the NQ adaptive scalper."""

from __future__ import annotations

import asyncio
import signal
import sys

import click
import structlog

from scalper.config import ScalperConfig, AccountSize
from scalper.agent import TradingAgent
from scalper.feeds.price_feed import WebSocketFeed, SimulatedFeed, CandleReplayFeed
from scalper.execution.engine import SimulatedExecution, LiveExecution

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)


@click.group()
def cli():
    """NQ Adaptive 1-Minute Candle Scalper"""
    pass


@cli.command()
@click.option("--feed-url", default="ws://localhost:8080/feed", help="WebSocket feed URL")
@click.option("--feed-type", default="sim", type=click.Choice(["rithmic", "tradovate", "sim"]))
@click.option("--api-key", default="", help="API key")
@click.option("--api-secret", default="", help="API secret")
@click.option("--account-size", default="50k", type=click.Choice(["50k", "100k", "150k"]))
@click.option("--max-contracts", default=2, type=int, help="Maximum contracts")
@click.option("--daily-loss-limit", default=800.0, type=float, help="Daily loss limit ($)")
@click.option("--paper/--live", default=True, help="Paper trade or live")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]))
def trade(
    feed_url: str,
    feed_type: str,
    api_key: str,
    api_secret: str,
    account_size: str,
    max_contracts: int,
    daily_loss_limit: float,
    paper: bool,
    log_level: str,
):
    """Start live trading (paper or live)."""
    config = ScalperConfig(
        feed_url=feed_url,
        feed_type=feed_type,
        api_key=api_key,
        api_secret=api_secret,
        account_size=AccountSize(account_size),
        max_contracts=max_contracts,
        daily_loss_limit=daily_loss_limit,
        log_level=log_level,
    )

    # Create feed
    if feed_type == "sim":
        feed = SimulatedFeed(
            symbol=config.symbol,
            start_price=20000.0,
            volatility=0.5,
            tick_rate=0.01,  # fast for sim
        )
    else:
        feed = WebSocketFeed(
            symbol=config.symbol,
            url=feed_url,
            api_key=api_key,
            api_secret=api_secret,
            provider=feed_type,
        )

    # Create execution engine
    if paper:
        execution = SimulatedExecution(config, slippage_ticks=1)
        click.echo("Starting in PAPER TRADING mode")
    else:
        execution = LiveExecution(config)
        click.echo("Starting in LIVE TRADING mode - real money at risk!")

    agent = TradingAgent(config, feed, execution)

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()

    def shutdown_handler(sig, frame):
        click.echo(f"\nReceived {sig}, shutting down gracefully...")
        loop.create_task(agent.stop())

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    click.echo(f"NQ Adaptive Scalper | {config.symbol} | {config.account_size.value} account")
    click.echo(f"Max contracts: {config.max_contracts} | Daily loss limit: ${config.daily_loss_limit}")
    click.echo(f"Feed: {feed_type} | Execution: {'paper' if paper else 'LIVE'}")
    click.echo("=" * 60)

    try:
        loop.run_until_complete(agent.run())
    except KeyboardInterrupt:
        click.echo("\nShutting down...")
        loop.run_until_complete(agent.stop())
    finally:
        loop.close()


@cli.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("--account-size", default="50k", type=click.Choice(["50k", "100k", "150k"]))
@click.option("--max-contracts", default=2, type=int)
@click.option("--daily-loss-limit", default=800.0, type=float)
@click.option("--output", default=None, type=click.Path(), help="Save results to JSON")
def backtest(
    data_file: str,
    account_size: str,
    max_contracts: int,
    daily_loss_limit: float,
    output: str,
):
    """Run backtest on historical candle data (CSV/JSON)."""
    import json
    import pandas as pd

    config = ScalperConfig(
        account_size=AccountSize(account_size),
        max_contracts=max_contracts,
        daily_loss_limit=daily_loss_limit,
    )

    # Load data
    if data_file.endswith(".csv"):
        df = pd.read_csv(data_file)
        candles = df.to_dict("records")
    else:
        with open(data_file) as f:
            candles = json.load(f)

    click.echo(f"Loaded {len(candles)} candles from {data_file}")

    # Create replay feed
    feed = CandleReplayFeed(symbol=config.symbol, candles=candles, speed=0)
    execution = SimulatedExecution(config, slippage_ticks=1)
    agent = TradingAgent(config, feed, execution)

    # Run backtest
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(agent.run())
    finally:
        loop.close()

    # Print results
    status = agent.get_status()
    risk = agent.risk_mgr.state
    adaptive = agent.learner.get_stats_summary()

    click.echo("\n" + "=" * 60)
    click.echo("BACKTEST RESULTS")
    click.echo("=" * 60)
    click.echo(f"Total trades: {status['trades']}")
    click.echo(f"Win rate: {agent.risk_mgr.win_rate:.1%}")
    click.echo(f"Total P&L: ${risk.total_pnl:.2f}")
    click.echo(f"Daily P&L: ${risk.daily_pnl:.2f}")
    click.echo(f"Max drawdown used: ${config.max_drawdown - risk.trailing_drawdown_remaining:.2f}")
    click.echo(f"Risk multiplier (final): {risk.risk_multiplier:.2f}")
    click.echo(f"\nAdaptive stats: {json.dumps(adaptive, indent=2)}")

    if output:
        results = {
            "config": {
                "account_size": account_size,
                "max_contracts": max_contracts,
                "daily_loss_limit": daily_loss_limit,
            },
            "status": status,
            "trades": [
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "side": t.side.value,
                    "entry": t.entry_price,
                    "exit": t.exit_price,
                    "pnl": t.pnl,
                    "reason": t.exit_reason,
                    "regime": t.regime.value,
                    "confidence": t.signal_confidence,
                }
                for t in agent.risk_mgr.trade_history
            ],
            "adaptive": adaptive,
        }
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to {output}")


@cli.command()
def status():
    """Show current configuration and system status."""
    config = ScalperConfig()
    click.echo("NQ Adaptive Scalper Configuration")
    click.echo("=" * 40)
    for key, value in config.model_dump().items():
        if "secret" in key.lower():
            value = "***"
        click.echo(f"  {key}: {value}")


if __name__ == "__main__":
    cli()
