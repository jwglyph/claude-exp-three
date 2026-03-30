"""CLI entry point for the NQ adaptive scalper."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import os

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

logger = structlog.get_logger()


@click.group()
def cli():
    """NQ Adaptive 1-Minute Candle Scalper"""
    pass


@cli.command()
@click.option("--feed-type", default="topstepx", type=click.Choice(["topstepx", "rithmic", "tradovate", "sim"]))
@click.option("--username", default=None, help="TopstepX/ProjectX username (or NQ_SCALPER_USERNAME env)")
@click.option("--api-key", default=None, help="TopstepX/ProjectX API key (or NQ_SCALPER_API_KEY env)")
@click.option("--account-size", default="50k", type=click.Choice(["50k", "100k", "150k"]))
@click.option("--max-contracts", default=2, type=int, help="Maximum contracts")
@click.option("--daily-loss-limit", default=800.0, type=float, help="Daily loss limit ($)")
@click.option("--paper/--live", default=True, help="Paper trade or live")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]))
def trade(
    feed_type: str,
    username: str,
    api_key: str,
    account_size: str,
    max_contracts: int,
    daily_loss_limit: float,
    paper: bool,
    log_level: str,
):
    """Start live trading with real market data from TopstepX."""
    username = username or os.environ.get("NQ_SCALPER_USERNAME", "")
    api_key = api_key or os.environ.get("NQ_SCALPER_API_KEY", "")

    config = ScalperConfig(
        feed_type=feed_type,
        api_key=api_key,
        account_size=AccountSize(account_size),
        max_contracts=max_contracts,
        daily_loss_limit=daily_loss_limit,
        log_level=log_level,
    )

    if feed_type == "topstepx":
        if not username or not api_key:
            click.echo("ERROR: --username and --api-key required for TopstepX feed")
            click.echo("  Set NQ_SCALPER_USERNAME and NQ_SCALPER_API_KEY env vars, or pass as args")
            click.echo("  Get API access at https://dashboard.projectx.com (Subscriptions > API Access)")
            sys.exit(1)

        feed, execution = _setup_topstepx(config, username, api_key, paper)
    elif feed_type == "sim":
        feed = SimulatedFeed(symbol=config.symbol, start_price=20000.0, volatility=0.5, tick_rate=0.01)
        execution = SimulatedExecution(config, slippage_ticks=1)
    else:
        feed = WebSocketFeed(
            symbol=config.symbol, url=config.feed_url,
            api_key=api_key, provider=feed_type,
        )
        execution = SimulatedExecution(config, slippage_ticks=1) if paper else LiveExecution(config)

    if paper:
        click.echo("Mode: PAPER TRADING (simulated execution on real data)")
    else:
        click.echo("Mode: LIVE TRADING - REAL MONEY AT RISK!")

    agent = TradingAgent(config, feed, execution)

    loop = asyncio.new_event_loop()

    def shutdown_handler(sig, frame):
        click.echo(f"\nReceived signal, shutting down gracefully...")
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


def _setup_topstepx(config, username, api_key, paper):
    """Set up TopstepX/ProjectX feed and execution."""
    from scalper.feeds.projectx_client import ProjectXClient, ProjectXConfig
    from scalper.feeds.projectx_feed import ProjectXFeed

    px_config = ProjectXConfig(username=username, api_key=api_key)
    client = ProjectXClient(px_config)

    # We need to authenticate and find the NQ contract synchronously for setup
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(client.authenticate())
        contract = loop.run_until_complete(client.find_active_nq_contract())
    finally:
        loop.close()

    click.echo(f"Connected to TopstepX | Contract: {contract.id} ({contract.description})")
    click.echo(f"Tick size: {contract.tick_size} | Tick value: ${contract.tick_value}")

    feed = ProjectXFeed(
        client=client,
        contract_id=contract.id,
        subscribe_quotes=True,
        subscribe_trades=True,
    )

    if paper:
        execution = SimulatedExecution(config, slippage_ticks=1)
    else:
        execution = LiveExecution(config)

    return feed, execution


@cli.command()
@click.option("--username", required=True, help="TopstepX username")
@click.option("--api-key", required=True, help="TopstepX API key")
@click.option("--bars", default=500, type=int, help="Number of 1-min bars to fetch")
@click.option("--account-size", default="50k", type=click.Choice(["50k", "100k", "150k"]))
@click.option("--max-contracts", default=2, type=int)
@click.option("--daily-loss-limit", default=800.0, type=float)
@click.option("--output", default=None, type=click.Path(), help="Save results to JSON")
def backtest_live(
    username: str,
    api_key: str,
    bars: int,
    account_size: str,
    max_contracts: int,
    daily_loss_limit: float,
    output: str,
):
    """Fetch REAL historical NQ bars from TopstepX and run backtest."""
    from scalper.feeds.projectx_client import ProjectXClient, ProjectXConfig

    config = ScalperConfig(
        account_size=AccountSize(account_size),
        max_contracts=max_contracts,
        daily_loss_limit=daily_loss_limit,
    )

    px_config = ProjectXConfig(username=username, api_key=api_key)
    client = ProjectXClient(px_config)

    click.echo("Connecting to TopstepX API...")

    loop = asyncio.new_event_loop()
    try:
        # Authenticate and find contract
        loop.run_until_complete(client.authenticate())
        contract = loop.run_until_complete(client.find_active_nq_contract())
        click.echo(f"Contract: {contract.id} ({contract.description})")

        # Fetch historical 1-minute bars
        click.echo(f"Fetching {bars} recent 1-minute bars...")
        candles = loop.run_until_complete(
            client.get_recent_bars(contract.id, count=bars, unit=2, unit_number=1)
        )
        click.echo(f"Received {len(candles)} bars")

        if not candles:
            click.echo("ERROR: No bars returned. Market may be closed.")
            loop.run_until_complete(client.close())
            return

        # Show data range
        from datetime import datetime, timezone
        first_ts = datetime.fromtimestamp(candles[0].timestamp, tz=timezone.utc)
        last_ts = datetime.fromtimestamp(candles[-1].timestamp, tz=timezone.utc)
        click.echo(f"Data range: {first_ts.strftime('%Y-%m-%d %H:%M')} to {last_ts.strftime('%Y-%m-%d %H:%M')} UTC")
        click.echo(f"Price range: {min(c.low for c in candles):.2f} - {max(c.high for c in candles):.2f}")
        click.echo()

        loop.run_until_complete(client.close())
    finally:
        loop.close()

    # Convert candles to replay format
    candle_dicts = [
        {
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]

    # Run backtest
    click.echo("=" * 60)
    click.echo("RUNNING BACKTEST ON REAL NQ DATA")
    click.echo("=" * 60)

    feed = CandleReplayFeed(symbol=config.symbol, candles=candle_dicts, speed=0)
    execution = SimulatedExecution(config, slippage_ticks=1)
    agent = TradingAgent(config, feed, execution)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(agent.run())
    finally:
        loop.close()

    # Print results
    _print_backtest_results(agent, config, output)


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
    """Run backtest on historical candle data from a CSV/JSON file."""
    import pandas as pd

    config = ScalperConfig(
        account_size=AccountSize(account_size),
        max_contracts=max_contracts,
        daily_loss_limit=daily_loss_limit,
    )

    if data_file.endswith(".csv"):
        df = pd.read_csv(data_file)
        candles = df.to_dict("records")
    else:
        with open(data_file) as f:
            candles = json.load(f)

    click.echo(f"Loaded {len(candles)} candles from {data_file}")

    feed = CandleReplayFeed(symbol=config.symbol, candles=candles, speed=0)
    execution = SimulatedExecution(config, slippage_ticks=1)
    agent = TradingAgent(config, feed, execution)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(agent.run())
    finally:
        loop.close()

    _print_backtest_results(agent, config, output)


@cli.command()
@click.option("--username", required=True, help="TopstepX username")
@click.option("--api-key", required=True, help="TopstepX API key")
@click.option("--bars", default=200, type=int, help="Number of 1-min bars to fetch")
@click.option("--save", default=None, type=click.Path(), help="Save bars to JSON file")
def fetch_data(username: str, api_key: str, bars: int, save: str):
    """Fetch real NQ historical bars from TopstepX and optionally save to file."""
    from scalper.feeds.projectx_client import ProjectXClient, ProjectXConfig

    px_config = ProjectXConfig(username=username, api_key=api_key)
    client = ProjectXClient(px_config)

    async def run():
        await client.authenticate()
        contract = await client.find_active_nq_contract()
        click.echo(f"Contract: {contract.id} ({contract.description})")
        candles = await client.get_recent_bars(contract.id, count=bars, unit=2, unit_number=1)
        await client.close()
        return candles

    loop = asyncio.new_event_loop()
    candles = loop.run_until_complete(run())
    loop.close()

    click.echo(f"Fetched {len(candles)} bars")
    if candles:
        from datetime import datetime, timezone
        first = datetime.fromtimestamp(candles[0].timestamp, tz=timezone.utc)
        last = datetime.fromtimestamp(candles[-1].timestamp, tz=timezone.utc)
        click.echo(f"Range: {first} to {last}")
        click.echo(f"Last close: {candles[-1].close:.2f}")
        click.echo(f"Price range: {min(c.low for c in candles):.2f} - {max(c.high for c in candles):.2f}")

    if save:
        data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
        with open(save, "w") as f:
            json.dump(data, f, indent=2)
        click.echo(f"Saved to {save}")


@cli.command()
def status():
    """Show current configuration and system status."""
    config = ScalperConfig()
    click.echo("NQ Adaptive Scalper Configuration")
    click.echo("=" * 40)
    for key, value in config.model_dump().items():
        if "secret" in key.lower() or "key" in key.lower():
            value = "***" if value else "(not set)"
        click.echo(f"  {key}: {value}")


def _print_backtest_results(agent: TradingAgent, config: ScalperConfig, output: str | None):
    """Print and optionally save backtest results."""
    status = agent.get_status()
    risk = agent.risk_mgr.state
    adaptive = agent.learner.get_stats_summary()
    trades = agent.risk_mgr.trade_history

    click.echo()
    click.echo("=" * 60)
    click.echo("BACKTEST RESULTS")
    click.echo("=" * 60)
    click.echo(f"Total candles processed: {status['candles']}")
    click.echo(f"Signals generated:       {status['signals']}")
    click.echo(f"Total trades:            {status['trades']}")
    click.echo()

    if trades:
        winners = [t for t in trades if t.pnl >= 0]
        losers = [t for t in trades if t.pnl < 0]
        click.echo(f"Win rate:     {agent.risk_mgr.win_rate:.1%}")
        click.echo(f"Total P&L:    ${risk.total_pnl:.2f}")
        click.echo(f"Avg winner:   ${sum(t.pnl for t in winners) / len(winners):.2f}" if winners else "Avg winner:   N/A")
        click.echo(f"Avg loser:    ${sum(t.pnl for t in losers) / len(losers):.2f}" if losers else "Avg loser:    N/A")
        click.echo(f"Largest win:  ${max(t.pnl for t in trades):.2f}")
        click.echo(f"Largest loss: ${min(t.pnl for t in trades):.2f}")
        click.echo(f"Max drawdown used: ${config.max_drawdown - risk.trailing_drawdown_remaining:.2f}")
        click.echo(f"Profit factor: {sum(t.pnl for t in winners) / abs(sum(t.pnl for t in losers)):.2f}" if losers and sum(t.pnl for t in losers) != 0 else "")
        click.echo()
        click.echo("--- Per-Regime Performance ---")
        for regime, stats in adaptive.get("regime_stats", {}).items():
            if stats["trades"] > 0:
                click.echo(f"  {regime:20s}  trades={stats['trades']:3d}  win={stats['win_rate']:.0%}  E[PnL]=${stats['expectancy']:.0f}")
        click.echo()
        click.echo(f"Final risk multiplier:     {risk.risk_multiplier:.2f}")
        click.echo(f"Adapted confidence threshold: {adaptive.get('confidence_threshold', 0.55):.3f}")

        # Print individual trades
        click.echo()
        click.echo("--- Trade Log ---")
        click.echo(f"{'#':>3}  {'Side':>5}  {'Entry':>10}  {'Exit':>10}  {'P&L':>8}  {'Regime':>15}  {'Reason':>15}  {'Bars':>4}")
        for i, t in enumerate(trades, 1):
            pnl_str = f"${t.pnl:+.0f}"
            click.echo(
                f"{i:3d}  {t.side.value:>5}  {t.entry_price:10.2f}  {t.exit_price:10.2f}  "
                f"{pnl_str:>8}  {t.regime.value:>15}  {t.exit_reason:>15}  {t.holding_candles:4d}"
            )
    else:
        click.echo("No trades were generated.")
        click.echo("This may be because:")
        click.echo("  - Not enough candles for warmup (need 50+)")
        click.echo("  - No signals met the confidence threshold")
        click.echo("  - Session filter blocked all candles")

    if output:
        results = {
            "summary": {
                "total_trades": len(trades),
                "win_rate": agent.risk_mgr.win_rate,
                "total_pnl": risk.total_pnl,
                "max_drawdown_used": config.max_drawdown - risk.trailing_drawdown_remaining,
            },
            "config": {
                "account_size": config.account_size.value,
                "max_contracts": config.max_contracts,
                "daily_loss_limit": config.daily_loss_limit,
            },
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
                    "holding_candles": t.holding_candles,
                }
                for t in trades
            ],
            "adaptive": adaptive,
        }
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults saved to {output}")


if __name__ == "__main__":
    cli()
