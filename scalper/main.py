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

import logging

def _configure_logging(level: str = "INFO") -> None:
    """Configure structlog with proper level filtering."""
    numeric = getattr(logging, level, logging.INFO)
    # Reset root logger
    root = logging.getLogger()
    root.setLevel(numeric)
    # Clear existing handlers and add one
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(numeric)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

_configure_logging("INFO")

logger = structlog.get_logger()


@click.group()
def cli():
    """NQ Adaptive 1-Minute Candle Scalper"""
    pass


@cli.command()
@click.option("--feed-type", default="topstepx", type=click.Choice(["topstepx", "rithmic", "tradovate", "sim"]))
@click.option("--env", "environment", default="live", type=click.Choice(["live", "demo"]), help="TopstepX environment (demo = free with eval account)")
@click.option("--username", default=None, help="TopstepX/ProjectX username (or NQ_SCALPER_USERNAME env)")
@click.option("--api-key", default=None, help="TopstepX/ProjectX API key (or NQ_SCALPER_API_KEY env)")
@click.option("--account-size", default="50k", type=click.Choice(["50k", "100k", "150k"]))
@click.option("--max-contracts", default=2, type=int, help="Maximum contracts")
@click.option("--daily-loss-limit", default=800.0, type=float, help="Daily loss limit ($)")
@click.option("--paper/--live", default=True, help="Paper trade or live")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]))
def trade(
    feed_type: str,
    environment: str,
    username: str,
    api_key: str,
    account_size: str,
    max_contracts: int,
    daily_loss_limit: float,
    paper: bool,
    log_level: str,
):
    """Start live trading with real market data from TopstepX."""
    _configure_logging(log_level)
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
            sys.exit(1)
    elif feed_type == "sim":
        pass
    else:
        click.echo(f"ERROR: feed type '{feed_type}' not fully supported yet")
        sys.exit(1)

    # Everything runs in ONE event loop to avoid session/loop mismatch
    async def run_all():
        from scalper.feeds.projectx_client import ProjectXClient, ProjectXConfig, get_urls
        from scalper.feeds.projectx_feed import ProjectXFeed
        from scalper.dashboard import LiveDashboard
        from rich.live import Live
        from rich.console import Console

        nonlocal feed_type

        if feed_type == "topstepx":
            # 1. Authenticate
            api_url, market_hub, user_hub = get_urls(environment)
            from scalper import __version__
            click.echo(f"NQ Scalper v{__version__}")
            click.echo(f"Connecting to TopstepX ({environment.upper()})...")
            px_config = ProjectXConfig(username=username, api_key=api_key,
                                       api_url=api_url, market_hub_url=market_hub, user_hub_url=user_hub)
            client = ProjectXClient(px_config)
            await client.authenticate()
            click.echo("Authenticated!")

            # 2. Find NQ contract
            contract = await client.find_active_nq_contract()
            click.echo(f"Contract: {contract.id} ({contract.description})")

            # 3. Fetch accounts
            account_info = "PAPER (simulated execution)"
            try:
                session = await client._ensure_session()
                url = f"{client.config.api_url}/api/Account/search"
                async with session.post(url, json={"onlyActive": True}, headers=client._auth_headers()) as resp:
                    accs = await resp.json()
                if isinstance(accs, list) and accs:
                    names = [a.get("name", "?") for a in accs[:5]]
                    account_info = "PAPER | Your accounts: " + ", ".join(names)
                    if len(accs) > 5:
                        account_info += f" (+{len(accs)-5} more)"
                    click.echo(f"Found {len(accs)} accounts")
            except Exception as e:
                click.echo(f"Account fetch: {e}")

            if not paper:
                account_info = account_info.replace("PAPER", "LIVE")

            # 4. Preload historical candles
            click.echo("Loading historical candles...")
            try:
                history = await client.get_recent_bars(
                    contract.id, count=config.warmup_candles + 20, unit=2, unit_number=1
                )
                click.echo(f"Got {len(history)} candles")
            except Exception as e:
                click.echo(f"History failed: {e}")
                history = []

            # 5. Create feed and execution
            feed = ProjectXFeed(client=client, contract_id=contract.id,
                                subscribe_quotes=True, subscribe_trades=True)
            if paper:
                execution = SimulatedExecution(config, slippage_ticks=1)
            else:
                execution = LiveExecution(config)

        else:
            # Sim mode
            feed = SimulatedFeed(symbol=config.symbol, start_price=20000.0, volatility=0.5, tick_rate=0.01)
            execution = SimulatedExecution(config, slippage_ticks=1)
            history = []
            account_info = "SIMULATED"

        # 6. Create agent and preload
        agent = TradingAgent(config, feed, execution)
        if history:
            agent.preload_candles(history)
            click.echo(f"Preloaded {len(history)} candles - ready!")

        # 7. Connect feed (with logging visible so user sees connection status)
        click.echo("Connecting to live feed...")
        await feed.connect()

        # Wait for first tick
        click.echo("Waiting for market data...")
        import queue as _q
        for _ in range(100):  # 10 sec timeout
            stats = getattr(feed, 'stats', {})
            if isinstance(stats, dict) and stats.get('quotes', 0) > 0:
                click.echo(f"Live data flowing! Price: {stats.get('last_price', 0):,.2f}")
                break
            await asyncio.sleep(0.1)
        else:
            click.echo("Warning: no market data yet (market may be closed)")

        # 8. Suppress logs and start dashboard
        _configure_logging("CRITICAL")

        from scalper.market_hours import is_market_open, get_session_info, time_until_open
        import time as _time

        console = Console()
        feed_stats_fn = lambda: getattr(feed, 'stats', {})
        dashboard = LiveDashboard(agent, feed_stats_fn, account_info=account_info)

        # Start agent processing
        agent._running = True
        agent._start_time = _time.time()

        async def agent_loop():
            """Main agent loop with reconnection and market hours awareness."""
            while agent._running:
                try:
                    # Check market hours
                    if not is_market_open():
                        session = get_session_info()
                        wait = time_until_open()
                        dashboard.account_info = (
                            f"MARKET CLOSED ({session['session']}) | "
                            f"Opens in {session.get('time_to_open', '?')}"
                        )
                        # Sleep in 30-second chunks so dashboard still updates
                        # and we can catch Ctrl+C
                        sleep_secs = min(wait.total_seconds(), 30)
                        await asyncio.sleep(max(1, sleep_secs))
                        continue

                    dashboard.account_info = account_info

                    # Ensure feed is connected
                    if not feed._connected:
                        await feed._connect_with_retry()

                    # Process ticks
                    async for tick in feed.stream():
                        if not agent._running:
                            break

                        # Periodic market hours check (every 1000 ticks)
                        if agent._tick_count % 1000 == 0:
                            if not is_market_open():
                                break  # will re-enter the while loop and sleep

                        await agent._process_tick(tick)

                        # Periodic token refresh (every 6 hours)
                        if agent._tick_count % 500000 == 0 and hasattr(feed, 'update_token'):
                            try:
                                await feed.update_token()
                            except Exception:
                                pass

                except (asyncio.CancelledError, KeyboardInterrupt):
                    break
                except Exception as e:
                    # Log error to file (dashboard suppresses logs)
                    import pathlib
                    err_file = pathlib.Path("logs/errors.txt")
                    err_file.parent.mkdir(exist_ok=True)
                    with open(err_file, "a") as ef:
                        ef.write(f"{_time.time()} agent_loop error: {e}\n")
                    # Wait and retry
                    await asyncio.sleep(5)

        agent_task = asyncio.ensure_future(agent_loop())

        try:
            with Live(dashboard.render(), refresh_per_second=4, console=console, screen=False) as live:
                while agent._running:
                    await asyncio.sleep(0.25)
                    try:
                        live.update(dashboard.render())
                    except Exception:
                        pass
        except KeyboardInterrupt:
            pass
        finally:
            agent._running = False
            feed._running = False
            agent_task.cancel()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_all())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        os._exit(0)


@cli.command()
@click.option("--username", required=True, help="TopstepX username")
@click.option("--api-key", required=True, help="TopstepX API key")
@click.option("--env", "environment", default="live", type=click.Choice(["live", "demo"]))
@click.option("--bars", default=500, type=int, help="Number of 1-min bars to fetch")
@click.option("--account-size", default="50k", type=click.Choice(["50k", "100k", "150k"]))
@click.option("--max-contracts", default=2, type=int)
@click.option("--daily-loss-limit", default=800.0, type=float)
@click.option("--output", default=None, type=click.Path(), help="Save results to JSON")
def backtest_live(
    username: str,
    api_key: str,
    environment: str,
    bars: int,
    account_size: str,
    max_contracts: int,
    daily_loss_limit: float,
    output: str,
):
    """Fetch REAL historical NQ bars from TopstepX and run backtest."""
    from scalper.feeds.projectx_client import ProjectXClient, ProjectXConfig, get_urls

    config = ScalperConfig(
        account_size=AccountSize(account_size),
        max_contracts=max_contracts,
        daily_loss_limit=daily_loss_limit,
    )

    api_url, market_hub, user_hub = get_urls(environment)
    px_config = ProjectXConfig(username=username, api_key=api_key, api_url=api_url, market_hub_url=market_hub, user_hub_url=user_hub)
    client = ProjectXClient(px_config)

    click.echo(f"Connecting to TopstepX API ({environment.upper()})...")

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
@click.option("--env", "environment", default="live", type=click.Choice(["live", "demo"]))
@click.option("--bars", default=200, type=int, help="Number of 1-min bars to fetch")
@click.option("--save", default=None, type=click.Path(), help="Save bars to JSON file")
def fetch_data(username: str, api_key: str, environment: str, bars: int, save: str):
    """Fetch real NQ historical bars from TopstepX and optionally save to file."""
    from scalper.feeds.projectx_client import ProjectXClient, ProjectXConfig, get_urls

    api_url, market_hub, user_hub = get_urls(environment)
    px_config = ProjectXConfig(username=username, api_key=api_key, api_url=api_url, market_hub_url=market_hub, user_hub_url=user_hub)
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
