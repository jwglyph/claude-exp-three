#!/usr/bin/env python3
"""Fetch real NQ data from TopstepX and run the scalper backtest.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --bars 1000 --output results.json
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from scalper.feeds.projectx_client import ProjectXClient, ProjectXConfig, get_urls
from scalper.feeds.price_feed import CandleReplayFeed
from scalper.execution.engine import SimulatedExecution
from scalper.agent import TradingAgent
from scalper.config import ScalperConfig, AccountSize


async def fetch_bars(username: str, api_key: str, count: int):
    """Try live then demo to fetch bars."""
    for env in ["live", "demo"]:
        api_url, mh, uh = get_urls(env)
        config = ProjectXConfig(username=username, api_key=api_key,
                                api_url=api_url, market_hub_url=mh, user_hub_url=uh)
        client = ProjectXClient(config)
        try:
            await client.authenticate()
            nq = await client.find_active_nq_contract()
            print(f"[{env.upper()}] Contract: {nq.id} ({nq.description})")
            bars = await client.get_recent_bars(nq.id, count=count, unit=2, unit_number=1)
            await client.close()
            return bars
        except Exception as e:
            print(f"[{env.upper()}] Failed: {e}")
            await client.close()
    return []


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default=os.environ.get("NQ_SCALPER_USERNAME", "wujacky1369@gmail.com"))
    parser.add_argument("--api-key", default=os.environ.get("NQ_SCALPER_API_KEY", "EhozDLrskNuGKI2j0REeoLAlGjiyL4lbn5Bu+NkpIz4="))
    parser.add_argument("--bars", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Fetching {args.bars} real NQ 1-min bars...")
    bars = asyncio.run(fetch_bars(args.username, args.api_key, args.bars))

    if not bars:
        print("ERROR: Could not fetch bars. Check credentials and internet.")
        sys.exit(1)

    first = datetime.fromtimestamp(bars[0].timestamp, tz=timezone.utc)
    last = datetime.fromtimestamp(bars[-1].timestamp, tz=timezone.utc)
    print(f"Got {len(bars)} bars: {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC")
    print(f"Price: {bars[-1].close:.2f} | Range: {min(b.low for b in bars):.2f} - {max(b.high for b in bars):.2f}")
    print()

    # Convert to replay format
    candle_dicts = [{"timestamp": c.timestamp, "open": c.open, "high": c.high,
                     "low": c.low, "close": c.close, "volume": c.volume} for c in bars]

    # Run backtest
    config = ScalperConfig(
        account_size=AccountSize.K50,
        max_contracts=2,
        daily_loss_limit=800.0,
        # Allow all sessions so backtest works on any time range
        trade_asian=True,
        trade_london=True,
        trade_ny_open=True,
        trade_ny_afternoon=True,
        trade_ny_close=True,
    )

    feed = CandleReplayFeed(symbol="NQ", candles=candle_dicts, speed=0)
    execution = SimulatedExecution(config, slippage_ticks=1)
    agent = TradingAgent(config, feed, execution)

    print("=" * 60)
    print("RUNNING BACKTEST ON REAL NQ DATA")
    print("=" * 60)

    asyncio.run(agent.run())

    # Results
    risk = agent.risk_mgr.state
    trades = agent.risk_mgr.trade_history

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Candles: {agent._candle_count} | Signals: {agent._signal_count} | Trades: {len(trades)}")

    if trades:
        winners = [t for t in trades if t.pnl >= 0]
        losers = [t for t in trades if t.pnl < 0]
        print(f"Win rate:     {agent.risk_mgr.win_rate:.1%}")
        print(f"Total P&L:    ${risk.total_pnl:.2f}")
        if winners:
            print(f"Avg winner:   ${sum(t.pnl for t in winners)/len(winners):.2f}")
        if losers:
            print(f"Avg loser:    ${sum(t.pnl for t in losers)/len(losers):.2f}")
        print(f"Largest win:  ${max(t.pnl for t in trades):.2f}")
        print(f"Largest loss: ${min(t.pnl for t in trades):.2f}")
        print(f"Max DD used:  ${config.max_drawdown - risk.trailing_drawdown_remaining:.2f}")
        print()
        print(f"{'#':>3}  {'Side':>5}  {'Entry':>10}  {'Exit':>10}  {'P&L':>8}  {'Regime':>15}  {'Reason':>15}")
        for i, t in enumerate(trades, 1):
            print(f"{i:3d}  {t.side.value:>5}  {t.entry_price:10.2f}  {t.exit_price:10.2f}  "
                  f"${t.pnl:+7.0f}  {t.regime.value:>15}  {t.exit_reason:>15}")
    else:
        print("No trades generated. Market data may be from closed session.")

    if args.output:
        out = {
            "bars": len(bars),
            "trades": len(trades),
            "pnl": risk.total_pnl,
            "win_rate": agent.risk_mgr.win_rate,
            "trade_log": [{"side": t.side.value, "entry": t.entry_price, "exit": t.exit_price,
                           "pnl": t.pnl, "regime": t.regime.value, "reason": t.exit_reason}
                          for t in trades],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
