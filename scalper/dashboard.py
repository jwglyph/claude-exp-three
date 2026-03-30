"""Clean Rich live terminal dashboard for the trading agent."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from scalper.agent import TradingAgent
from scalper.models import Side


def format_price(p: float) -> str:
    return f"{p:,.2f}" if p > 0 else "---"


def format_pnl(pnl: float) -> str:
    if pnl > 0:
        return f"[bold green]+${pnl:,.2f}[/]"
    elif pnl < 0:
        return f"[bold red]-${abs(pnl):,.2f}[/]"
    return "$0.00"


def render_dashboard(agent: TradingAgent, feed_stats: dict) -> Table:
    """Render the full dashboard as a single table layout."""
    status = agent.get_status()
    risk = status["risk"]
    pos = status["position"]
    ind = status["indicators"]
    adaptive = status.get("adaptive", {})
    regime = status.get("regime", "unknown")
    trades = agent.risk_mgr.trade_history

    # Main grid
    grid = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    grid.add_column(ratio=1)

    # ── Header ──
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    price = feed_stats.get("last_price", 0)
    header = (
        f"[bold cyan] NQ ADAPTIVE SCALPER [/] | "
        f"[white]{format_price(price)}[/] | "
        f"Regime: [yellow]{regime.upper()}[/] | "
        f"{now}"
    )
    grid.add_row(header)
    grid.add_row("[dim]─" * 70 + "[/]")

    # ── Position ──
    if pos["side"]:
        side_color = "green" if pos["side"] == "long" else "red"
        pos_line = (
            f"  Position: [{side_color}]{pos['side'].upper()}[/] @ {format_price(pos['entry'])} | "
            f"P&L: {format_pnl(pos['pnl'])} | "
            f"Stop: {format_price(pos['stop'] or 0)} | "
            f"Target: {format_price(pos['target'] or 0)}"
        )
        trail = pos.get("trail")
        if trail and trail > 0:
            pos_line += f" | Trail: {format_price(trail)}"
    else:
        pos_line = "  Position: [dim]FLAT[/]"
    grid.add_row(pos_line)

    # ── Risk ──
    pnl_str = format_pnl(risk["daily_pnl"])
    dd_pct = 100 * (1 - risk["drawdown_remaining"] / agent.config.max_drawdown) if agent.config.max_drawdown > 0 else 0
    lock_str = " [bold red]LOCKED[/]" if risk["is_locked"] else ""
    risk_line = (
        f"  Daily P&L: {pnl_str} | "
        f"DD Remaining: ${risk['drawdown_remaining']:,.0f} ({100-dd_pct:.0f}%) | "
        f"Risk x{risk['risk_multiplier']:.1f} | "
        f"Losses: {risk['consecutive_losses']}"
        f"{lock_str}"
    )
    grid.add_row(risk_line)

    # ── Indicators ──
    ind_line = (
        f"  EMA: {format_price(ind['ema_fast'])}/{format_price(ind['ema_slow'])} | "
        f"RSI: {ind['rsi']:.0f} | "
        f"ATR: {ind['atr']:.2f} | "
        f"VWAP: {format_price(ind['vwap'])}"
    )
    grid.add_row(ind_line)

    # ── Stats ──
    quotes = feed_stats.get("quotes", 0)
    ftrades = feed_stats.get("trades", 0)
    candles = status["candles"]
    signals = status["signals"]
    trade_count = status["trades"]
    wr = agent.risk_mgr.win_rate
    conf = adaptive.get("confidence_threshold", 0.55)

    stats_line = (
        f"  Candles: {candles} | Signals: {signals} | Trades: {trade_count} | "
        f"Win: {wr:.0%} | Conf: {conf:.2f} | "
        f"Feed: {quotes}q/{ftrades}t"
    )
    grid.add_row(stats_line)

    # ── Recent Trades ──
    if trades:
        grid.add_row("[dim]─" * 70 + "[/]")
        grid.add_row("  [bold]Recent Trades:[/]")

        trade_table = Table(box=box.SIMPLE, padding=(0, 1), show_edge=False)
        trade_table.add_column("#", style="dim", width=3)
        trade_table.add_column("Side", width=5)
        trade_table.add_column("Entry", width=10, justify="right")
        trade_table.add_column("Exit", width=10, justify="right")
        trade_table.add_column("P&L", width=9, justify="right")
        trade_table.add_column("Regime", width=14)
        trade_table.add_column("Reason", width=15)

        for i, t in enumerate(trades[-8:], len(trades) - min(8, len(trades)) + 1):
            side_style = "green" if t.side == Side.LONG else "red"
            pnl_style = "green" if t.pnl >= 0 else "red"
            trade_table.add_row(
                str(i),
                f"[{side_style}]{t.side.value.upper()}[/]",
                f"{t.entry_price:,.2f}",
                f"{t.exit_price:,.2f}",
                f"[{pnl_style}]${t.pnl:+,.0f}[/]",
                t.regime.value,
                t.exit_reason,
            )

        grid.add_row(trade_table)

    # ── Regime Stats ──
    regime_stats = adaptive.get("regime_stats", {})
    if regime_stats:
        grid.add_row("[dim]─" * 70 + "[/]")
        parts = []
        for r, s in regime_stats.items():
            if s["trades"] > 0:
                wr_c = "green" if s["win_rate"] > 0.5 else "red"
                parts.append(f"{r}: [{wr_c}]{s['win_rate']:.0%}[/]({s['trades']})")
        grid.add_row("  Regimes: " + " | ".join(parts))

    return grid


class LiveDashboard:
    """Runs the Rich live display alongside the agent."""

    def __init__(self, agent: TradingAgent, feed_stats_fn):
        self.agent = agent
        self.feed_stats_fn = feed_stats_fn
        self.console = Console()

    def render(self):
        stats = self.feed_stats_fn()
        return Panel(
            render_dashboard(self.agent, stats),
            title="[bold] NQ Scalper [/]",
            border_style="blue",
            padding=(0, 1),
        )
