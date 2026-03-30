"""Live terminal dashboard for monitoring the trading agent.

Uses Rich for a clean, updating terminal display showing:
- Current position and P&L
- Market regime and indicators
- Risk state
- Recent trade history
- Adaptive learning stats
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scalper.agent import TradingAgent
from scalper.models import Side


class Dashboard:
    """Rich terminal dashboard for the trading agent."""

    def __init__(self, agent: TradingAgent, refresh_rate: float = 1.0):
        self.agent = agent
        self.refresh_rate = refresh_rate
        self.console = Console()

    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )
        layout["left"].split_column(
            Layout(name="position", size=8),
            Layout(name="signals", ratio=1),
        )
        layout["right"].split_column(
            Layout(name="risk", size=12),
            Layout(name="adaptive", ratio=1),
        )
        return layout

    def render(self) -> Layout:
        status = self.agent.get_status()
        layout = self.generate_layout()

        # Header
        layout["header"].update(
            Panel(
                f"[bold]NQ Adaptive Scalper[/bold] | "
                f"Regime: [cyan]{status['regime']}[/cyan] | "
                f"Candles: {status['candles']} | "
                f"Trades: {status['trades']}",
                style="bold white on blue",
            )
        )

        # Position panel
        pos = status["position"]
        if pos["side"]:
            pnl_color = "green" if pos["pnl"] >= 0 else "red"
            pos_text = (
                f"Side: [bold]{pos['side'].upper()}[/bold]\n"
                f"Entry: {pos['entry']:.2f}\n"
                f"P&L: [{pnl_color}]${pos['pnl']:.2f}[/{pnl_color}]\n"
                f"Stop: {pos['stop']:.2f} | Target: {pos['target']:.2f}\n"
                f"Trail: {pos['trail']:.2f}" if pos['trail'] else ""
            )
        else:
            pos_text = "[dim]FLAT - No open position[/dim]"
        layout["position"].update(Panel(pos_text, title="Position"))

        # Indicators
        ind = status["indicators"]
        signals_text = (
            f"EMA Fast: {ind['ema_fast']:.2f}  Slow: {ind['ema_slow']:.2f}\n"
            f"RSI: {ind['rsi']:.1f}\n"
            f"ATR: {ind['atr']:.2f}\n"
            f"VWAP: {ind['vwap']:.2f}"
        )
        layout["signals"].update(Panel(signals_text, title="Indicators"))

        # Risk panel
        risk = status["risk"]
        pnl_color = "green" if risk["daily_pnl"] >= 0 else "red"
        risk_text = (
            f"Daily P&L: [{pnl_color}]${risk['daily_pnl']:.2f}[/{pnl_color}]\n"
            f"Drawdown Left: ${risk['drawdown_remaining']:.2f}\n"
            f"Risk Multiplier: {risk['risk_multiplier']:.2f}\n"
            f"Consec Losses: {risk['consecutive_losses']}\n"
            f"Locked: {'YES' if risk['is_locked'] else 'No'}"
        )
        layout["risk"].update(Panel(risk_text, title="Risk Management"))

        # Adaptive stats
        adaptive = status.get("adaptive", {})
        regime_stats = adaptive.get("regime_stats", {})
        table = Table(show_header=True, header_style="bold")
        table.add_column("Regime")
        table.add_column("Trades")
        table.add_column("Win%")
        table.add_column("E[PnL]")
        for regime, stats in regime_stats.items():
            wr_color = "green" if stats["win_rate"] > 0.5 else "red"
            table.add_row(
                regime,
                str(stats["trades"]),
                f"[{wr_color}]{stats['win_rate']:.0%}[/{wr_color}]",
                f"${stats['expectancy']:.0f}",
            )
        layout["adaptive"].update(Panel(table, title="Adaptive Learning"))

        # Footer
        layout["footer"].update(
            Panel(
                f"Confidence threshold: {adaptive.get('confidence_threshold', 0.55):.3f} | "
                f"Signals: {status['signals']} | "
                f"[dim]Ctrl+C to stop[/dim]",
                style="dim",
            )
        )

        return layout

    async def run(self) -> None:
        """Run the dashboard alongside the agent."""
        with Live(self.render(), refresh_per_second=1 / self.refresh_rate, console=self.console) as live:
            while self.agent._running:
                live.update(self.render())
                await asyncio.sleep(self.refresh_rate)
