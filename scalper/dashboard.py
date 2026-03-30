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


SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
BARS = "▁▂▃▄▅▆▇█"


def fp(p: float) -> str:
    """Format price."""
    return f"{p:,.2f}" if p > 0 else "---"


def fpnl(pnl: float) -> str:
    """Format P&L with color."""
    if pnl > 0:
        return f"[bold green]+${pnl:,.2f}[/]"
    elif pnl < 0:
        return f"[bold red]-${abs(pnl):,.2f}[/]"
    return "[dim]$0.00[/]"


def regime_color(regime: str) -> str:
    colors = {
        "trending_up": "bold green",
        "trending_down": "bold red",
        "ranging": "yellow",
        "volatile": "bold magenta",
        "low_volatility": "cyan",
    }
    return colors.get(regime, "white")


def rsi_color(rsi: float) -> str:
    if rsi > 70:
        return "red"
    elif rsi < 30:
        return "green"
    return "white"


def make_mini_chart(prices: list[float], width: int = 20) -> str:
    """Make a tiny sparkline chart from recent prices."""
    if len(prices) < 2:
        return "[dim]waiting...[/]"
    # Take last `width` prices
    p = prices[-width:]
    mn, mx = min(p), max(p)
    rng = mx - mn
    if rng == 0:
        return "▅" * len(p)
    chars = []
    for v in p:
        idx = int((v - mn) / rng * (len(BARS) - 1))
        chars.append(BARS[idx])
    # Color based on direction
    color = "green" if p[-1] >= p[0] else "red"
    return f"[{color}]{''.join(chars)}[/]"


def render_dashboard(agent: TradingAgent, feed_stats: dict, account_info: str, tick_counter: int) -> Panel:
    """Render the full dashboard."""
    status = agent.get_status()
    risk = status["risk"]
    pos = status["position"]
    ind = status["indicators"]
    adaptive = status.get("adaptive", {})
    regime = status.get("regime", "unknown")
    trades = agent.risk_mgr.trade_history

    # Feed stats
    quotes = feed_stats.get("quotes", 0)
    ftrades = feed_stats.get("trades", 0)
    last_price = feed_stats.get("last_price", 0)
    q_size = feed_stats.get("queue_size", 0)
    connected = feed_stats.get("connected", False)

    # Collect recent close prices for sparkline
    candles = agent.aggregator.get_candles()
    recent_closes = [c.close for c in candles[-30:]] if candles else []
    if last_price > 0:
        recent_closes.append(last_price)

    # Spinner for liveness
    spin = SPINNER[tick_counter % len(SPINNER)]
    now_utc = datetime.now(timezone.utc)
    now = now_utc.strftime("%H:%M:%S")
    from datetime import timedelta
    pst = (now_utc - timedelta(hours=8)).strftime("%H:%M:%S")

    # Build output
    lines = []

    # ── Header with price + sparkline ──
    conn_icon = "[green]●[/]" if connected else "[red]●[/]"
    price_str = f"[bold white]{fp(last_price)}[/]" if last_price > 0 else "[dim]waiting...[/]"
    rc = regime_color(regime)
    lines.append(
        f"  {conn_icon} [bold cyan]NQ SCALPER[/]  {price_str}  "
        f"[{rc}]{regime.upper()}[/]  "
        f"{make_mini_chart(recent_closes)}  "
        f"[dim]{pst} PST / {now} UTC {spin}[/]"
    )

    # ── Account ──
    if account_info:
        lines.append(f"  [dim]Account: {account_info}[/]")

    lines.append("")

    # ── Current Candle Building ──
    current = agent.aggregator.current_candle
    if current:
        elapsed = time.time() - current.timestamp
        bar_pct = min(elapsed / agent.config.candle_interval_sec, 1.0)
        bar_filled = int(bar_pct * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        candle_dir = "[green]▲[/]" if current.is_bullish else "[red]▼[/]" if current.is_bearish else "[dim]─[/]"
        lines.append(
            f"  Building: [{bar}] {bar_pct:.0%}  "
            f"{candle_dir} O={current.open:.2f} H={current.high:.2f} L={current.low:.2f} C={current.close:.2f}  "
            f"V={current.volume} Δ={current.delta:+d}"
        )
    else:
        lines.append("  [dim]Waiting for first tick...[/]")

    # ── HTF Confluence ──
    htf = status.get("htf", {})
    if htf:
        def tf_arrow(t):
            if t > 0: return "[green]▲[/]"
            elif t < 0: return "[red]▼[/]"
            return "[dim]─[/]"

        htf_line = (
            f"  HTF: 5m{tf_arrow(htf.get('5m', 0))} "
            f"15m{tf_arrow(htf.get('15m', 0))} "
            f"1h{tf_arrow(htf.get('1h', 0))}  "
        )
        agrees = htf.get("agrees", "neutral")
        if agrees == "long":
            htf_line += "[green]BULLISH BIAS[/]"
        elif agrees == "short":
            htf_line += "[red]BEARISH BIAS[/]"
        else:
            htf_line += "[dim]NEUTRAL[/]"
        htf_line += f"  str={htf.get('strength', 0):.0%}"
        lines.append(htf_line)

    # ── Tick Events ──
    tick_evt = status.get("tick_event")
    if tick_evt and tick_evt.get("age", 999) < 30:
        evt_color = "green" if tick_evt["dir"] > 0 else "red"
        lines.append(
            f"  [{evt_color}]⚡ {tick_evt['type'].upper()}: {tick_evt['desc']}[/] "
            f"[dim]({tick_evt['age']:.0f}s ago)[/]"
        )

    lines.append("")

    # ── Position ──
    if pos["side"]:
        side_color = "green" if pos["side"] == "long" else "red"
        arrow = "▲" if pos["side"] == "long" else "▼"
        lines.append(
            f"  [{side_color}]{arrow} {pos['side'].upper()}[/] @ {fp(pos['entry'])}  "
            f"P&L: {fpnl(pos['pnl'])}  "
            f"Stop: {fp(pos['stop'] or 0)}  Target: {fp(pos['target'] or 0)}"
        )
        trail = pos.get("trail")
        if trail and trail > 0:
            lines[-1] += f"  Trail: {fp(trail)}"
    else:
        lines.append("  [dim]● FLAT — scanning for entry...[/]")

    lines.append("")

    # ── Risk Bar ──
    daily_pnl = risk["daily_pnl"]
    dd_rem = risk["drawdown_remaining"]
    dd_max = agent.config.max_drawdown
    dd_pct = (dd_max - dd_rem) / dd_max if dd_max > 0 else 0
    dd_bar_filled = int(dd_pct * 20)
    dd_color = "green" if dd_pct < 0.3 else "yellow" if dd_pct < 0.6 else "red"
    dd_bar = f"[{dd_color}]{'█' * dd_bar_filled}[/][dim]{'░' * (20 - dd_bar_filled)}[/]"
    lock_str = " [bold red]⊘ LOCKED[/]" if risk["is_locked"] else ""

    lines.append(
        f"  P&L: {fpnl(daily_pnl)}  "
        f"DD: [{dd_bar}] ${dd_rem:,.0f} left  "
        f"Risk: x{risk['risk_multiplier']:.1f}  "
        f"Losses: {risk['consecutive_losses']}"
        f"{lock_str}"
    )

    lines.append("")

    # ── Indicators ──
    rsi = ind["rsi"]
    rsi_c = rsi_color(rsi)
    # RSI visual bar
    rsi_pos = int(rsi / 100 * 20)
    rsi_bar = "░" * rsi_pos + "█" + "░" * (20 - rsi_pos)

    lines.append(
        f"  EMA: {fp(ind['ema_fast'])}/{fp(ind['ema_slow'])}  "
        f"RSI: [{rsi_c}]{rsi:.0f}[/] [{rsi_c}][{rsi_bar}][/]  "
        f"ATR: {ind['atr']:.2f}  "
        f"VWAP: {fp(ind['vwap'])}"
    )

    lines.append("")

    # ── Stats ──
    candle_count = status["candles"]
    signals = status["signals"]
    trade_count = status["trades"]
    wr = agent.risk_mgr.win_rate
    conf = adaptive.get("confidence_threshold", 0.55)

    balance = risk.get("balance", 0)
    max_ct = risk.get("max_contracts", 2)
    intra = status.get("intra_candle_trades", 0)

    lines.append(
        f"  [dim]Bal: ${balance:,.0f} | Max: {max_ct}ct | "
        f"Candles: {candle_count} | Sig: {signals} | Trades: {trade_count}"
        f"{'(' + str(intra) + ' intra)' if intra else ''} | "
        f"Win: {wr:.0%} | Conf: {conf:.2f} | "
        f"Feed: {quotes}q/{ftrades}t[/]"
    )

    # ── Recent Trades ──
    if trades:
        lines.append("")
        lines.append("  [bold]Trade Log:[/]")

        header = f"  [dim]{'#':>3}  {'Side':>5}  {'Entry':>10}  {'Exit':>10}  {'P&L':>8}  {'Regime':<14}  {'Reason':<15}[/]"
        lines.append(header)

        for i, t in enumerate(trades[-6:], max(1, len(trades) - 5)):
            side_c = "green" if t.side == Side.LONG else "red"
            pnl_c = "green" if t.pnl >= 0 else "red"
            lines.append(
                f"  {i:3d}  [{side_c}]{t.side.value.upper():>5}[/]  "
                f"{t.entry_price:10,.2f}  {t.exit_price:10,.2f}  "
                f"[{pnl_c}]${t.pnl:+7,.0f}[/]  "
                f"{t.regime.value:<14}  {t.exit_reason:<15}"
            )

    # ── Regime Stats ──
    regime_stats = adaptive.get("regime_stats", {})
    if regime_stats:
        lines.append("")
        parts = []
        for r, s in regime_stats.items():
            if s["trades"] > 0:
                wr_c = "green" if s["win_rate"] > 0.5 else "red"
                parts.append(f"{r}:[{wr_c}]{s['win_rate']:.0%}[/]({s['trades']})")
        if parts:
            lines.append("  [dim]" + " | ".join(parts) + "[/]")

    content = "\n".join(lines)

    border = "green" if connected else "red"
    return Panel(
        content,
        title=f"[bold] NQ Adaptive Scalper [/]",
        subtitle="[dim]Ctrl+C to stop[/]",
        border_style=border,
        padding=(0, 0),
    )


class LiveDashboard:
    """Runs the Rich live display alongside the agent."""

    def __init__(self, agent: TradingAgent, feed_stats_fn, account_info: str = ""):
        self.agent = agent
        self.feed_stats_fn = feed_stats_fn
        self.account_info = account_info
        self.console = Console()
        self._tick = 0

    def render(self):
        self._tick += 1
        stats = self.feed_stats_fn()
        return render_dashboard(self.agent, stats, self.account_info, self._tick)
