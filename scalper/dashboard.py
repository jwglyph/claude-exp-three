"""Clean Rich live terminal dashboard for the trading agent."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box
import numpy as np

from scalper.agent import TradingAgent
from scalper.models import Side
from scalper import __version__


SPIN = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
BARS = "▁▂▃▄▅▆▇█"
DIM = "[dim]"
END = "[/]"


def fp(p: float) -> str:
    return f"{p:,.2f}" if p > 0 else "---"

def fpnl(pnl: float) -> str:
    if pnl > 0: return f"[bold green]+${pnl:,.2f}[/]"
    elif pnl < 0: return f"[bold red]-${abs(pnl):,.2f}[/]"
    return f"{DIM}$0.00{END}"

def rc(regime: str) -> str:
    return {"trending_up": "bold green", "trending_down": "bold red", "ranging": "yellow",
            "volatile": "bold magenta", "low_volatility": "cyan"}.get(regime, "white")

def sparkline(prices: list[float], width: int = 20) -> str:
    if len(prices) < 2: return f"{DIM}···{END}"
    p = prices[-width:]
    mn, mx = min(p), max(p)
    rng = mx - mn
    if rng == 0: return "▅" * len(p)
    chars = [BARS[int((v - mn) / rng * 7)] for v in p]
    c = "green" if p[-1] >= p[0] else "red"
    return f"[{c}]{''.join(chars)}[/]"

def bar(filled: int, total: int = 20, color: str = "white") -> str:
    return f"[{color}]{'█' * filled}{'░' * (total - filled)}[/]"

def tf_arrow(t: int) -> str:
    if t > 0: return "[green]▲[/]"
    elif t < 0: return "[red]▼[/]"
    return f"{DIM}─{END}"

def sep() -> str:
    return f"  {DIM}{'─' * 72}{END}"


def render_dashboard(agent: TradingAgent, feed_stats: dict, account_info: str, tick_counter: int) -> Panel:
    s = agent.get_status()
    risk = s["risk"]
    pos = s["position"]
    ind = s["indicators"]
    adaptive = s.get("adaptive", {})
    regime = s.get("regime", "unknown")
    trades = agent.risk_mgr.trade_history
    htf = s.get("htf", {})
    of = s.get("orderflow", {})
    scan = s.get("scan")
    dr = s.get("dynamic_risk", {})

    last_price = feed_stats.get("last_price", 0)
    connected = feed_stats.get("connected", False)
    quotes = feed_stats.get("quotes", 0)
    ftrades = feed_stats.get("trades", 0)

    candles = agent.aggregator.get_candles()
    closes = [c.close for c in candles[-30:]] if candles else []
    if last_price > 0: closes.append(last_price)
    current = agent.aggregator.current_candle

    spin = SPIN[tick_counter % len(SPIN)]
    now_utc = datetime.now(timezone.utc)
    try:
        import zoneinfo
        pt = now_utc.astimezone(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%H:%M %Z")
    except Exception:
        from datetime import timedelta
        pt = (now_utc - timedelta(hours=7)).strftime("%H:%M PDT")

    L = []  # lines

    # ═══════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════
    icon = "[green]●[/]" if connected else "[red]●[/]"
    price = f"[bold white]{fp(last_price)}[/]" if last_price > 0 else f"{DIM}---{END}"
    regime_str = f"[{rc(regime)}]{regime.replace('_', ' ').upper()}[/]"

    L.append(f"  {icon} [bold cyan]NQ[/] {price}  {regime_str}  {sparkline(closes)}  {DIM}v{__version__} {pt} {spin}{END}")

    # Account + market
    try:
        from scalper.market_hours import get_session_info
        mkt = get_session_info()
        mkt_s = f"[green]●{END} {mkt['session']}" if mkt["open"] else f"[red]●{END} {mkt['session']}"
        if mkt.get("time_to_open"): mkt_s += f" opens {mkt['time_to_open']}"
    except Exception:
        mkt_s = ""
    L.append(f"  {DIM}{account_info}  {mkt_s}{END}")

    L.append(sep())

    # ═══════════════════════════════════════════════════════
    # CANDLE + HTF + LIVE SIGNALS
    # ═══════════════════════════════════════════════════════
    if current:
        elapsed = time.time() - current.timestamp
        pct = min(elapsed / agent.config.candle_interval_sec, 1.0)
        filled = int(pct * 15)
        c_bar = bar(filled, 15, "cyan")
        c_dir = "[green]▲[/]" if current.is_bullish else "[red]▼[/]" if current.is_bearish else "─"
        L.append(
            f"  Candle [{c_bar}] {pct:.0%}  {c_dir} "
            f"O {current.open:.2f}  H {current.high:.2f}  L {current.low:.2f}  C {current.close:.2f}  "
            f"V {current.volume}  Δ {current.delta:+d}"
        )
    else:
        L.append(f"  {DIM}Waiting for tick data...{END}")

    # HTF + Flow on one line
    htf_parts = []
    if htf:
        htf_parts.append(f"HTF: 5m{tf_arrow(htf.get('5m',0))} 15m{tf_arrow(htf.get('15m',0))} 1h{tf_arrow(htf.get('1h',0))}")
        agrees = htf.get("agrees", "neutral")
        if agrees == "long": htf_parts.append("[green]BULL[/]")
        elif agrees == "short": htf_parts.append("[red]BEAR[/]")
        else: htf_parts.append(f"{DIM}NEUTRAL{END}")
    if of:
        fb = of.get("flow_bias", 0)
        fb_c = "green" if fb > 0.1 else "red" if fb < -0.1 else "dim"
        d1m = of.get("delta_1m", 0)
        d1c = "green" if d1m > 0 else "red" if d1m < 0 else "dim"
        bt, st = of.get("buy_trades", 0), of.get("sell_trades", 0)
        htf_parts.append(f"Flow:[{fb_c}]{fb:+.2f}[/]")
        htf_parts.append(f"Δ:[{d1c}]{d1m:+d}[/]")
        htf_parts.append(f"{DIM}{bt}B/{st}S{END}")
        abs_v = of.get("absorption", 0)
        if abs(abs_v) > 0.1:
            ac = "green" if abs_v > 0 else "red"
            htf_parts.append(f"[{ac}]absorb{abs_v:+.1f}[/]")
    if htf_parts:
        L.append(f"  {DIM}│{END} {'  '.join(htf_parts)}")

    # Live tick events
    tick_evt = s.get("tick_event")
    if tick_evt and tick_evt.get("age", 999) < 20:
        ec = "green" if tick_evt["dir"] > 0 else "red"
        L.append(f"  [{ec}]⚡ {tick_evt['type'].upper()}: {tick_evt['desc']}[/] {DIM}({tick_evt['age']:.0f}s ago){END}")

    L.append(sep())

    # ═══════════════════════════════════════════════════════
    # SIGNAL ANALYSIS
    # ═══════════════════════════════════════════════════════
    if scan:
        bs = scan.get("bull_score", 0)
        ss = scan.get("bear_score", 0)
        bull_f = scan.get("bull", [])
        bear_f = scan.get("bear", [])

        # Confidence
        total = bs + ss
        if total > 0:
            net = bs - ss
            conf = (abs(net) / total) * 0.5 + max(bs, ss) * 0.5
            side_str = "[green]▲ LONG[/]" if net > 0 else "[red]▼ SHORT[/]"
            cc = "green" if conf >= 0.50 else "yellow" if conf >= 0.35 else "dim"
            # Confidence bar
            cf = int(min(conf, 1.0) * 15)
            cb = bar(cf, 15, cc)
            L.append(f"  Signal: {side_str}  [{cb}] [{cc}]{conf:.0%}[/]  {DIM}need 50%{END}")

        # Factors as compact tags
        if bull_f:
            tags = " ".join(f"[green]{n}[/]" for n, w in bull_f[:5])
            L.append(f"  [green]▲[/] {DIM}{bs:.2f}{END}  {tags}")
        if bear_f:
            tags = " ".join(f"[red]{n}[/]" for n, w in bear_f[:5])
            L.append(f"  [red]▼[/] {DIM}{ss:.2f}{END}  {tags}")
    else:
        L.append(f"  {DIM}Signal: warming up...{END}")

    L.append(sep())

    # ═══════════════════════════════════════════════════════
    # POSITION
    # ═══════════════════════════════════════════════════════
    if pos["side"]:
        sc = "green" if pos["side"] == "long" else "red"
        arrow = "▲" if pos["side"] == "long" else "▼"
        trail_str = f"  Trail: {fp(pos['trail'])}" if pos.get("trail") and pos["trail"] > 0 else ""
        L.append(
            f"  [{sc}]{arrow} {pos['side'].upper()}[/] @ {fp(pos['entry'])}  "
            f"P&L: {fpnl(pos['pnl'])}  "
            f"Stop: {fp(pos['stop'] or 0)}  Target: {fp(pos['target'] or 0)}{trail_str}"
        )
    else:
        L.append(f"  {DIM}○ FLAT — scanning...{END}")

    # ═══════════════════════════════════════════════════════
    # RISK + INDICATORS (compact)
    # ═══════════════════════════════════════════════════════
    dd_rem = risk["drawdown_remaining"]
    dd_max = agent.config.max_drawdown
    dd_pct = (dd_max - dd_rem) / dd_max if dd_max > 0 else 0
    dc = "green" if dd_pct < 0.3 else "yellow" if dd_pct < 0.6 else "red"
    dd_b = bar(int(dd_pct * 12), 12, dc)

    rsi = ind["rsi"]
    rsi_c = "red" if rsi > 70 else "green" if rsi < 30 else "white"

    L.append(
        f"  P&L {fpnl(risk['daily_pnl'])}  DD [{dd_b}] ${dd_rem:,.0f}  "
        f"Risk x{risk['risk_multiplier']:.1f}  "
        f"RSI [{rsi_c}]{rsi:.0f}[/]  ATR {ind['atr']:.1f}  "
        f"EMA {fp(ind['ema_fast'])}/{fp(ind['ema_slow'])}"
    )

    # Edge (if we have data)
    if dr and dr.get("sample_size", 0) > 0:
        edge = dr.get("edge_per_dollar", 0)
        ec = "green" if edge > 0 else "red"
        L.append(
            f"  {DIM}Edge:[/] [{ec}]${edge:.2f}/$ risked[/]  "
            f"Kelly {dr.get('kelly_pct',0):.1f}%  "
            f"WR {dr.get('win_rate',0):.0%}  "
            f"Payoff {dr.get('payoff_ratio',0):.1f}:1  "
            f"{DIM}({dr.get('sample_size',0)} trades){END}"
        )

    L.append(sep())

    # ═══════════════════════════════════════════════════════
    # TRADE LOG
    # ═══════════════════════════════════════════════════════
    if trades:
        L.append(f"  {DIM}{'#':>3}  {'Side':>5}  {'Entry':>10}  {'Exit':>10}  {'P&L':>8}  {'Reason':<15}{END}")
        for i, t in enumerate(trades[-5:], max(1, len(trades) - 4)):
            sc = "green" if t.side == Side.LONG else "red"
            pc = "green" if t.pnl >= 0 else "red"
            L.append(
                f"  {i:3d}  [{sc}]{t.side.value.upper():>5}[/]  "
                f"{t.entry_price:10,.2f}  {t.exit_price:10,.2f}  "
                f"[{pc}]${t.pnl:+7,.0f}[/]  {t.exit_reason:<15}"
            )

    # ═══════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════
    balance = risk.get("balance", 0)
    max_ct = risk.get("max_contracts", 2)
    wr = agent.risk_mgr.win_rate
    trade_count = s["trades"]
    candle_count = s["candles"]
    signals = s["signals"]

    L.append(
        f"  {DIM}Bal ${balance:,.0f} | {max_ct}ct | "
        f"{candle_count}candles {signals}sig {trade_count}trades | "
        f"WR {wr:.0%} | Feed {quotes}q/{ftrades}t{END}"
    )

    border = "green" if connected else "red"
    return Panel(
        "\n".join(L),
        title="[bold] NQ Adaptive Scalper [/]",
        subtitle=f"{DIM}Ctrl+C to stop{END}",
        border_style=border,
        padding=(0, 0),
    )


class LiveDashboard:
    def __init__(self, agent: TradingAgent, feed_stats_fn, account_info: str = ""):
        self.agent = agent
        self.feed_stats_fn = feed_stats_fn
        self.account_info = account_info
        self.console = Console()
        self._tick = 0

    def render(self):
        self._tick += 1
        return render_dashboard(self.agent, self.feed_stats_fn(), self.account_info, self._tick)
