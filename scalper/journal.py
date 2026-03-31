"""Trade journal and analytics logger.

Logs every decision the agent makes to JSON files for analysis:
- Every signal (taken or skipped, with reasons)
- Every tick event
- Every trade (entry, management, exit with full context)
- Every regime change
- Every risk decision
- Performance snapshots every N minutes

This data is the foundation for iterating on the strategy.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from scalper.models import Signal, TradeResult, MarketRegime, Side
from scalper import __version__


class TradeJournal:
    """Persistent JSON logger for all trading decisions."""

    def __init__(self, log_dir: str = "logs"):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._version = __version__
        self._run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Files include version in name for easy filtering
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        self._signals_file = self._dir / f"signals_{date_str}.jsonl"
        self._trades_file = self._dir / f"trades_{date_str}.jsonl"
        self._events_file = self._dir / f"events_{date_str}.jsonl"
        self._snapshots_file = self._dir / f"snapshots_{date_str}.jsonl"
        self._decisions_file = self._dir / f"decisions_{date_str}.jsonl"

        self._last_snapshot = 0.0
        self._snapshot_interval = 60.0

        # Log run start
        self._write(self._events_file, {
            "type": "run_start",
            "version": self._version,
            "run_id": self._run_id,
        })

    def _write(self, filepath: Path, data: dict) -> None:
        data["_ts"] = time.time()
        data["_utc"] = datetime.now(timezone.utc).isoformat()
        data["_v"] = self._version
        data["_run"] = self._run_id
        with open(filepath, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def log_signal(
        self,
        signal: Signal,
        taken: bool,
        skip_reason: str = "",
        size: int = 0,
        htf_confluence: dict = None,
    ) -> None:
        """Log a signal and whether it was taken."""
        self._write(self._signals_file, {
            "type": "signal",
            "taken": taken,
            "skip_reason": skip_reason,
            "side": signal.side.value if signal.side else None,
            "signal_type": signal.signal_type.value,
            "confidence": round(signal.confidence, 4),
            "entry": signal.entry_price,
            "stop": signal.stop_price,
            "target": signal.target_price,
            "rr_ratio": round(signal.rr_ratio, 2),
            "regime": signal.regime.value,
            "reasons": signal.reasons,
            "size": size,
            "htf": htf_confluence,
        })

    def log_trade_entry(
        self,
        side: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        size: int,
        confidence: float,
        regime: str,
        reasons: list[str],
        order_type: str,
        indicators: dict = None,
        htf: dict = None,
    ) -> None:
        """Log trade entry with full context."""
        self._write(self._trades_file, {
            "type": "entry",
            "side": side,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "size": size,
            "confidence": round(confidence, 4),
            "regime": regime,
            "reasons": reasons,
            "order_type": order_type,
            "indicators": indicators,
            "htf": htf,
        })

    def log_trade_exit(self, trade: TradeResult, indicators: dict = None) -> None:
        """Log trade exit with performance data."""
        self._write(self._trades_file, {
            "type": "exit",
            "side": trade.side.value,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl": round(trade.pnl, 2),
            "quantity": trade.quantity,
            "exit_reason": trade.exit_reason,
            "regime": trade.regime.value,
            "holding_candles": trade.holding_candles,
            "max_favorable": round(trade.max_favorable, 2),
            "max_adverse": round(trade.max_adverse, 2),
            "signal_confidence": round(trade.signal_confidence, 4),
            "indicators": indicators,
        })

    def log_tick_event(self, event_type: str, direction: int, magnitude: float,
                       price: float, description: str, taken: bool = False) -> None:
        """Log intra-candle tick event."""
        self._write(self._events_file, {
            "type": "tick_event",
            "event_type": event_type,
            "direction": direction,
            "magnitude": round(magnitude, 3),
            "price": price,
            "description": description,
            "taken": taken,
        })

    def log_regime_change(self, old: str, new: str, confidence: float) -> None:
        """Log regime change."""
        self._write(self._events_file, {
            "type": "regime_change",
            "from": old,
            "to": new,
            "confidence": round(confidence, 3),
        })

    def log_risk_decision(self, decision: str, details: dict) -> None:
        """Log risk management decisions."""
        self._write(self._decisions_file, {
            "type": "risk_decision",
            "decision": decision,
            **details,
        })

    def log_snapshot(self, agent_status: dict) -> None:
        """Log periodic performance snapshot."""
        now = time.time()
        if now - self._last_snapshot < self._snapshot_interval:
            return
        self._last_snapshot = now
        self._write(self._snapshots_file, {
            "type": "snapshot",
            **agent_status,
        })

    def log_skip(self, reason: str, details: dict = None) -> None:
        """Log why a potential trade was skipped."""
        self._write(self._decisions_file, {
            "type": "skip",
            "reason": reason,
            **(details or {}),
        })
