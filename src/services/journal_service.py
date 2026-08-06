"""SQLite-based trade journal replacing JSON file storage.

Provides persistent, queryable trade history with full CRUD operations.
"""

from __future__ import annotations

import sqlite3
import json
import csv
import io
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.logger import get_logger
from src.exceptions import TradingBotError
from src.db.migration_runner import run_migrations

log = get_logger(__name__)



class JournalService:
    """Persistent SQLite trade journal."""

    def __init__(self, db_path: str = "logs/trade_journal.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            run_migrations(conn)

    def record_trade(
        self,
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
        symbol: str = "",
        direction: str = "",
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        lot_size: float = 0.0,
        profit_loss: float = 0.0,
        risk_pct: float = 0.0,
        ai_score: Optional[int] = None,
        regime: str = "",
        smc_bias: str = "",
        confluence_notes: Optional[str] = None,
        strategy: str = "",
    ) -> int:
        """Record a completed trade. Returns row id."""
        result = "WIN" if profit_loss > 0 else ("LOSS" if profit_loss < 0 else "BREAKEVEN")
        risk_amount = abs(entry_price - stop_loss) * lot_size
        r_multiple = (profit_loss / risk_amount) if risk_amount > 0 else None
        duration = None
        if entry_time and exit_time:
            try:
                duration = (exit_time - entry_time).total_seconds() / 3600.0
            except Exception:
                pass

        sql = """
            INSERT OR REPLACE INTO trades
            (ticket, entry_time, exit_time, symbol, direction, entry_price, exit_price,
             stop_loss, take_profit, lot_size, profit_loss, risk_pct, ai_score, regime,
             smc_bias, confluence_notes, strategy, result, r_multiple, trade_duration_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            cur = conn.execute(sql, (
                None, entry_time, exit_time, symbol, direction, entry_price, exit_price,
                stop_loss, take_profit, lot_size, profit_loss, risk_pct, ai_score,
                regime, smc_bias, confluence_notes, strategy, result, r_multiple, duration,
            ))
            conn.commit()
            log.info("Trade recorded: %s %s P/L=%.2f", direction, symbol, profit_loss)
            return cur.lastrowid

    def record_order_open(self, ticket: int, **kwargs: Any) -> None:
        """Store pending order context for enrichment on close."""
        columns = [
            "ticket", "entry_time", "symbol", "direction", "entry_price",
            "stop_loss", "take_profit", "lot_size", "risk_pct", "ai_score",
            "regime", "smc_bias", "confluence_notes", "strategy",
        ]
        values = [ticket] + [kwargs.get(c) for c in columns[1:]]
        placeholders = ",".join("?" * len(columns))
        sql = f"INSERT OR REPLACE INTO pending_orders ({','.join(columns)}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.execute(sql, values)
            conn.commit()

    def pop_pending_order(self, ticket: int) -> Dict[str, Any]:
        """Retrieve and delete pending order by ticket."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM pending_orders WHERE ticket=?", (ticket,)).fetchone()
            if row:
                conn.execute("DELETE FROM pending_orders WHERE ticket=?", (ticket,))
                conn.commit()
                return dict(row)
            return {}

    def record_position_risk(
        self, ticket: int, symbol: str, direction: str, entry_price: float, initial_sl: float,
    ) -> None:
        """Store the fixed entry/SL reference for a newly opened position,
        used by trailing-stop/break-even to compute R-multiples. This
        persists to disk so the reference survives an application restart —
        only the broker's live SL changes over time, this baseline never does."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO position_risk "
                "(ticket, symbol, direction, entry_price, initial_sl) VALUES (?, ?, ?, ?, ?)",
                (ticket, symbol, direction, entry_price, initial_sl),
            )
            conn.commit()

    def get_position_risk(self, ticket: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM position_risk WHERE ticket=?", (ticket,)).fetchone()
            return dict(row) if row else None

    def all_position_risk_tickets(self) -> List[int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT ticket FROM position_risk").fetchall()
            return [r["ticket"] for r in rows]

    def clear_position_risk(self, ticket: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM position_risk WHERE ticket=?", (ticket,))
            conn.commit()

    @property
    def trades(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM trades ORDER BY entry_time DESC").fetchall()
            return [dict(r) for r in rows]

    @property
    def open_orders(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM pending_orders").fetchall()
            return [dict(r) for r in rows]

    def filter_trades(
        self,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        regime: Optional[str] = None,
        smc_bias: Optional[str] = None,
        result: Optional[str] = None,
        min_ai_score: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM trades WHERE 1=1"
        params: List[Any] = []
        if symbol:
            query += " AND symbol=?"
            params.append(symbol)
        if direction:
            query += " AND direction=?"
            params.append(direction)
        if regime:
            query += " AND regime=?"
            params.append(regime)
        if smc_bias:
            query += " AND smc_bias=?"
            params.append(smc_bias)
        if result:
            query += " AND result=?"
            params.append(result)
        if min_ai_score is not None:
            query += " AND ai_score >= ?"
            params.append(min_ai_score)
        query += " ORDER BY entry_time DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def to_csv(self, trades: Optional[List[Dict[str, Any]]] = None) -> str:
        rows = trades if trades is not None else self.trades
        if not rows:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()

    def to_json(self, trades: Optional[List[Dict[str, Any]]] = None) -> str:
        rows = trades if trades is not None else self.trades
        return json.dumps(rows, indent=2, default=str)

    def __len__(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM trades").fetchone()
            return row[0] if row else 0
