"""Tests for the versioned DB migration system."""

import sqlite3
import pytest

from src.db.migration_runner import run_migrations, _current_version
from src.services.journal_service import JournalService


class TestMigrationRunner:
    def test_fresh_db_applies_all_migrations(self):
        conn = sqlite3.connect(":memory:")
        version = run_migrations(conn)
        assert version >= 3  # m001 + m002 + m003 all applied
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "trades" in tables
        assert "pending_orders" in tables
        assert "schema_version" in tables
        assert "position_risk" in tables

    def test_rerun_is_idempotent(self):
        conn = sqlite3.connect(":memory:")
        v1 = run_migrations(conn)
        v2 = run_migrations(conn)
        assert v1 == v2

    def test_ai_columns_present(self):
        conn = sqlite3.connect(":memory:")
        run_migrations(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
        assert "ai_confidence" in cols
        assert "news_sentiment_score" in cols

    def test_journal_service_uses_migrations(self, tmp_path):
        db_path = tmp_path / "test_journal.db"
        journal = JournalService(db_path=str(db_path))
        assert journal.trades == []
        conn = sqlite3.connect(str(db_path))
        assert _current_version(conn) >= 3


class TestPositionRiskPersistence:
    """Position-risk reference data must survive across JournalService
    instances (i.e. across an application restart) — that's the whole
    point of persisting it instead of keeping it in memory."""

    def test_record_and_retrieve(self, tmp_path):
        db_path = tmp_path / "journal.db"
        journal = JournalService(db_path=str(db_path))
        journal.record_position_risk(1001, "XAUUSD", "BUY", 2450.0, 2440.0)
        risk = journal.get_position_risk(1001)
        assert risk["entry_price"] == 2450.0
        assert risk["initial_sl"] == 2440.0

    def test_missing_ticket_returns_none(self, tmp_path):
        journal = JournalService(db_path=str(tmp_path / "journal.db"))
        assert journal.get_position_risk(9999) is None

    def test_survives_new_journal_instance(self, tmp_path):
        db_path = str(tmp_path / "journal.db")
        journal1 = JournalService(db_path=db_path)
        journal1.record_position_risk(2002, "EURUSD", "SELL", 1.0850, 1.0900)

        # Simulate an application restart: a brand-new JournalService instance.
        journal2 = JournalService(db_path=db_path)
        risk = journal2.get_position_risk(2002)
        assert risk is not None
        assert risk["symbol"] == "EURUSD"

    def test_clear_removes_entry(self, tmp_path):
        journal = JournalService(db_path=str(tmp_path / "journal.db"))
        journal.record_position_risk(3003, "XAUUSD", "BUY", 2450.0, 2440.0)
        journal.clear_position_risk(3003)
        assert journal.get_position_risk(3003) is None

    def test_all_tickets_lists_open_positions(self, tmp_path):
        journal = JournalService(db_path=str(tmp_path / "journal.db"))
        journal.record_position_risk(1, "XAUUSD", "BUY", 2450.0, 2440.0)
        journal.record_position_risk(2, "EURUSD", "SELL", 1.08, 1.09)
        tickets = journal.all_position_risk_tickets()
        assert set(tickets) == {1, 2}
