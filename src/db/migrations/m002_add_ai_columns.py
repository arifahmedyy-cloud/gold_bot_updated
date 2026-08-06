"""Adds columns to store the AI reviewer's opinion alongside each trade."""

VERSION = 2


def _column_exists(conn, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def up(conn) -> None:
    if not _column_exists(conn, "trades", "ai_confidence"):
        conn.execute("ALTER TABLE trades ADD COLUMN ai_confidence REAL")
    if not _column_exists(conn, "trades", "news_sentiment_score"):
        conn.execute("ALTER TABLE trades ADD COLUMN news_sentiment_score REAL")
    if not _column_exists(conn, "pending_orders", "ai_confidence"):
        conn.execute("ALTER TABLE pending_orders ADD COLUMN ai_confidence REAL")
    if not _column_exists(conn, "pending_orders", "news_sentiment_score"):
        conn.execute("ALTER TABLE pending_orders ADD COLUMN news_sentiment_score REAL")
