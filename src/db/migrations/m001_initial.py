"""Initial schema: trades + pending_orders tables (matches the pre-migration
journal_service.py schema exactly, so existing databases upgrade cleanly)."""

VERSION = 1

_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket INTEGER UNIQUE,
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    symbol TEXT,
    direction TEXT,
    entry_price REAL,
    exit_price REAL,
    stop_loss REAL,
    take_profit REAL,
    lot_size REAL,
    profit_loss REAL,
    risk_pct REAL,
    ai_score INTEGER,
    regime TEXT,
    smc_bias TEXT,
    confluence_notes TEXT,
    strategy TEXT,
    result TEXT,
    r_multiple REAL,
    trade_duration_hours REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pending_orders (
    ticket INTEGER PRIMARY KEY,
    entry_time TIMESTAMP,
    symbol TEXT,
    direction TEXT,
    entry_price REAL,
    stop_loss REAL,
    take_profit REAL,
    lot_size REAL,
    risk_pct REAL,
    ai_score INTEGER,
    regime TEXT,
    smc_bias TEXT,
    confluence_notes TEXT,
    strategy TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_direction ON trades(direction);
CREATE INDEX IF NOT EXISTS idx_trades_regime ON trades(regime);
CREATE INDEX IF NOT EXISTS idx_trades_result ON trades(result);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
"""


def up(conn) -> None:
    conn.executescript(_SQL)
