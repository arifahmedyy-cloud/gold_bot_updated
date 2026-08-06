"""Adds position_risk: stores each open position's *original* entry price
and stop-loss at the moment it was opened. Trailing-stop / break-even logic
needs this fixed reference to compute R-multiples correctly — the broker's
live SL moves over time as the trade is managed, so we can't use it as the
risk baseline. Storing it in the DB (not just in memory) is what makes
trailing-stop state survive an application restart.
"""

VERSION = 3

_SQL = """
CREATE TABLE IF NOT EXISTS position_risk (
    ticket INTEGER PRIMARY KEY,
    symbol TEXT,
    direction TEXT,
    entry_price REAL,
    initial_sl REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def up(conn) -> None:
    conn.executescript(_SQL)
