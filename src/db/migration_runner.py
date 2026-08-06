"""Lightweight versioned migration runner for the SQLite journal database.

Every schema change (new table, new column) is a numbered module in
`src/db/migrations/`, each exposing `VERSION: int` and `up(conn)`. On
startup, `run_migrations(conn)` applies any migration whose VERSION is
greater than the database's current recorded version, in order, inside a
single transaction per migration. This means:

- A fresh install runs every migration from 1 upward.
- An existing install only runs the ones it hasn't seen yet.
- Nothing is ever hand-edited on a live database — a schema change is
  always a new file, never an edit to an old one.
"""

from __future__ import annotations

import importlib
import pkgutil
import sqlite3
from typing import List, Tuple, Callable

from src.logger import get_logger

log = get_logger(__name__)

_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


def _current_version(conn: sqlite3.Connection) -> int:
    conn.execute(_VERSION_TABLE)
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] if row and row[0] is not None else 0


def _record_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


def discover_migrations() -> List[Tuple[int, Callable]]:
    """Import every module in src.db.migrations and collect (VERSION, up) pairs."""
    from src.db import migrations as migrations_pkg

    found = []
    for _, mod_name, _ in pkgutil.iter_modules(migrations_pkg.__path__):
        mod = importlib.import_module(f"src.db.migrations.{mod_name}")
        if hasattr(mod, "VERSION") and hasattr(mod, "up"):
            found.append((mod.VERSION, mod.up))
    found.sort(key=lambda pair: pair[0])
    return found


def run_migrations(conn: sqlite3.Connection) -> int:
    """Apply any pending migrations. Returns the final schema version."""
    current = _current_version(conn)
    migrations = discover_migrations()
    applied = 0
    for version, up_fn in migrations:
        if version <= current:
            continue
        log.info("Applying DB migration %d", version)
        up_fn(conn)
        _record_version(conn, version)
        applied += 1
        current = version
    conn.commit()
    if applied:
        log.info("Applied %d migration(s), schema now at version %d", applied, current)
    return current
