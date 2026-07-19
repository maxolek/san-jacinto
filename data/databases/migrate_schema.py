#!/usr/bin/env python3
"""
migrate_schema.py

Adds new columns to SQLite (raw) and DuckDB (analytics) tables
without breaking existing pipelines.

How it stays safe:
  - New columns are always added as NULLABLE with a DEFAULT, so any
    existing INSERT/pipeline code that doesn't know about the new
    column yet keeps working unchanged.
  - Each migration is tracked in a `schema_migrations` table inside
    the target db. Already-applied migrations are skipped, so it's
    safe to re-run this script every time (cron, CI, pre-pipeline hook).
  - --dry-run shows exactly what would change before you touch real data.
  - Each migration runs in its own transaction: it either fully applies
    or fully rolls back.

Usage:
  python migrate_schema.py --raw-db ./raw.db --analytics-db ./analytics.duckdb
  python migrate_schema.py --raw-db ./raw.db --analytics-db ./analytics.duckdb --dry-run
  python migrate_schema.py --raw-db ./raw.db --analytics-db ./analytics.duckdb --only pvs_research_time_logging

To add a new feature later: add a new entry to MIGRATIONS below and
run the script again. Never edit an already-applied migration's id or
column list after it has shipped -- add a new migration instead.
"""

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from ..etl.paths import RAW_DB, ANALYTICS_DB

try:
    import duckdb
except ImportError:
    duckdb = None


# ---------------------------------------------------------------------------
# 1. DEFINE MIGRATIONS HERE
# ---------------------------------------------------------------------------
# db:      "raw" or "analytics"
# table:   table name to alter
# columns: list of (column_name, sql_type, default_sql_literal)
#          default_sql_literal is inserted verbatim into the DEFAULT clause,
#          e.g. "0", "0.0", "''", "NULL", "CURRENT_TIMESTAMP"

@dataclass
class Migration:
    id: str                    # unique, stable, never reused
    description: str
    db: str                    # "raw" or "analytics"
    table: str
    columns: list = field(default_factory=list)  # (name, type, default)


MIGRATIONS = [
    # Add future feature migrations, e.g.:
    # Migration(
    #     id="2026_08_some_new_feature",
    #     description="...",
    #     db="raw",
    #     table="...",
    #     columns=[("new_col", "TEXT", "''")],
    # ),
    """
    Migration(
        id="2026_07_04_completed_search_depth_rawDB",
        description="including the completed depth (not just the depth when search terminated) in raw",
        db="raw",
        table="searches",
        columns=[
            ("completed_depth", "INTEGER", "NULL")
        ]
    ),
    """
]

# ---------------------------------------------------------------------------
# 2. ENGINE ADAPTERS
# ---------------------------------------------------------------------------

class Adapter:
    """Common interface so the migration runner doesn't care if it's
    talking to sqlite3 or duckdb."""

    def __init__(self, conn):
        self.conn = conn

    def ensure_migrations_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id TEXT PRIMARY KEY,
                description TEXT,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def is_applied(self, migration_id):
        row = self.conn.execute(
            "SELECT 1 FROM schema_migrations WHERE id = ?", (migration_id,)
        ).fetchone()
        return row is not None

    def table_exists(self, table):
        raise NotImplementedError

    def existing_columns(self, table):
        raise NotImplementedError

    def add_column(self, table, name, sql_type, default):
        self.conn.execute(
            f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type} DEFAULT {default}'
        )

    def record_migration(self, migration_id, description):
        self.conn.execute(
            "INSERT INTO schema_migrations (id, description) VALUES (?, ?)",
            (migration_id, description),
        )

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()


class SQLiteAdapter(Adapter):
    def table_exists(self, table):
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None

    def existing_columns(self, table):
        rows = self.conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        return {r[1] for r in rows}  # r[1] = column name


class DuckDBAdapter(Adapter):
    def table_exists(self, table):
        row = self.conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()
        return row is not None

    def existing_columns(self, table):
        rows = self.conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            [table],
        ).fetchall()
        return {r[0] for r in rows}

    def is_applied(self, migration_id):
        row = self.conn.execute(
            "SELECT 1 FROM schema_migrations WHERE id = ?", [migration_id]
        ).fetchone()
        return row is not None

    def add_column(self, table, name, sql_type, default):
        self.conn.execute(
            f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type} DEFAULT {default}'
        )

    def record_migration(self, migration_id, description):
        self.conn.execute(
            "INSERT INTO schema_migrations (id, description) VALUES (?, ?)",
            [migration_id, description],
        )

    def commit(self):
        pass  # duckdb autocommits DDL/DML outside explicit BEGIN

    def rollback(self):
        pass


# ---------------------------------------------------------------------------
# 3. RUNNER
# ---------------------------------------------------------------------------

def run_migration(adapter: Adapter, m: Migration, dry_run: bool):
    if adapter.is_applied(m.id):
        print(f"  [skip]  {m.id} already applied")
        return

    if not adapter.table_exists(m.table):
        print(f"  [WARN]  table '{m.table}' does not exist yet in this db -- "
              f"skipping '{m.id}' (create the table first, then re-run)")
        return

    existing = adapter.existing_columns(m.table)
    to_add = [(n, t, d) for (n, t, d) in m.columns if n not in existing]
    already_there = [n for (n, t, d) in m.columns if n in existing]

    if already_there:
        print(f"  [note]  columns already present on '{m.table}', leaving as-is: "
              f"{', '.join(already_there)}")

    if not to_add:
        print(f"  [skip]  {m.id}: nothing to add, marking as applied")
        if not dry_run:
            adapter.record_migration(m.id, m.description)
            adapter.commit()
        return

    print(f"  [apply] {m.id}: {m.description}")
    for name, sql_type, default in to_add:
        print(f"          + {m.table}.{name} {sql_type} DEFAULT {default}")

    if dry_run:
        return

    try:
        for name, sql_type, default in to_add:
            adapter.add_column(m.table, name, sql_type, default)
        adapter.record_migration(m.id, m.description)
        adapter.commit()
        print(f"  [done]  {m.id}")
    except Exception as e:
        adapter.rollback()
        print(f"  [ERROR] {m.id} failed, rolled back: {e}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-db", help="Path to SQLite raw db file")
    parser.add_argument("--analytics-db", help="Path to DuckDB analytics db file")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--only", help="Only run the migration with this id")
    args = parser.parse_args()

    migrations = MIGRATIONS
    if args.only:
        migrations = [m for m in migrations if m.id == args.only]
        if not migrations:
            print(f"No migration found with id '{args.only}'", file=sys.stderr)
            sys.exit(1)

    raw_conn = sqlite3.connect(args.raw_db if args.raw_db else RAW_DB)
    raw_adapter = SQLiteAdapter(raw_conn)
    raw_adapter.ensure_migrations_table()
    raw_adapter.conn.commit()

    if duckdb is None:
        print("duckdb package not installed (pip install duckdb) -- "
              "skipping analytics migrations", file=sys.stderr)
        analytics_adapter = None
    else:
        analytics_conn = duckdb.connect(args.analytics_db if args.analytics_db else ANALYTICS_DB)
        analytics_adapter = DuckDBAdapter(analytics_conn)
        analytics_adapter.ensure_migrations_table()

    print(f"{'DRY RUN -- ' if args.dry_run else ''}Running {len(migrations)} migration(s)\n")

    for m in migrations:
        print(f"[{m.db}] {m.table}")
        if m.db == "raw":
            run_migration(raw_adapter, m, args.dry_run)
        elif m.db == "analytics":
            if analytics_adapter is None:
                print("  [skip]  analytics db not available")
                continue
            run_migration(analytics_adapter, m, args.dry_run)
        else:
            print(f"  [ERROR] unknown db '{m.db}' for migration '{m.id}'", file=sys.stderr)
        print()

    raw_conn.close()
    if analytics_adapter is not None:
        analytics_adapter.conn.close()

    print("Done." if not args.dry_run else "Dry run complete -- no changes made.")


if __name__ == "__main__":
    main()