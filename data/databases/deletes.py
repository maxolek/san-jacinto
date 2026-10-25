import sqlite3
import duckdb
from pathlib import Path
import argparse
from ..etl.paths import RAW_DB, ANALYTICS_DB, connect_raw
from ..transforms.aggregates import AGG_TABLES
from ..transforms.transform_search import ANALYTICS_VIEWS

def drop_all_tables(db_path) -> None:
    """
    Drops all tables from chess.db.
    WARNING: This deletes schemas and all data permanently.

    The raw database is dropped in one transaction, so an interruption leaves it intact
    rather than half dropped with a migration history that no longer matches its tables.
    """

    is_duckdb = str(db_path).endswith(".duckdb")
    if is_duckdb: cnxn = duckdb.connect(db_path)
    else: cnxn = connect_raw(db_path, isolation_level=None)  # BEGIN/COMMIT below
    cur = cnxn.cursor()

    if not is_duckdb: cur.execute("PRAGMA foreign_keys = OFF;")

    if is_duckdb:
        # Views must be dropped with DROP VIEW, including legacy feature names.
        for name in reversed(ANALYTICS_VIEWS):
            kind = cur.execute("""
                SELECT table_type FROM information_schema.tables
                WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = ?
            """, [name]).fetchone()
            if kind and kind[0] == 'VIEW':
                cur.execute(f'DROP VIEW "{name}"')
        tables = [
            "search_timings",
            "iterative_deepening_stats",
            "search_tree_stats",
            "search_stats",
            "root_moves",
            "dim_positions",
            "game_stats",
            "sprt_runs",
            "sts_runs",
            "perft",
            "experiments",
            "engines",
            "engine_ratings",
            "position_features",
            "search_iteration_features",
            "search_tree_features",
            "search_features",
            *AGG_TABLES,
            "analytics_meta",
            "schema_migrations"
        ]
    else:
        cur.execute("BEGIN IMMEDIATE")
        # every table, including those later migrations added: init re-creates them all
        tables = sorted((row[0] for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").fetchall()),
            key=lambda name: name != "schema_migrations")  # the history first

    dropped = []
    try:
        for table in tables:
            cur.execute(f'DROP TABLE IF EXISTS "{table}";')
            dropped.append(table)
        if not is_duckdb:
            cur.execute("COMMIT")
    except Exception:
        if not is_duckdb:
            cur.execute("ROLLBACK")
        raise

    if not is_duckdb: cur.execute("PRAGMA foreign_keys = ON;")

    cnxn.commit()
    cnxn.close()
    print(f"[DB] Dropped {len(dropped)} tables from {db_path}\n  Tables: {', '.join(dropped)}")

def clear_all_tables(db_path, exclude_engines=False) -> None:
    """
    Deletes all rows from all tables but keeps schemas intact.
    Resets AUTOINCREMENT counters.
    """

    is_duckdb = str(db_path).endswith(".duckdb")
    if is_duckdb: cnxn = duckdb.connect(db_path)
    else: cnxn = connect_raw(db_path)
    cur = cnxn.cursor()

    if not is_duckdb: cur.execute("PRAGMA foreign_keys = OFF;")

    if is_duckdb:
        tables = [
            "search_timings",
            "iterative_deepening_stats",
            "search_tree_stats",
            "search_stats",
            "root_moves",
            "dim_positions",
            "game_stats",
            "sprt_runs", 
            "sts_runs",
            "perft",
            "experiments",
            "position_features",
            "search_iteration_features",
            "search_tree_features",
            "search_features",
            *AGG_TABLES,
        ] + (["engines", "engine_ratings"] if not exclude_engines else [])
    else:
        # every table but the migration history (the schema stays) and, if asked, the
        # catalogue of engines and nets; tables later migrations add are cleared too
        keep = {"schema_migrations"} | ({"engines", "engine_ratings", "nets"} if exclude_engines else set())
        tables = [row[0] for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").fetchall()
            if row[0] not in keep]

    total_deleted = 0
    deleted_info = []

    for table in tables:
        if is_duckdb:
            kind = cur.execute("""
                SELECT table_type FROM information_schema.tables
                WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = ?
            """, [table]).fetchone()
            if not kind or kind[0] == 'VIEW':
                continue
        elif not cur.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone():
            continue  # e.g. tables added by data/etl/schema.py, not created yet
        cur.execute(f"DELETE FROM {table};")
        count = cur.fetchone()[0] if is_duckdb else cur.rowcount
        total_deleted += count
        deleted_info.append((table, count))

    # Reset AUTOINCREMENT counters
    if not is_duckdb:
        cur.execute("DELETE FROM sqlite_sequence;")
        cur.execute("PRAGMA foreign_keys = ON;")

    cnxn.commit()
    cnxn.close()

    print(f"[DB] Cleared {total_deleted:,} rows from {db_path}")
    for table, count in deleted_info:
        print(f"  - {table}: {count:,} rows")

if __name__ == "__main__":
    # clear tables or drop tables
    p = argparse.ArgumentParser(description="Clear and/or delete chess.db tables")
    p.add_argument("--db", type=str, required=True, choices=['raw', 'analytics'], help="Which database to operate on (raw or analytics)")
    p.add_argument("--clear", action="store_true")
    p.add_argument("--clear_no_engines", action="store_true")
    p.add_argument("--delete", action="store_true")
    args = p.parse_args()

    if args.clear:
        db_path = str(RAW_DB) if args.db == 'raw' else str(ANALYTICS_DB)
        print(f"[DB] Clearing all tables in {db_path}")
        clear_all_tables(db_path)
    if args.clear_no_engines:
        db_path = str(RAW_DB) if args.db == 'raw' else str(ANALYTICS_DB)
        print(f"[DB] Clearing all tables in {db_path} (excluding engines)")
        clear_all_tables(db_path, exclude_engines=True)
    if args.delete:
        db_path = str(RAW_DB) if args.db == 'raw' else str(ANALYTICS_DB)
        print(f"[DB] Dropping all tables in {db_path}")
        drop_all_tables(db_path)
