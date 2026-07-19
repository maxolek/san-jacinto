import sqlite3
import duckdb
from pathlib import Path
import argparse
from ..etl.paths import RAW_DB, ANALYTICS_DB

def drop_all_tables(db_path) -> None:
    """
    Drops all tables from chess.db.
    WARNING: This deletes schemas and all data permanently.
    """

    is_duckdb = str(db_path).endswith(".duckdb")
    if is_duckdb: cnxn = duckdb.connect(db_path)
    else: cnxn = sqlite3.connect(db_path)
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
            "engines",
            "engine_ratings",
            "position_features",
            "search_iteration_features",
            "search_tree_features",
            "search_features",
            "schema_migrations"
        ]
    else:
        tables = [
            "timing",
            "searches_by_iteration",
            "searches_by_tree_depth",
            "searches",
            "root_moves",
            "games",
            "sprt",
            "sts",
            "perft",
            "experiments",
            "engines",
            "engine_ratings",
            "schema_migrations"
        ]

    dropped = []
    for table in tables:
        cur.execute(f"DROP TABLE IF EXISTS {table};")
        dropped.append(table)

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
    else: cnxn = sqlite3.connect(db_path)
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
            "search_features"
        ] + (["engines", "engine_ratings"] if not exclude_engines else [])
    else:
        tables = [
            "timing",
            "searches_by_iteration",
            "searches_by_tree_depth",
            "searches",
            "root_moves",
            "games",
            "sprt",
            "sts",
            "perft",
            "experiments"
        ] + (["engines", "engine_ratings"] if not exclude_engines else [])

    total_deleted = 0
    deleted_info = []

    for table in tables:
        cur.execute(f"DELETE FROM {table};")
        count = cur.rowcount  # rows affected
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
