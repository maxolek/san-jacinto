"""CLI entry point for ETL operations.

Usage:
    python -m data.etl --register_engine --engine path/to/engine.exe
  python -m data.etl --log_games
"""
import sqlite3
import argparse

from .paths import RAW_DB
from .db import get_db, register_engine
from .ingest import log_games_directory, ingest_log_dir


def main():

    p = argparse.ArgumentParser(
        description="ETL functions for chess.db\nRegister engines, upload logs, clear directories"
    )

    p.add_argument("--register_engine", action="store_true",
                   help="Flag to register engine to chess.db")
    p.add_argument("--engine", default=None, type=str,
                   help="Path to engine binary for UCI probing")
    p.add_argument("--name", default=None, type=str, help="Engine name")
    p.add_argument("--version", default=None, type=str, help="Engine version")
    p.add_argument("--description", default=None, type=str,
                   help="Engine description (changes from last iteration, etc)")
    p.add_argument("--uci_options", default=None, type=str,
                   help="UCI settings of the engine (e.g. threads, hash size, etc.)")

    p.add_argument("--log_games", action="store_true",
                   help="Flag to log game directory to chess.db")
    p.add_argument("--log_dir", default=None, type=str,
                   help="Directory containing logs to log to chess.db")

    args = p.parse_args()

    # Resolve DB path
    raw_path = str(RAW_DB)

    cnxn = sqlite3.connect(raw_path)

    if args.register_engine:
        engine_data = {
            "name": args.name,
            "version": args.version,
            "description": args.description,
        }
        if args.engine:
            engine_data["engine_path"] = args.engine
        register_engine(cnxn, engine_data)
    elif args.log_games:
        log_games_directory(cnxn)
    elif args.log_dir:
        ingest_log_dir(cnxn, args.log_dir)

    cnxn.close()


if __name__ == '__main__':
    main()
