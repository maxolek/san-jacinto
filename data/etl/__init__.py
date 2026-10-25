"""ETL package for chess engine data ingestion and processing.

Submodules:
  - paths: project paths and constants
  - utils: small helper functions
  - openings: ECO/opening classification from move lists
  - db: database connection helpers and engine probing
  - schema: bring the raw database up to the current schema (data/databases/migrate_schema.py)
  - columns: log values stored by their column's declared type
  - provenance: build identity (binary + net hash) and how a build was made
  - ingest: bulk logging and ingestion functions (games, searches, timing, STS, SPRT)
  - match_games: games recorded by the match runner (utils/match_runner.py) and how engine logs link to them
"""
from .paths import (
    DATA_DIR, PROJECT_ROOT, LOGS_DIR, GAMES_LOG_DIR,
    GAME_JSON, SEARCH_JSON, TIMING_JSON, ROOT_MOVES_JSON,
    LOG_DIRS, JSONL_FILES, get_jsonl_paths, connect_raw,
)
from .utils import safe_val, safe, consolidate_instance_logs
from .openings import get_opening_from_moves
from .db import get_db, get_engine_id, register_engine, probe_engine_metadata, extract_engine_id_from_search, clear_log_dir
from .schema import ensure_schema
from .provenance import resolve_engine, engine_provenance, write_build_info, read_build_info, locate_engine
from .ingest import (
    log_games_directory,
    ingest_log_dir,
    ingest_all_log_dirs,
    start_experiment,
    update_experiment,
    log_perft,
    log_engine_ratings,
    log_sprt,
    bulk_log_sts,
    bulk_log_game,
    bulk_log_search_and_timing,,
    bulk_log_match_games,
)
