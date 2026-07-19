"""Project paths and constants."""
import os
import platform
from pathlib import Path

# projects/
# ├── tomahawk/
# └── san-jacinto/
#     └── data/
#         └── etl/
#             └── paths.py
DATA_DIR = Path(__file__).resolve().parent.parent  # data/
PROJECT_ROOT = DATA_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"
GAMES_LOG_DIR = LOGS_DIR / "game_logs"

# Default JSONL paths (game_logs directory)
GAME_JSON = GAMES_LOG_DIR / "game.jsonl"
SEARCH_JSON = GAMES_LOG_DIR / "search.jsonl"
TIMING_JSON = GAMES_LOG_DIR / "timing.jsonl"
ROOT_MOVES_JSON = GAMES_LOG_DIR / "root_moves.jsonl"

# Known log directories that may contain JSONL data
LOG_DIRS = [
    GAMES_LOG_DIR,
    LOGS_DIR / "sprt_logs",
    LOGS_DIR / "sts_logs",
    LOGS_DIR / "tournament_logs",
    LOGS_DIR / "test_logs",
]

# JSONL filenames (consistent across all log dirs)
JSONL_FILES = ["game.jsonl", "search.jsonl", "timing.jsonl", "root_moves.jsonl"]


def get_jsonl_paths(log_dir):
    """Return dict of JSONL paths for a given log directory."""
    log_dir = Path(log_dir)
    return {
        "game": log_dir / "game.jsonl",
        "search": log_dir / "search.jsonl",
        "timing": log_dir / "timing.jsonl",
        "root_moves": log_dir / "root_moves.jsonl",
    }

# ─────────────────────────────────────────────────────────────────────────────
# Database paths — configurable via CHESS_RAW_DB / CHESS_ANALYTICS_DB env vars
# ─────────────────────────────────────────────────────────────────────────────
_system = platform.system()

if _system == "Windows":
    _DEFAULT_RAW = Path("F:/databases/chess.db")
    _DEFAULT_ANALYTICS = Path("F:/databases/chess_analytics.duckdb")
elif _system == "Darwin":
    _DEFAULT_RAW = Path.home() / "Documents/databases/chess.db"
    _DEFAULT_ANALYTICS = Path.home() / "Documents/databases/chess_analytics.duckdb"
else:
    _DEFAULT_RAW = Path.home() / "Documents/databases/chess.db"
    _DEFAULT_ANALYTICS = Path.home() / "Documents/databases/chess_analytics.duckdb"

RAW_DB = Path(os.environ.get("CHESS_RAW_DB") or _DEFAULT_RAW)
ANALYTICS_DB = Path(os.environ.get("CHESS_ANALYTICS_DB") or _DEFAULT_ANALYTICS)
