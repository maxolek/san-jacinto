"""Load raw SQLite data into analytics DuckDB.

Supports two modes:
  --full    : drop and reload all tables
  (default) : incremental -- only insert rows with id > max(existing id);
              a table that doesn't exist yet is created in full

Usage (from the repository root):
  python -m data.databases.load_analytics          # incremental
  python -m data.databases.load_analytics --full   # full refresh
  python -m data.databases.load_analytics --raw-db chess.db --analytics-db chess_analytics.duckdb

Columns added to the raw schema later are picked up automatically: tables are
created with SELECT *, and incremental loads add any missing column to the
analytics table and insert by column name. The small tables (engines,
experiments, ratings, SPRT and STS runs) are always reloaded in full, and
game_stats is rebuilt once when it was derived by an older loader
(LOADER_VERSION). Other rows that changed in SQLite after they were loaded
need a --full refresh.
"""
import argparse
from functools import lru_cache
import json
import ntpath
from pathlib import Path

import chess
import duckdb

from ..etl.openings import get_opening_details
from ..etl.paths import RAW_DB, ANALYTICS_DB
from .migrate_schema import GAMES_MATCH_COLUMNS, SPRT_PENTANOMIAL_COLUMNS

# Bump when the derivation of game_stats changes: incremental loads then rebuild it once.
# 2: opening_variation, termination codes 7..10, environment columns, EPD-book openings.
LOADER_VERSION = 2

# Engine GameResult / termination codes (see docs/dashboard-data.md).
RESULT_NAMES = {1: "white", 2: "black", 3: "draw"}          # 0 ongoing and 4 aborted become NULL
RESULT_ALIASES = {"1-0": "white", "0-1": "black", "1/2-1/2": "draw"}
TERMINATION_NAMES = {
    1: "checkmate", 2: "stalemate", 3: "threefold", 4: "fiftymove", 5: "time", 6: "resign",
    7: "adjudication", 8: "insufficient_material", 9: "illegal_move", 10: "crash",
}

# SQLite types of the shared column sets -> the types DuckDB's SQLite scanner reports.
_DUCKDB_TYPES = {"TEXT": "VARCHAR", "INTEGER": "BIGINT", "REAL": "DOUBLE"}

# sprt columns read as DOUBLE whatever their declared type: older databases
# declare the measured elo_diff and elo0/elo1 as INTEGER but hold REAL values.
SPRT_DOUBLE_COLUMNS = {
    "elo0", "elo1", "alpha", "beta", "elo_diff", "llr", "los", "time_per_move", "run_time_s",
    *(name for name, sql_type in SPRT_PENTANOMIAL_COLUMNS if sql_type == "REAL"),
}

STOCKFISH_COLUMNS_SQL = """
    CAST(NULL AS INTEGER)   AS sf_eval,
    CAST(NULL AS TEXT)      AS sf_best_move,
    CAST(NULL AS DOUBLE)    AS sf_time_ms,
    CAST(NULL AS TIMESTAMP) AS sf_computed_at,
    CAST(NULL AS TEXT)      AS sf_pv,
    CAST(NULL AS TEXT)      AS sf_eval_pov,
    CAST(NULL AS INTEGER)   AS sf_depth,
    CAST(NULL AS INTEGER)   AS sf_multipv,
    CAST(NULL AS INTEGER)   AS sf_move_rank,
    CAST(NULL AS INTEGER)   AS sf_move_eval
"""


# ─────────────────────────────────────────────────────────────────────────────
# UDFs
# ─────────────────────────────────────────────────────────────────────────────
def _opening_part(index):
    """UDF returning element `index` of get_opening_details: 0 eco, 1 name, 2 variation."""
    def udf(moves_list: str):
        try:
            return get_opening_details(moves_list)[index] or ""
        except Exception:
            return ""
    return udf


opening_code_udf = _opening_part(0)
opening_name_udf = _opening_part(1)
opening_variation_udf = _opening_part(2)


@lru_cache(maxsize=200_000)
def replays_from_start(moves_list):
    """True when every move (UCI or SAN) is legal from the standard start position.

    Games from EPD books store moves relative to the book position; a prefix of
    them can still be legal from the start and would name a wrong opening.
    """
    text = str(moves_list).strip()
    try:
        tokens = json.loads(text) if text.startswith("[") else text.split()
    except ValueError:
        return False
    board = chess.Board()
    for token in tokens:
        try:
            board.push_uci(str(token))
        except ValueError:
            try:
                board.push_san(str(token))
            except ValueError:
                return False
    return True


def path_basename(path):
    """File name of a Windows or POSIX path: 'C:\\books\\x.epd' and '/home/u/x.epd' -> 'x.epd'."""
    if path is None:
        return None
    text = str(path)
    # ntpath splits on both separators and understands drive letters.
    return ntpath.basename(text.rstrip("/\\")) or text


def _register_udfs(cnxn):
    for name, function in (("get_opening_code", opening_code_udf),
                           ("get_opening_name", opening_name_udf),
                           ("get_opening_variation", opening_variation_udf),
                           ("path_basename", path_basename)):
        cnxn.create_function(name, function, parameters=["VARCHAR"], return_type="VARCHAR")
    cnxn.create_function("replays_from_start", lambda moves: replays_from_start(moves),
                         parameters=["VARCHAR"], return_type="BOOLEAN")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _table_exists(cnxn, table: str) -> bool:
    result = cnxn.execute(
        """SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_catalog = current_database() -- prevent raw.db creating false positives
            AND table_schema = 'main' AND table_name = ?
    """, [table]).fetchone()
    return result[0] > 0


def _max_id(cnxn, table: str, id_col: str = 'id') -> int:
    """Return max id in an analytics table, or 0 if table doesn't exist."""
    if not _table_exists(cnxn, table):
        return 0
    result = cnxn.execute(f'SELECT COALESCE(MAX("{id_col}"), 0) FROM "{table}"').fetchone()
    return result[0]


def _count(cnxn, table: str) -> int:
    return cnxn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]


def _columns(cnxn, table: str) -> set:
    rows = cnxn.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = ?
    """, [table]).fetchall()
    return {r[0] for r in rows}


def _sync_columns(cnxn, table: str, select_sql: str) -> list:
    """Add the columns of `select_sql` that `table` lacks (e.g. new raw columns)."""
    existing = _columns(cnxn, table)
    added = []
    for name, dtype, *_ in cnxn.execute(f"DESCRIBE {select_sql}").fetchall():
        if name not in existing:
            cnxn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {dtype}')
            added.append(name)
    return added


def _ensure_columns(cnxn, table: str, columns) -> None:
    """Contract columns exist even when the raw database predates them (they stay NULL)."""
    for name, sql_type in columns:
        cnxn.execute(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS "{name}" {_DUCKDB_TYPES[sql_type]}')


def _decode_sql(column: str, names: dict, aliases: dict = None) -> str:
    """CASE mapping integer codes, already-decoded names and aliases to names (else NULL)."""
    text = f'CAST("{column}" AS VARCHAR)'
    whens = [f"WHEN TRY_CAST(\"{column}\" AS INTEGER) = {code} THEN '{name}'" for code, name in names.items()]
    labels = {name: name for name in names.values()}
    labels.update(aliases or {})
    whens += [f"WHEN {text} = '{label}' THEN '{name}'" for label, name in labels.items()]
    return "CASE " + " ".join(whens) + " END"


def _create(cnxn, table: str, select_sql: str) -> int:
    cnxn.execute(f'CREATE OR REPLACE TABLE "{table}" AS {select_sql}')
    return _count(cnxn, table)


def _append(cnxn, table: str, select_sql: str) -> int:
    """Insert rows by column name in one transaction, adding columns the table lacks."""
    before = _count(cnxn, table)
    cnxn.execute("BEGIN TRANSACTION")
    try:
        added = _sync_columns(cnxn, table, select_sql)
        cnxn.execute(f'INSERT INTO "{table}" BY NAME {select_sql}')
        cnxn.execute("COMMIT")
    except Exception:
        cnxn.execute("ROLLBACK")
        raise
    if added:
        print(f"    {table}: added new column(s) {', '.join(added)}")
    return _count(cnxn, table) - before


def _stage_lenient(cnxn, raw_table: str) -> list:
    """Copy raw.<table> to a temporary table of text columns; returns the declared (name, type)s.

    SQLite keeps REAL values in INTEGER columns, which DuckDB's SQLite scanner
    rejects, so the table is read as text and cast back per column.
    """
    declared = [(row[0], row[1]) for row in cnxn.execute(f'DESCRIBE raw."{raw_table}"').fetchall()]
    cnxn.execute("SET sqlite_all_varchar = true")
    try:
        cnxn.execute(f'CREATE OR REPLACE TEMP TABLE "_raw_{raw_table}" AS SELECT * FROM raw."{raw_table}"')
    finally:
        cnxn.execute("SET sqlite_all_varchar = false")
    return declared


# ─────────────────────────────────────────────────────────────────────────────
# Source queries (shared by full and incremental loads)
# ─────────────────────────────────────────────────────────────────────────────
ENGINE_RATINGS_SQL = """
    SELECT
        r.id, r.engine_id,
        e.name as engine_name,
        e.version as engine_version,
        cast(r.elo_ultra_fast AS INTEGER) AS elo_ultra_fast,
        cast(r.elo_bullet AS INTEGER) AS elo_bullet,
        cast(r.elo_blitz AS INTEGER) AS elo_blitz,
        cast(r.elo_rapid AS INTEGER) AS elo_rapid,
        cast(r.elo_classical AS INTEGER) AS elo_classical,
        r.games_ultra_fast AS games_ultra_fast,
        r.games_bullet AS games_bullet,
        r.games_blitz AS games_blitz,
        r.games_rapid AS games_rapid,
        r.games_classical AS games_classical,
        r.ingestion_timestamp_utc
    FROM raw.engine_ratings r
    LEFT JOIN raw.engines e
        ON e.id = r.engine_id
"""


def _sprt_select(cnxn) -> str:
    """sprt rows with the measured elo_diff kept and the opening book reduced to its file name."""
    columns = []
    for name, dtype in _stage_lenient(cnxn, "sprt"):
        if name == "opening_book":
            expr = 'path_basename("opening_book")'
        elif name in SPRT_DOUBLE_COLUMNS:
            expr = f'TRY_CAST("{name}" AS DOUBLE)'
        else:
            expr = f'TRY_CAST("{name}" AS {dtype})'
        columns.append(f'{expr} AS "{name}"')
    return f'SELECT {", ".join(columns)} FROM "_raw_sprt"'


def _sts_select() -> str:
    return """
        SELECT * REPLACE (path_basename(suite) AS suite),
               trim(split_part(position_name, '-', 2)) AS position_type
        FROM raw.sts
    """


def _games_select(min_id: int = 0) -> str:
    """games with decoded result/termination and the opening of the move list.

    When the moves can't be replayed from the standard start position (e.g. an
    EPD book) or name no opening, the recorded label ("Name, Variation") is
    used instead.
    """
    return f"""
        WITH src AS (
            SELECT * EXCLUDE (opening), opening AS _raw_opening,
                coalesce(replays_from_start(moves), true) AS _replays,
                get_opening_code(moves) AS _eco,
                get_opening_name(moves) AS _name,
                get_opening_variation(moves) AS _variation
            FROM raw.games WHERE id > {int(min_id)}
        ),
        classified AS (
            SELECT *, _replays AND coalesce(_name, '') <> '' AS _from_moves FROM src
        )
        SELECT * EXCLUDE (_raw_opening, _replays, _eco, _name, _variation, _from_moves) REPLACE (
                {_decode_sql('result', RESULT_NAMES, RESULT_ALIASES)} AS result,
                {_decode_sql('termination', TERMINATION_NAMES)} AS termination
            ),
            CASE WHEN _from_moves THEN _name
                 ELSE coalesce(split_part(_raw_opening, ', ', 1), '') END AS opening,
            CASE WHEN _from_moves THEN coalesce(_eco, '') ELSE '' END AS opening_eco,
            CASE WHEN _from_moves THEN coalesce(_variation, '')
                 WHEN strpos(_raw_opening, ', ') > 0 THEN substr(_raw_opening, strpos(_raw_opening, ', ') + 2)
                 ELSE '' END AS opening_variation
        FROM classified
    """


def _searches_select(min_id: int = 0, stockfish_columns: bool = True) -> str:
    extra = f", {STOCKFISH_COLUMNS_SQL}" if stockfish_columns else ""
    return f"SELECT *{extra} FROM raw.searches WHERE id > {int(min_id)}"


def _child_select(raw_table: str, key: str, min_id: int = 0) -> str:
    return f'SELECT * FROM raw."{raw_table}" WHERE "{key}" > {int(min_id)}'


def _dim_positions_select(min_id: int = 0) -> str:
    return f"SELECT id AS search_id, fen, game_id, sts_id FROM raw.searches WHERE id > {int(min_id)}"


# (analytics table, raw table, key column) for the per-search child facts.
CHILD_TABLES = (
    ("iterative_deepening_stats", "searches_by_iteration", "search_id"),
    ("search_tree_stats", "searches_by_tree_depth", "search_id"),
    ("search_timings", "timing", "search_id"),
    ("root_moves", "root_moves", "id"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Loader version
# ─────────────────────────────────────────────────────────────────────────────
def _stored_loader_version(cnxn) -> int:
    """Version of the loader that derived the analytics tables (1 before it was recorded)."""
    if not _table_exists(cnxn, "analytics_meta"):
        return 1 if _table_exists(cnxn, "game_stats") else LOADER_VERSION
    row = cnxn.execute("SELECT value FROM analytics_meta WHERE key = 'loader_version'").fetchone()
    return int(row[0]) if row else 1


def _store_loader_version(cnxn) -> None:
    cnxn.execute("CREATE TABLE IF NOT EXISTS analytics_meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
    cnxn.execute("INSERT OR REPLACE INTO analytics_meta VALUES ('loader_version', ?)", [str(LOADER_VERSION)])


# ─────────────────────────────────────────────────────────────────────────────
# Full refresh
# ─────────────────────────────────────────────────────────────────────────────
def _load_small_tables(cnxn, mode: str) -> dict:
    """engines, experiments, engine_ratings, sprt_runs and sts_runs are small: always fully refreshed."""
    totals = {}
    print(f"  [{mode}] Loading engines...")
    totals["engines"] = _create(cnxn, "engines", "SELECT * FROM raw.engines")
    print(f"  [{mode}] Loading experiments...")
    totals["experiments"] = _create(cnxn, "experiments", "SELECT * FROM raw.experiments")
    print(f"  [{mode}] Loading engine_ratings...")
    totals["engine_ratings"] = _create(cnxn, "engine_ratings", ENGINE_RATINGS_SQL)
    print(f"  [{mode}] Loading sprt_runs...")
    totals["sprt_runs"] = _create(cnxn, "sprt_runs", _sprt_select(cnxn))
    _ensure_columns(cnxn, "sprt_runs", SPRT_PENTANOMIAL_COLUMNS)
    print(f"  [{mode}] Loading sts_runs...")
    totals["sts_runs"] = _create(cnxn, "sts_runs", _sts_select())
    return totals


def load_full(cnxn):
    """Drop and recreate all tables from raw SQLite; returns ({table: rows}, rebuilt tables)."""
    totals = _load_small_tables(cnxn, "full")

    print("  [full] Loading game_stats...")
    totals["game_stats"] = _create(cnxn, "game_stats", _games_select())
    _ensure_columns(cnxn, "game_stats", GAMES_MATCH_COLUMNS)

    print("  [full] Loading search_stats...")
    totals["search_stats"] = _create(cnxn, "search_stats", _searches_select())

    for table, raw_table, key in CHILD_TABLES:
        print(f"  [full] Loading {table}...")
        totals[table] = _create(cnxn, table, _child_select(raw_table, key))

    print("  [full] Loading dim_positions...")
    totals["dim_positions"] = _create(cnxn, "dim_positions", _dim_positions_select())
    _store_loader_version(cnxn)
    return totals, set(totals)


# ─────────────────────────────────────────────────────────────────────────────
# Incremental load
# ─────────────────────────────────────────────────────────────────────────────
def load_incremental(cnxn):
    """Only insert new rows (by id) into existing analytics tables; returns ({table: rows}, rebuilt tables).

    If a table doesn't exist yet, falls back to full creation for that table.
    Each table's inserts run in one transaction: either all succeed, or none do.
    """
    outdated = _stored_loader_version(cnxn) < LOADER_VERSION
    totals = _load_small_tables(cnxn, "incr")
    rebuilt = set(totals)

    print("  [incr] Syncing game_stats...")
    if not _table_exists(cnxn, "game_stats") or outdated:
        if outdated:
            print("    game_stats was derived by an older loader: rebuilding it")
        totals["game_stats"] = _create(cnxn, "game_stats", _games_select())
        rebuilt.add("game_stats")
    else:
        totals["game_stats"] = _append(cnxn, "game_stats", _games_select(_max_id(cnxn, "game_stats")))
    _ensure_columns(cnxn, "game_stats", GAMES_MATCH_COLUMNS)

    print("  [incr] Syncing search_stats...")
    if not _table_exists(cnxn, "search_stats"):
        totals["search_stats"] = _create(cnxn, "search_stats", _searches_select())
        rebuilt.add("search_stats")
    else:
        # Stockfish columns of existing rows are kept; new rows get NULLs by name.
        select_sql = _searches_select(_max_id(cnxn, "search_stats"), stockfish_columns=False)
        totals["search_stats"] = _append(cnxn, "search_stats", select_sql)

    for table, raw_table, key in CHILD_TABLES:
        print(f"  [incr] Syncing {table}...")
        if not _table_exists(cnxn, table):
            totals[table] = _create(cnxn, table, _child_select(raw_table, key))
            rebuilt.add(table)
        else:
            totals[table] = _append(cnxn, table, _child_select(raw_table, key, _max_id(cnxn, table, key)))

    print("  [incr] Syncing dim_positions...")
    if not _table_exists(cnxn, "dim_positions"):
        totals["dim_positions"] = _create(cnxn, "dim_positions", _dim_positions_select())
        rebuilt.add("dim_positions")
    else:
        max_id = _max_id(cnxn, "dim_positions", id_col="search_id")
        totals["dim_positions"] = _append(cnxn, "dim_positions", _dim_positions_select(max_id))
    _store_loader_version(cnxn)
    return totals, rebuilt


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def _sql_string(value) -> str:
    return str(value).replace("'", "''")


def load(raw_db=None, analytics_db=None, full=False) -> dict:
    """Load `raw_db` (SQLite) into `analytics_db` (DuckDB).

    Returns {table: rows}: all rows of fully (re)built tables, rows appended otherwise.
    """
    raw_db = Path(raw_db or RAW_DB)
    analytics_db = Path(analytics_db or ANALYTICS_DB)
    if not raw_db.is_file():
        raise FileNotFoundError(f"Raw database does not exist: {raw_db} "
                                "(create it with python -m data.databases.init_raw_db)")
    analytics_db.parent.mkdir(parents=True, exist_ok=True)
    cnxn = duckdb.connect(str(analytics_db))
    try:
        # ATTACH does not accept parameter placeholders in DuckDB.
        cnxn.execute(f"ATTACH '{_sql_string(raw_db)}' AS raw (TYPE SQLITE, READ_ONLY)")
        _register_udfs(cnxn)
        if full:
            print("Running FULL refresh...")
            totals, rebuilt = load_full(cnxn)
        else:
            print("Running INCREMENTAL load...")
            totals, rebuilt = load_incremental(cnxn)
    except Exception:
        # On failure, attempt rollback if in a transaction
        try:
            cnxn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        cnxn.close()

    print("\n  Load summary:")
    for table, count in totals.items():
        if table in rebuilt:
            status = f"{count:,} rows (reloaded)"
        else:
            status = f"+{count:,} rows" if count > 0 else "up to date"
        print(f"    {table:.<35s} {status}")
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load raw data into analytics DuckDB")
    parser.add_argument('--full', action='store_true', help='Full refresh (drop + reload all tables)')
    parser.add_argument('--raw-db', type=Path, default=RAW_DB, help=f'SQLite raw database (default: {RAW_DB})')
    parser.add_argument('--analytics-db', type=Path, default=ANALYTICS_DB,
                        help=f'DuckDB analytics database (default: {ANALYTICS_DB})')
    args = parser.parse_args(argv)

    print(f"RAW_DB:       {args.raw_db}")
    print(f"ANALYTICS_DB: {args.analytics_db}\n")
    load(args.raw_db, args.analytics_db, full=args.full)
    print("Load completed successfully.")


if __name__ == '__main__':
    main()
