#!/usr/bin/env python3
"""Schema changes to the raw SQLite and analytics DuckDB databases.

The base schema is the CREATE TABLEs in init_raw_db.py; every change after it is
a migration below. init_raw_db applies all of them (and re-applies whatever is
missing of applied ones), and every program that writes to the raw database
applies pending ones first (data.etl.ensure_schema), so a new database and an old
one upgraded in place get the same tables and columns.

How it stays safe:
  - Migrations only add: tables, indexes and nullable columns (plus a rename that
    keeps its data). Existing rows keep their data; a column added later is NULL
    for them ("not measured", which is not the same as 0).
  - Each migration is recorded in a `schema_migrations` table inside the target
    database and applied once, so running this again is always safe.
  - Each migration runs in its own transaction (SQLite and DuckDB both have
    transactional DDL): it either fully applies or fully rolls back. Whether it
    is still pending is checked again under the write lock, so programs starting
    at the same time apply it once.
  - --dry-run opens existing databases read-only and only reports what would
    change, plus the tables and columns the database has beyond the current
    schema. It never creates database files or the schema_migrations table.

Usage (from the repository root):
  python -m data.databases.migrate_schema
  python -m data.databases.migrate_schema --dry-run
  python -m data.databases.migrate_schema --raw-db ./chess.db --analytics-db ./chess_analytics.duckdb
  python -m data.databases.migrate_schema --only 2026_10_15_games_plies

Without --raw-db/--analytics-db the CHESS_RAW_DB / CHESS_ANALYTICS_DB paths from
data/etl/paths.py are used. A database file that does not exist is skipped
(init_raw_db.py creates the raw database).

To store a new engine statistic, add a migration with its column (nullable,
DEFAULT NULL), named after the engine's log field and typed like its values
(INTEGER for counts, REAL for ratios); data/etl/ingest.py then stores it with no
code change:
  searches                 log field x            -> column x
  searches_by_iteration    log field itdepth_x    -> column x
  searches_by_tree_depth   log field treedepth_x  -> column x
  games                    log field x            -> column x
Until the column exists, the ETL keeps the field in the row's extra_fields (JSON)
and reports it; the migration then fills the new column from there, so runs
ingested in between lose nothing. The analytics database needs no migration for
raw columns: load_analytics.py adds them to its tables.

Never edit or reuse an applied migration's id or columns: add a new migration.
Don't add columns to the CREATE TABLEs in init_raw_db.py either; a migration
reaches old and new databases alike. Analytics migrations must use DuckDB types
(DOUBLE, VARCHAR, ...).
"""

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..etl.columns import MISMATCH, coerce
from ..etl.paths import RAW_DB, ANALYTICS_DB, RAW_DB_TIMEOUT_S

# ---------------------------------------------------------------------------
# 1. COLUMN SETS (shared with the CREATE TABLE statements in init_raw_db.py)
# ---------------------------------------------------------------------------
# (column name, SQLite type). All of them are nullable.

GAMES_MATCH_COLUMNS = [
    ("source", "TEXT"),              # 'engine' (engine log) | 'match' (match-runner record, authoritative result)
    ("run_id", "TEXT"),              # match-runner run directory name, e.g. 20260928_101129_354256
    ("run_game", "INTEGER"),         # game number within the run (pairs = (1,2), (3,4), ...)
    ("pair_index", "INTEGER"),       # 1-based opening pair within the run
    ("env_host", "TEXT"),
    ("env_cpu_model", "TEXT"),
    ("env_os", "TEXT"),
    ("env_cpu", "INTEGER"),          # logical CPU the game ran on
    ("env_core_type", "TEXT"),       # 'P' | 'E' | '-' (not pinned)
    ("env_cluster", "INTEGER"),      # L2 cluster id of the core
    ("env_concurrency", "INTEGER"),  # games running at the same time on the machine
    ("env_mhz_mean", "REAL"),        # measured clock of that CPU during the game
    ("env_mhz_min", "REAL"),
    ("env_mhz_max", "REAL"),
    ("env_util_mean", "REAL"),       # CPU utilisation of that CPU during the game (%)
    ("env_power_source", "TEXT"),    # 'ac' | 'battery'
    ("env_core_mixed", "INTEGER"),   # 1 if an engine was found off its core (excluded from SPRT)
]

SPRT_PENTANOMIAL_COLUMNS = [
    ("pairs", "INTEGER"),
    ("penta_ll", "INTEGER"),
    ("penta_ld", "INTEGER"),
    ("penta_dd", "INTEGER"),
    ("penta_wl", "INTEGER"),
    ("penta_wd", "INTEGER"),
    ("penta_ww", "INTEGER"),
    ("score", "REAL"),
    ("elo_low", "REAL"),
    ("elo_high", "REAL"),
    ("nelo", "REAL"),
    ("nelo_low", "REAL"),
    ("nelo_high", "REAL"),
    ("draw_ratio", "REAL"),
    ("variance_ratio", "REAL"),
    ("concurrency", "INTEGER"),
    ("core_summary", "TEXT"),        # JSON: {"P": {pairs, elo, elo_low, elo_high, pentanomial}, "E": {...}}
    ("excluded_pairs", "INTEGER"),
    ("run_id", "TEXT"),
    ("book_seed", "INTEGER"),
    ("env_host", "TEXT"),
    ("env_cpu_model", "TEXT"),
]



# ---------------------------------------------------------------------------
# 2. MIGRATIONS
# ---------------------------------------------------------------------------
# db:         "raw" or "analytics"
# table:      table to add `columns` to (None: no fixed columns)
# columns:    list of (column_name, sql_type, default_sql_literal); the default is
#             inserted verbatim into the DEFAULT clause, e.g. "NULL", "0", "''"
# statements: SQL run after the columns, e.g. CREATE TABLE / INDEX IF NOT EXISTS
# dynamic:    (sqlite3 connection) -> SQL statements computed from the database, run last
#
# init_raw_db re-applies every migration's missing parts, applied or not, so statements
# must be safe to run again (IF NOT EXISTS, or an UPDATE of rows still to be filled).
# A column added to searches, searches_by_iteration, searches_by_tree_depth or games is
# filled from extra_fields, where the ETL keeps log fields that had no column yet.

@dataclass
class Migration:
    id: str                    # unique, stable, never reused
    description: str
    db: str                    # "raw" or "analytics"
    table: str = None
    columns: list = field(default_factory=list)     # (name, type, default)
    statements: list = field(default_factory=list)  # run after the columns
    dynamic: object = None     # (sqlite3 connection) -> [SQL statement, ...]


def _nullable(columns):
    return [(name, sql_type, "NULL") for name, sql_type in columns]


def _table_columns(conn, table):
    return {row[1]: row[2] for row in conn.execute(f'PRAGMA table_info("{table}")')}


# Log field prefix of each table's columns; these tables keep unmatched fields in extra_fields
EXTRA_FIELD_TABLES = {"searches": "", "searches_by_iteration": "itdepth_",
                      "searches_by_tree_depth": "treedepth_", "games": ""}


def _repair_base_tables(conn):
    """Base tables and columns (init_raw_db.create_base_tables) an older init_raw_db didn't
    create yet. Only for an initialized database (it has an engines table); a missing
    column is added nullable."""
    from .init_raw_db import create_base_tables  # init_raw_db imports this module
    if not _table_columns(conn, "engines"):
        return []
    reference = sqlite3.connect(":memory:")
    try:
        create_base_tables(reference)
        statements = []
        for table, sql in reference.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").fetchall():
            have = _table_columns(conn, table)
            if not have:
                statements.append(sql)
                continue
            statements += [f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type}'
                           for _, name, sql_type, _, _, pk in reference.execute(f'PRAGMA table_info("{table}")')
                           if name not in have and not pk]
        return statements
    finally:
        reference.close()


def _rename_nmp_fail(conn):
    """nmp_fail was renamed nmp_failhigh in July 2026 (the same counter: null-move fail-highs)."""
    statements = []
    for table in ("searches", "searches_by_iteration", "searches_by_tree_depth"):
        have = _table_columns(conn, table)
        if "nmp_fail" in have and "nmp_failhigh" not in have:
            statements.append(f'ALTER TABLE "{table}" RENAME COLUMN nmp_fail TO nmp_failhigh')
        elif "nmp_fail" in have:  # both: keep the old values where the new column is empty
            statements.append(f'UPDATE "{table}" SET nmp_failhigh = nmp_fail '
                              f'WHERE nmp_failhigh IS NULL AND nmp_fail IS NOT NULL')
    return statements


def _add_extra_fields(conn):
    """extra_fields (JSON) on the tables log records go to: fields without a column yet."""
    return [f'ALTER TABLE "{table}" ADD COLUMN extra_fields TEXT' for table in EXTRA_FIELD_TABLES
            if (have := _table_columns(conn, table)) and "extra_fields" not in have]


# Search statistics Tomahawk logs (include/search_stats.h) that had no column:
# transposition-table cutoffs, the three kinds of PVS re-search (after an LMR
# reduction, full-window interior, full-window root; they replace the single
# pvs_researches count of older builds), reverse futility and futility prunes.
SEARCH_STATS_2026_10 = [
    ("tt_returns", "INTEGER"),
    ("pvs_researches_depth", "INTEGER"),
    ("pvs_researches_full", "INTEGER"),
    ("pvs_researches_root", "INTEGER"),
    ("rfp", "INTEGER"),
    ("futility_prune", "INTEGER"),
]

MIGRATIONS = [
    Migration(
        id="2026_07_04_completed_search_depth_rawDB",
        description="including the completed depth (not just the depth when search terminated) in raw",
        db="raw",
        table="searches",
        columns=[("completed_depth", "INTEGER", "NULL")],
    ),
    Migration(
        id="2026_09_28_games_match_environment",
        description="match-runner provenance and the environment each game ran in (host, cpu, core type, clock)",
        db="raw",
        table="games",
        columns=_nullable(GAMES_MATCH_COLUMNS),
    ),
    Migration(
        id="2026_09_28_sprt_pentanomial",
        description="pentanomial SPRT statistics, per-core-type summary and machine of each SPRT run",
        db="raw",
        table="sprt",
        columns=_nullable(SPRT_PENTANOMIAL_COLUMNS),
    ),
    Migration(
        id="2026_10_15_rename_nmp_fail",
        description="nmp_fail -> nmp_failhigh (renamed in July 2026), keeping the data",
        db="raw",
        dynamic=_rename_nmp_fail,
    ),
    Migration(
        id="2026_10_15_base_tables",
        description="base tables and columns that databases created by an older init_raw_db lack",
        db="raw",
        dynamic=_repair_base_tables,
    ),
    Migration(
        id="2026_10_15_extra_fields",
        description="extra_fields (JSON) keeps log fields that have no column yet",
        db="raw",
        dynamic=_add_extra_fields,
    ),
    Migration(
        id="2026_10_15_engine_build_provenance",
        description="engine builds: binary and net hashes (engine_key), paths, git state, build type",
        db="raw",
        table="engines",
        columns=_nullable([
            ("engine_key", "TEXT"),        # sha256(binary)[:24] + "+" + sha256(net)[:24] (data/etl/provenance.py)
            ("binary_sha256", "TEXT"),
            ("binary_path", "TEXT"),
            ("net_sha256", "TEXT"),
            ("net_path", "TEXT"),
            ("git_commit", "TEXT"),
            ("git_branch", "TEXT"),
            ("git_dirty", "INTEGER"),
            ("build_type", "TEXT"),
            ("provenance", "TEXT"),        # JSON: everything recorded about the build
        ]),
        statements=["CREATE UNIQUE INDEX IF NOT EXISTS idx_engines_engine_key ON engines (engine_key) "
                    "WHERE engine_key IS NOT NULL"],
    ),
    Migration(
        id="2026_10_15_sprt_nodes_threads",
        description="SPRT node limit per move and search threads",
        db="raw",
        table="sprt",
        columns=_nullable([("nodes_per_move", "INTEGER"), ("threads", "INTEGER")]),
    ),
    Migration(
        id="2026_10_15_sts_search_nodes",
        description="STS node limit per position",
        db="raw",
        table="sts",
        columns=_nullable([("search_nodes", "INTEGER")]),
    ),
    Migration(
        id="2026_10_15_rating_tables",
        description="pairwise match results and global rating fits (data/ratings.py)",
        db="raw",
        statements=[
            """
            CREATE TABLE IF NOT EXISTS match_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id       INTEGER,
                source              TEXT NOT NULL,      -- tournament | sprt | manual
                engine_a_id         INTEGER NOT NULL,
                engine_b_id         INTEGER NOT NULL,
                time_control        TEXT,               -- e.g. 10+0.1, depth=8, nodes=50000
                tc_category         TEXT NOT NULL,      -- bullet, blitz, ..., depth=8, nodes=50000, blitz@T4
                wins                INTEGER NOT NULL,   -- engine a's results
                draws               INTEGER NOT NULL,
                losses              INTEGER NOT NULL,
                penta_ll            INTEGER, penta_ld INTEGER, penta_dd INTEGER, penta_wd INTEGER, penta_ww INTEGER,
                created_utc         DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id),
                FOREIGN KEY (engine_a_id) REFERENCES engines(id),
                FOREIGN KEY (engine_b_id) REFERENCES engines(id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_match_results_category ON match_results (tc_category)",
            """
            CREATE TABLE IF NOT EXISTS rating_fits (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                tc_category         TEXT NOT NULL,
                sources             TEXT NOT NULL,      -- comma-separated match_results sources used
                anchor_engine_id    INTEGER,
                anchor_elo          REAL,
                prior_draws         REAL,
                games               INTEGER,
                engines             INTEGER,
                fitted_utc          DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ratings (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                fit_id              INTEGER NOT NULL,
                engine_id           INTEGER NOT NULL,
                elo                 REAL NOT NULL,
                se                  REAL NOT NULL,      -- vs the component anchor
                games               INTEGER NOT NULL,
                points              REAL NOT NULL,
                component           INTEGER NOT NULL,   -- 0 = the anchor's; others are not comparable to it
                prev_engine_id      INTEGER,            -- previous engine (by id) in the same component
                delta_prev          REAL,               -- elo - elo of prev_engine_id
                delta_prev_se       REAL,               -- its standard error (uses the fit's covariance)
                FOREIGN KEY (fit_id) REFERENCES rating_fits(id),
                FOREIGN KEY (engine_id) REFERENCES engines(id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_ratings_fit ON ratings (fit_id)",
        ],
    ),
    Migration(
        id="2026_10_15_net_registry",
        description="NNUE nets and net-vs-net matches (nnue/registry.py)",
        db="raw",
        statements=[
            """
            CREATE TABLE IF NOT EXISTS nets (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                net_sha256          TEXT NOT NULL UNIQUE,
                run_id              TEXT,               -- train.py --net-id
                superbatch          INTEGER,
                file_path           TEXT,
                val_loss            REAL,               -- at this superbatch
                train_loss          REAL,
                best_val_loss       REAL,               -- over the whole run
                best_val_superbatch INTEGER,
                train_config        TEXT,               -- JSON written by train.py
                notes               TEXT,
                registered_utc      DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS net_matches (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                net_a_id            INTEGER NOT NULL,
                net_b_id            INTEGER NOT NULL,
                engine_path         TEXT,
                time_control        TEXT,
                wins                INTEGER, draws INTEGER, losses INTEGER,
                penta_ll            INTEGER, penta_ld INTEGER, penta_dd INTEGER, penta_wd INTEGER, penta_ww INTEGER,
                elo                 REAL,
                elo_low             REAL,
                elo_high            REAL,
                created_utc         DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (net_a_id) REFERENCES nets(id),
                FOREIGN KEY (net_b_id) REFERENCES nets(id)
            )
            """,
        ],
    ),
    Migration(
        id="2026_10_15_search_stats",
        description="per-search tt_returns, PVS re-search kinds, reverse futility and futility prunes",
        db="raw",
        table="searches",
        columns=_nullable(SEARCH_STATS_2026_10),
    ),
    Migration(
        id="2026_10_15_iteration_stats",
        description="the same statistics per iterative-deepening iteration (log fields itdepth_*)",
        db="raw",
        table="searches_by_iteration",
        columns=_nullable(SEARCH_STATS_2026_10),
    ),
    Migration(
        id="2026_10_15_tree_depth_stats",
        description="the same statistics per tree depth (log fields treedepth_*)",
        db="raw",
        table="searches_by_tree_depth",
        columns=_nullable(SEARCH_STATS_2026_10),
    ),
    Migration(
        id="2026_10_15_games_plies",
        description="number of plies the engine logged for each game",
        db="raw",
        table="games",
        columns=_nullable([("plies", "INTEGER")]),
    ),
    Migration(
        id="2026_10_15_telemetry_loads",
        description="which side of a run's engine logs is loaded (tests/sprt.py): a retry never loads one twice",
        db="raw",
        statements=["""
            CREATE TABLE IF NOT EXISTS telemetry_loads (
                experiment_id       INTEGER NOT NULL,
                side                TEXT NOT NULL,      -- candidate | baseline
                loaded_utc          DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (experiment_id, side),
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """],
    ),
    Migration(
        id="2026_10_25_match_results_penta_wl",
        description="match_results.penta_wl: pairs one win and one loss. rows with it have "
                    "penta_dd = two draws; rows without it (NULL : the runner didnt split the middle bin) "
                    "keep both in penta_dd. middle bin = penta_dd + coalesce(penta_wl, 0) (data/ratings.py)",
                    db="raw",
                    table="match_results",
                    columns=_nullable([("penta_wl","INTERGER")]),
    )
]


# ---------------------------------------------------------------------------
# 3. ENGINE ADAPTERS
# ---------------------------------------------------------------------------

class Adapter:
    """Common interface so the runner doesn't care if it's talking to sqlite3 or duckdb."""

    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=None):
        return self.conn.execute(sql) if params is None else self.conn.execute(sql, params)

    def ensure_migrations_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id TEXT PRIMARY KEY,
                description TEXT,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def is_applied(self, migration_id):
        if not self.table_exists("schema_migrations"):
            return False
        row = self.conn.execute(
            "SELECT 1 FROM schema_migrations WHERE id = ?", [migration_id]
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
            [migration_id, description],
        )

    def begin(self):
        self.conn.execute("BEGIN TRANSACTION")

    def commit(self):
        self.conn.execute("COMMIT")

    def rollback(self):
        try:
            self.conn.execute("ROLLBACK")
        except Exception:
            pass  # the transaction never started (e.g. BEGIN itself failed)

    def close(self):
        self.conn.close()


class SQLiteAdapter(Adapter):
    """BEGIN/COMMIT are explicit. On a connection in Python's default (legacy)
    transaction mode that is safe while no transaction is open: the runner never
    leaves one open, and migrate_connection commits a caller's first."""

    def begin(self):
        self.conn.execute("BEGIN IMMEDIATE")  # take the write lock now: the pending check below holds

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
        row = self.conn.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = ?
        """, [table]).fetchone()
        return row is not None

    def existing_columns(self, table):
        rows = self.conn.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = ?
        """, [table]).fetchall()
        return {r[0] for r in rows}


def _open_raw(path, dry_run):
    path = Path(path)
    if not path.is_file():
        return None
    if dry_run:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None,
                               timeout=RAW_DB_TIMEOUT_S)
    else:
        conn = sqlite3.connect(str(path), isolation_level=None, timeout=RAW_DB_TIMEOUT_S)
    return SQLiteAdapter(conn)


def _open_analytics(path, dry_run):
    path = Path(path)
    try:
        import duckdb  # only the analytics side needs it
    except ImportError:
        print("duckdb package not installed (pip install duckdb) -- skipping analytics migrations",
              file=sys.stderr)
        return None
    if not path.is_file():
        return None
    return DuckDBAdapter(duckdb.connect(str(path), read_only=dry_run))


# ---------------------------------------------------------------------------
# 4. RUNNER
# ---------------------------------------------------------------------------

def _short(statement):
    return " ".join(statement.split())[:100]


def _schema_objects(adapter):
    """(type, name) of the tables and indexes of a SQLite database (empty for DuckDB)."""
    if not isinstance(adapter, SQLiteAdapter):
        return set()
    return {(kind, name) for kind, name in adapter.conn.execute(
        "SELECT type, name FROM sqlite_master WHERE type IN ('table', 'index') AND name NOT LIKE 'sqlite%'")}


def _fill_from_extra_fields(conn, table, columns):
    """Move the values the ETL kept in extra_fields (their column didn't exist yet) into the
    columns just added. A value that doesn't fit its column's type stays there.
    Returns the number of rows filled."""
    if table not in EXTRA_FIELD_TABLES or "extra_fields" not in _table_columns(conn, table):
        return 0
    fields = {EXTRA_FIELD_TABLES[table] + name: (name, sql_type) for name, sql_type, _ in columns}
    filled = 0
    for rowid, text in conn.execute(
            f'SELECT rowid, extra_fields FROM "{table}" WHERE extra_fields IS NOT NULL').fetchall():
        try:
            extra = json.loads(text)
        except ValueError:
            continue
        values = {}
        for field, (name, sql_type) in fields.items():
            if field in extra and (value := coerce(extra[field], sql_type)) is not MISMATCH:
                values[name] = value
                del extra[field]
        if values:
            assignments = ", ".join(f'"{name}" = ?' for name in values)
            conn.execute(f'UPDATE "{table}" SET {assignments}, extra_fields = ? WHERE rowid = ?',
                         (*values.values(), json.dumps(extra) if extra else None, rowid))
            filled += 1
    return filled


def _preview(adapter, m, say, applied):
    if m.table and not adapter.table_exists(m.table):
        say(f"  [WARN]  table '{m.table}' does not exist yet in this db -- '{m.id}' would be skipped")
        return "missing-table"
    existing = adapter.existing_columns(m.table) if m.table else set()
    to_add = [(n, t, d) for (n, t, d) in m.columns if n not in existing]
    statements = ([] if applied else list(m.statements)) + (m.dynamic(adapter.conn) if m.dynamic else [])
    if to_add or statements:
        say(f"  [would apply] {m.id}: {m.description}")
        for name, sql_type, default in to_add:
            say(f"          + {m.table}.{name} {sql_type} DEFAULT {default}")
        for statement in statements:
            say(f"          {_short(statement)}")
        return "pending"
    say(f"  [ok]    {m.id}" if applied else f"  [would record] {m.id}: nothing to add")
    return "skipped" if applied else "pending"


def run_migration(adapter: Adapter, m: Migration, dry_run: bool, verbose: bool = True,
                  recheck: bool = False, report: bool = True) -> str:
    """Apply one migration. Returns applied | recorded | skipped | pending | missing-table.

    recheck: also re-apply what's missing of an applied migration (init_raw_db), e.g. when
    tables were dropped and re-created while the history stayed.
    verbose: also print migrations that change nothing; report: print the changes made.
    """
    say = print if verbose else (lambda *a, **k: None)
    tell = print if verbose or report else say
    if adapter.is_applied(m.id) and not recheck:
        say(f"  [skip]  {m.id} already applied")
        return "skipped"
    if dry_run:
        return _preview(adapter, m, say, adapter.is_applied(m.id))

    adapter.ensure_migrations_table()
    try:
        adapter.begin()
        applied = adapter.is_applied(m.id)  # again under the write lock: another program may have applied it
        if applied and not recheck:
            adapter.rollback()
            say(f"  [skip]  {m.id} already applied")
            return "skipped"
        if m.table and not adapter.table_exists(m.table):
            adapter.rollback()
            print(f"  [WARN]  table '{m.table}' does not exist yet in this db -- "
                  f"skipping '{m.id}' (create the table first, then re-run)")
            return "missing-table"
        before = _schema_objects(adapter)
        existing = adapter.existing_columns(m.table) if m.table else set()
        to_add = [(n, t, d) for (n, t, d) in m.columns if n not in existing]
        for name, sql_type, default in to_add:
            adapter.add_column(m.table, name, sql_type, default)
        sqlite = isinstance(adapter, SQLiteAdapter)
        filled = _fill_from_extra_fields(adapter.conn, m.table, to_add) if to_add and sqlite else 0
        rows_before = adapter.conn.total_changes if sqlite else 0
        for statement in m.statements:  # safe to run again (see the top of this file)
            adapter.execute(statement)
        updated = adapter.conn.total_changes - rows_before if sqlite else 0
        dynamic = m.dynamic(adapter.conn) if m.dynamic else []
        for statement in dynamic:
            adapter.execute(statement)
        created = sorted(name for _, name in _schema_objects(adapter) - before)
        if not applied:
            adapter.record_migration(m.id, m.description)
        adapter.commit()
    except Exception as e:
        adapter.rollback()
        print(f"  [ERROR] {m.id} failed, rolled back: {e}", file=sys.stderr)
        raise

    changes = ([f"+ {m.table}.{n}" for n, _, _ in to_add] + [f"created {name}" for name in created]
               + [_short(s) for s in dynamic if not s.lstrip().upper().startswith("CREATE TABLE")]
               + ([f"updated {updated} row(s)"] if updated else [])
               + ([f"filled {filled} row(s) from extra_fields"] if filled else [])
               + ([_short(s) for s in m.statements] if not sqlite and not applied else []))  # DuckDB: no diff
    if not changes:
        say(f"  [skip]  {m.id}: {'nothing missing' if applied else 'nothing to add, marked as applied'}")
        return "skipped" if applied else "recorded"
    tell(f"  [done]  {m.id}: {'; '.join(changes)}")
    return "applied"


def pending_ids(conn, migrations=None):
    """Ids of the raw migrations not yet recorded in the database of sqlite3 connection `conn`."""
    ids = [m.id for m in (MIGRATIONS if migrations is None else migrations) if m.db == "raw"]
    has_history = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'").fetchone()
    applied = {row[0] for row in conn.execute("SELECT id FROM schema_migrations")} if has_history else set()
    return [i for i in ids if i not in applied]


def migrate_connection(conn, migrations=None, verbose=False, recheck=False, report=True) -> dict:
    """Apply the pending raw migrations through an open sqlite3 connection.

    Used by data.etl.ensure_schema and init_raw_db (recheck=True: what's missing of applied
    migrations too). A transaction the caller left open is committed first (BEGIN cannot
    nest). Returns {migration id: status}.
    """
    if conn.in_transaction:
        conn.commit()
    adapter = SQLiteAdapter(conn)
    pending = [m for m in (MIGRATIONS if migrations is None else migrations) if m.db == "raw"]
    return {m.id: run_migration(adapter, m, dry_run=False, verbose=verbose, recheck=recheck, report=report)
            for m in pending}


def schema_differences(conn):
    """(tables, columns) a raw database has that the current schema doesn't: tables of other
    tools, and columns of older layouts (kept as they are, e.g. fail_high_first)."""
    from .init_raw_db import create_base_tables  # init_raw_db imports this module
    reference = sqlite3.connect(":memory:")
    try:
        create_base_tables(reference)
        migrate_connection(reference, report=False)
        names = "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        current = {table: _table_columns(reference, table) for (table,) in reference.execute(names).fetchall()}
    finally:
        reference.close()
    tables, columns = [], []
    for (table,) in conn.execute(names).fetchall():
        if table not in current:
            tables.append(table)
        else:
            columns += [f"{table}.{name}" for name in _table_columns(conn, table) if name not in current[table]]
    return tables, columns


def migrate(raw_db=None, analytics_db=None, dry_run=False, only=None, migrations=None) -> dict:
    """Run migrations against the given (or configured) databases.

    Returns {migration id: status}; status is one of applied, recorded,
    skipped, pending (dry run), missing-table or no-database.
    """
    migrations = list(MIGRATIONS if migrations is None else migrations)
    if only:
        migrations = [m for m in migrations if m.id == only]
        if not migrations:
            raise ValueError(f"No migration found with id '{only}'")

    print(f"{'DRY RUN -- ' if dry_run else ''}Running {len(migrations)} migration(s)\n")
    results = {}
    targets = (
        ("raw", Path(raw_db or RAW_DB), _open_raw),
        ("analytics", Path(analytics_db or ANALYTICS_DB), _open_analytics),
    )
    for m in migrations:
        if m.db not in ("raw", "analytics"):
            print(f"  [ERROR] unknown db '{m.db}' for migration '{m.id}'", file=sys.stderr)
            results[m.id] = "unknown-db"

    for db_name, path, opener in targets:
        pending = [m for m in migrations if m.db == db_name]
        if not pending:
            continue
        adapter = opener(path, dry_run)
        if adapter is None:
            print(f"[{db_name}] {path} does not exist -- nothing to migrate"
                  f"{' (dry run: nothing created)' if dry_run else ''}\n")
            results.update({m.id: "no-database" for m in pending})
            continue
        try:
            for m in pending:
                print(f"[{db_name}] {m.table or 'several tables'}")
                results[m.id] = run_migration(adapter, m, dry_run)
                print()
            if db_name == "raw":
                tables, columns = schema_differences(adapter.conn)
                if tables or columns:
                    print("[raw] not part of the current schema, kept as they are:")
                    if tables:
                        print(f"  tables:  {', '.join(tables)}")
                    if columns:
                        print(f"  columns: {', '.join(columns)}")
                    print("  (a column of an older layout may hold data under a name that has since changed)\n")
        finally:
            adapter.close()

    print("Dry run complete -- no changes made." if dry_run else "Done.")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-db", type=Path, help=f"SQLite raw db file (default: {RAW_DB})")
    parser.add_argument("--analytics-db", type=Path, help=f"DuckDB analytics db file (default: {ANALYTICS_DB})")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--only", help="Only run the migration with this id")
    args = parser.parse_args(argv)
    try:
        migrate(args.raw_db, args.analytics_db, dry_run=args.dry_run, only=args.only)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
