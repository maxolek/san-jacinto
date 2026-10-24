"""Schema migrations and databases created by older versions."""
import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import duckdb

from data.databases.init_raw_db import init_raw_db
from data.databases.load_analytics import LOADER_VERSION, load
from data.databases.migrate_schema import (GAMES_MATCH_COLUMNS, MIGRATIONS, SPRT_PENTANOMIAL_COLUMNS, Migration,
                                           migrate)
from data.transforms.aggregates import build_aggregates
from data.transforms.transform_positions import build_position_features
from data.transforms.transform_search import build_analytics_views

# The tables as created before the environment / pentanomial columns existed.
OLD_GAMES = """
    CREATE TABLE games (
        id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id INTEGER NULL,
        white_engine_id INTEGER NULL, black_engine_id INTEGER NULL,
        wtime INTEGER, winc INTEGER, btime INTEGER, binc INTEGER, movestogo INTEGER, depth INTEGER,
        nodes INTEGER, movetime INTEGER, result TEXT, termination TEXT, opening TEXT, start_fen TEXT,
        moves TEXT, run_time_s REAL, ingestion_timestamp_utc DATETIME DEFAULT CURRENT_TIMESTAMP)
"""
OLD_SPRT = """
    CREATE TABLE sprt (
        id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id INTEGER NOT NULL,
        baseline_engine_id INTEGER NOT NULL, candidate_engine_id INTEGER NOT NULL,
        opening_book TEXT, book_depth INTEGER, time_control TEXT, time_per_move REAL, depth_per_move INTEGER,
        elo0 INTEGER, elo1 INTEGER, alpha REAL, beta REAL, result TEXT, elo_diff INTEGER, llr REAL, los REAL,
        candidate_wins INTEGER, candidate_losses INTEGER, candidate_draws INTEGER,
        candidate_white_wins INTEGER, candidate_white_losses INTEGER, candidate_white_draws INTEGER,
        candidate_black_wins INTEGER, candidate_black_losses INTEGER, candidate_black_draws INTEGER,
        games_played INTEGER, run_time_s REAL, ingestion_timestamp_utc DATETIME DEFAULT CURRENT_TIMESTAMP)
"""
OLD_SEARCHES = "CREATE TABLE searches (id INTEGER PRIMARY KEY AUTOINCREMENT, fen TEXT NOT NULL, depth INTEGER)"


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _sqlite_columns(path, table):
    with contextlib.closing(sqlite3.connect(path)) as cnxn:
        return [r[1] for r in cnxn.execute(f"PRAGMA table_info({table})")]


def _old_raw_db(path):
    """A raw database whose games/sprt/searches tables predate the newer columns."""
    with contextlib.closing(sqlite3.connect(path)) as cnxn:
        cnxn.executescript(OLD_GAMES + ";" + OLD_SPRT + ";" + OLD_SEARCHES)
        cnxn.execute("INSERT INTO games (result, termination, moves) VALUES (1, 7, '[\"e2e4\"]')")
        cnxn.execute("INSERT INTO sprt (experiment_id, baseline_engine_id, candidate_engine_id, elo_diff, elo1) "
                     "VALUES (1, 1, 2, 12.5, 2.5)")
        cnxn.execute("INSERT INTO searches (fen, depth) VALUES ('8/8/8/8/8/8/8/K6k w - - 0 1', 9)")
        cnxn.commit()


class MigrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_migrations_apply_to_an_old_database_and_are_idempotent(self):
        raw = self.dir / "old.db"
        _old_raw_db(raw)
        with _quiet():
            first = migrate(raw_db=raw, analytics_db=self.dir / "missing.duckdb")
        self.assertEqual(set(first.values()), {"applied"})
        games = _sqlite_columns(raw, "games")
        self.assertTrue({name for name, _ in GAMES_MATCH_COLUMNS} <= set(games))
        self.assertTrue({name for name, _ in SPRT_PENTANOMIAL_COLUMNS} <= set(_sqlite_columns(raw, "sprt")))
        self.assertIn("completed_depth", _sqlite_columns(raw, "searches"))
        with contextlib.closing(sqlite3.connect(raw)) as cnxn:
            self.assertEqual(cnxn.execute("SELECT result, termination, env_core_type FROM games").fetchall(),
                             [("1", "7", None)])
            self.assertEqual(cnxn.execute("SELECT elo_diff, pairs FROM sprt").fetchall(), [(12.5, None)])
            recorded = {r[0] for r in cnxn.execute("SELECT id FROM schema_migrations")}
        self.assertEqual(recorded, {m.id for m in MIGRATIONS})
        with _quiet():
            second = migrate(raw_db=raw, analytics_db=self.dir / "missing.duckdb")
        self.assertEqual(set(second.values()), {"skipped"})
        self.assertEqual(_sqlite_columns(raw, "games"), games)
        self.assertFalse((self.dir / "missing.duckdb").exists())

    def test_dry_run_changes_nothing(self):
        with _quiet():
            result = migrate(raw_db=self.dir / "none.db", analytics_db=self.dir / "none.duckdb", dry_run=True)
        self.assertEqual(set(result.values()), {"no-database"})
        self.assertEqual(list(self.dir.iterdir()), [])

        raw = self.dir / "old.db"
        _old_raw_db(raw)
        before = raw.read_bytes()
        with _quiet():
            result = migrate(raw_db=raw, dry_run=True)
        self.assertEqual(set(result.values()), {"pending"})
        self.assertEqual(raw.read_bytes(), before)
        with contextlib.closing(sqlite3.connect(raw)) as cnxn:
            tables = {r[0] for r in cnxn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        self.assertNotIn("schema_migrations", tables)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["old.db"])

    def test_a_failing_migration_rolls_back_in_sqlite(self):
        raw = self.dir / "old.db"
        _old_raw_db(raw)
        broken = Migration(id="broken", description="second column is invalid", db="raw", table="games",
                           columns=[("fine", "INTEGER", "NULL"), ("broken", "INTEGER", "(")])
        with _quiet(), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(sqlite3.Error):
            migrate(raw_db=raw, migrations=[broken])
        self.assertNotIn("fine", _sqlite_columns(raw, "games"))
        with contextlib.closing(sqlite3.connect(raw)) as cnxn:
            self.assertEqual(cnxn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0], 0)

    def test_duckdb_migrations_run_in_transactions(self):
        analytics = self.dir / "analytics.duckdb"
        with duckdb.connect(str(analytics)) as cnxn:
            cnxn.execute("CREATE TABLE game_stats (id INTEGER)")
        broken = Migration(id="broken", description="second column is invalid", db="analytics", table="game_stats",
                           columns=[("fine", "INTEGER", "NULL"), ("broken", "NOT_A_TYPE", "NULL")])
        good = Migration(id="good", description="adds a column", db="analytics", table="game_stats",
                         columns=[("fine", "DOUBLE", "NULL")])
        with _quiet(), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(duckdb.Error):
            migrate(raw_db=self.dir / "none.db", analytics_db=analytics, migrations=[broken])
        with _quiet():
            self.assertEqual(migrate(analytics_db=analytics, raw_db=self.dir / "none.db", migrations=[good]),
                             {"good": "applied"})
            self.assertEqual(migrate(analytics_db=analytics, raw_db=self.dir / "none.db", migrations=[good]),
                             {"good": "skipped"})
        with duckdb.connect(str(analytics), read_only=True) as cnxn:
            columns = [r[0] for r in cnxn.execute("DESCRIBE game_stats").fetchall()]
            recorded = [r[0] for r in cnxn.execute("SELECT id FROM schema_migrations").fetchall()]
        self.assertEqual(columns, ["id", "fine"])
        self.assertEqual(recorded, ["good"])

    def test_init_raw_db_creates_the_current_schema_at_the_configured_path(self):
        with _quiet():
            path = init_raw_db(self.dir / "custom.db")
            result = migrate(raw_db=path, migrations=[m for m in MIGRATIONS if m.db == "raw"])
        self.assertEqual(path, self.dir / "custom.db")
        self.assertEqual(set(result.values()), {"recorded"})  # already in the CREATE TABLEs
        self.assertTrue({name for name, _ in GAMES_MATCH_COLUMNS} <= set(_sqlite_columns(path, "games")))
        self.assertTrue({name for name, _ in SPRT_PENTANOMIAL_COLUMNS} <= set(_sqlite_columns(path, "sprt")))
        with _quiet():
            self.assertEqual(init_raw_db(self.dir), self.dir / "chess.db")

    def test_pipeline_runs_on_an_unmigrated_raw_database(self):
        """REAL Elo in INTEGER columns, missing environment columns and analytics tables derived by
        the previous loader are all handled by the default (incremental) load."""
        with _quiet():
            raw = init_raw_db(self.dir / "raw.db")
        with contextlib.closing(sqlite3.connect(raw)) as cnxn:
            cnxn.executescript("DROP TABLE games; DROP TABLE sprt;" + OLD_GAMES + ";" + OLD_SPRT)
            engine = cnxn.execute("INSERT INTO engines (name, version) VALUES ('Tomahawk', '1.0.0')").lastrowid
            game = cnxn.execute("INSERT INTO games (white_engine_id, black_engine_id, result, termination, moves) "
                                "VALUES (?, ?, 1, 7, ?)", (engine, engine, json.dumps(["b2b4", "g8h6"]))).lastrowid
            cnxn.execute("INSERT INTO sprt (experiment_id, baseline_engine_id, candidate_engine_id, elo0, elo1, "
                         "elo_diff, opening_book) VALUES (1, ?, ?, 0, 2.5, 12.5, 'C:\\books\\x.epd')",
                         (engine, engine))
            search = cnxn.execute("INSERT INTO searches (engine_id, game_id, fen, time_ms, eval, completed_depth, "
                                  "depth, nodes, qnodes) VALUES (?, ?, ?, 10.0, 25, 1, 2, 100, 50)",
                                  (engine, game, "rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1")).lastrowid
            for depth in (1, 2):
                cnxn.execute("INSERT INTO searches_by_iteration (search_id, depth, time_ms, eval, nodes, qnodes) "
                             "VALUES (?, ?, 5.0, 20, 40, 10)", (search, depth))
            cnxn.commit()
        analytics = self.dir / "analytics.duckdb"
        with _quiet():
            load(raw, analytics, full=True)
            # tables as the previous loader derived them: elo1 - elo0 as "elo_diff", full book paths,
            # integer Elo bounds, no opening_variation / termination 7, and no loader version
            with duckdb.connect(str(analytics)) as cnxn:
                cnxn.execute("DROP TABLE analytics_meta")
                cnxn.execute("ALTER TABLE sprt_runs ALTER COLUMN elo0 TYPE BIGINT")
                cnxn.execute("UPDATE sprt_runs SET elo_diff = elo1 - elo0, opening_book = 'C:\\books\\x.epd'")
                cnxn.execute("UPDATE game_stats SET termination = NULL, opening_variation = NULL")
            with contextlib.closing(sqlite3.connect(raw)) as cnxn:
                cnxn.execute("INSERT INTO sprt (experiment_id, baseline_engine_id, candidate_engine_id, elo0, elo1, "
                             "elo_diff) VALUES (1, 1, 1, 0.5, 2.5, -1.25)")
                cnxn.commit()
            load(raw, analytics)  # incremental: reloads the small tables and rebuilds game_stats once
            with duckdb.connect(str(analytics)) as cnxn:
                build_position_features(cnxn)
                build_analytics_views(cnxn)
                build_aggregates(cnxn)
                games = cnxn.execute("SELECT termination, opening, opening_variation, env_core_type, source "
                                     "FROM game_stats").fetchall()
                sprt = cnxn.execute("SELECT elo_diff, elo0, opening_book, pairs FROM sprt_runs ORDER BY id").fetchall()
                elo0_type = cnxn.execute("SELECT data_type FROM information_schema.columns "
                                         "WHERE table_name = 'sprt_runs' AND column_name = 'elo0'").fetchone()[0]
                version = cnxn.execute("SELECT value FROM analytics_meta WHERE key = 'loader_version'").fetchone()
                core_types = cnxn.execute("SELECT DISTINCT core_type FROM agg_engine_summary ORDER BY 1").fetchall()
                searches = cnxn.execute("SELECT count(*) FROM search_stats").fetchone()[0]
        self.assertEqual(games, [("adjudication", "Polish", "Tuebingen variation", None, None)])
        self.assertEqual(sprt, [(12.5, 0.0, "x.epd", None), (-1.25, 0.5, None, None)])
        self.assertEqual(elo0_type, "DOUBLE")
        self.assertEqual(version, (str(LOADER_VERSION),))
        self.assertEqual(core_types, [("all",), ("unknown",)])
        self.assertEqual(searches, 1)


if __name__ == "__main__":
    unittest.main()
