"""Schema evolution of the raw and analytics databases.

Old databases upgraded in place match new ones, init after a delete gives the full
schema, migrations are safe to rerun and roll back on failure, and a statistic a new
engine version logs reaches the raw and analytics databases with only a migration,
while earlier rows keep NULL.

    python -m unittest discover -s data/tests -t .
"""
import contextlib
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import duckdb

from data.databases import deletes, load_analytics, migrate_schema
from data.databases import init_raw_db as init_module
from data.databases.migrate_schema import MIGRATIONS, Migration
from data.etl import ingest
from data.etl.paths import connect_raw
from data.etl.schema import ensure_schema
from data.transforms.transform_positions import build_position_features
from data.transforms.transform_search import build_analytics_views

# init_raw_db's CREATE TABLEs as of October 2026. They are frozen: a column added there
# would only reach new databases, while a migration (migrate_schema.py) reaches all.
BASE_SCHEMA = {
    "engine_ratings": ('id', 'engine_id', 'elo_ultra_fast', 'elo_bullet', 'elo_blitz', 'elo_rapid', 'elo_classical', 'games_ultra_fast', 'games_bullet', 'games_blitz', 'games_rapid', 'games_classical', 'ingestion_timestamp_utc'),
    "engines": ('id', 'name', 'version', 'description', 'compile_flags', 'move_overhead_ms', 'max_threads', 'hash_size_mb', 'pondering', 'delta_prune_threshold', 'see_prune_threshold', 'aspiration_window', 'aspiration_start_depth', 'aspiration_depth_scale', 'aspiration_research_scale', 'draw_eval', 'contempt', 'r_nmp', 'r_lmr_const', 'r_lmr_denom', 'lmr_move_order_threshold', 'lmr_depth_threshold', 'ingestion_timestamp_utc'),
    "experiments": ('id', 'engine_id', 'comparison_engine_id', 'type', 'start_time_utc', 'end_time_utc', 'metadata'),
    "games": ('id', 'experiment_id', 'white_engine_id', 'black_engine_id', 'wtime', 'winc', 'btime', 'binc', 'movestogo', 'depth', 'nodes', 'movetime', 'result', 'termination', 'opening', 'start_fen', 'moves', 'run_time_s', 'ingestion_timestamp_utc'),
    "perft": ('id', 'experiment_id', 'fen', 'depth', 'nodes', 'expected_nodes', 'correct', 'time_ms', 'ingestion_timestamp_utc'),
    "root_moves": ('id', 'search_id', 'depth', 'move_index', 'move', 'eval', 'time_ms', 'nodes', 'ingestion_timestamp_utc'),
    "searches": ('id', 'engine_id', 'game_id', 'sts_id', 'fen', 'ply', 'time_ms', 'eval', 'completed_depth', 'depth', 'qdepth', 'move', 'principal_variation', 'nodes', 'qnodes', 'tt_stores', 'tt_hits', 'tt_fill', 'fail_highs', 'fail_lows', 'fail_high_researches', 'fail_low_researches', 'fh_index_0', 'fh_index_1', 'fh_index_2', 'fh_index_3', 'fh_index_4to7', 'fh_index_8plus', 'see_prunes', 'delta_prunes', 'pvs_researches', 'nmp', 'nmp_failhigh', 'tt_overwritten', 'ingestion_timestamp_utc'),
    "searches_by_iteration": ('search_id', 'depth', 'time_ms', 'eval', 'move', 'nodes', 'qnodes', 'qdepth', 'tt_stores', 'tt_hits', 'tt_fill', 'fail_highs', 'fail_lows', 'fail_high_researches', 'fail_low_researches', 'fh_index_0', 'fh_index_1', 'fh_index_2', 'fh_index_3', 'fh_index_4to7', 'fh_index_8plus', 'see_prunes', 'delta_prunes', 'pvs_researches', 'nmp', 'nmp_failhigh', 'ingestion_timestamp_utc'),
    "searches_by_tree_depth": ('search_id', 'depth', 'nodes', 'qnodes', 'tt_stores', 'tt_hits', 'fail_highs', 'fail_lows', 'fh_index_0', 'fh_index_1', 'fh_index_2', 'fh_index_3', 'fh_index_4to7', 'fh_index_8plus', 'see_prunes', 'delta_prunes', 'pvs_researches', 'nmp', 'nmp_failhigh', 'ingestion_timestamp_utc'),
    "sprt": ('id', 'experiment_id', 'baseline_engine_id', 'candidate_engine_id', 'opening_book', 'book_depth', 'time_control', 'time_per_move', 'depth_per_move', 'elo0', 'elo1', 'alpha', 'beta', 'result', 'elo_diff', 'llr', 'los', 'candidate_wins', 'candidate_losses', 'candidate_draws', 'candidate_white_wins', 'candidate_white_losses', 'candidate_white_draws', 'candidate_black_wins', 'candidate_black_losses', 'candidate_black_draws', 'games_played', 'run_time_s', 'ingestion_timestamp_utc'),
    "sts": ('id', 'experiment_id', 'suite', 'position_name', 'fen', 'search_time_ms', 'search_depth', 'engine_move', 'engine_score', 'expected_move', 'expected_score', 'alt_expected_move', 'avoid_move', 'move_is_correct', 'ingestion_timestamp_utc'),
    "timing": ('search_id', 'function', 'total_time_ms', 'num_calls', 'ingestion_timestamp_utc'),
}

# Every field the current engine logs (tomahawk include/search_stats.h, include/game_stats.h).
# Per-depth arrays skip depth 0; tree arrays of the PVS kinds have the iteration length.
CURRENT_SEARCH = {
    "engine_id": "1.0", "instance_id": 7, "type": "search", "session": 1, "game_uuid": "g1", "search_uuid": "s1",
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "ply": 0, "nodes": 5000, "qnodes": 900,
    "max_depth": 2, "max_qdepth": 6, "completed_depth": 2, "time_ms": 15, "root_eval": 25, "best_move": "e2e4",
    "principal_variation": ["e2e4", "e7e5"], "tt_hits": 40, "tt_returns": 31, "tt_stores": 60, "tt_fill": 0.02,
    "tt_overwritten": 3, "fail_highs": 70, "fail_lows": 20, "fail_high_index": [50, 10, 5, 3, 1, 1],
    "delta_prunes": 4, "see_prunes": 12, "aspiration_fail_low_researches": 1, "aspiration_fail_high_researches": 2,
    "pvs_researches_depth": 8, "pvs_researches_full": 5, "pvs_researches_root": 1, "nmp": 9, "nmp_failhigh": 6,
    "rfp": 14, "futility_prune": 17,
    "itdepth_time_ms": [3, 12], "itdepth_eval": [20, 25], "itdepth_move": ["d2d4", "e2e4"],
    "itdepth_nodes": [1000, 4000], "itdepth_qnodes": [100, 800], "itdepth_qdepth": [3, 6],
    "itdepth_tthits": [10, 30], "itdepth_tt_returns": [8, 23], "itdepth_ttstores": [15, 45],
    "itdepth_fail_highs": [20, 50], "itdepth_fail_lows": [5, 15], "itdepth_fh_index": [[15, 3, 1, 1, 0, 0], [35, 7, 4, 2, 1, 1]],
    "itdepth_aspiration_failhigh_researches": [0, 2], "itdepth_aspiration_faillow_researches": [0, 1],
    "itdepth_see_prunes": [2, 10], "itdepth_delta_prunes": [1, 3], "itdepth_pvs_researches_depth": [2, 6],
    "itdepth_pvs_researches_full": [1, 4], "itdepth_pvs_researches_root": [0, 1], "itdepth_nmp": [3, 6],
    "itdepth_nmp_failhigh": [2, 4], "itdepth_rfp": [4, 10], "itdepth_futility_prune": [5, 12],
    "treedepth_nodes": [10, 300, 4690], "treedepth_qnodes": [0, 50, 850], "treedepth_tt_hits": [1, 10, 29],
    "treedepth_tt_returns": [1, 8, 22], "treedepth_tt_stores": [2, 20, 38], "treedepth_fail_highs": [1, 19, 50],
    "treedepth_fail_lows": [1, 4, 15], "treedepth_fh_index": [[1, 0, 0, 0, 0, 0], [15, 2, 1, 1, 0, 0], [34, 8, 4, 2, 1, 1]],
    "treedepth_see_prunes": [0, 2, 10], "treedepth_delta_prunes": [0, 1, 3], "treedepth_pvs_researches_depth": [0, 8],
    "treedepth_pvs_researches_full": [1, 4], "treedepth_pvs_researches_root": [1, 0], "treedepth_nmp": [0, 3, 6],
    "treedepth_nmp_failhigh": [0, 2, 4], "treedepth_rfp": [0, 4, 10], "treedepth_futility_prune": [0, 5, 12],
}
CURRENT_GAME = {
    "engine_id": "1.0", "instance_id": 7, "type": "game", "session": 1, "game_uuid": "g1", "side": "white",
    "result": 1, "reason": 1, "plies": 3, "moves": ["f2f3", "e7e5", "g2g4"], "time_s": 1.5,
    "start_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "wtime": 10000, "btime": 10000,
    "winc": 100, "binc": 100, "movestogo": 0, "depth": 0, "nodes": 0, "movetime": 0,
}

# A statistic a later engine version starts logging (e.g. correction history)
CORRHIST = [
    Migration(id="test_corrhist_searches", description="test", db="raw", table="searches",
              columns=[("corrhist_hits", "INTEGER", "NULL")]),
    Migration(id="test_corrhist_iterations", description="test", db="raw", table="searches_by_iteration",
              columns=[("corrhist_hits", "INTEGER", "NULL")]),
]


def quietly(fn, *args, **kwargs):
    """fn(*args, **kwargs) with its printing captured: (result, stdout)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


def columns(cnxn, table):
    return [row[1] for row in cnxn.execute(f'PRAGMA table_info("{table}")')]


def schema(cnxn):
    """{table: {column: declared type}}; column order is not part of it (old databases get
    columns appended, and everything reads columns by name)."""
    tables = [row[0] for row in cnxn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
    return {t: {row[1]: row[2] for row in cnxn.execute(f'PRAGMA table_info("{t}")')} for t in tables}


def search(uuid, **fields):
    """A search record of an engine that logs only the basics (an older version)."""
    return {"engine_id": "x", "game_uuid": "g", "search_uuid": uuid, "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
            "nodes": 3, "itdepth_nodes": [1, 2], **fields}


class TempDirTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)  # registered first, so it runs after the connections close
        self.dir = Path(tmp.name)

    def init(self, name, migrate=True):
        """A raw database made by init_raw_db; migrate=False: the base tables only (an old database)."""
        if migrate:
            quietly(init_module.init_raw_db, self.dir / name)
        else:
            with mock.patch.object(init_module, "ensure_schema", lambda cnxn, **kwargs: None):
                quietly(init_module.init_raw_db, self.dir / name)
        return self.connect(name)

    def connect(self, name):
        cnxn = sqlite3.connect(self.dir / name / "chess.db")
        self.addCleanup(cnxn.close)
        return cnxn


class RawSchemaTest(TempDirTest):
    def test_the_base_tables_are_frozen(self):
        base = self.init("base", migrate=False)
        self.assertEqual({t: tuple(columns(base, t)) for t in schema(base)}, BASE_SCHEMA,
                         "init_raw_db's CREATE TABLEs changed: add a migration in migrate_schema.py instead")

    def test_init_applies_every_migration(self):
        cnxn = self.init("fresh")
        recorded = {row[0] for row in cnxn.execute("SELECT id FROM schema_migrations")}
        self.assertEqual(recorded, {m.id for m in MIGRATIONS if m.db == "raw"})
        self.assertEqual(migrate_schema.pending_ids(cnxn), [])
        self.assertEqual(quietly(ensure_schema, cnxn)[1], "")  # current: nothing to do, nothing printed

    def test_an_old_database_is_upgraded_in_place_to_the_current_schema(self):
        old = self.init("old", migrate=False)
        # a database init_raw_db made in late June 2026: no root_moves table yet, nmp_fail
        # (renamed nmp_failhigh on 07-13), no completed_depth (07-07) or pvs_researches (07-09)
        old.execute("DROP TABLE root_moves")
        for table in ("searches", "searches_by_iteration", "searches_by_tree_depth"):
            old.execute(f"ALTER TABLE {table} RENAME COLUMN nmp_failhigh TO nmp_fail")
        old.execute("ALTER TABLE searches DROP COLUMN completed_depth")
        old.execute("ALTER TABLE searches DROP COLUMN pvs_researches")
        old.execute("INSERT INTO engines (name, version) VALUES ('old', '0.1')")
        old.execute("INSERT INTO searches (engine_id, fen, nodes, nmp_fail) VALUES (1, 'startpos', 123, 7)")
        old.execute("INSERT INTO searches_by_iteration (search_id, depth, nodes, nmp_fail) VALUES (1, 1, 50, 2)")
        old.commit()
        _, out = quietly(ensure_schema, old)  # what every program does when it opens the database
        self.assertIn("schema upgraded", out)
        self.assertEqual(schema(old), schema(self.init("fresh")))
        self.assertEqual(old.execute("SELECT nodes, nmp_failhigh, completed_depth, pvs_researches, tt_returns, "
                                     "extra_fields FROM searches").fetchall(), [(123, 7, None, None, None, None)])
        self.assertEqual(old.execute("SELECT nodes, nmp_failhigh FROM searches_by_iteration").fetchall(), [(50, 2)])
        self.assertEqual(quietly(ensure_schema, old)[1], "")

    def test_init_restores_what_is_missing_even_where_the_history_says_applied(self):
        cnxn = self.init("db")
        # e.g. an interrupted delete: tables dropped while schema_migrations stayed
        cnxn.execute("DROP TABLE match_results")
        cnxn.execute("DROP TABLE engines")
        cnxn.commit()
        cnxn.close()
        again = self.init("db")
        self.assertEqual(schema(again), schema(self.init("fresh")))
        self.assertIn("engine_key", columns(again, "engines"))

    def test_rerunning_the_migrations_changes_nothing(self):
        before = schema(self.init("db"))
        results, _ = quietly(migrate_schema.migrate, raw_db=self.dir / "db" / "chess.db",
                             analytics_db=self.dir / "none.duckdb")
        self.assertEqual(set(results.values()), {"skipped"})
        self.assertEqual(schema(self.connect("db")), before)

    def test_a_failing_migration_rolls_back_completely(self):
        cnxn = self.init("db")
        broken = Migration(id="test_broken", description="adds a column, then fails", db="raw", table="searches",
                           columns=[("half_applied", "INTEGER", "NULL")], statements=["CREATE TABLE broken ("])
        with self.assertRaises(sqlite3.Error):
            quietly(migrate_schema.migrate_connection, cnxn, [broken])
        self.assertNotIn("half_applied", columns(cnxn, "searches"))
        self.assertEqual(migrate_schema.pending_ids(cnxn, [broken]), ["test_broken"])

    def test_a_migration_another_program_applied_meanwhile_is_skipped(self):
        cnxn = self.init("db")
        race = Migration(id="test_race", description="race", db="raw", table="searches",
                         columns=[("raced", "INTEGER", "NULL")])
        other = sqlite3.connect(self.dir / "db" / "chess.db", isolation_level=None)
        self.addCleanup(other.close)
        quietly(migrate_schema.migrate_connection, other, [race])  # the other program took the lock first
        adapter = migrate_schema.SQLiteAdapter(cnxn)
        real, calls = adapter.is_applied, []

        def stale_then_real(migration_id):  # this program looked before the other one committed
            calls.append(migration_id)
            return False if len(calls) == 1 else real(migration_id)

        adapter.is_applied = stale_then_real
        status, _ = quietly(migrate_schema.run_migration, adapter, race, False)
        self.assertEqual(status, "skipped")  # not "duplicate column name: raced"
        self.assertEqual(columns(cnxn, "searches").count("raced"), 1)

    def test_dry_run_changes_nothing(self):
        old = self.init("old", migrate=False)
        before = schema(old)
        old.close()
        results, _ = quietly(migrate_schema.migrate, raw_db=self.dir / "old" / "chess.db",
                             analytics_db=self.dir / "none.duckdb", dry_run=True)
        self.assertIn("pending", results.values())
        self.assertEqual(schema(self.connect("old")), before)  # not even the schema_migrations table

    def test_delete_then_init_gives_the_full_schema_again(self):
        cnxn = self.init("db")
        expected = schema(cnxn)
        cnxn.execute("CREATE TABLE added_by_a_later_migration (x INTEGER)")
        cnxn.commit()
        cnxn.close()
        quietly(deletes.drop_all_tables, str(self.dir / "db" / "chess.db"))
        self.assertEqual(schema(self.connect("db")), {})
        again = self.init("db")
        self.assertEqual(schema(again), expected)
        self.assertEqual(migrate_schema.pending_ids(again), [])

    def test_an_interrupted_delete_leaves_the_database_intact(self):
        before = schema(self.init("db"))
        reader = sqlite3.connect(self.dir / "db" / "chess.db", isolation_level=None)
        self.addCleanup(reader.close)
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM searches").fetchone()  # e.g. load_analytics reading meanwhile
        with mock.patch.object(deletes, "connect_raw",
                               lambda path, **kwargs: sqlite3.connect(path, **{**kwargs, "timeout": 0.2})):
            with self.assertRaises(sqlite3.OperationalError):  # the drop can't commit while the reader reads
                quietly(deletes.drop_all_tables, str(self.dir / "db" / "chess.db"))
        reader.execute("COMMIT")
        self.assertEqual(schema(self.connect("db")), before)  # nothing half dropped

    def test_the_dry_run_lists_columns_of_older_layouts(self):
        old = self.init("old", migrate=False)
        old.execute("ALTER TABLE searches ADD COLUMN fail_high_first INTEGER")  # before 2026-07-07
        old.execute("CREATE TABLE notes (x TEXT)")
        old.commit()
        old.close()
        _, out = quietly(migrate_schema.migrate, raw_db=self.dir / "old" / "chess.db",
                         analytics_db=self.dir / "none.duckdb", dry_run=True)
        self.assertIn("columns: searches.fail_high_first", out)
        self.assertIn("tables:  notes", out)

    def test_clearing_rows_keeps_the_schema_and_its_history(self):
        cnxn = self.init("db")
        cnxn.execute("INSERT INTO engines (name, version) VALUES ('a', '1')")
        cnxn.execute("INSERT INTO match_results (source, engine_a_id, engine_b_id, tc_category, wins, draws, losses) "
                     "VALUES ('manual', 1, 1, 'blitz', 1, 0, 0)")
        cnxn.execute("CREATE TABLE added_by_a_later_migration (x INTEGER)")
        cnxn.execute("INSERT INTO added_by_a_later_migration VALUES (1)")
        cnxn.commit()
        cnxn.close()
        quietly(deletes.clear_all_tables, str(self.dir / "db" / "chess.db"), exclude_engines=True)
        check = self.connect("db")
        count = {t: check.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
                 for t in ("engines", "match_results", "added_by_a_later_migration")}
        self.assertEqual(count, {"engines": 1, "match_results": 0, "added_by_a_later_migration": 0})
        self.assertEqual(migrate_schema.pending_ids(check), [])


class ConcurrencyTest(TempDirTest):
    def test_a_writer_waits_for_another_programs_write(self):
        self.init("db")
        path = self.dir / "db" / "chess.db"
        other = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.addCleanup(other.close)
        other.execute("BEGIN IMMEDIATE")  # e.g. an SPRT loading its telemetry
        other.execute("INSERT INTO engines (name, version) VALUES ('other', '1')")
        finish = threading.Timer(1.0, other.execute, ("COMMIT",))
        finish.start()
        self.addCleanup(finish.join)
        cnxn = connect_raw(path)
        self.addCleanup(cnxn.close)
        self.assertGreaterEqual(cnxn.execute("PRAGMA busy_timeout").fetchone()[0], 600_000)  # not sqlite3's 5 s
        cnxn.execute("INSERT INTO engines (name, version) VALUES ('mine', '2')")
        cnxn.commit()
        self.assertEqual(cnxn.execute("SELECT name FROM engines ORDER BY id").fetchall(), [("other",), ("mine",)])

    def test_a_side_of_a_run_is_never_loaded_twice(self):
        from tests import sprt
        cnxn = self.init("raw")
        cnxn.execute("INSERT INTO engines (name, version) VALUES ('candidate', '1'), ('baseline', '2')")
        cnxn.execute("INSERT INTO experiments (engine_id, type) VALUES (1, 'SPRT')")
        cnxn.commit()
        run = self.dir / "run"

        def write_logs():
            for side in ("candidate", "baseline"):
                (run / side).mkdir(parents=True)
                (run / side / "game_1.jsonl").write_text(
                    json.dumps(dict(CURRENT_GAME, game_uuid=f"{side}-game")) + "\n", encoding="utf-8")

        write_logs()
        ok, _ = quietly(sprt.ingest_telemetry, cnxn, run, dict(candidate=1, baseline=2, experiment=1, loaded=[]))
        self.assertTrue(ok)
        # the process stopped after the commits, before it noted them: the same logs come again
        write_logs()
        ok, out = quietly(sprt.ingest_telemetry, cnxn, run, dict(candidate=1, baseline=2, experiment=1, loaded=[]))
        self.assertTrue(ok)
        self.assertIn("loaded already", out)
        self.assertEqual(cnxn.execute("SELECT count(*) FROM games").fetchone()[0], 2)
        self.assertFalse((run / "candidate").exists() or (run / "baseline").exists())

    def test_a_side_with_a_record_the_schema_refuses_keeps_its_logs_until_a_retry_loads_it(self):
        from tests import sprt
        cnxn = self.init("raw")
        cnxn.execute("INSERT INTO engines (name, version) VALUES ('candidate', '1'), ('baseline', '2')")
        cnxn.execute("INSERT INTO experiments (engine_id, type) VALUES (1, 'SPRT')")
        cnxn.commit()
        run = self.dir / "run"
        for side in ("candidate", "baseline"):
            (run / side).mkdir(parents=True)
            (run / side / "game_1.jsonl").write_text(
                json.dumps(dict(CURRENT_GAME, game_uuid=f"{side}-game")) + "\n", encoding="utf-8")
        good = dict(CURRENT_SEARCH, game_uuid="candidate-game", search_uuid="s1")
        bad = {k: v for k, v in dict(good, search_uuid="s2").items() if k != "fen"}  # searches.fen is NOT NULL
        searches = run / "candidate" / "search_1.jsonl"
        searches.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
        ids = dict(candidate=1, baseline=2, experiment=1, loaded=[])
        (run / "engines.json").write_text(json.dumps(ids), encoding="utf-8")
        db = self.dir / "raw" / "chess.db"

        ok, out = quietly(sprt.ingest_telemetry, cnxn, run, ids)
        self.assertFalse(ok)
        self.assertIn("ingestion failed (candidate)", out)
        self.assertNotIn("loaded already", out)
        self.assertTrue(searches.exists())  # kept for a retry
        self.assertFalse((run / "baseline").exists())
        self.assertEqual(ids["loaded"], ["baseline"])
        self.assertEqual(cnxn.execute("SELECT side FROM telemetry_loads").fetchall(), [("baseline",)])
        self.assertEqual(cnxn.execute("SELECT count(*) FROM searches").fetchone()[0], 0)  # the side rolled back
        cnxn.close()

        # the record fixed, a retry loads the candidate's telemetry, once
        searches.write_text(json.dumps(good) + "\n" + json.dumps(dict(good, search_uuid="s2")) + "\n",
                            encoding="utf-8")
        ok, out = quietly(sprt.retry_ingest, run, db=db)
        self.assertTrue(ok, out)
        self.assertFalse(run.exists())
        check = self.connect("raw")
        self.assertEqual(sorted(check.execute("SELECT side FROM telemetry_loads").fetchall()),
                         [("baseline",), ("candidate",)])
        self.assertEqual(check.execute("SELECT count(*) FROM searches").fetchone()[0], 2)
        self.assertEqual(check.execute("SELECT count(*) FROM games").fetchone()[0], 2)


class NewStatisticTest(TempDirTest):
    """A statistic a new engine version logs needs only a migration to be stored."""

    def setUp(self):
        super().setUp()
        self.cnxn = self.init("raw")
        self.cnxn.execute("INSERT INTO engines (name, version) VALUES ('old', '0.1'), ('new', '0.2')")
        self.cnxn.commit()
        self.files = 0

    def ingest(self, records, engine_id):
        self.files += 1
        path = self.dir / f"search_{self.files}.jsonl"
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return quietly(ingest.bulk_log_search_and_timing, self.cnxn, path, {}, engine_id=engine_id, sts_id=1)

    def analytics(self):
        """A new analytics connection each time, as every pipeline run opens one."""
        cnxn = duckdb.connect(str(self.dir / "analytics.duckdb"))
        self.addCleanup(cnxn.close)
        cnxn.execute(f"ATTACH '{self.dir / 'raw' / 'chess.db'}' AS raw (TYPE SQLITE)")
        load_analytics._register_udfs(cnxn)
        return cnxn

    def test_everything_the_current_engine_logs_is_stored(self):
        counts, out = self.ingest([CURRENT_SEARCH], engine_id=2)
        self.assertEqual((counts["unstored_fields"], counts["mismatched_fields"]), ([], []))
        self.assertNotIn("[ETL]", out)
        r = CURRENT_SEARCH
        self.assertEqual(self.cnxn.execute(
            "SELECT tt_returns, pvs_researches_depth, pvs_researches_full, pvs_researches_root, rfp, futility_prune, "
            "pvs_researches, extra_fields FROM searches").fetchone(),
            (r["tt_returns"], r["pvs_researches_depth"], r["pvs_researches_full"], r["pvs_researches_root"],
             r["rfp"], r["futility_prune"], None, None))  # the combined count is no longer logged: NULL, not 0
        self.assertEqual(self.cnxn.execute(
            "SELECT depth, tt_returns, rfp, futility_prune, pvs_researches_depth, pvs_researches, tt_fill "
            "FROM searches_by_iteration ORDER BY depth").fetchall(),
            [(1, 8, 4, 5, 2, None, None), (2, 23, 10, 12, 6, None, None)])
        self.assertEqual(self.cnxn.execute(
            "SELECT depth, nodes, tt_returns, futility_prune, pvs_researches_full "
            "FROM searches_by_tree_depth ORDER BY depth").fetchall(),
            [(1, 10, 1, 0, 1), (2, 300, 8, 5, 4), (3, 4690, 22, 12, None)])  # past the shorter array: NULL

        game = self.dir / "game.jsonl"
        game.write_text(json.dumps(CURRENT_GAME) + "\n", encoding="utf-8")
        _, out = quietly(ingest.bulk_log_game, self.cnxn, game, second_engine_id=1, engine_id=2)
        self.assertNotIn("[ETL]", out)
        self.assertEqual(self.cnxn.execute("SELECT white_engine_id, black_engine_id, plies, extra_fields FROM games")
                         .fetchone(), (2, 1, 3, None))

    def test_logs_of_older_engines_are_stored_under_the_current_names(self):
        self.ingest([search("old-1", nmp_fail=4, itdepth_nmp_fail=[1, 3])], engine_id=1)  # before July 2026
        self.assertEqual(self.cnxn.execute("SELECT nmp_failhigh FROM searches").fetchone(), (4,))
        self.assertEqual(self.cnxn.execute("SELECT nmp_failhigh FROM searches_by_iteration ORDER BY depth")
                         .fetchall(), [(1,), (3,)])

    def test_fields_an_engine_does_not_log_are_null_not_zero(self):
        self.ingest([search("old-1")], engine_id=1)
        self.assertEqual(self.cnxn.execute("SELECT nodes, fh_index_0, tt_returns, pvs_researches FROM searches")
                         .fetchone(), (3, None, None, None))
        self.assertEqual(self.cnxn.execute(
            "SELECT depth, nodes, time_ms, tt_hits, fh_index_0, pvs_researches, nmp, rfp FROM searches_by_iteration "
            "ORDER BY depth").fetchall(), [(1, 1, None, None, None, None, None, None),
                                           (2, 2, None, None, None, None, None, None)])

    def test_a_new_statistic_needs_only_a_migration_and_loses_nothing_meanwhile(self):
        self.ingest([search("old-1")], engine_id=1)  # the old version, before the statistic existed
        # the new version runs before the migration is written: its values are kept aside
        counts, out = self.ingest([search("new-1", corrhist_hits=5, itdepth_corrhist_hits=[1, 4])], engine_id=2)
        self.assertEqual(counts["unstored_fields"], ["corrhist_hits", "itdepth_corrhist_hits"])
        self.assertIn("Kept in extra_fields; add a migration", out)
        self.assertEqual(json.loads(self.cnxn.execute("SELECT extra_fields FROM searches WHERE id = 2").fetchone()[0]),
                         {"corrhist_hits": 5})
        with mock.patch.object(migrate_schema, "MIGRATIONS", MIGRATIONS + CORRHIST):
            _, out = quietly(ensure_schema, self.cnxn)
            self.assertIn("filled 1 row(s) from extra_fields", out)
            counts, _ = self.ingest([search("new-2", corrhist_hits=7, itdepth_corrhist_hits=[2, 6])], engine_id=2)
        self.assertEqual(counts["unstored_fields"], [])
        self.assertEqual(self.cnxn.execute("SELECT engine_id, corrhist_hits, extra_fields FROM searches ORDER BY id")
                         .fetchall(), [(1, None, None), (2, 5, None), (2, 7, None)])
        self.assertEqual(self.cnxn.execute(
            "SELECT corrhist_hits FROM searches_by_iteration ORDER BY search_id, depth").fetchall(),
            [(None,), (None,), (1,), (4,), (2,), (6,)])

    def test_values_that_do_not_fit_their_column_are_kept_aside(self):
        # INTEGER columns: 2.5 or a list there would make DuckDB reject the whole table at the analytics load
        counts, out = self.ingest([search("odd", rfp=2.5, tt_returns=4.0, futility_prune=[1, 2])], engine_id=2)
        self.assertEqual(counts["mismatched_fields"], ["futility_prune", "rfp"])
        self.assertIn("don't fit their column's type", out)
        rfp, tt_returns, futility, extra = self.cnxn.execute(
            "SELECT rfp, tt_returns, futility_prune, extra_fields FROM searches").fetchone()
        self.assertEqual((rfp, tt_returns, futility), (None, 4, None))  # 4.0 is a whole number: stored as 4
        self.assertEqual(json.loads(extra), {"rfp": 2.5, "futility_prune": [1, 2]})
        self.cnxn.commit()
        analytics = self.analytics()
        quietly(load_analytics.load_full, analytics)
        self.assertEqual(analytics.execute("SELECT rfp, tt_returns FROM search_stats").fetchone(), (None, 4))

    def test_builds_sharing_a_version_are_attributed_exactly(self):
        # a release registers its DEV and Prod builds; both log the same version string
        dev = self.cnxn.execute("INSERT INTO engines (name, version) VALUES ('dev', '0.3')").lastrowid
        self.cnxn.execute("INSERT INTO engines (name, version) VALUES ('prod', '0.3')")
        self.cnxn.commit()
        path = self.dir / "shared.jsonl"
        path.write_text(json.dumps(search("d1", engine_id="0.3")) + "\n", encoding="utf-8")
        quietly(ingest.bulk_log_search_and_timing, self.cnxn, path, {}, engine_ids_by_version={"0.3": dev})
        game = self.dir / "shared_game.jsonl"
        game.write_text(json.dumps(dict(CURRENT_GAME, engine_id="0.3")) + "\n", encoding="utf-8")
        quietly(ingest.bulk_log_game, self.cnxn, game, second_engine_id=1, engine_ids_by_version={"0.3": dev})
        self.assertEqual(self.cnxn.execute("SELECT engine_id FROM searches").fetchall(), [(dev,)])
        self.assertEqual(self.cnxn.execute("SELECT white_engine_id FROM games").fetchall(), [(dev,)])
        ingest._ambiguous_versions_reported.discard("0.3")
        _, out = quietly(ingest.bulk_log_search_and_timing, self.cnxn, path, {})  # no ids given: said so
        self.assertIn("2 builds are registered as version 0.3", out)

    def test_the_analytics_database_follows_new_raw_columns(self):
        self.ingest([search("old-1")], engine_id=1)
        self.cnxn.commit()
        first_run = self.analytics()
        quietly(load_analytics.load_full, first_run)
        first_run.close()  # a run ends; the next one attaches the raw database anew
        with mock.patch.object(migrate_schema, "MIGRATIONS", MIGRATIONS + CORRHIST):
            quietly(ensure_schema, self.cnxn)
            self.ingest([search("new-1", corrhist_hits=7, itdepth_corrhist_hits=[2, 6])], engine_id=2)
        self.cnxn.commit()
        analytics = self.analytics()
        _, out = quietly(load_analytics.load_incremental, analytics)
        self.assertIn("search_stats: added new column(s) corrhist_hits", out)
        self.assertEqual(analytics.execute("SELECT corrhist_hits FROM search_stats ORDER BY id").fetchall(),
                         [(None,), (7,)])
        self.assertEqual(analytics.execute(
            "SELECT corrhist_hits FROM iterative_deepening_stats ORDER BY search_id, depth").fetchall(),
            [(None,), (None,), (2,), (6,)])
        quietly(build_position_features, analytics)  # the pipeline's next steps still run over the wider tables
        quietly(build_analytics_views, analytics)
        self.assertEqual(analytics.execute("SELECT count(*) FROM search_features").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()
