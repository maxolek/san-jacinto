"""The analytics pipeline on synthetic telemetry, without Stockfish.

The shared fixture runs `run_analytics_pipeline --full` (every step's real
command-line entry point) on a raw database built from generated logs.
"""
import contextlib
import io
import shutil
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

import duckdb

from data import etl
from data.databases.load_analytics import load, path_basename, replays_from_start
from data.databases.test_schema import VIEW_COLUMNS, check_schema, validate_schema
from data.transforms import transform_positions as positions
from data.transforms.aggregates import (AGG_COLUMNS, LOSS_BUCKETS, SAMPLE_SIZE, STOCKFISH_TABLES, allocate_sample,
                                        build_aggregates)
from data.transforms.transform_search import ANALYTICS_VIEWS, build_analytics_views
from data.transforms.validate import run_validation

from .fixtures import EPD_BOOK_OPENING, FALLBACK_OPENING, analytics_fixture, write_logs


def _columns(cnxn, relation):
    return [r[0] for r in cnxn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
        [relation]).fetchall()]


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root, cls.raw, cls.analytics, cls.output = analytics_fixture()
        cls.db = duckdb.connect(str(cls.analytics), read_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def scalar(self, sql, params=None):
        return self.db.execute(sql, params or []).fetchone()[0]

    def rows(self, sql, params=None):
        return self.db.execute(sql, params or []).fetchall()

    def test_runs_without_stockfish(self):
        self.assertIn("is not an executable file", self.output)
        self.assertIn("stockfish: not found -- skipping", self.output)
        self.assertIn("Pipeline completed successfully.", self.output)
        searches = self.scalar("SELECT count(*) FROM search_stats")
        self.assertGreater(searches, 200)
        self.assertEqual(self.scalar("SELECT count(sf_eval) FROM search_stats"), 0)
        self.assertEqual(self.scalar("SELECT count(DISTINCT search_id) FROM position_features"), searches)

    def test_views_have_documented_columns(self):
        for view in ANALYTICS_VIEWS:
            with self.subTest(view=view):
                kind = self.scalar("SELECT table_type FROM information_schema.tables WHERE table_name = ?", [view])
                self.assertEqual(kind, "VIEW")
                missing = set(VIEW_COLUMNS[view]) - set(_columns(self.db, view))
                self.assertFalse(missing, f"{view} lacks {missing}")
                self.scalar(f'SELECT count(*) FROM "{view}"')

    def test_aggregate_tables_have_documented_columns(self):
        documented = (Path(__file__).resolve().parents[2] / "docs" / "dashboard-data.md").read_text(encoding="utf-8")
        for table, columns in AGG_COLUMNS.items():
            with self.subTest(table=table):
                self.assertEqual(tuple(_columns(self.db, table)), columns)
                if table not in STOCKFISH_TABLES:   # those need Stockfish: see StockfishPipelineTest
                    self.assertGreater(self.scalar(f'SELECT count(*) FROM "{table}"'), 0)
                self.assertIn(f"`{table}`", documented)
                for column in columns:
                    self.assertIn(column, documented, f"{table}.{column} is not documented")

    def test_schema_and_data_validation_pass(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(validate_schema(self.analytics, profile="analytics"))
        result = run_validation(self.db)
        self.assertEqual(result.errors, [])

    def test_game_stats_decoding_and_openings(self):
        terminations = {r[0] for r in self.rows("SELECT termination FROM game_stats")}
        self.assertTrue({"adjudication", "insufficient_material", "illegal_move", "crash"} <= terminations)
        self.assertTrue({r[0] for r in self.rows("SELECT result FROM game_stats")} <= {"white", "black", "draw"})
        polish = self.rows("SELECT opening, opening_eco, opening_variation FROM game_stats WHERE id = 1")
        self.assertEqual(polish, [("Polish", "A00", "Tuebingen variation")])
        fallback = self.rows("SELECT opening, opening_variation FROM game_stats WHERE id = 3")
        self.assertEqual(fallback, [FALLBACK_OPENING])
        epd_book = self.rows("SELECT opening, opening_variation, opening_eco FROM game_stats WHERE id = 5")
        self.assertEqual(epd_book, [EPD_BOOK_OPENING + ("",)])
        env = self.rows("SELECT env_core_type, count(*) FROM game_stats GROUP BY 1 ORDER BY 1")
        self.assertEqual({row[0] for row in env}, {"E", "P", None})

    def test_sprt_keeps_measured_elo_and_book_file_names(self):
        rows = self.rows("SELECT elo_diff, elo1, opening_book, pairs, penta_ww, core_summary FROM sprt_runs ORDER BY id")
        self.assertEqual([r[0] for r in rows], [12.37, -3.5])
        self.assertEqual([r[1] for r in rows], [2.5, 2.5])
        self.assertEqual([r[2] for r in rows], ["UHO_4060_v2.epd", "8moves_v3.pgn"])
        self.assertEqual(rows[0][3:5], (200, 25))
        self.assertIn('"P"', rows[0][5])
        self.assertEqual(self.rows("SELECT DISTINCT suite FROM sts_runs ORDER BY 1"), [("STS1.epd",), ("decoy.epd",)])

    def test_path_basename_handles_windows_and_posix_paths(self):
        for path, name in ((r"C:\books\UHO.epd", "UHO.epd"), ("/home/u/books/8moves.pgn", "8moves.pgn"),
                           ("book.epd", "book.epd"), ("C:/mixed\\sep/x.epd", "x.epd"), (None, None)):
            self.assertEqual(path_basename(path), name)

    def test_replays_from_start(self):
        self.assertTrue(replays_from_start('["e2e4", "e7e5", "g1f3"]'))
        self.assertTrue(replays_from_start("e4 e5 Nf3"))
        self.assertTrue(replays_from_start("[]"))
        self.assertFalse(replays_from_start('["c2c3", "b7b5", "b3c2"]'))   # a bishop that isn't on b3
        self.assertFalse(replays_from_start("not json ["))

    def test_sts_join_uses_experiment_and_position(self):
        rows = self.rows("SELECT sts_suite, sts_position_name FROM search_context WHERE sts_id IS NOT NULL ORDER BY id")
        self.assertEqual(len(rows), 5)
        self.assertEqual({r[0] for r in rows}, {"STS1.epd"})
        self.assertEqual([r[1] for r in rows], [f"STS({n}) - Undermining" for n in range(5)])
        self.assertEqual(self.scalar("SELECT count(*) FROM search_context"), self.scalar("SELECT count(*) FROM search_stats"))

    def test_eval_cp_and_mate_columns(self):
        rows = self.rows("SELECT mate, min(eval), max(eval), count(eval_cp), min(mate_plies), max(mate_plies) "
                         "FROM search_metrics GROUP BY mate ORDER BY mate")
        by_mate = {r[0]: r[1:] for r in rows}
        self.assertEqual(set(by_mate), {-1, 0, 1})
        self.assertEqual(by_mate[1][2], 0)
        self.assertEqual(by_mate[1][3:], (7, 7))
        self.assertEqual(by_mate[-1][3:], (-6, -6))
        self.assertLess(max(abs(by_mate[0][0]), abs(by_mate[0][1])), 99000)
        self.assertLess(self.scalar("SELECT max(abs(avg_eval)) FROM search_iteration_summary"), 99000)
        self.assertLess(self.scalar("SELECT max(abs(stddev_eval)) FROM search_iteration_features"), 99000)

    def test_environment_columns_in_search_views(self):
        mismatches = self.scalar("""
            SELECT count(*) FROM search_context c JOIN game_stats g ON g.id = c.game_id
            WHERE c.env_core_type IS DISTINCT FROM g.env_core_type OR c.env_mhz_mean IS DISTINCT FROM g.env_mhz_mean
               OR c.env_host IS DISTINCT FROM g.env_host OR c.env_cpu_model IS DISTINCT FROM g.env_cpu_model""")
        self.assertEqual(mismatches, 0)
        self.assertGreater(self.scalar("SELECT count(env_mhz_mean) FROM search_metrics"), 0)

    def test_nps_is_total_nodes_over_total_time(self):
        rows = self.rows("""
            SELECT a.engine_id, a.nps, s.nps, a.knps_per_ghz, k.knps FROM agg_engine_summary a
            JOIN (SELECT engine_id, sum(nodes + qnodes) / (sum(time_ms) / 1000) AS nps FROM search_stats GROUP BY 1) s
              USING (engine_id)
            JOIN (SELECT s.engine_id, sum(s.nodes + s.qnodes) / sum(s.time_ms / 1000 * g.env_mhz_mean / 1000) / 1000 AS knps
                  FROM search_stats s JOIN game_stats g ON g.id = s.game_id WHERE g.env_mhz_mean > 0 GROUP BY 1) k
              USING (engine_id)
            WHERE a.core_type = 'all'""")
        self.assertEqual(len(rows), 3)
        for engine_id, nps, expected, knps, expected_knps in rows:
            self.assertAlmostEqual(nps, expected, delta=expected * 1e-9)
            self.assertAlmostEqual(knps, expected_knps, delta=expected_knps * 1e-9)
        by_core = self.scalar("SELECT sum(searches) FROM agg_engine_summary WHERE core_type <> 'all'")
        self.assertEqual(by_core, self.scalar("SELECT sum(searches) FROM agg_engine_summary WHERE core_type = 'all'"))

    def test_eval_wdl_matches_game_results(self):
        expected = self.rows("""
            SELECT s.engine_id,
                   CASE WHEN abs(s.eval) >= 99000 THEN NULL
                        ELSE CAST(greatest(-1000, least(1000, round(s.eval / 25.0) * 25)) AS INTEGER) END AS bucket,
                   CASE WHEN s.eval >= 99000 THEN 1 WHEN s.eval <= -99000 THEN -1 ELSE 0 END AS mate,
                   count(*), count(DISTINCT s.game_id),
                   count(*) FILTER (WHERE g.result = CASE split_part(s.fen, ' ', 2) WHEN 'w' THEN 'white' ELSE 'black' END),
                   count(*) FILTER (WHERE g.result = 'draw'),
                   count(*) FILTER (WHERE g.result = CASE split_part(s.fen, ' ', 2) WHEN 'w' THEN 'black' ELSE 'white' END)
            FROM search_stats s JOIN game_stats g ON g.id = s.game_id
            WHERE g.result IS NOT NULL AND s.eval IS NOT NULL
            GROUP BY ALL ORDER BY 1, 3, 2""")
        actual = self.rows("""
            SELECT engine_id, eval_bucket, mate, positions, games, wins, draws, losses FROM agg_eval_wdl
            WHERE core_type = 'all' AND phase = 'all' ORDER BY engine_id, mate, eval_bucket""")
        self.assertEqual(actual, expected)
        # every per-core-type / per-phase slice adds up to the 'all' rows
        for column in ("core_type", "phase"):
            split = self.rows(f"SELECT sum(positions), sum(wins), sum(draws) FROM agg_eval_wdl "
                              f"WHERE {column} <> 'all' AND {'phase' if column == 'core_type' else 'core_type'} = 'all'")
            total = self.rows("SELECT sum(positions), sum(wins), sum(draws) FROM agg_eval_wdl "
                              "WHERE core_type = 'all' AND phase = 'all'")
            self.assertEqual(split, total)

    def test_openings_match_game_stats(self):
        expected = self.rows("""
            WITH sides AS (
                SELECT white_engine_id AS engine_id, result = 'white' AS won, result = 'black' AS lost, result FROM game_stats
                UNION ALL
                SELECT black_engine_id, result = 'black', result = 'white', result FROM game_stats)
            SELECT engine_id, count(*), count(*) FILTER (WHERE won), count(*) FILTER (WHERE result = 'draw'),
                   count(*) FILTER (WHERE lost)
            FROM sides GROUP BY 1 ORDER BY 1""")
        for level in ("family", "variation"):
            actual = self.rows(f"SELECT engine_id, sum(games), sum(wins), sum(draws), sum(losses) FROM agg_openings "
                               f"WHERE level = '{level}' GROUP BY 1 ORDER BY 1")
            self.assertEqual(actual, expected, level)
        variation = self.rows("SELECT opening_variation, opening_eco FROM agg_openings "
                              "WHERE opening = 'Polish' AND level = 'variation'")
        self.assertIn(("Tuebingen variation", "A00"), variation)
        self.assertEqual(self.scalar("SELECT count(*) FROM agg_openings WHERE level = 'family' "
                                     "AND opening_variation IS NOT NULL"), 0)

    def test_distribution_tables_are_consistent(self):
        self.assertEqual(self.rows("SELECT DISTINCT round(s, 9) FROM (SELECT sum(share) s FROM agg_hist "
                                   "GROUP BY engine_id, core_type, metric)"), [(1.0,)])
        self.assertEqual(self.rows("SELECT DISTINCT round(s, 9) FROM (SELECT sum(share) s FROM agg_moveorder "
                                   "GROUP BY engine_id, dimension, dim_value)"), [(1.0,)])
        self.assertEqual(self.rows("SELECT DISTINCT round(s, 9) FROM (SELECT sum(node_share) s FROM agg_tree_ply "
                                   "GROUP BY engine_id)"), [(1.0,)])
        self.assertEqual(self.scalar("SELECT count(*) FROM agg_timing WHERE function = 'ROOT' AND abs(share - 1) > 1e-9"), 0)
        depth_rows = self.rows("SELECT depth, searches FROM agg_depth WHERE engine_id = 1 AND core_type = 'all' ORDER BY depth")
        self.assertEqual([r[1] for r in depth_rows], sorted([r[1] for r in depth_rows], reverse=True))

    def test_correlations(self):
        self.assertEqual(self.scalar("SELECT count(*) FROM agg_corr WHERE n < 30 OR metric_x = metric_y"), 0)
        self.assertEqual(self.scalar("SELECT count(*) FROM agg_corr WHERE abs(pearson) > 1 + 1e-9 "
                                     "OR abs(spearman) > 1 + 1e-9"), 0)
        rho = self.rows("SELECT engine_id, spearman FROM agg_corr WHERE metric_x = 'depth' AND metric_y = 'nodes'")
        self.assertEqual(len(rho), 3)
        self.assertTrue(all(r[1] > 0.5 for r in rho))
        self.assertEqual(self.scalar("SELECT count(*) FROM agg_corr a JOIN agg_corr b ON a.engine_id = b.engine_id "
                                     "AND a.metric_x = b.metric_y AND a.metric_y = b.metric_x "
                                     "WHERE abs(a.spearman - b.spearman) > 1e-12"), 0)

    def test_search_sample(self):
        self.assertLessEqual(self.scalar("SELECT count(*) FROM search_sample"), SAMPLE_SIZE)
        weights = self.rows("SELECT s.engine_id, round(sum(sample_weight)), count(*) FROM search_sample s GROUP BY 1 ORDER BY 1")
        totals = self.rows("SELECT engine_id, count(*) FROM search_stats GROUP BY 1 ORDER BY 1")
        self.assertEqual([(e, float(n)) for e, n in totals], [(e, w) for e, w, _ in weights])
        self.assertEqual(allocate_sample({1: 10, 2: 500, 3: 500}, 100), {1: 10, 2: 45, 3: 45})
        self.assertEqual(sum(allocate_sample({1: 9000, 2: 9000, 3: 9000}, 20000).values()), 20000)

    def test_sample_is_stratified(self):
        # (determinism: the public export rebuilds the sample and must match, see test_public_export)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            copy = Path(tmp) / "copy.duckdb"
            shutil.copyfile(self.analytics, copy)
            with duckdb.connect(str(copy)) as cnxn:
                build_aggregates(cnxn, sample_size=90, verbose=False)
                per_engine = cnxn.execute("SELECT count(*), sum(sample_weight) FROM search_sample "
                                          "GROUP BY engine_id ORDER BY engine_id").fetchall()
        totals = self.rows("SELECT count(*) FROM search_stats GROUP BY engine_id ORDER BY engine_id")
        self.assertEqual([r[0] for r in per_engine], [30, 30, 30])
        self.assertEqual([round(r[1]) for r in per_engine], [t[0] for t in totals])


class StockfishPipelineTest(unittest.TestCase):
    """The positions step with a (fake) Stockfish, then the views, aggregates and checks, on a copy."""
    LINES = 3

    @classmethod
    def setUpClass(cls):
        _, _, analytics, _ = analytics_fixture()
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.path = Path(cls.tmp.name) / "with_stockfish.duckdb"
        shutil.copyfile(analytics, cls.path)
        engine = [sys.executable, str(Path(__file__).with_name("fake_uci_engine.py"))]
        cls.db = duckdb.connect(str(cls.path))
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(positions, "find_stockfish", return_value=engine):
            positions.run(cls.db, depth=2, multipv=cls.LINES, workers=2)
            build_analytics_views(cls.db)
            build_aggregates(cls.db, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        cls.tmp.cleanup()

    def scalar(self, sql):
        return self.db.execute(sql).fetchone()[0]

    def rows(self, sql):
        return self.db.execute(sql).fetchall()

    def test_every_search_gets_the_lines_and_its_move_ranked(self):
        searches = self.scalar("SELECT count(*) FROM search_stats")
        self.assertEqual(self.scalar("SELECT count(sf_eval) FROM search_stats"), searches)
        self.assertEqual(self.scalar("SELECT count(sf_move_rank) FROM search_stats"), searches)  # moves played are legal
        self.assertEqual(self.scalar(f"SELECT count(*) FROM search_stats WHERE sf_depth <> 2 OR sf_multipv <> {self.LINES} "
                                     f"OR sf_move_rank NOT BETWEEN 0 AND {self.LINES}"), 0)
        # the fake engine scores every move differently: no loss exactly when the move is Stockfish's first line
        self.assertEqual(self.scalar("SELECT count(*) FROM search_metrics WHERE sf_cp_loss IS NULL "
                                     "OR (sf_move_rank = 1) <> (sf_cp_loss = 0) OR engine_move_rank <> sf_move_rank"), 0)
        self.assertGreater(self.scalar("SELECT count(*) FROM search_metrics WHERE sf_move_rank = 0"), 0)

    def test_aggregates_match_the_searches(self):
        direct = self.rows(f"""
            SELECT engine_id, CASE WHEN sf_move_rank > {self.LINES} THEN 0 ELSE sf_move_rank END AS rank,
                   count(*), round(sum(sf_cp_loss), 6), count(*) FILTER (WHERE sf_cp_loss >= 100)
            FROM search_metrics WHERE engine_id IS NOT NULL GROUP BY ALL ORDER BY ALL""")
        aggregated = self.rows("""
            SELECT engine_id, sf_rank, sum(searches), round(sum(loss_sum), 6),
                   coalesce(sum(searches) FILTER (WHERE loss_bucket >= 100), 0)
            FROM agg_sf_moves WHERE core_type = 'all' AND phase = 'all' GROUP BY ALL ORDER BY ALL""")
        self.assertEqual(aggregated, direct)
        self.assertEqual(self.rows("SELECT DISTINCT lines, depth_min, depth_max FROM agg_sf_moves"), [(self.LINES, 2, 2)])
        self.assertTrue({b for (b,) in self.rows("SELECT DISTINCT loss_bucket FROM agg_sf_moves")} <= set(LOSS_BUCKETS))
        # phases and core types add up to the 'all' rows
        self.assertEqual(self.rows("SELECT sum(searches) FILTER (WHERE phase <> 'all') - sum(searches) FILTER "
                                   "(WHERE phase = 'all') FROM agg_sf_moves WHERE core_type = 'all'"), [(0,)])
        self.assertEqual(self.rows("SELECT sum(searches) FILTER (WHERE core_type <> 'all') - sum(searches) FILTER "
                                   "(WHERE core_type = 'all') FROM agg_sf_moves WHERE phase = 'all'"), [(0,)])

    def test_root_effort_and_order(self):
        # per-search node shares over Stockfish's lines and the other moves add up to one
        self.assertEqual(self.rows("SELECT DISTINCT round(sum(node_share_sum) / max(searches), 9) FROM agg_sf_root_effort "
                                   "GROUP BY engine_id, core_type, phase"), [(1.0,)])
        # reference in Python: each search's deepest iteration with every root move logged
        iterations = defaultdict(lambda: defaultdict(list))
        for search, engine, best, depth, index, move in self.rows("""
                SELECT s.id, s.engine_id, s.sf_best_move, r.depth, r.move_index, r.move
                FROM root_moves r JOIN search_stats s ON s.id = r.search_id WHERE s.engine_id IS NOT NULL"""):
            iterations[(search, engine, best)][depth].append((index, move))
        self.assertTrue(any(len(depths) > 1 for depths in iterations.values()), "the fixture logs partial iterations")
        searches, moves, order = Counter(), Counter(), Counter()
        for (search, engine, best), depths in iterations.items():
            most = max(len(logged) for logged in depths.values())
            chosen = depths[max(d for d, logged in depths.items() if len(logged) == most)]
            searches[engine] += 1
            moves[engine] += len(chosen)
            index = min((i for i, move in chosen if move == best), default=None)
            order[(engine, -1 if index is None else min(index, 10))] += 1
        self.assertEqual(self.rows("SELECT engine_id, max(searches), sum(moves) FROM agg_sf_root_effort "
                                   "WHERE core_type = 'all' AND phase = 'all' GROUP BY 1 ORDER BY 1"),
                         [(e, searches[e], moves[e]) for e in sorted(searches)])
        self.assertEqual(self.rows("SELECT engine_id, move_index, searches FROM agg_sf_root_order "
                                   "WHERE core_type = 'all' AND phase = 'all' ORDER BY ALL"),
                         sorted((e, i, n) for (e, i), n in order.items()))
        # the fixture logs 8 root moves of each search, so Stockfish's best move is sometimes missing
        self.assertGreater(self.scalar("SELECT count(*) FROM agg_sf_root_order WHERE move_index = -1"), 0)

    def test_checks_pass(self):
        for table in STOCKFISH_TABLES:
            self.assertGreater(self.scalar(f"SELECT count(*) FROM {table}"), 0, table)
        self.assertEqual(run_validation(self.db).errors, [])
        errors, _, _ = check_schema(self.db, profile="analytics")
        self.assertEqual(errors, [])


class IncrementalLoadTest(unittest.TestCase):
    """An incremental load after new raw rows and a new raw column matches a full reload."""

    def test_incremental_matches_full_and_picks_up_new_columns(self):
        _, raw_source, _, _ = analytics_fixture()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw.db"
            shutil.copyfile(raw_source, raw)
            quiet = contextlib.redirect_stdout(io.StringIO())
            with quiet:
                load(raw, tmp / "incremental.duckdb")               # creates every table
                write_logs(tmp / "more", n_games=2, seed=99)
                cnxn = sqlite3.connect(raw)
                etl.ingest_log_dir(cnxn, tmp / "more", clear=False)
                cnxn.execute("ALTER TABLE searches_by_iteration ADD COLUMN extra_metric INTEGER")
                cnxn.execute("UPDATE searches_by_iteration SET extra_metric = depth "
                             "WHERE search_id > (SELECT max(search_id) - 5 FROM searches_by_iteration)")
                cnxn.commit()
                cnxn.close()
                added = load(raw, tmp / "incremental.duckdb")
                load(raw, tmp / "full.duckdb", full=True)
            self.assertGreater(added["search_stats"], 0)
            self.assertGreater(added["game_stats"], 0)
            tables = ("game_stats", "search_stats", "iterative_deepening_stats", "search_tree_stats",
                      "search_timings", "root_moves", "dim_positions", "sprt_runs", "sts_runs")
            summary = {}
            for name in ("incremental", "full"):
                with duckdb.connect(str(tmp / f"{name}.duckdb"), read_only=True) as cnxn:
                    summary[name] = {t: cnxn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
                    summary[name]["nodes"] = cnxn.execute("SELECT sum(nodes) FROM iterative_deepening_stats").fetchone()[0]
                    summary[name]["extra"] = cnxn.execute(
                        "SELECT count(extra_metric) FROM iterative_deepening_stats").fetchone()[0]
                    summary[name]["openings"] = cnxn.execute(
                        "SELECT count(*) FROM game_stats WHERE opening <> ''").fetchone()[0]
            self.assertEqual(summary["incremental"], summary["full"])
            self.assertGreater(summary["full"]["extra"], 0)


if __name__ == "__main__":
    unittest.main()
