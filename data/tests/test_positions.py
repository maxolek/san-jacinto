"""Stockfish discovery and ground truth (with a fake UCI engine) and position features."""
import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import chess
import chess.engine
import duckdb

from data.transforms import transform_positions as positions

FAKE_ENGINE = Path(__file__).with_name("fake_uci_engine.py")
WHITE_TO_MOVE = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
BLACK_TO_MOVE = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
MATES = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 50"      # fake engine: mate in 2 (side to move)
MATED = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 51"  # fake engine: black gets mated in 1


def _search_table(cnxn, fens):
    cnxn.execute("CREATE TABLE search_stats (id INTEGER, fen VARCHAR, game_id INTEGER, sts_id INTEGER)")
    cnxn.executemany("INSERT INTO search_stats VALUES (?, ?, NULL, NULL)", list(enumerate(fens, 1)))


def _fake_best_move(fen):
    return sorted(chess.Board(fen).legal_moves, key=lambda m: m.uci())[-1].uci()


def _fake_order(fen):
    """The fake engine's move order: position p scores 57 - 10 * p centipawns."""
    return [m.uci() for m in sorted(chess.Board(fen).legal_moves, key=lambda m: m.uci(), reverse=True)]


def _moves_table(cnxn, rows):
    """search_stats with the engine's move of every search: rows of (fen, move)."""
    cnxn.execute("CREATE TABLE search_stats (id INTEGER, fen VARCHAR, move VARCHAR, game_id INTEGER, sts_id INTEGER)")
    cnxn.executemany("INSERT INTO search_stats VALUES (?, ?, ?, NULL, NULL)",
                     [(i, fen, move) for i, (fen, move) in enumerate(rows, 1)])


class FindStockfishTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self.tmp.name)
        empty_path = self.dir / "bin"
        empty_path.mkdir()
        self.env = mock.patch.dict(os.environ, {"PATH": str(empty_path)})
        self.env.start()
        for name in ("STOCKFISH", "STOCKFISH_PATH"):
            os.environ.pop(name, None)
        self.root = mock.patch.object(positions, "PROJECT_ROOT", self.dir)
        self.root.start()

    def tearDown(self):
        self.root.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_absent(self):
        self.assertIsNone(positions.find_stockfish())

    def test_configured_path_that_does_not_exist(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertIsNone(positions.find_stockfish(str(self.dir / "nope.exe")))
            os.environ["STOCKFISH"] = str(self.dir / "also-nope")
            self.assertIsNone(positions.find_stockfish())
        self.assertIn("is not an executable file", out.getvalue())

    def test_argument_then_environment_then_path(self):
        binary = self.dir / "sf.bin"
        binary.write_text("")
        other = self.dir / "other.bin"
        other.write_text("")
        os.environ["STOCKFISH"] = str(other)
        self.assertEqual(positions.find_stockfish(str(binary)), str(binary))
        self.assertEqual(positions.find_stockfish(), str(other))
        del os.environ["STOCKFISH"]
        on_path = self.dir / "bin" / ("stockfish.exe" if os.name == "nt" else "stockfish")
        on_path.write_text("")
        on_path.chmod(0o755)
        self.assertEqual(Path(positions.find_stockfish()).resolve(), on_path.resolve())


class StockfishScoreTest(unittest.TestCase):
    def test_engine_scale(self):
        self.assertEqual(positions.stockfish_score(chess.engine.Cp(-35)), -35)
        self.assertEqual(positions.stockfish_score(chess.engine.Mate(3)), 100000 - 5)
        self.assertEqual(positions.stockfish_score(chess.engine.Mate(-2)), -(100000 - 4))
        self.assertEqual(positions.stockfish_score(chess.engine.Mate(0)), -100000)
        self.assertIsNone(positions.stockfish_score(None))


class GroundTruthTest(unittest.TestCase):
    def test_eval_is_from_the_side_to_move_and_best_move_from_the_pv(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            log = Path(tmp) / "uci.log"
            fens = [WHITE_TO_MOVE, BLACK_TO_MOVE, MATES, MATED]
            with duckdb.connect() as cnxn, mock.patch.dict(os.environ, {"FAKE_UCI_LOG": str(log)}), \
                    contextlib.redirect_stdout(io.StringIO()):
                _search_table(cnxn, fens)
                updated = positions.update_search_stats_with_stockfish(
                    cnxn, [sys.executable, str(FAKE_ENGINE)], depth=3, batch_size=3)
                rows = cnxn.execute("SELECT fen, sf_eval, sf_best_move, sf_pv, sf_eval_pov, sf_time_ms "
                                    "FROM search_stats ORDER BY id").fetchall()
            commands = log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(updated, 4)
        self.assertEqual([r[1] for r in rows], [57, 57, 100000 - 3, -(100000 - 2)])
        for fen, _, best, pv, pov, elapsed in rows:
            self.assertEqual(best, _fake_best_move(fen))
            self.assertEqual(json.loads(pv)[0], best)
            self.assertEqual(pov, "stm")
            self.assertGreaterEqual(elapsed, 0)
        # one analysis per position (the table has no engine moves to score alone)
        self.assertEqual(sum(1 for c in commands if c.startswith("go")), len(fens))

    def test_legacy_white_view_evaluations_are_converted_once(self):
        with duckdb.connect() as cnxn, contextlib.redirect_stdout(io.StringIO()):
            _search_table(cnxn, [WHITE_TO_MOVE, BLACK_TO_MOVE])
            cnxn.execute("ALTER TABLE search_stats ADD COLUMN sf_eval INTEGER")
            cnxn.execute("UPDATE search_stats SET sf_eval = 40")       # White's view, as older versions stored
            positions.ensure_stockfish_columns(cnxn)
            self.assertEqual(positions.convert_legacy_stockfish_evals(cnxn), 2)
            self.assertEqual(positions.convert_legacy_stockfish_evals(cnxn), 0)
            rows = cnxn.execute("SELECT sf_eval, sf_eval_pov FROM search_stats ORDER BY id").fetchall()
        self.assertEqual(rows, [(40, "stm"), (-40, "stm")])


class StockfishLinesTest(unittest.TestCase):
    """Stockfish's top lines per position (stockfish_moves) and the engine's move among them."""
    ENGINE = [sys.executable, str(FAKE_ENGINE)]
    SF = "SELECT id, sf_eval, sf_best_move, sf_depth, sf_multipv, sf_move_rank, sf_move_eval FROM search_stats ORDER BY id"
    LINES = "SELECT fen, move, rank, eval, pv, depth, multipv, stockfish FROM stockfish_moves ORDER BY ALL"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.log = Path(tmp.name) / "uci.log"
        self.log.touch()
        white, black = _fake_order(WHITE_TO_MOVE), _fake_order(BLACK_TO_MOVE)
        self.white, self.black = white, black
        self.rows = [(WHITE_TO_MOVE, white[0]), (WHITE_TO_MOVE, white[2]), (WHITE_TO_MOVE, white[7]),
                     (WHITE_TO_MOVE, "0000"), (WHITE_TO_MOVE, None), (BLACK_TO_MOVE, black[1])]

    def analyse(self, cnxn, **kwargs):
        with mock.patch.dict(os.environ, {"FAKE_UCI_LOG": str(self.log)}), contextlib.redirect_stdout(io.StringIO()):
            return positions.update_search_stats_with_stockfish(cnxn, self.ENGINE, **kwargs)

    def go_commands(self):
        return [c for c in self.log.read_text(encoding="utf-8").splitlines() if c.startswith("go")]

    def test_lines_ranks_and_evals_of_the_engine_moves(self):
        with duckdb.connect() as cnxn:
            _moves_table(cnxn, self.rows)
            self.assertEqual(self.analyse(cnxn, depth=3, multipv=3), 6)
            sf = cnxn.execute(self.SF).fetchall()
            lines = cnxn.execute(self.LINES).fetchall()
        w, b = self.white, self.black
        self.assertEqual(sf, [
            (1, 57, w[0], 3, 3, 1, 57),       # Stockfish's first line
            (2, 57, w[0], 3, 3, 3, 37),       # its third line
            (3, 57, w[0], 3, 3, 0, -13),      # outside the three lines: scored alone
            (4, 57, w[0], 3, 3, None, None),  # not a legal move
            (5, 57, w[0], 3, 3, None, None),  # no move
            (6, 57, b[0], 3, 3, 2, 47),
        ])
        ranked = [(fen, move, rank, value) for fen, move, rank, value, *_ in lines if rank is not None]
        self.assertEqual(sorted(ranked), sorted([(WHITE_TO_MOVE, w[p], p + 1, 57 - 10 * p) for p in range(3)]
                                                + [(BLACK_TO_MOVE, b[p], p + 1, 57 - 10 * p) for p in range(3)]))
        alone = [row for row in lines if row[2] is None]
        self.assertEqual([(fen, move, value, depth, multipv) for fen, move, _, value, _, depth, multipv, _ in alone],
                         [(WHITE_TO_MOVE, w[7], -13, 3, None)])
        for fen, move, rank, value, pv, depth, multipv, name in lines:
            self.assertEqual(json.loads(pv)[0], move)
            self.assertEqual(name, "FakeFish")
        # one MultiPV analysis per position, plus one search of the move outside the lines
        commands = self.go_commands()
        self.assertEqual(len(commands), 3)
        self.assertEqual(sum(1 for c in commands if f"searchmoves {w[7]}" in c), 1)

    def test_results_are_cached_and_restored_after_a_reload(self):
        with duckdb.connect() as cnxn, contextlib.redirect_stdout(io.StringIO()):
            _moves_table(cnxn, self.rows)
            self.analyse(cnxn, depth=3, multipv=3)
            first = cnxn.execute(self.SF).fetchall()
            self.assertEqual(self.analyse(cnxn, depth=3, multipv=3), 0)
            self.assertEqual(len(self.go_commands()), 3)            # nothing analysed twice

            # load_analytics --full recreates search_stats without the sf_* values: no engine needed
            cnxn.execute("CREATE OR REPLACE TABLE search_stats AS SELECT id, fen, move, game_id, sts_id FROM search_stats")
            positions.run(cnxn, skip_stockfish=True)
            self.assertEqual(cnxn.execute(self.SF).fetchall(), first)
            self.assertEqual(len(self.go_commands()), 3)

            # a new engine move in an analysed position is scored alone, without a new MultiPV search
            cnxn.execute("INSERT INTO search_stats (id, fen, move) VALUES (7, ?, ?)", [WHITE_TO_MOVE, self.white[9]])
            self.analyse(cnxn, depth=3, multipv=3)
            commands = self.go_commands()
            self.assertEqual(len(commands), 4)
            self.assertIn(f"searchmoves {self.white[9]}", commands[-1])
            self.assertEqual(cnxn.execute(self.SF + " LIMIT 1 OFFSET 6").fetchone()[5:], (0, -33))

            # a deeper or wider analysis replaces the lines and the single-move scores
            self.analyse(cnxn, depth=4, multipv=3)
            self.assertEqual(len(self.go_commands()), 4 + 2 + 2)
            self.assertEqual(cnxn.execute("SELECT DISTINCT depth FROM stockfish_moves").fetchall(), [(4,)])
            self.assertEqual(cnxn.execute("SELECT count(*) FROM stockfish_moves").fetchone()[0], 3 + 3 + 2)
            self.analyse(cnxn, depth=4, multipv=2)                  # narrower: nothing to do
            self.assertEqual(len(self.go_commands()), 8)

    def test_parallel_processes_give_the_same_results(self):
        results = []
        for workers in (1, 3):
            with duckdb.connect() as cnxn:
                _moves_table(cnxn, self.rows + [(MATES, None), (MATED, None)])
                self.analyse(cnxn, depth=3, multipv=4, workers=workers, batch_size=1)
                results.append((cnxn.execute(self.SF).fetchall(), cnxn.execute(self.LINES).fetchall()))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(results[0][1]), 4 * 4 + 1)

    def test_invalid_positions_are_skipped(self):
        with duckdb.connect() as cnxn:
            _moves_table(cnxn, [("not a fen", None), ("8/8/8/8/8/8/8/8 w - - 0 1", None), (WHITE_TO_MOVE, None)])
            self.analyse(cnxn, depth=3, multipv=2)
            self.assertEqual(cnxn.execute("SELECT count(DISTINCT fen) FROM stockfish_moves").fetchone()[0], 1)
            self.assertEqual(cnxn.execute("SELECT count(sf_eval) FROM search_stats").fetchone()[0], 1)
        self.assertEqual(len(self.go_commands()), 1)

    def test_a_crashing_engine_is_restarted(self):
        results = []
        for crash_on in ("0", "3"):     # every engine process dies on its third search (tasks need at most two)
            with duckdb.connect() as cnxn, mock.patch.dict(os.environ, {"FAKE_UCI_CRASH_ON_GO": crash_on}):
                _moves_table(cnxn, self.rows + [(MATES, None), (MATED, None)])
                self.analyse(cnxn, depth=3, multipv=3, workers=1)
                results.append((cnxn.execute(self.SF).fetchall(), cnxn.execute(self.LINES).fetchall()))
        self.assertEqual(results[0], results[1])

    def test_an_engine_that_does_not_start_leaves_the_rest_of_the_step_working(self):
        broken = [sys.executable, "-c", "import sys; sys.exit(3)"]
        with duckdb.connect() as cnxn, contextlib.redirect_stdout(io.StringIO()) as out, \
                mock.patch.object(positions, "find_stockfish", return_value=broken):
            _moves_table(cnxn, self.rows)
            positions.run(cnxn)
            self.assertEqual(cnxn.execute("SELECT count(*) FROM position_features").fetchone()[0], len(self.rows))
            self.assertEqual(cnxn.execute("SELECT count(*) FROM stockfish_moves").fetchone()[0], 0)
        self.assertIn("cannot start", out.getvalue())

    def test_moves_are_scored_at_the_depth_of_their_lines_and_analyses_never_shrink(self):
        with duckdb.connect() as cnxn, contextlib.redirect_stdout(io.StringIO()):
            _moves_table(cnxn, self.rows)
            self.analyse(cnxn, depth=4, multipv=3)
            # asking for less depth but more lines analyses again at depth 4 with 5 lines
            self.analyse(cnxn, depth=3, multipv=5)
            self.assertEqual(cnxn.execute("SELECT DISTINCT depth, multipv FROM stockfish_moves "
                                          "WHERE rank IS NOT NULL").fetchall(), [(4, 5)])
            # a new move in an analysed position is scored at the depth of that position's lines
            cnxn.execute("INSERT INTO search_stats (id, fen, move) VALUES (7, ?, ?)", [WHITE_TO_MOVE, self.white[9]])
            self.analyse(cnxn, depth=3, multipv=5)
            self.assertEqual(self.go_commands()[-1], f"go depth 4 searchmoves {self.white[9]}")
            self.assertEqual(cnxn.execute("SELECT DISTINCT depth FROM stockfish_moves").fetchall(), [(4,)])

    def test_engines_without_multipv_or_that_ignore_searchmoves(self):
        with duckdb.connect() as cnxn, contextlib.redirect_stdout(io.StringIO()) as out, \
                mock.patch.dict(os.environ, {"FAKE_UCI_NO_MULTIPV": "1"}), \
                mock.patch.object(positions, "find_stockfish", return_value=self.ENGINE):
            _moves_table(cnxn, self.rows)
            positions.run(cnxn, multipv=3)
            self.assertEqual(cnxn.execute("SELECT count(*) FROM stockfish_moves").fetchone()[0], 0)
        self.assertIn("has no MultiPV option", out.getvalue())
        with duckdb.connect() as cnxn, mock.patch.dict(os.environ, {"FAKE_UCI_IGNORE_SEARCHMOVES": "1"}):
            _moves_table(cnxn, self.rows)
            self.analyse(cnxn, depth=3, multipv=3)
            # asked for one move, the engine searched another: outside the lines, loss unknown
            self.assertEqual(cnxn.execute("SELECT sf_move_rank, sf_move_eval FROM search_stats WHERE id = 3").fetchone(),
                             (0, None))

    def test_an_engine_that_never_answers_is_given_up(self):
        silent = [sys.executable, "-c", "import time; time.sleep(60)"]
        started = time.perf_counter()
        with mock.patch.object(positions, "HANDSHAKE_TIMEOUT", 1.0), self.assertRaises(positions.EngineTerminated):
            positions.UciEngine(silent)
        self.assertLess(time.perf_counter() - started, 20)

    def test_an_interrupted_write_is_rolled_back(self):
        class Interrupted:
            """A connection that is interrupted (Ctrl+C) when the rows are copied into stockfish_moves."""
            def __init__(self, conn):
                self.conn = conn

            def execute(self, sql, *args):
                if sql.lstrip().startswith("INSERT INTO stockfish_moves"):
                    raise KeyboardInterrupt
                return self.conn.execute(sql, *args)

        row = (WHITE_TO_MOVE, "e2e3", 1, 10, '["e2e3"]', 3, 1, 1.0)
        with duckdb.connect() as cnxn:
            positions.ensure_stockfish_tables(cnxn)
            with self.assertRaises(KeyboardInterrupt):
                positions._write_stockfish_rows(Interrupted(cnxn), [WHITE_TO_MOVE], [row], "FakeFish", None)
            positions._write_stockfish_rows(cnxn, [WHITE_TO_MOVE], [row], "FakeFish", None)  # no transaction left open
            self.assertEqual(cnxn.execute("SELECT count(*) FROM stockfish_moves").fetchone()[0], 1)


class PositionFeaturesTest(unittest.TestCase):
    def test_run_without_stockfish_builds_features(self):
        with duckdb.connect() as cnxn, contextlib.redirect_stdout(io.StringIO()) as out, \
                mock.patch.object(positions, "find_stockfish", return_value=None):
            _search_table(cnxn, [WHITE_TO_MOVE, BLACK_TO_MOVE, "not a fen"])
            positions.run(cnxn)
            features = cnxn.execute("SELECT search_id, game_phase, position_type FROM position_features "
                                    "ORDER BY search_id").fetchall()
            self.assertEqual(cnxn.execute("SELECT count(sf_eval) FROM search_stats").fetchone()[0], 0)
            positions.run(cnxn)  # incremental: nothing new
            self.assertEqual(cnxn.execute("SELECT count(*) FROM position_features").fetchone()[0], 3)
        self.assertIn("stockfish: not found", out.getvalue())
        self.assertEqual([f[0] for f in features], [1, 2, 3])
        self.assertEqual(features[0][1:], ("opening", features[0][2]))
        self.assertEqual(features[2][1:], (None, "Unknown"))

    def test_empty_database_still_gets_the_table(self):
        with duckdb.connect() as cnxn, contextlib.redirect_stdout(io.StringIO()):
            positions.run(cnxn)
            self.assertEqual(cnxn.execute("SELECT count(*) FROM position_features").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
