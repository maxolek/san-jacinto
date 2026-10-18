"""Tests for tests/mistakes.py.

Run from the repo root: python -m unittest discover -s tests/unit -t .
The Stockfish test runs only when STOCKFISH_PATH points to a Stockfish binary.
"""
import contextlib
import csv
import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from tests import mistakes, sts

STOCKFISH = os.environ.get("STOCKFISH_PATH", "")

# after 1. e4 e5 2. Nf3, black to move: 2...Qg5?? hangs the queen to Nxg5
QG5_FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
# after 1. e4, black to move (python-chess leaves out the e3 square: nobody can take en passant)
E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def sample_mistake(fen=QG5_FEN, played="d8g5", good=("b8c6", "d7d6", "g8f6"), **overrides):
    """A Mistake as the miner builds it, with SAN from python-chess."""
    board = chess.Board(fen)
    fields = dict(
        key=board.epd(), fen=board.fen(),
        bm=[board.san(chess.Move.from_uci(uci)) for uci in good],
        am=board.san(chess.Move.from_uci(played)),
        best=board.san(chess.Move.from_uci(good[0])),
        loss=44.4, cp_loss=812, best_cp=-30, played_cp=-842, depth=18,
        player="Blunderer", source=Path("logs/sprt_results/run/games.pgn"), game_index=7,
        ply=board.ply(), phase=mistakes.classify_phase(board, board.ply()),
        white="Tester", black="Blunderer", game_round="4",
    )
    fields.update(overrides)
    return mistakes.Mistake(**fields)


class WinPercentTest(unittest.TestCase):
    def test_even_position_is_fifty(self):
        self.assertEqual(mistakes.win_percent(0), 50)

    def test_symmetric(self):
        for cp in (1, 35, 100, 250, 999, 1000, 4000):
            with self.subTest(cp=cp):
                self.assertAlmostEqual(mistakes.win_percent(cp) + mistakes.win_percent(-cp), 100)

    def test_monotone_and_clipped_at_1000(self):
        values = [mistakes.win_percent(cp) for cp in range(-1000, 1001, 5)]
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))
        self.assertEqual(mistakes.win_percent(1500), mistakes.win_percent(1000))
        self.assertEqual(mistakes.win_percent(-5000), mistakes.win_percent(-1000))

    def test_lichess_reference_value(self):
        self.assertAlmostEqual(mistakes.win_percent(100), 59.1, delta=0.05)

    def test_mate_scores_count_as_clipped_evals(self):
        mate = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
        self.assertEqual(mistakes.score_cp(mate, chess.WHITE), 9997)
        self.assertEqual(mistakes.score_cp(mate, chess.BLACK), -9997)
        self.assertEqual(mistakes.win_percent(9997), mistakes.win_percent(1000))


class PhaseTest(unittest.TestCase):
    def test_opening_then_middlegame_by_ply(self):
        board = chess.Board()
        self.assertEqual(mistakes.classify_phase(board, 0), "opening")
        self.assertEqual(mistakes.classify_phase(board, 19), "opening")
        self.assertEqual(mistakes.classify_phase(board, 20), "middlegame")

    def test_endgame_by_pieces_other_than_pawns_and_kings(self):
        six = chess.Board("r3k2r/pppp4/8/8/8/8/PPPP4/RN1QK2R w - - 0 1")
        seven = chess.Board("rn2k2r/8/8/8/8/8/8/RN1QK2R w - - 0 1")
        self.assertEqual(mistakes.classify_phase(six, 0), "endgame")
        self.assertEqual(mistakes.classify_phase(seven, 10), "opening")
        self.assertEqual(mistakes.classify_phase(seven, 60), "middlegame")


class PlayerSelectionTest(unittest.TestCase):
    def test_players_and_exclusion(self):
        default = "(?i)stockfish"
        self.assertTrue(mistakes.is_selected("Candidate", None, default))
        self.assertFalse(mistakes.is_selected("Stockfish 17", None, default))
        self.assertFalse(mistakes.is_selected("stockfish-dev", None, default))
        self.assertTrue(mistakes.is_selected("Candidate", ["Candidate"], default))
        self.assertFalse(mistakes.is_selected("Baseline", ["Candidate"], default))
        self.assertTrue(mistakes.is_selected("Stockfish 17", None, ""))


class EpdLineTest(unittest.TestCase):
    def test_exact_format(self):
        key = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        self.assertEqual(
            mistakes.format_epd_line(sample_mistake()),
            f'{key} bm Nc6 d6 Nf6; am Qg5; id "mistake.{digest}"; '
            'c0 "loss=44.4 cp_loss=812 best=Nc6 played=Qg5 depth=18 player=Blunderer '
            'game=games.pgn#7 ply=3 phase=opening";')

    def test_round_trip_through_sts(self):
        m = sample_mistake()
        fen, ops, bm, am = sts.parse_epd_line(mistakes.format_epd_line(m))
        board = chess.Board(fen)
        self.assertEqual(board.epd(), m.key)
        bm_uci = sts.moves_to_uci(board, bm)
        self.assertEqual(bm_uci, ["b8c6", "d7d6", "g8f6"])
        # the played move is only an am: sts never counts it as a solution
        self.assertNotIn("d8g5", bm_uci)
        self.assertEqual(am, [])
        self.assertEqual(ops["id"], mistakes.entry_id(m.key))
        self.assertTrue(ops["c0"].startswith("loss=44.4 cp_loss=812 best=Nc6 played=Qg5"))

    def test_castling_check_promotion_and_mate_san_round_trip(self):
        cases = [
            ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
             "f3e5", ("e1g1", "c4f7"), ["O-O", "Bxf7+"]),
            ("8/P6k/8/8/8/8/6K1/8 w - - 0 1", "g2f3", ("a7a8q",), ["a8=Q"]),
            ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "g1f1", ("a1a8",), ["Ra8#"]),
        ]
        for fen, played, good, san in cases:
            with self.subTest(fen=fen):
                m = sample_mistake(fen=fen, played=played, good=good)
                self.assertEqual(m.bm, san)
                parsed_fen, _, bm, _ = sts.parse_epd_line(mistakes.format_epd_line(m))
                self.assertEqual(sts.moves_to_uci(chess.Board(parsed_fen), bm), list(good))

    def test_c0_values_cannot_break_the_line(self):
        m = sample_mistake(player='Toma "hawk"; 1.2', source=Path("my games;1.pgn"))
        _, ops, bm, _ = sts.parse_epd_line(mistakes.format_epd_line(m))
        self.assertEqual(bm, ["Nc6", "d6", "Nf6"])
        self.assertEqual(ops["id"], mistakes.entry_id(m.key))
        self.assertIn("player=Toma_hawk_1.2 game=my_games_1.pgn#7", ops["c0"])

class SelectionTest(unittest.TestCase):
    def test_dedupe_keeps_the_largest_loss_per_position(self):
        a = sample_mistake(loss=20.0, game_index=1)
        b = sample_mistake(fen=E4_FEN, played="f7f6", good=("e7e5",), loss=30.0, game_index=1)
        c = sample_mistake(loss=35.0, game_index=2)  # same position as a
        self.assertEqual(mistakes.dedupe_by_position([a, b, c]), [b, c])

    def test_cap_per_game_keeps_the_largest_losses_in_order(self):
        game1 = [sample_mistake(loss=loss, game_index=1, ply=ply)
                 for ply, loss in enumerate([20.0, 50.0, 30.0, 40.0])]
        game2 = [sample_mistake(loss=10.0, game_index=2)]
        capped = mistakes.cap_per_game(game1 + game2, 3)
        self.assertEqual([m.loss for m in capped], [50.0, 30.0, 40.0, 10.0])
        self.assertEqual(len(mistakes.cap_per_game(game1 + game2, 0)), 5)


class SuiteMergeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.suite = Path(tmp.name) / "suite" / "mistakes.epd"
        self.csv = self.suite.with_suffix(".csv")

    def test_merge_appends_only_new_positions(self):
        self.suite.parent.mkdir()
        # another suite's entry for the 1. e4 position, written with the en passant
        # square, and no newline at the end of the file
        existing = ("# hand-written entries\n"
                    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 bm e5; id "old.1";')
        self.suite.write_text(existing, encoding="utf-8")
        known = sample_mistake(fen=E4_FEN, played="f7f6", good=("e7e5",))
        new = sample_mistake()
        repeated = sample_mistake(loss=20.0)

        self.assertEqual(mistakes.merge_into_suite(self.suite, [known, new, repeated]), [new])
        self.assertEqual(self.suite.read_text(encoding="utf-8"),
                         existing + "\n" + mistakes.format_epd_line(new) + "\n")

        with open(self.csv, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["id"], mistakes.entry_id(new.key))
        self.assertEqual((row["fen"], row["bm"], row["am"], row["best"], row["played"]),
                         (QG5_FEN, "Nc6 d6 Nf6", "Qg5", "Nc6", "Qg5"))
        self.assertEqual((row["loss"], row["cp_loss"], row["sf_best_cp"], row["played_cp"]),
                         ("44.4", "812", "-30", "-842"))
        self.assertEqual((row["player"], row["game"], row["ply"], row["phase"], row["depth"]),
                         ("Blunderer", "games.pgn#7", "3", "opening", "18"))
        self.assertEqual((row["source"], row["white"], row["black"], row["round"]),
                         (str(new.source), "Tester", "Blunderer", "4"))

        # merging the same mistakes again changes nothing
        suite_before = self.suite.read_text(encoding="utf-8")
        csv_before = self.csv.read_text(encoding="utf-8")
        self.assertEqual(mistakes.merge_into_suite(self.suite, [known, new]), [])
        self.assertEqual(self.suite.read_text(encoding="utf-8"), suite_before)
        self.assertEqual(self.csv.read_text(encoding="utf-8"), csv_before)

    def test_merge_creates_the_directory_and_files(self):
        self.assertEqual(mistakes.merge_into_suite(self.suite, [sample_mistake()]), [sample_mistake()])
        self.assertEqual(self.suite.read_text(encoding="utf-8").splitlines(),
                         [mistakes.format_epd_line(sample_mistake())])
        self.assertEqual(self.csv.read_text(encoding="utf-8").splitlines()[0], ",".join(mistakes.CSV_FIELDS))


class CollectPgnFilesTest(unittest.TestCase):
    def test_files_and_glob_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [root / "20260101_000000", root / "20260102_000000"]
            for run in runs:
                run.mkdir()
                (run / "games.pgn").write_text("", encoding="utf-8")
            (root / "notes.txt").write_text("", encoding="utf-8")
            patterns = [str(runs[0] / "games.pgn"), str(root / "*" / "games.pgn"), str(root / "none*.pgn")]
            with contextlib.redirect_stdout(io.StringIO()) as log:
                files = mistakes.collect_pgn_files(patterns)
            self.assertEqual(files, [(run / "games.pgn").resolve() for run in runs])
            self.assertIn("no PGN files match", log.getvalue())


class FakeEngine:
    """Stands in for SimpleEngine.analyse with fixed evals.

    evals: position key -> cp for the side to move (screening).
    move_evals: position key -> {uci: cp for the mover} (verification).
    """

    def __init__(self, evals=None, move_evals=None):
        self.evals = evals or {}
        self.move_evals = move_evals or {}
        self.analysed = []

    def analyse(self, board, limit, *, multipv=None, root_moves=None, game=None, info=None):
        key = board.epd()
        self.analysed.append(key)
        if multipv is None and root_moves is None:
            return {"score": chess.engine.PovScore(chess.engine.Cp(self.evals[key]), board.turn)}
        lines = sorted(self.move_evals[key].items(), key=lambda item: -item[1])
        if root_moves:
            lines = [(uci, cp) for uci, cp in lines if chess.Move.from_uci(uci) in root_moves]
        infos = [{"pv": [chess.Move.from_uci(uci)],
                  "score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn)} for uci, cp in lines]
        return infos[:multipv] if multipv else infos[0]


GAME_PGN = """[White "Tester"]
[Black "Blunderer"]

1. e4 {book} e5 {book} 2. Nf3 {+0.30/10 0.1s} Qg5 {-0.30/10 0.1s} 3. Nxg5 {+9.00/10 0.1s} 1-0
"""


class ScreenAndVerifyTest(unittest.TestCase):
    def test_game_moves_mark_book_moves_and_players(self):
        moves, final = mistakes.game_moves(chess.pgn.read_game(io.StringIO(GAME_PGN)))
        self.assertEqual([m.book for m in moves], [True, True, False, False, False])
        self.assertEqual([m.player for m in moves], ["Tester", "Blunderer"] * 2 + ["Tester"])
        self.assertEqual([m.ply for m in moves], [0, 1, 2, 3, 4])
        self.assertEqual(moves[3].board.fen(), QG5_FEN)
        self.assertEqual(final.move_stack, [])

    def test_fen_header_sets_the_start_position(self):
        pgn = f'[White "A"]\n[Black "B"]\n[SetUp "1"]\n[FEN "{QG5_FEN}"]\n\n2... Qg5 3. Nxg5 *\n'
        moves, _ = mistakes.game_moves(chess.pgn.read_game(io.StringIO(pgn)))
        first = moves[0]
        self.assertEqual((first.board.fen(), first.ply, first.player), (QG5_FEN, 3, "B"))

    def test_screen_uses_the_negated_eval_of_the_next_position(self):
        moves, final = mistakes.game_moves(chess.pgn.read_game(io.StringIO(GAME_PGN)))
        boards = [m.board for m in moves] + [final]
        # evals for the side to move; 2...Qg5 turns -30 into -900 for black
        cps = [30, -30, 30, -30, 900, -900]
        engine = FakeEngine(evals={board.epd(): cp for board, cp in zip(boards, cps)})
        candidates = mistakes.screen_game(engine, moves, final, [2, 3, 4],
                                          chess.engine.Limit(depth=1), 15, "game")
        self.assertEqual([(gm.ply, gm.move.uci()) for gm, _ in candidates], [(3, "d8g5")])
        self.assertAlmostEqual(candidates[0][1], mistakes.win_percent(-30) - mistakes.win_percent(-900))
        self.assertEqual(sorted(engine.analysed), sorted({b.epd() for b in boards[2:]}))

    def test_finished_positions_need_no_engine(self):
        mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(mistakes.evaluate(FakeEngine(), mated, None, "game"), -mistakes.MATE_SCORE)
        self.assertEqual(mistakes.evaluate(FakeEngine(), stalemate, None, "game"), 0)

    def verify(self, played, multipv=4, threshold=15):
        board = chess.Board(QG5_FEN)
        engine = FakeEngine(move_evals={board.epd(): {
            "b8c6": -30, "d7d6": -45, "g8f6": -50, "f7f6": -120, "d8g5": -900}})
        return mistakes.verify(engine, board, chess.Move.from_uci(played), 18, multipv, 25, threshold, "game")

    def test_verify_keeps_a_real_mistake(self):
        verdict = self.verify("d8g5")
        self.assertEqual(verdict.best.uci(), "b8c6")
        self.assertEqual([move.uci() for move in verdict.good], ["b8c6", "d7d6", "g8f6"])  # not f6: 90 cp worse
        self.assertEqual((verdict.best_cp, verdict.played_cp), (-30, -900))
        self.assertAlmostEqual(verdict.loss, mistakes.win_percent(-30) - mistakes.win_percent(-900))

    def test_verify_drops_good_moves_and_small_losses(self):
        self.assertIsNone(self.verify("d7d6"))              # within the margin of the best
        # good though outside the MultiPV lines: 15 cp worse (a low threshold isolates the margin rule)
        self.assertIsNone(self.verify("d7d6", multipv=1, threshold=1))
        self.assertIsNone(self.verify("f7f6"))              # 90 cp worse but only ~8 win% points
        self.assertIsNotNone(self.verify("f7f6", threshold=8))

    def test_verify_widens_the_search_while_every_line_is_good(self):
        # 2 lines are both good, so 4 are searched: every good move is listed in bm
        verdict = self.verify("d8g5", multipv=2)
        self.assertEqual([move.uci() for move in verdict.good], ["b8c6", "d7d6", "g8f6"])

    def test_verify_stops_widening_at_the_legal_move_count(self):
        board = chess.Board(QG5_FEN)
        evals = {move.uci(): -30 for move in board.legal_moves}
        engine = FakeEngine(move_evals={board.epd(): evals})
        self.assertIsNone(mistakes.verify(engine, board, chess.Move.from_uci("d8g5"), 18, 4, 25, 15, "game"))
        self.assertLessEqual(len(engine.analysed), 1 + (board.legal_moves.count() - 1).bit_length())


class CliTest(unittest.TestCase):
    def test_missing_stockfish_is_a_clear_error(self):
        missing = Path(tempfile.gettempdir()) / "no-such-dir" / "stockfish.exe"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exit_:
            mistakes.main(["--stockfish", str(missing), "--pgn", "games.pgn"])
        self.assertEqual(exit_.exception.code, 2)
        self.assertIn(f"Stockfish not found at {missing}", stderr.getvalue())


INTEGRATION_PGN = f"""[Event "mistakes test"]
[Round "1"]
[White "Tester"]
[Black "Blunderer"]
[Result "1-0"]
[SetUp "1"]
[FEN "{QG5_FEN}"]

2... Qg5 {{-0.25/12 0.050s}} 3. Nxg5 {{+9.10/12 0.050s}} 1-0

[Event "mistakes test"]
[Round "2"]
[White "Blunderer"]
[Black "Tester"]
[Result "0-1"]

1. e4 {{book}} e5 {{book}} 2. Qh5 {{book}} Nc6 {{book}} 3. Qxe5+ {{book}}
Nxe5 {{-9.00/12 0.050s}} 0-1

[Event "mistakes test"]
[Round "3"]
[White "Stockfish 17"]
[Black "Tester"]
[Result "0-1"]

1. e4 {{+0.30/12 0.050s}} e5 {{-0.30/12 0.050s}} 2. Qh5 {{+0.10/12 0.050s}}
Nc6 {{-0.10/12 0.050s}} 3. Qxe5+ {{-9.00/12 0.050s}} Nxe5 {{-9.00/12 0.050s}} 0-1
"""


@unittest.skipUnless(STOCKFISH and Path(STOCKFISH).is_file(),
                     "set STOCKFISH_PATH to a Stockfish binary to run")
class StockfishIntegrationTest(unittest.TestCase):
    def test_hung_queen_becomes_an_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            pgn = Path(tmp) / "games.pgn"
            pgn.write_text(INTEGRATION_PGN, encoding="utf-8")
            out = Path(tmp) / "suite" / "mistakes.epd"
            argv = ["--pgn", str(pgn), "--out", str(out), "--stockfish", STOCKFISH,
                    "--screen-depth", "8", "--verify-depth", "12"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mistakes.main(argv), 0)

            # only 2...Qg5: the queen hangs of game 2 (book) and game 3 (Stockfish) are not mined
            lines = out.read_text(encoding="utf-8").splitlines()
            self.assertEqual([line.split(" bm ")[0] for line in lines], [chess.Board(QG5_FEN).epd()])
            self.assertIn("; am Qg5;", lines[0])
            fen, ops, bm, _ = sts.parse_epd_line(lines[0])
            bm_uci = sts.moves_to_uci(chess.Board(fen), bm)
            self.assertTrue(bm_uci)
            self.assertNotIn("d8g5", bm_uci)
            self.assertIn("depth=12 player=Blunderer game=games.pgn#1 ply=3 phase=opening", ops["c0"])

            with open(out.with_suffix(".csv"), newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([(r["fen"], r["am"], r["player"], r["round"]) for r in rows],
                             [(QG5_FEN, "Qg5", "Blunderer", "1")])

            # mining the same games again adds nothing
            before = out.read_text(encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mistakes.main(argv), 0)
            self.assertEqual(out.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
