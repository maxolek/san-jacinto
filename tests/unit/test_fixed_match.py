"""utils/fixed_match.py: fixed-length matches of one or more pairings on the match runner.

Controller tests feed fake games: pairs counted in the order they started, excluded and
unfinished pairs replayed with their openings, pairings interleaved (and balanced over
core types), failing jobs and too many exclusions stopping the match, colour reversal,
games.jsonl as data/etl/match_games.py reads it, and the run header. The fastchess runs (skipped without
fastchess or python-chess) play tests/unit/mock_uci_engine.py: a match of two pairings,
an SMP ladder step (tests/smp.py, Threads 2 vs 1) and a net match (nnue/registry.py)
stored in a temporary database.
"""
import contextlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import cpu_topology, fixed_match  # noqa: E402
from utils.fixed_match import (FixedMatchController, MatchStopped, Pairing, normalize_pairings,  # noqa: E402
                               run_fixed_match, tc_label)
from utils.match_runner import EngineSpec, GameResult, find_fastchess, tc_arguments  # noqa: E402
from utils.openings import Opening  # noqa: E402

try:
    import chess
    import chess.pgn
except ImportError:  # pragma: no cover
    chess = None
try:
    FASTCHESS = find_fastchess()
except FileNotFoundError:
    FASTCHESS = None
MOCK = ROOT / "tests" / "unit" / "mock_uci_engine.py"
PINNING = cpu_topology.topology().supported
BOOK = "".join(f'[Event "?"]\n[Result "*"]\n\n{line} *\n\n' for line in (
    "1. e4 e5 2. Nf3 Nc6", "1. d4 d5 2. c4 e6", "1. c4 e5 2. Nc3 Nf6", "1. e4 c5 2. Nf3 d6"))


def quietly(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class Book:
    """Distinct numbered openings, like a shuffled OpeningBook."""
    format, plies, path, seed = "pgn", 4, "fake.pgn", 1

    def __init__(self):
        self.taken = 0

    def take(self, n):
        out = [Opening(self.taken + i, f"opening {self.taken + i}") for i in range(n)]
        self.taken += n
        return out

    def __len__(self):
        return 1000


class FakeJob:
    """What the runner hands the controller: a job of one JobSpec on one slot."""

    def __init__(self, job_id, spec, slot, log_path):
        self.id, self.spec, self.slot, self.log_path = job_id, spec, slot, log_path
        self.games, self.pairs_started, self.killed, self.returncode = [], 0, False, 0

    @property
    def pairs(self):
        return len(self.spec.openings)


P = SimpleNamespace(index=0, cpu=1, core_type="P", label="P/cpu01")
E = SimpleNamespace(index=1, cpu=8, core_type="E", label="E/cpu08")


def engines(*names):
    return [EngineSpec(name, f"{name}.exe", pin_option="") for name in names]


class Controller(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.log = self.dir / "job.log"
        self.log.write_text("Warning; cannot start engine\n")
        self.jobs = 0

    def controller(self, pairings, slots=(P,), **kwargs):
        kwargs.setdefault("quiet", True)
        ctl = FixedMatchController(pairings, Book(), list(slots), self.dir, **kwargs)
        self.addCleanup(ctl.close)  # cleanups run last-in first-out: before the folder is removed
        return ctl

    def job(self, ctl, slot=P):
        spec = ctl.next_job(slot)
        if spec is None:
            return None
        self.jobs += 1
        return FakeJob(self.jobs, spec, slot, self.log)

    def start(self, ctl, job, pair_no):
        """The runner saw "Started game" of the pair's first game."""
        job.pairs_started = max(job.pairs_started, pair_no)
        ctl.on_pair_start(job, pair_no)

    def play(self, ctl, job, game_no, result, core_mixed=False, white=None):
        first, second = (e.name for e in job.spec.engines)
        if white is None:  # the job's first engine has White in the first game of each pair
            white = first if game_no % 2 else second
        black = second if white == first else first
        env = dict(slot=job.slot.index, slot_label=job.slot.label, cpu=job.slot.cpu, core_type=job.slot.core_type,
                   core_mixed=core_mixed, mhz_mean=3000.0)
        game = GameResult(job=job, pair_no=(game_no + 1) // 2, game_no=game_no, white=white, black=black,
                          result=result, reason="White mates" if result == "1-0" else "Draw by adjudication",
                          started=1000.0, finished=1001.5, env=env)
        job.games.append(game)
        quietly(ctl.on_game, game)

    def play_pair(self, ctl, job, pair_no, results=("1-0", "1/2-1/2")):
        self.start(ctl, job, pair_no)
        for half, result in enumerate(results, 1):
            self.play(ctl, job, 2 * pair_no - 2 + half, result)

    def end(self, ctl, job, returncode=0):
        job.returncode = returncode
        return quietly(ctl.on_job_end, job)[1]

    def records(self):
        return [json.loads(line) for line in (self.dir / "games.jsonl").read_text().splitlines()]

    def test_pairs_are_counted_in_the_order_they_started(self):
        a, b = engines("A", "B")
        ctl = self.controller([Pairing(a, b, 8)], slots=(P, E), pairs_per_job=2)
        first, second = self.job(ctl, P), self.job(ctl, E)
        self.start(ctl, first, 1)   # pair 1 of the pairing
        self.start(ctl, second, 1)  # pair 2
        self.start(ctl, first, 2)   # pair 3
        self.play(ctl, second, 1, "1-0")
        self.play(ctl, second, 2, "0-1")  # pair 2 is complete first (A won both)...
        self.play(ctl, first, 3, "1-0")
        self.play(ctl, first, 4, "1-0")  # ... then pair 3 (one win each)
        state = ctl.pairings[0].state
        self.assertEqual(state.pairs_tested, 0, "a later pair must wait for the pair that started before it")
        self.play(ctl, first, 1, "1/2-1/2")
        self.assertEqual(state.pairs_tested, 0)
        self.play(ctl, first, 2, "1/2-1/2")
        self.assertEqual(state.pairs_tested, 3)
        self.assertEqual(state.counts, [0, 0, 2, 0, 1])  # DD, WW, WL in start order
        self.assertEqual(state.tally.detail["WL"], 1)
        self.assertEqual({core: t.pairs for core, t in ctl.pairings[0].by_core.items()}, {"P": 2, "E": 1})
        by_game = {r["game"]: r for r in self.records()}
        self.assertEqual(sorted(by_game), [1, 2, 3, 4, 5, 6])  # numbered by start order, not finish order
        self.assertEqual([by_game[n]["job"] for n in (1, 3, 5)], [first.id, second.id, first.id])
        self.assertEqual([by_game[n]["pairing_pair"] for n in (1, 3, 5)], [1, 2, 3])
        self.assertEqual((by_game[3]["candidate_score"], by_game[4]["candidate_score"]), (1.0, 1.0))
        self.assertEqual(by_game[4]["white"], "B")
        for key in ("run", "pair", "job_game", "termination", "seconds", "started_utc", "eco", "opening",
                    "core_type", "cpu", "core_mixed", "pairing", "pairing_game"):
            self.assertIn(key, by_game[1])

    def test_excluded_and_unfinished_pairs_are_played_again_with_their_openings(self):
        a, b = engines("A", "B")
        ctl = self.controller([Pairing(a, b, 3)], pairs_per_job=2)
        job = self.job(ctl)
        played = [o.index for o in job.spec.openings]
        self.play_pair(ctl, job, 1)
        self.start(ctl, job, 2)
        self.play(ctl, job, 3, "1-0", core_mixed=True)  # an engine left its core: the pair is excluded
        self.play(ctl, job, 4, "0-1")
        self.end(ctl, job)
        job = self.job(ctl)
        self.assertEqual([o.index for o in job.spec.openings], [played[1], played[1] + 1], "the replay first")
        self.play_pair(ctl, job, 1)
        self.start(ctl, job, 2)
        self.play(ctl, job, 3, "1-0")  # fastchess died in the middle of the pair
        out = self.end(ctl, job, returncode=1)
        self.assertIn("exited with code 1", out)
        self.assertIn("1 pair(s) excluded and played again", out)
        self.assertEqual(ctl.failures, 1)
        job = self.job(ctl)
        self.assertEqual([o.index for o in job.spec.openings], [played[1] + 1])
        self.play_pair(ctl, job, 1)
        self.end(ctl, job)
        self.assertEqual(ctl.failures, 0)
        self.assertIsNone(ctl.next_job(P))
        self.assertFalse(ctl.finished())
        result = ctl.result()["pairings"]["A vs B"]
        self.assertEqual((result["pairs"], result["pairs_requested"], result["complete"]), (3, 3, True))
        self.assertEqual((result["voided_pairs"], result["excluded_pairs"]), (2, 1))
        self.assertEqual(result["games_played"], 6)
        self.assertEqual(len(self.records()), 9)  # every finished game is recorded, counted or not

    def test_pairings_take_turns_and_share_the_core_types(self):
        a, b, c, d = engines("A", "B", "C", "D")
        ctl = self.controller([Pairing(a, b, 4, key="ab"), Pairing(c, d, 4, key="cd"), Pairing(a, d, 8, key="ad")],
                              pairs_per_job=2)
        order = []
        while (spec := ctl.next_job(P)) is not None:
            order.append((spec.meta["pairing"], len(spec.openings)))
        # the pairing furthest behind (share of its openings handed out) goes next
        self.assertEqual(order, [("ab", 2), ("cd", 2), ("ad", 2), ("ad", 2), ("ab", 2), ("cd", 2), ("ad", 2),
                                 ("ad", 2)])
        # on a tie, the pairing with fewer pairs on the idle slot's core type
        ctl = self.controller([Pairing(a, b, 4, key="ab"), Pairing(c, d, 4, key="cd")], slots=(P, E),
                              pairs_per_job=1)
        cores = {"ab": [], "cd": []}
        for slot in (P, E) * 4:
            cores[ctl.next_job(slot).meta["pairing"]].append(slot.core_type)
        self.assertEqual({k: sorted(v) for k, v in cores.items()}, {"ab": ["E", "E", "P", "P"],
                                                                    "cd": ["E", "E", "P", "P"]})

    def test_the_last_jobs_are_spread_over_the_slots(self):
        a, b = engines("A", "B")
        ctl = self.controller([Pairing(a, b, 10)], slots=(P, E, P, E), pairs_per_job=4)
        self.assertEqual([len(ctl.next_job(P).openings) for _ in range(4)], [3, 2, 2, 1])

    def test_jobs_failing_in_a_row_stop_the_match(self):
        a, b = engines("A", "B")
        ctl = self.controller([Pairing(a, b, 10)], pairs_per_job=2)
        for _ in range(3):  # max(3, slots)
            job = self.job(ctl)
            self.assertIsNotNone(job)
            out = self.end(ctl, job, returncode=1)
            self.assertIn("cannot start engine", out)
        self.assertIn("3 jobs in a row failed", ctl.stop_reason)
        self.assertTrue(ctl.finished())
        self.assertIsNone(ctl.next_job(P))
        self.assertEqual(len(ctl.pairings[0].queue), 10, "unplayed openings are kept")
        result = ctl.result()
        self.assertEqual((result["pairs"], result["error"]), (0, ctl.stop_reason))

    def test_too_many_excluded_pairs_stop_the_match(self):
        a, b = engines("A", "B")
        ctl = self.controller([Pairing(a, b, 6)], pairs_per_job=3, max_excluded=1)
        job = self.job(ctl)
        for pair in (1, 2):
            self.start(ctl, job, pair)
            self.play(ctl, job, 2 * pair - 1, "1-0", core_mixed=True)
        finished, out = quietly(ctl.finished)
        self.assertTrue(finished)
        self.assertIn("2 pairs had to be excluded (limit 1)", ctl.stop_reason)
        self.assertIn("stopping", out)
        self.assertIsNone(ctl.next_job(P))

    def test_a_pair_must_reverse_colours(self):
        a, b = engines("A", "B")
        ctl = self.controller([Pairing(a, b, 1)])
        job = self.job(ctl)
        self.start(ctl, job, 1)
        self.play(ctl, job, 1, "1-0")
        with self.assertRaisesRegex(ValueError, "did not reverse colors"):
            self.play(ctl, job, 2, "1-0", white="A")
        with self.assertRaisesRegex(ValueError, "Unexpected game identity"):
            self.play(ctl, job, 2, "1-0", white="X")

    def test_pairings_are_checked(self):
        a, b, c = engines("A", "B", "C")
        pairings = normalize_pairings([(a, b, 2), (a, c, 3, "ac"), Pairing(b, c, 0)])
        self.assertEqual([(p.key, p.pairs) for p in pairings], [("A vs B", 2), ("ac", 3), ("B vs C", 0)])
        for bad, error in (([(a, b, 2), (a, b, 3)], "distinct keys"), ([(a, a, 2)], "both engines are called"),
                           ([(a, b, -1)], "non-negative"), ([(a, b, 2.5)], "non-negative"), ([], "at least one")):
            with self.subTest(error), self.assertRaisesRegex(ValueError, error):
                normalize_pairings(bad)
        with self.assertRaises(TypeError):
            normalize_pairings([(a, b, 2, ("A", "B"))])
        # one name, one engine across the pairings (results, stored games and telemetry folders go by name)
        same_a = EngineSpec("A", "A.exe", pin_option="")
        self.assertEqual(len(normalize_pairings([(a, b, 1), (same_a, c, 1)])), 2)  # an equal spec is the same engine
        for other in (EngineSpec("A", "other.exe", pin_option=""), EngineSpec("A", "A.exe", ("--uci",)),
                      EngineSpec("A", "A.exe", options={"Hash": 64})):
            with self.subTest(other=other), \
                    self.assertRaisesRegex(ValueError, "pairings 'A vs B' and 'A vs C': two different engines are "
                                                       "called 'A'"):
                normalize_pairings([(a, b, 1), (other, c, 1)])
        with self.assertRaisesRegex(ValueError, "two different engines are called 'B'"):
            normalize_pairings([(a, b, 1), (c, EngineSpec("B", "B.exe", options={"Threads": 2}), 1)])
        self.assertEqual([tc_label(tc_arguments(**kw)) for kw in (dict(depth=8), dict(movetime=0.1),
                                                                    dict(nodes=5000), dict(tc="10+0.1"))],
                         ["depth=8", "st=0.1", "nodes=5000", "10+0.1"])

    @unittest.skipIf(chess is None, "python-chess is not installed (data/etl/match_games reads games.pgn with it)")
    def test_match_games_reads_the_games_of_every_pairing(self):
        from data.etl.match_games import game_rows
        a, b, c = engines("A", "B", "C")
        ctl = self.controller([Pairing(a, b, 1), Pairing(c, b, 1)], slots=(P, E), pairs_per_job=1)
        first, second = self.job(ctl, P), self.job(ctl, E)
        self.assertEqual((first.spec.meta["pairing"], second.spec.meta["pairing"]), ("A vs B", "C vs B"))
        self.start(ctl, first, 1)
        self.play_pair(ctl, second, 1, ("0-1", "1-0"))
        self.play(ctl, first, 1, "1/2-1/2")
        self.play(ctl, first, 2, "1-0")
        ctl.close()
        rows = game_rows(self.dir, {"A": 1, "B": 2, "C": 3})
        self.assertEqual(sorted((r["run_game"], r["pair_index"], r["white_engine_id"], r["black_engine_id"])
                                for r in rows), [(1, 1, 1, 2), (2, 1, 2, 1), (3, 2, 3, 2), (4, 2, 2, 3)])
        results = ctl.result()["pairings"]
        self.assertEqual(results["C vs B"]["candidate_losses"], 2)
        self.assertEqual(results["A vs B"]["pentanomial"], [0, 1, 0, 0, 0])  # a draw and a loss for A


@unittest.skipIf(chess is None, "python-chess is not installed (the mock engine needs it)")
class Stops(unittest.TestCase):
    """What run_fixed_match refuses before the run, and matches that stop (failing jobs, Ctrl-C)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "book.pgn").write_text(BOOK)
        self.results = self.dir / "results"

    def run_match(self, first_options=None, **kwargs):
        a = EngineSpec.from_path("A", MOCK, first_options)
        b = EngineSpec.from_path("B", MOCK)
        kwargs = dict(dict(tc_args=tc_arguments(depth=1), book=self.dir / "book.pgn", plies=4, concurrency=2,
                           pin=False, fastchess=sys.executable, results_dir=self.results, quiet=True), **kwargs)
        return quietly(run_fixed_match, [Pairing(a, b, 2)], **kwargs)

    def test_threads_the_engine_or_the_slots_cannot_take_are_refused(self):
        with self.assertRaisesRegex(ValueError, r"Threads: value 100 exceeds the engine's \[1, 64\]"):
            self.run_match({"Threads": 100})
        with self.assertRaisesRegex(ValueError, "searches with 4 threads, but games get 2 cpu"):
            self.run_match({"Threads": 4}, threads=2)
        self.assertFalse(self.results.exists())  # refused before anything was created

    def test_failing_jobs_end_the_match_with_an_error_and_its_results(self):
        if os.name == "nt":  # fastchess that cannot start a game: every job fails at once
            fake = self.dir / "fastchess.bat"
            fake.write_text("@exit /b 1\n")
        else:
            fake = self.dir / "fastchess"
            fake.write_text("#!/bin/sh\nexit 1\n")
            fake.chmod(0o755)
        with self.assertRaises(MatchStopped) as stopped:
            self.run_match(fastchess=fake)
        self.assertIn("jobs in a row failed", str(stopped.exception))
        result = stopped.exception.result
        self.assertEqual((result["pairs"], result["pairs_requested"]), (0, 2))
        run_dir, = self.results.iterdir()
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertIn("jobs in a row failed", summary["error"])
        self.assertEqual(json.loads((run_dir / "config.json").read_text())["meta"]["time_control"], "depth=1")

    def test_ctrl_c_writes_the_results_so_far_and_raises_them(self):
        from utils.match_runner import MatchRunner
        with mock.patch.object(MatchRunner, "run", side_effect=KeyboardInterrupt), \
                self.assertRaises(KeyboardInterrupt) as stopped:
            self.run_match(fastchess=sys.executable)
        self.assertIsInstance(stopped.exception, fixed_match.MatchInterrupted)
        result = stopped.exception.result
        self.assertEqual((result["interrupted"], result["error"], result["pairs"]), (True, "interrupted", 0))
        run_dir, = self.results.iterdir()
        self.assertEqual(json.loads((run_dir / "summary.json").read_text())["error"], "interrupted")

    def test_an_interrupted_smp_ladder_keeps_its_steps(self):
        from tests import smp
        calls = []

        def play(pairings, **kwargs):  # the first step completes, Ctrl-C in the second
            calls.append(kwargs["run_name"])
            pairing, = pairings
            snapshot = dict(pairs=len(calls), elo_diff=10.0 * len(calls), elo_low=0.0, elo_high=20.0 * len(calls),
                            nelo=5.0, nelo_low=0.0, nelo_high=10.0, candidate_wins=1, candidate_draws=0,
                            candidate_losses=0, pentanomial=[0, 0, 0, 1, 0], by_core={})
            result = dict(pairings={pairing.key: snapshot}, run_dir=str(Path(kwargs["results_dir"]) / calls[-1]))
            if len(calls) == 2:
                raise fixed_match.MatchInterrupted(result)
            return result

        argv = ["--engine", str(MOCK), "--threads", "1", "2", "4", "8", "--depth", "1", "--pairs", "2", "--no-pin",
                "--book", str(self.dir / "book.pgn"), "--results-dir", str(self.results)]
        with mock.patch.object(smp, "run_fixed_match", play), self.assertRaises(KeyboardInterrupt):
            quietly(smp.main, argv)
        self.assertEqual(calls, ["T2_vs_T1", "T4_vs_T2"])  # no third step after Ctrl-C
        run_root, = self.results.iterdir()
        steps = json.loads((run_root / "summary.json").read_text())["steps"]
        self.assertEqual([(s["threads_hi"], s["pairs"], s.get("interrupted", False)) for s in steps],
                         [(2, 1, False), (4, 2, True)])
        self.assertIn("(interrupted)", smp.format_report(steps, "depth=1"))


class Header(unittest.TestCase):
    """The run header: one line per distinct book, the machine/slot/fastchess lines only with show_setup."""

    def header(self, **kwargs):
        import copy
        from utils.match_runner import MatchRunner, Slot
        from utils.openings import OpeningBook
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            (tmp / "book.pgn").write_text(BOOK)
            book = OpeningBook(tmp / "book.pgn", plies=4, seed=7)
            baseline = EngineSpec("Baseline", sys.executable, pin_option="")
            # as a CMA-ES generation: every pairing its own copy of one shuffle
            pairings = [Pairing(EngineSpec(f"Candidate{i}", sys.executable, pin_option=""), baseline, 1,
                                book=copy.copy(book)) for i in range(1, 7)]
            with mock.patch.object(MatchRunner, "run", lambda runner, controller: None):
                _, out = quietly(run_fixed_match, pairings, tc_args=tc_arguments(depth=1), slots=[Slot(0, None, "-")],
                                 fastchess=sys.executable, results_dir=tmp / "results", report=False, **kwargs)
        return out.splitlines()

    def test_identical_books_are_one_line_and_the_setup_can_be_left_out(self):
        setup = ("[MATCH] machine ", "[MATCH] 1 concurrent games", "[MATCH] fastchess ")
        for show_setup in (True, False):
            lines = self.header(show_setup=show_setup)
            self.assertEqual([line for line in lines if line.startswith("[MATCH] book ")],
                             ["[MATCH] book book.pgn: 4 openings, 4 plies, seed 7 (6 pairings)"])
            self.assertEqual([any(line.startswith(s) for line in lines) for s in setup], [show_setup] * 3, lines)
            self.assertTrue(any(line.startswith("[MATCH] results ") for line in lines))


@unittest.skipIf(FASTCHESS is None, "fastchess is not built (libs/fastchess)")
@unittest.skipIf(chess is None, "python-chess is not installed")
class FastchessMatches(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.book = self.dir / "book.pgn"
        self.book.write_text(BOOK)
        self.mock_log = self.dir / "mock_log"
        patcher = mock.patch.object(sys, "stdin", io.StringIO(""))
        patcher.start()
        self.addCleanup(patcher.stop)

    def games(self, run_dir):
        return [json.loads(line) for line in (Path(run_dir) / "games.jsonl").read_text().splitlines()]

    def commands(self, run_dir):
        return [p.read_text().splitlines()[0] for p in (Path(run_dir) / "jobs").glob("job*.log")]

    def test_a_match_of_two_pairings(self):
        strong = EngineSpec.from_path("Strong", MOCK, {"Strength": 90})
        weak = EngineSpec.from_path("Weak", MOCK, {"Strength": 10})
        middle = EngineSpec.from_path("Middle", MOCK, {"Strength": 50})
        result, out = quietly(run_fixed_match, [Pairing(strong, weak, 3, key="strong-weak"), (middle, weak, 2)],
                              tc_args=tc_arguments(depth=1), book=self.book, plies=4, seed=3, concurrency=2,
                              pairs_per_job=2, extra_args=["-maxmoves", "40"], results_dir=self.dir / "results",
                              run_name="two", quiet=True, status_every=0)
        self.assertEqual(list(result["pairings"]), ["strong-weak", "Middle vs Weak"])
        for key, pairs in (("strong-weak", 3), ("Middle vs Weak", 2)):
            s = result["pairings"][key]
            self.assertEqual((s["pairs"], s["pairs_requested"], s["complete"], s["voided_pairs"]),
                             (pairs, pairs, True, 0), out)
            self.assertEqual(sum(s["pentanomial"]), pairs)
            self.assertEqual(s["candidate_wins"] + s["candidate_draws"] + s["candidate_losses"], 2 * pairs)
            self.assertEqual(sum(t["pairs"] for t in s["by_core"].values()), pairs)
            self.assertIn("pentanomial_detail", s)
        run_dir = Path(result["run_dir"])
        self.assertEqual(run_dir, self.dir / "results" / "two")
        self.assertEqual(result["games"], 10)
        games = self.games(run_dir)
        self.assertEqual(sorted(g["game"] for g in games), list(range(1, 11)))
        by_number = {g["game"]: g for g in games}
        for pair in range(1, 6):
            first, second = by_number[2 * pair - 1], by_number[2 * pair]
            self.assertEqual((first["pair"], second["pair"]), (pair, pair))
            self.assertEqual(first["pairing"], second["pairing"])
            self.assertEqual((first["job"], first["job_game"] + 1), (second["job"], second["job_game"]))
            self.assertEqual((first["white"], first["black"]), (second["black"], second["white"]))
            self.assertEqual(first["white"], {"strong-weak": "Strong", "Middle vs Weak": "Middle"}[first["pairing"]])
        for key, pairs in (("strong-weak", 3), ("Middle vs Weak", 2)):
            mine = sorted(g["pairing_game"] for g in games if g["pairing"] == key)
            self.assertEqual(mine, list(range(1, 2 * pairs + 1)))
        for g in games:
            self.assertFalse(g["core_mixed"])
            if PINNING:
                self.assertTrue(g["pinned"])
                self.assertIn(g["core_type"], ("P", "E"))
                self.assertEqual(g["affinity_limited"], os.name == "nt")
        jobs = {g["job"] for g in games}
        commands = self.commands(run_dir)
        self.assertEqual(len(commands), len(jobs))
        self.assertTrue(all("-recover" in c and "-maxmoves 40" in c for c in commands))
        tags = []
        with open(run_dir / "games.pgn", encoding="utf-8") as f:
            while (game := chess.pgn.read_game(f)) is not None:
                tags.append((int(game.headers["Job"]), int(game.headers["JobGame"])))
        self.assertEqual(sorted(tags), sorted((g["job"], g["job_game"]) for g in games))
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["pairings"]["strong-weak"]["pairs"], 3)
        meta = json.loads((run_dir / "config.json").read_text())["meta"]
        self.assertEqual((meta["time_control"], meta["book"]["seed"]), ("depth=1", 3))
        self.assertEqual([p["key"] for p in meta["pairings"]], ["strong-weak", "Middle vs Weak"])

    def test_an_smp_ladder_step(self):
        from tests import smp
        argv = ["--engine", str(MOCK), "--threads", "1", "2", "--depth", "1", "--pairs", "2", "--concurrency", "1",
                "--pairs-per-job", "2", "--book", str(self.book), "--book-depth", "4", "--option", "Hash=8",
                "--fastchess", str(FASTCHESS), "--results-dir", str(self.dir / "smp"), "--quiet"]
        with mock.patch.dict(os.environ, MOCK_ENGINE_LOG=str(self.mock_log)):
            steps, out = quietly(smp.main, argv)
        step, = steps
        self.assertEqual((step["threads_hi"], step["threads_lo"], step["pairs"]), (2, 1, 2), out)
        self.assertEqual(step["wins"] + step["draws"] + step["losses"], 4)
        self.assertIsNotNone(step["nelo"])
        self.assertIn("2 vs 1", out)
        run_root = Path(step["run_dir"]).parent
        self.assertEqual(Path(step["run_dir"]).name, "T2_vs_T1")
        self.assertEqual(json.loads((run_root / "summary.json").read_text())["steps"][0]["pairs"], 2)
        command, = self.commands(step["run_dir"])
        self.assertIn("name=T2", command)
        self.assertIn("option.Threads=2", command)
        self.assertIn("option.Threads=1", command)
        self.assertEqual(command.count("option.Hash=8"), 2)
        games = self.games(step["run_dir"])
        self.assertEqual(len(games), 4)
        self.assertEqual({g["white"] for g in games}, {"T2", "T1"})
        if PINNING:  # both engines of every game on the slot's two cpus, one core type
            cpus = {tuple(g["cpus"]) for g in games}
            self.assertEqual(len(cpus), 1)
            slot, = cpus
            self.assertEqual(len(slot), 2)
            for path in self.mock_log.glob("go_*.jsonl"):
                for line in path.read_text().splitlines():
                    record = json.loads(line)
                    self.assertEqual(tuple(record["allowed"]), tuple(sorted(slot)), record)
                    self.assertEqual(record["affinity_option"], ",".join(map(str, slot)))

    def test_a_net_match_is_stored(self):
        from data.databases.init_raw_db import init_raw_db
        from nnue import registry
        db, _ = quietly(init_raw_db, self.dir / "db")
        net_a, net_b = self.dir / "alpha.bin", self.dir / "beta.bin"
        net_a.write_bytes(b"net a")
        net_b.write_bytes(b"net b")
        argv = ["--db", str(db), "match", "--engine", str(MOCK), "--net-a", str(net_a), "--net-b", str(net_b),
                "--depth", "1", "--pairs", "2", "--concurrency", "2", "--net-option", "Version", "--net-value", "stem",
                "--book", str(self.book), "--fastchess", str(FASTCHESS), "--results-dir", str(self.dir / "nets")]
        _, out = quietly(registry.main, argv)
        self.assertIn("net A vs net B, depth=1: 2 pairs", out)
        cnxn = sqlite3.connect(db)
        self.addCleanup(cnxn.close)
        row = cnxn.execute("SELECT net_a_id, net_b_id, time_control, wins + draws + losses, penta_ll + penta_ld + "
                           "penta_dd + penta_wd + penta_ww, elo FROM net_matches").fetchall()
        self.assertEqual(len(row), 1, out)
        net_a_id, net_b_id, tc, games, pairs, elo = row[0]
        self.assertNotEqual(net_a_id, net_b_id)
        self.assertEqual((tc, games, pairs), ("depth=1", 4, 2))
        self.assertIsNotNone(elo)
        run_dir, = (self.dir / "nets").iterdir()
        command = self.commands(run_dir)[0]
        self.assertIn("option.Version=alpha", command)
        self.assertIn("option.Version=beta", command)
        self.assertEqual(len(self.games(run_dir)), 4)
        with self.assertRaisesRegex(ValueError, "does not declare the UCI option 'nnue_weight_file'"):
            quietly(registry.main, argv[:-2] + ["--net-option", "nnue_weight_file"])


if __name__ == "__main__":
    unittest.main()
