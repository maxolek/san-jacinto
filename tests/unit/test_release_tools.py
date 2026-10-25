"""tests/release.py against the tools it drives, and -recover in every fastchess command.

release.py builds the arguments of tests/sprt.py (a Namespace sprt.resolve_args completes)
and of tuning/spsa_tuner.py (through the tuner's parse_args); these tests check that every
field it sets is one of their flags with the same meaning (its tournament Namespace:
tests/unit/test_tournament.py). Every tool plays on the match runner, whose jobs carry
-recover so one engine crash does not end the whole run: checked here for the tournament's
and a CMA-ES generation's jobs (the runner's own command: test_match_runner.py; real games
of mock engines through utils/fixed_match.py: test_fixed_match.py). The tools default to
the fastchess binary the runner finds.
"""
import argparse
import contextlib
import io
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests import release, sprt, tournament  # noqa: E402
from tuning import spsa_tuner  # noqa: E402
from utils import fastchess_match  # noqa: E402

try:
    from tuning import cmaes_tuner
except (ImportError, SystemExit):  # the optional cma package
    cmaes_tuner = None


def release_args(*extra):
    args = release.build_parser().parse_args(["--version", "9.9.9", "--name", "n", "--description", "d", *extra])
    args.prod_engine, args.base_engine, args.dev_engine = "prod.exe", "base.exe", "dev.exe"
    return args


class SprtArguments(unittest.TestCase):
    def test_every_field_is_an_sprt_flag(self):
        ns = release._sprt_namespace(release_args(), 1, tc="10+0.1")
        flags = set(vars(sprt.build_parser().parse_args([])))
        self.assertEqual(set(vars(ns)) - flags, {"params"})  # params: resolve_args merges it with --param

    def test_the_namespace_means_what_the_sprt_flags_mean(self):
        args = release_args("--sprt_tc", "5+0.05", "--sprt_threads", "1", "4", "--concurrency", "3",
                            "--sprt_games", "200", "--tb", "tb", "--tbpieces", "5", "--no-pin", "--elo1", "5")
        resolved = sprt.resolve_args(release._sprt_namespace(args, 4, {"r_nmp": 4}, tc=args.sprt_tc))
        self.assertEqual((resolved.engine_a, resolved.engine_b), ("prod.exe", "base.exe"))
        self.assertEqual((resolved.tc, resolved.time, resolved.depth, resolved.nodes), ("5+0.05", None, None, None))
        self.assertEqual((resolved.threads, resolved.params, resolved.concurrency), (4, {"r_nmp": 4}, 3))
        self.assertEqual((resolved.max_games, resolved.book_depth, resolved.elo1), (200, 8, 5))
        self.assertEqual((resolved.tb, resolved.tbpieces, resolved.no_pin), ("tb", 5, True))
        self.assertEqual((resolved.db, resolved.fastchess), (release.DB_PATH, args.fastchess))
        self.assertTrue(resolved.log and resolved.plot and not resolved.no_db)
        self.assertEqual(resolved.pairs_per_job, 4)  # sprt's own default
        self.assertEqual(resolved.results_dir, sprt.RESULTS_DIR)

    def test_run_sprt_passes_the_one_limit_given_and_gates_only_their_tc(self):
        calls = []

        def fake_main(ns):
            ns = sprt.resolve_args(ns)
            sprt.tc_arguments(depth=ns.depth, movetime=ns.time, tc=ns.tc, nodes=ns.nodes)  # exactly one limit
            calls.append((ns.tc, ns.time, ns.nodes, ns.threads, ns.params))
            return {"accepted": True}

        args = release_args("--sprt_nodes", "20000", "--sprt_threads", "1", "2", "--sprt_gates", "10+.1", "60+.6")
        with mock.patch.object(release.sprt, "main", fake_main), contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(release.run_sprt(args))
            self.assertTrue(release.run_gated_sprt(args, {"r_nmp": 4}))
        self.assertEqual(calls[:2], [(None, None, 20000, 1, {}), (None, None, 20000, 2, {})])
        self.assertEqual(calls[2:], [(tc, None, None, threads, {"r_nmp": 4})
                                     for tc in ("10+.1", "60+.6") for threads in (1, 2)])

    def test_a_failed_gate_stops_the_gating(self):
        results = iter([{"accepted": True}, {"accepted": False}, {"accepted": True}])
        args = release_args("--sprt_gates", "10+.1", "60+.6")
        with mock.patch.object(release.sprt, "main", lambda ns: next(results)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(release.run_gated_sprt(args))
        self.assertEqual(next(results), {"accepted": True})  # the third gate was never played

    def test_an_sprt_that_ends_with_an_error_stops_the_release_rather_than_failing_a_gate(self):
        def stopped(ns):  # sprt.main after too many failed jobs or excluded pairs (as a stopped SPSA does)
            raise RuntimeError("SPRT stopped after 0 pairs: 3 jobs in a row failed (see the job logs)")

        args = release_args("--sprt_gates", "10+.1", "60+.6")
        with mock.patch.object(release.sprt, "main", stopped), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "jobs in a row failed"):
                release.run_gated_sprt(args)
            with self.assertRaisesRegex(RuntimeError, "jobs in a row failed"):
                release.run_sprt(args)


class TuneArguments(unittest.TestCase):
    def test_run_tune_goes_through_the_tuner_parser(self):
        seen = []
        args = release_args("--no-pin", "--fastchess", "fc.exe")
        with mock.patch.object(release.spsa_tuner, "main", lambda a: seen.append(a) or {"r_nmp": 4}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(release.run_tune(args), {"r_nmp": 4})
        tune = seen[0]
        self.assertEqual((tune.engine, tune.fastchess, tune.tc, tune.pairs, tune.concurrency, tune.no_pin),
                         ("dev.exe", "fc.exe", "0:20+.2", 10000, 4, True))
        self.assertTrue(tune.plot)
        self.assertEqual(tune.output, os.path.join(release.LOGS_DIR, "dev_logs", "spsa_results.json"))
        self.assertEqual(spsa_tuner.resolve_args(tune).pairs, 10000)  # main accepts the parsed Namespace


class FastchessLocation(unittest.TestCase):
    """Every tool plays with the fastchess the match runner finds: $FASTCHESS, libs/fastchess, then PATH."""

    def test_fastchess_match_honours_the_environment_variable_like_the_runner(self):
        from utils import match_runner

        def runner_choice():
            try:
                return Path(match_runner.find_fastchess())
            except FileNotFoundError:
                return fastchess_match.LIBS_FASTCHESS  # neither finds one: the libs path, for the error

        with tempfile.TemporaryDirectory() as tmp:
            own = Path(tmp) / ("fastchess.exe" if os.name == "nt" else "fastchess")
            own.write_bytes(b"")
            with mock.patch.dict(os.environ, FASTCHESS=str(own)):
                self.assertEqual(fastchess_match.default_fastchess(), Path(os.path.abspath(own)))
                self.assertEqual(fastchess_match.default_fastchess(), runner_choice())
            with mock.patch.dict(os.environ, FASTCHESS=str(Path(tmp) / "missing.exe")):  # skipped, as by the runner
                self.assertEqual(fastchess_match.default_fastchess(), runner_choice())

    def test_the_tools_default_to_it(self):
        from tests import smp
        self.assertEqual(smp.parse_args(["--engine", "e.exe"]).fastchess, str(fastchess_match.FASTCHESS))
        self.assertEqual(release.build_parser().parse_args(["--version", "1", "--name", "n", "--description", "d"])
                         .fastchess, str(fastchess_match.FASTCHESS))
        self.assertIs(tournament.FASTCHESS, fastchess_match.FASTCHESS)
        if cmaes_tuner is not None:
            with mock.patch.object(sys, "argv", ["cmaes_tuner", "--engine", "e.exe", "--tc", "1+0.01"]):
                self.assertEqual(cmaes_tuner.parse_args().fastchess, str(fastchess_match.FASTCHESS))


class Recover(unittest.TestCase):
    def test_tournament(self):
        """The tournament plays on the match runner (utils/fixed_match.py), whose jobs carry -recover."""
        from data.databases.init_raw_db import init_raw_db
        from utils.match_runner import JobSpec, MatchRunner, Slot
        seen = []

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            tmp = Path(tmp)
            fastchess = tmp / "fastchess.exe"
            fastchess.write_bytes(b"")
            (tmp / "book.pgn").write_text('[Event "?"]\n\n1. e4 e5 *\n')

            def play(pairings, **kwargs):  # one job of the first pairing, as the runner would start it
                runner = MatchRunner(kwargs["fastchess"], [Slot(0, None)], tmp / "run", kwargs["tc_args"],
                                     extra_args=kwargs["extra_args"], sample_clocks=False)
                p = pairings[0]
                seen.append(runner.command(Slot(0, None), JobSpec((p.first, p.second), [None]), tmp / "b.pgn",
                                           tmp / "t.pgn"))
                raise RuntimeError("played")

            cnxn = sqlite3.connect(init_raw_db(tmp / "db"))
            args = argparse.Namespace(engine="cand.exe", fastchess=str(fastchess), opening_book=str(tmp / "book.pgn"),
                                      tournament_games=2, tournament_tc=["1+0.01"], tournament_engines=1,
                                      opps=[{"id": 2, "version": "1.0", "path": "old.exe"}], concurrency=1,
                                      results_dir=tmp / "results", logroot=tmp / "logs")
            try:
                with mock.patch.object(tournament, "run_fixed_match", play), \
                        self.assertRaisesRegex(RuntimeError, "played"):
                    tournament.run_tournament(args, cnxn, 1)
            finally:
                cnxn.close()
        self.assertIn("-recover", seen[0])
        self.assertEqual(seen[0][seen[0].index("-maxmoves") + 1], "200")

    @unittest.skipIf(cmaes_tuner is None, "the cma package is not installed")
    def test_cmaes(self):
        """A CMA-ES generation is a fixed match on the runner, whose jobs carry -recover."""
        from utils import fixed_match
        from utils.match_runner import EngineSpec, Slot
        from utils.openings import OpeningBook
        seen = []

        def run(runner, controller):  # the first job's command instead of playing it
            slot = runner.slots[0]
            seen.append(runner.command(slot, controller.next_job(slot), "book.pgn", "games.pgn"))

        engine = EngineSpec("Engine", sys.executable, pin_option="")
        params = {"x": {"value": 3, "min": 0, "max": 9, "sigma0": 1, "integer": True}}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(fixed_match.MatchRunner, "run", run), \
                contextlib.redirect_stdout(io.StringIO()):
            book = Path(tmp) / "book.pgn"
            book.write_text('[Event "?"]\n\n1. e4 e5 *\n')
            pairings = cmaes_tuner.generation_pairings(engine, cmaes_tuner.candidate_engine(engine, "Baseline", {"x": 3}),
                                                       params, ["x"], [[4.4]], 1, OpeningBook(book, plies=2, seed=1))
            cmaes_tuner.evaluate_generation(pairings, 1, tc_args=["depth=1", "tc=inf"], slots=[Slot(0, None, "-")],
                                            fastchess=sys.executable, results_dir=Path(tmp) / "results", plies=2)
        self.assertIn("-recover", seen[0])
        self.assertIn("option.x=4", seen[0])


if __name__ == "__main__":
    unittest.main()
