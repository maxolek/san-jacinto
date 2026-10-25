"""SPSA tuner: schedules, stochastic rounding and convergence on a synthetic objective;
its command line, the per-job fastchess options, failing jobs, the option checks and a
short real tune with fastchess and the mock engine (resumed from its state.json)."""
import argparse
import contextlib
import io
import json
import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tuning import spsa_tuner as spsa  # noqa: E402
from utils.match_runner import EngineSpec, Slot, find_fastchess  # noqa: E402

try:
    import chess
except ImportError:  # pragma: no cover
    chess = None
try:
    FASTCHESS = find_fastchess()
except FileNotFoundError:
    FASTCHESS = None
MOCK = ROOT / "tests" / "unit" / "mock_uci_engine.py"
BOOK = "".join(f'[Event "?"]\n[Result "*"]\n\n{line} *\n\n' for line in (
    "1. e4 e5 2. Nf3 Nc6", "1. d4 d5 2. c4 e6", "1. c4 e5 2. Nc3 Nf6", "1. e4 c5 2. Nf3 d6"))


class Book:
    format = "pgn"

    def take(self, n):
        return [SimpleNamespace(text="", moves=(), fen=None, eco="")] * n


class Game:
    def __init__(self, job, pair_no, plus_score):
        self.job, self.pair_no, self.env, self.plus_score = job, pair_no, {}, plus_score

    def score_for(self, name):
        return self.plus_score if name == "Plus" else 1 - self.plus_score


class Schedules(unittest.TestCase):
    def test_gains_follow_the_published_conventions(self):
        schedule = spsa.Schedule(8000)
        cfg = {"c_end": 4.0}
        self.assertAlmostEqual(schedule.c(cfg, 8000), 4.0)
        self.assertAlmostEqual(schedule.c(cfg, 1), 4.0 * 8000 ** 0.101)
        self.assertAlmostEqual(schedule.a(cfg, 8000) / schedule.c(cfg, 8000) ** 2, 0.002)
        self.assertEqual(schedule.A, 800)

    def test_stochastic_rounding_is_unbiased(self):
        rng = random.Random(0)
        values = [spsa.stochastic_round(-2.25, rng) for _ in range(40000)]
        self.assertEqual(set(values), {-3, -2})
        self.assertAlmostEqual(sum(values) / len(values), -2.25, delta=0.01)


class Convergence(unittest.TestCase):
    def run_tune(self, optimizer="sgd", pairs=6000, seed=4):
        """Plus beats Minus with a probability that grows with f(plus) - f(minus),
        f peaks at 70: SPSA must move the parameter from 30 towards 70."""
        params = {"x": {"value": 30, "min": 0, "max": 100, "c_end": 4, "integer": True}}
        args = argparse.Namespace(pairs_per_job=1, optimizer=optimizer, no_polyak=False, polyak_from=0.5,
                                  quiet=True, status_every=1e9, r_end=0.002)
        with tempfile.TemporaryDirectory() as tmp:
            state = spsa.SpsaState(params, pairs, seed)
            controller = spsa.SpsaController(args, state, spsa.Schedule(pairs, r_end=0.02),
                                             EngineSpec("E", "engine.exe", pin_option=""), Book(), [None], tmp)
            rng = random.Random(seed)
            slot = SimpleNamespace(core_type="P")
            job_id = 0
            while True:
                spec = controller.next_job(slot)
                if spec is None:
                    break
                job_id += 1
                job = SimpleNamespace(id=job_id, spec=spec, slot=slot, killed=False, returncode=0, games=[1],
                                      log_path=None)
                edge = (-(spec.meta["plus"]["x"] - 70) ** 2 + (spec.meta["minus"]["x"] - 70) ** 2) / 4000
                p_win = 1 / (1 + math.exp(-edge))
                for game_no in (1, 2):
                    u = rng.random()
                    score = 1.0 if u < 0.4 * p_win else 0.0 if u > 1 - 0.4 * (1 - p_win) else 0.5
                    controller.on_game(Game(job, 1, score))
                controller.on_job_end(job)
            controller.close()
        return state

    def test_sgd_moves_towards_the_optimum_with_a_clear_signal(self):
        state = self.run_tune()
        self.assertGreater(state.theta["x"], 55)
        self.assertGreater(state.polyak()["x"], 50)
        self.assertGreater(state.t_values()["x"], 2)

    def test_state_survives_a_json_round_trip(self):
        state = self.run_tune(pairs=40)
        copy = spsa.SpsaState.from_json(state.to_json(random.Random(1)))
        self.assertEqual((copy.done, copy.theta, copy.signal), (state.done, state.theta, state.signal))
        self.assertEqual(copy.tally.counts, state.tally.counts)


class CommandLine(unittest.TestCase):
    def test_parse_args_fills_every_default(self):
        args = spsa.parse_args(["--engine", "e.exe", "--nodes", "5000", "--no-pin", "--fastchess", "fc.exe"])
        self.assertEqual((args.nodes, args.no_pin, args.fastchess, args.pairs, args.pairs_per_job, args.optimizer),
                         (5000, True, "fc.exe", 4000, 2, "sgd"))
        self.assertEqual(spsa.limit_flag(args), "--nodes 5000")

    def test_old_iterations_become_pairs(self):
        args = spsa.parse_args(["--engine", "e.exe", "--tc", "10+0.1", "--iterations", "50", "--games-per-iter", "4",
                                "--r", "0.003"])
        self.assertEqual((args.pairs, args.r_end), (100, 0.003))
        self.assertEqual(spsa.limit_flag(args), "--tc 10+0.1")

    def test_the_old_baseline_flag_is_accepted_and_ignored(self):
        args = spsa.parse_args(["--engine", "e.exe", "--baseline", "b.exe", "--tc", "10+0.1"])
        self.assertEqual((args.engine, args.tc, args.pairs), ("e.exe", "10+0.1", 4000))
        self.assertNotIn("--baseline", spsa.build_parser().format_help())

    def test_a_namespace_gets_the_missing_defaults(self):
        args = spsa.resolve_args(argparse.Namespace(engine="e.exe", time=0.1))
        self.assertEqual((args.pairs, args.r_end, args.maxmoves, args.no_pin, args.time), (4000, 0.002, 100, False, 0.1))
        self.assertEqual(spsa.limit_flag(args), "--time 0.1")


class Jobs(unittest.TestCase):
    PARAMS = {"Strength": {"value": 50, "min": 0, "max": 100, "c_end": 5, "integer": True}}

    def controller(self, pairs=10, slots=(None,)):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        args = argparse.Namespace(pairs_per_job=2, optimizer="sgd", no_polyak=False, polyak_from=0.5,
                                  quiet=False, status_every=1e9, r_end=0.002)
        state = spsa.SpsaState(dict(self.PARAMS), pairs, 3)
        engine = EngineSpec("E", "engine.exe", options={"Hash": "16"}, pin_option="Affinity")
        ctl = spsa.SpsaController(args, state, spsa.Schedule(pairs), engine, Book(), list(slots), tmp.name)
        self.addCleanup(ctl.close)  # cleanups run last-in first-out: before the folder is removed
        return ctl, Path(tmp.name)

    def test_a_job_plays_one_perturbation_given_to_fastchess_as_options(self):
        ctl, _ = self.controller()
        spec = ctl.next_job(SimpleNamespace(core_type="P"))
        self.assertEqual(len(spec.openings), 2)
        plus, minus = spec.engines
        meta = spec.meta
        self.assertEqual((plus.name, minus.name), ("Plus", "Minus"))
        self.assertGreater((meta["plus"]["Strength"] - meta["minus"]["Strength"]) * meta["delta"]["Strength"], 0)
        block = plus.fastchess_args(Slot(0, 3, "P"))
        self.assertIn(f"option.Strength={meta['plus']['Strength']}", block)
        self.assertIn("option.Hash=16", block)
        self.assertIn("option.Affinity=3", block)
        self.assertIn(f"option.Strength={meta['minus']['Strength']}", minus.fastchess_args(Slot(0, 3, "P")))
        self.assertFalse(any(a.startswith("options.") for a in block))

    def test_unplayed_pairs_are_played_again_and_failing_jobs_stop_the_tune(self):
        ctl, tmp = self.controller()
        log = tmp / "job.log"
        log.write_text("Warning; cannot start engine\n")
        slot = SimpleNamespace(core_type="P")

        def end(spec, scores, returncode=0):
            job = SimpleNamespace(id=id(spec), spec=spec, slot=slot, killed=False, returncode=returncode,
                                  games=list(scores), log_path=log)
            for number, score in enumerate(scores, 1):
                ctl.on_game(Game(job, (number + 1) // 2, score))
            ctl.on_job_end(job)

        with contextlib.redirect_stdout(io.StringIO()) as out:
            end(ctl.next_job(slot), [1.0, 0.5, 1.0])  # the second pair is unfinished
            self.assertEqual((ctl.state.done, ctl.state.planned), (1, 1))
            self.assertIn("R +1", out.getvalue())
            self.assertEqual(json.loads((tmp / "history.jsonl").read_text())["result"], 1)
            for _ in range(3):
                spec = ctl.next_job(slot)
                self.assertEqual(spec.meta["k"], 2)
                end(spec, [], returncode=1)
        self.assertEqual(ctl.state.planned, 1)
        self.assertIsNotNone(ctl.stop_reason)
        self.assertTrue(ctl.finished())
        self.assertIsNone(ctl.next_job(slot))

    @unittest.skipIf(chess is None, "python-chess is not installed (the mock engine needs it)")
    def test_options_the_engine_would_not_take_are_refused(self):
        engine = EngineSpec.from_path("E", MOCK).resolve()
        spsa.check_options(engine, self.PARAMS)
        with self.assertRaisesRegex(ValueError, "the engine calls it Strength"):
            spsa.check_options(engine, {"strength": dict(self.PARAMS["Strength"])})
        with self.assertRaisesRegex(ValueError, r"exceeds the engine's \[0, 100\]"):
            spsa.check_options(engine, {"Strength": dict(self.PARAMS["Strength"], max=150)})


@unittest.skipIf(FASTCHESS is None, "fastchess is not built (libs/fastchess)")
@unittest.skipIf(chess is None, "python-chess is not installed")
class FastchessTune(unittest.TestCase):
    def test_a_short_tune_and_its_resume(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        tmp = Path(tmp.name)
        (tmp / "book.pgn").write_text(BOOK)
        (tmp / "params.json").write_text(json.dumps(
            {"Strength": {"value": 40, "min": 0, "max": 100, "c_end": 10, "integer": True}}))
        argv = ["--engine", str(MOCK), "--depth", "1", "--params-file", str(tmp / "params.json"), "--pairs", "4",
                "--pairs-per-job", "1", "--concurrency", "2", "--maxmoves", "30", "--book", str(tmp / "book.pgn"),
                "--book-depth", "4", "--seed", "5", "--quiet", "--results-dir", str(tmp / "results"),
                "--r-end", "0.05"]
        with mock.patch.object(sys, "stdin", io.StringIO("")), contextlib.redirect_stdout(io.StringIO()) as out:
            recommended = spsa.main(argv)
        run_dir = next((tmp / "results").iterdir())
        state = json.loads((run_dir / "state.json").read_text())
        self.assertEqual(state["done"], 4, out.getvalue())
        self.assertEqual(set(recommended), {"Strength"})
        history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
        self.assertEqual(len(history), 4)
        commands = [p.read_text().splitlines()[0] for p in (run_dir / "jobs").glob("job*.log")]
        for job in history:  # each job's fastchess got its own pair of values
            self.assertTrue(any(f"option.Strength={job['plus']['Strength']}" in c and
                                f"option.Strength={job['minus']['Strength']}" in c and "-recover" in c
                                for c in commands))
        # resume: a longer tune of the same run continues from the saved state
        state["pairs"] = 6
        (run_dir / "state.json").write_text(json.dumps(state))
        with mock.patch.object(sys, "stdin", io.StringIO("")), contextlib.redirect_stdout(io.StringIO()) as out:
            spsa.main(argv + ["--resume", str(run_dir / "state.json")])
        resumed = json.loads((run_dir / "state.json").read_text())
        self.assertEqual((resumed["done"], resumed["pairs"]), (6, 6), out.getvalue())
        self.assertEqual(len((run_dir / "history.jsonl").read_text().splitlines()), 6)
        self.assertIn("resuming", out.getvalue())


if __name__ == "__main__":
    unittest.main()
