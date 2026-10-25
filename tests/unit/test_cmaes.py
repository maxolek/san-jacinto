"""CMA-ES tuner (tuning/cmaes_tuner.py) on the match runner: the parameters and the values
sent, a generation as one fixed match (a pairing per candidate, all on the same openings),
the tune loop with the match faked (the fitness sign, CPUs planned once per tune, the
checkpoint, Ctrl-C, stopped matches and any other error, tied generations, pycma's own
stop, the seed), the option checks, and a short real tune with
fastchess and tests/unit/mock_uci_engine.py. Skipped without the optional cma package."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import fastchess_match  # noqa: E402
from utils.fixed_match import MatchInterrupted, MatchStopped  # noqa: E402
from utils.match_runner import EngineSpec, Slot, find_fastchess  # noqa: E402
from utils.openings import OpeningBook, parse_pgn_book  # noqa: E402

try:
    from tuning import cmaes_tuner as cmaes
except (ImportError, SystemExit):  # the optional cma package
    cmaes = None
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
    "1. e4 e5 2. Nf3 Nc6", "1. d4 d5 2. c4 e6", "1. c4 e5 2. Nc3 Nf6", "1. e4 c5 2. Nf3 d6",
    "1. d4 Nf6 2. c4 g6", "1. Nf3 d5 2. g3 Nf6"))
STRENGTH = {"Strength": {"value": 30, "min": 0, "max": 100, "sigma0": 20, "integer": True}}


def quietly(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


def openings(path):
    return [" ".join(o.moves) for o in parse_pgn_book(Path(path).read_text(), 16)]


@unittest.skipIf(cmaes is None, "the cma package is not installed")
class Parameters(unittest.TestCase):
    def test_the_defaults_a_selection_and_a_file(self):
        self.assertEqual(set(cmaes.load_params()), set(cmaes.DEFAULT_PARAMS))
        self.assertEqual(list(cmaes.load_params(["aspiration_window", "r_nmp"])), ["aspiration_window", "r_nmp"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "params.json"
            path.write_text(json.dumps({"Strength": {"value": 0.5, "min": 0, "max": 1, "sigma0": 0.1}}))
            self.assertEqual(cmaes.load_params(path=path), {"Strength": {"value": 0.5, "min": 0, "max": 1,
                                                                         "sigma0": 0.1, "integer": False}})

    def test_unknown_or_inconsistent_parameters_are_refused(self):
        with self.assertRaisesRegex(ValueError, r"unknown parameter\(s\): \['r_nmpp'\]"):
            cmaes.load_params(["r_nmp", "r_nmpp"])  # a typo is not silently left out of the tune
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "params.json"
            for cfg, message in (({"value": 5, "min": 0, "max": 9}, "needs \\['sigma0'\\]"),
                                 ({"value": 12, "min": 0, "max": 9, "sigma0": 1}, "min <= value <= max"),
                                 ({"value": 5, "min": 0, "max": 9, "sigma0": 0}, "sigma0 > 0")):
                path.write_text(json.dumps({"x": cfg}))
                with self.assertRaisesRegex(ValueError, message):
                    cmaes.load_params(path=path)

    def test_the_values_sent_are_rounded_and_clamped(self):
        params = {"i": {"min": 0, "max": 9, "integer": True}, "f": {"min": -1.0, "max": 1.0, "integer": False}}
        values = cmaes.option_values(params, ["i", "f"], [3.6, 1.7])
        self.assertEqual(values, {"i": 4, "f": 1.0})
        self.assertIsInstance(values["i"], int)
        self.assertEqual(cmaes.option_values(params, ["i", "f"], [-0.4, -0.25]), {"i": 0, "f": -0.25})


@unittest.skipIf(cmaes is None, "the cma package is not installed")
class CommandLine(unittest.TestCase):
    def test_the_defaults(self):
        args = cmaes.parse_args(["--engine", "e.exe", "--depth", "1"])
        self.assertEqual((args.concurrency, args.no_pin, args.fastchess, args.pairs_per_job, args.maxmoves),
                         (None, False, str(fastchess_match.FASTCHESS), 4, 100))
        self.assertEqual((args.games_per_eval, args.sigma0, args.seed, args.generations), (16, 1.0, None, 100))
        self.assertEqual((Path(args.book), Path(args.results_dir)), (cmaes.DEFAULT_BOOK, cmaes.RESULTS_DIR))
        self.assertEqual(args.output, "logs/dev_logs/cmaes_results.json")


@unittest.skipIf(cmaes is None, "the cma package is not installed")
class Generation(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.book = Path(tmp.name) / "book.pgn"
        self.book.write_text(BOOK)

    def test_one_pairing_per_candidate_all_on_the_same_openings(self):
        engine = EngineSpec("Engine", "engine.exe", options={"Hash": 16}, pin_option="Affinity")
        baseline = cmaes.candidate_engine(engine, "Baseline", {"Strength": 30})
        book = OpeningBook(self.book, plies=4, seed=7)
        pairings = cmaes.generation_pairings(engine, baseline, STRENGTH, ["Strength"], [[41.6], [-3.0], [30.2]],
                                             2, book)
        self.assertEqual([p.key for p in pairings], ["Candidate1", "Candidate2", "Candidate3"])
        self.assertEqual([p.first.name for p in pairings], ["Candidate1", "Candidate2", "Candidate3"])
        self.assertEqual([p.first.options for p in pairings],
                         [{"Hash": 16, "Strength": s} for s in (42, 0, 30)])
        self.assertEqual([p.meta["values"] for p in pairings], [{"Strength": s} for s in (42, 0, 30)])
        self.assertEqual({p.first.pin_option for p in pairings}, {"Affinity"})
        self.assertTrue(all(p.second is baseline and p.pairs == 2 for p in pairings))
        self.assertEqual(baseline.options, {"Hash": 16, "Strength": 30})
        # each pairing has its own copy of the book: the same shuffle, its own cursor
        self.assertEqual(len({id(p.book) for p in pairings}), 3)
        taken = [[o.index for o in p.book.take(4)] for p in pairings]
        self.assertEqual(taken, [[o.index for o in OpeningBook(self.book, plies=4, seed=7).take(4)]] * 3)
        self.assertEqual(book.cursor, 0)

    def test_the_score_is_points_per_game(self):
        self.assertEqual(cmaes.candidate_score(dict(candidate_wins=3, candidate_draws=4, candidate_losses=1)), 5 / 8)
        self.assertIsNone(cmaes.candidate_score(dict(candidate_wins=0, candidate_draws=0, candidate_losses=0)))


class FakeMatch:
    """run_fixed_match stand-in: by default a candidate scores more the higher its Strength (best at 100).

    `score`: Strength -> points per game. `stop`: (call number, exception factory taking the
    result) to end a generation with."""

    def __init__(self, score=lambda strength: 0.2 + 0.6 * strength / 100, stop=None):
        self.calls, self.score, self.stop = [], score, stop

    def __call__(self, pairings, **kwargs):
        self.calls.append((pairings, kwargs))
        result = dict(run_dir=str(Path(kwargs["results_dir"]) / kwargs["run_name"]), games=0, pairings={})
        for p in pairings:
            games = 2 * p.pairs
            wins = round(self.score(p.first.options["Strength"]) * games)
            result["pairings"][p.key] = dict(candidate_wins=wins, candidate_draws=0, candidate_losses=games - wins)
            result["games"] += games
        if self.stop and len(self.calls) == self.stop[0]:
            raise self.stop[1](result)
        return result


@unittest.skipIf(cmaes is None, "the cma package is not installed")
@unittest.skipIf(chess is None, "python-chess is not installed (the mock engine needs it)")
class Tune(unittest.TestCase):
    """main() with the matches faked; the engine is the mock (its options are checked for real)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "book.pgn").write_text(BOOK)
        self.params = self.dir / "params.json"
        self.params.write_text(json.dumps(STRENGTH))
        self.output = self.dir / "out" / "cmaes.json"
        self.slots = [Slot(0, None, "-"), Slot(1, None, "-")]

    def argv(self, *extra):
        return ["--engine", str(MOCK), "--depth", "1", "--params-file", str(self.params),
                "--book", str(self.dir / "book.pgn"), "--book-depth", "4", "--fastchess", sys.executable,
                "--seed", "5", "--results-dir", str(self.dir / "results"), "--output", str(self.output), "--quiet",
                *extra]

    def tune(self, fake, *extra):
        """main(argv) with the match faked; its console output in self.out."""
        with mock.patch.object(cmaes, "run_fixed_match", fake), \
                mock.patch.object(cmaes, "plan_slots", return_value=self.slots) as plan, \
                mock.patch.object(cmaes, "release_slots") as release:
            out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                    return cmaes.main(self.argv(*extra))
            finally:
                self.out = out.getvalue()
                plan.assert_called_once_with(concurrency=None, pin=True)
                release.assert_called_once_with(self.slots)

    def test_the_tune_climbs_towards_the_better_values_on_cpus_planned_once(self):
        fake = FakeMatch()
        final = self.tune(fake, "--generations", "6", "--popsize", "6", "--games-per-eval", "100")
        self.assertEqual(len(fake.calls), 6)
        for gen, (pairings, kwargs) in enumerate(fake.calls, 1):
            self.assertIs(kwargs["slots"], self.slots)
            self.assertEqual(kwargs["run_name"], f"gen{gen:03d}")
            self.assertEqual(Path(kwargs["results_dir"]).parent, self.dir / "results")
            self.assertEqual((kwargs["tc_args"], kwargs["extra_args"], kwargs["plies"]),
                             (["depth=1", "tc=inf"], ["-maxmoves", "100"], 4))
            self.assertEqual([p.key for p in pairings], [f"Candidate{i}" for i in range(1, 7)])
            self.assertEqual({p.pairs for p in pairings}, {50})
            self.assertEqual({p.second.options["Strength"] for p in pairings}, {30})  # the start value
            self.assertEqual({p.book.seed for p in pairings}, {5 + gen})  # common openings, new each generation
            self.assertEqual(kwargs["meta"]["generation"], gen)
            self.assertEqual(kwargs["show_setup"], gen == 1)  # the machine/slots/fastchess header once per tune
        # CMA-ES minimizes -score: the mean moves from 30 towards Strength 100
        self.assertGreater(final["mean_values"]["Strength"], 60)
        self.assertGreater(final["best_values"]["Strength"], 60)
        self.assertEqual((final["generation"], len(final["history"]), final["total_games"]), (6, 6, 6 * 6 * 100))
        self.assertIsNone(final["stopped"])
        self.assertEqual(final["seed"], 5)
        self.assertEqual(json.loads(self.output.read_text())["mean_values"], final["mean_values"])
        self.assertEqual(json.loads((Path(final["run_dir"]) / "cmaes_results.json").read_text())["history"],
                         final["history"])
        history = final["history"][0]
        self.assertEqual(len(history["scores"]), 6)
        self.assertEqual(history["candidates"], [p.meta["values"] for p in fake.calls[0][0]])

    def test_generations_of_ties_do_not_end_the_tune(self):
        fake = FakeMatch(score=lambda strength: 0.5)  # e.g. every game drawn: pycma sees flat fitness
        final = self.tune(fake, "--generations", "4", "--popsize", "4", "--games-per-eval", "2")
        self.assertEqual((len(fake.calls), final["generation"]), (4, 4), self.out)
        self.assertEqual(final["cma_stop"], {"maxiter": 4})
        self.assertIn("OPTIMIZATION COMPLETE", self.out)

    def test_a_tune_pycma_ends_early_says_so(self):
        fake = FakeMatch()
        strategy = cmaes.cma.CMAEvolutionStrategy

        def stops_after_one_generation(x0, sigma0, opts):
            return strategy(x0, sigma0, dict(opts, tolx=1e9))  # pycma's own criterion: x changes below 1e9

        with mock.patch.object(cmaes.cma, "CMAEvolutionStrategy", stops_after_one_generation):
            final = self.tune(fake, "--generations", "5", "--popsize", "4", "--games-per-eval", "2")
        self.assertEqual((len(fake.calls), final["generation"], final["stopped"]), (1, 1, None))
        self.assertEqual(final["cma_stop"], {"tolx": 1e9})
        self.assertNotIn("OPTIMIZATION COMPLETE", self.out)
        self.assertIn("STOPPED EARLY after 1 of 5 generations (pycma: tolx=1000000000.0)", self.out)

    def test_ctrl_c_keeps_the_complete_generations(self):
        fake = FakeMatch(stop=(2, MatchInterrupted))
        with self.assertRaises(KeyboardInterrupt):
            self.tune(fake, "--generations", "5", "--popsize", "4", "--games-per-eval", "10")
        saved = json.loads(self.output.read_text())
        self.assertEqual((saved["generation"], len(saved["history"]), saved["stopped"]), (1, 1, "interrupted"))
        self.assertEqual(saved["total_games"], 2 * 4 * 10)  # the interrupted generation's games count too

    def test_any_error_keeps_the_complete_generations_with_its_text(self):
        for error in (lambda result: ValueError("Game 3 of job 1: the engines did not reverse colors"),
                      lambda result: OSError("fastchess cannot start")):
            with self.subTest(error=error(None)):
                fake = FakeMatch(stop=(3, error))
                with self.assertRaises(type(error(None))):
                    self.tune(fake, "--generations", "5", "--popsize", "4", "--games-per-eval", "10")
                saved = json.loads(self.output.read_text())
                self.assertEqual((saved["generation"], len(saved["history"])), (2, 2))
                self.assertEqual(saved["stopped"], f"{type(error(None)).__name__}: {error(None)}")
                self.assertEqual(saved["total_games"], 2 * 4 * 10)  # the failed generation's games are unknown

    def test_seed_0_is_refused(self):
        # pycma would seed itself from the clock while the checkpoint records 0
        for seed in ("0", "-3", str(2 ** 32)):
            with self.subTest(seed=seed), mock.patch.object(cmaes, "plan_slots") as plan, \
                    self.assertRaisesRegex(ValueError, r"--seed must be between 1 and 4294967295"):
                quietly(cmaes.main, self.argv("--seed", seed))
            plan.assert_not_called()

    def test_a_stopped_match_stops_the_tune(self):
        fake = FakeMatch(stop=(1, lambda result: MatchStopped("CMA-ES gen 1 stopped: 3 jobs in a row failed", result)))
        with self.assertRaisesRegex(MatchStopped, "jobs in a row failed"):
            self.tune(fake, "--generations", "5", "--popsize", "4")
        saved = json.loads(self.output.read_text())
        self.assertEqual((saved["generation"], saved["mean_values"]), (0, {"Strength": 30}))
        self.assertIn("jobs in a row failed", saved["stopped"])

    def test_options_the_engines_would_not_take_are_refused_before_any_cpu_is_claimed(self):
        for params, message in (({"strength": dict(STRENGTH["Strength"])}, "the engine calls it Strength"),
                                ({"Strength": dict(STRENGTH["Strength"], max=150)},
                                 r"Strength: range \[0, 150\] exceeds the engine's \[0, 100\]")):
            self.params.write_text(json.dumps(params))
            with mock.patch.object(cmaes, "plan_slots") as plan, self.assertRaisesRegex(ValueError, message):
                quietly(cmaes.main, self.argv())
            plan.assert_not_called()
        engine = EngineSpec.from_path("Engine", MOCK).resolve()
        baseline = EngineSpec.from_path("Baseline", MOCK, {"Strength": 150}).resolve()
        with self.assertRaisesRegex(ValueError, "the baseline plays the start values.*Strength: value 150"):
            cmaes.check_options(engine, baseline, STRENGTH, {"Strength": 150})
        self.assertFalse((self.dir / "results").exists())
        self.assertFalse(self.output.exists())


@unittest.skipIf(cmaes is None, "the cma package is not installed")
@unittest.skipIf(FASTCHESS is None, "fastchess is not built (libs/fastchess)")
@unittest.skipIf(chess is None, "python-chess is not installed")
class FastchessTune(unittest.TestCase):
    def test_two_generations_with_the_mock_engine(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        tmp = Path(tmp.name)
        (tmp / "book.pgn").write_text(BOOK)
        (tmp / "params.json").write_text(json.dumps(STRENGTH))
        argv = ["--engine", str(MOCK), "--depth", "1", "--params-file", str(tmp / "params.json"),
                "--generations", "2", "--popsize", "3", "--games-per-eval", "2", "--concurrency", "2",
                "--maxmoves", "30", "--book", str(tmp / "book.pgn"), "--book-depth", "4", "--seed", "5", "--quiet",
                "--results-dir", str(tmp / "results"), "--output", str(tmp / "cmaes.json")]
        with mock.patch.object(sys, "stdin", io.StringIO("")):
            final, out = quietly(cmaes.main, argv)
        self.assertEqual((final["generation"], final["stopped"]), (2, None), out)
        gen_dirs = sorted(Path(final["run_dir"]).glob("gen*"))
        self.assertEqual([d.name for d in gen_dirs], ["gen001", "gen002"])
        for gen, (history, gen_dir) in enumerate(zip(final["history"], gen_dirs), 1):
            summary = json.loads((gen_dir / "summary.json").read_text())
            self.assertEqual((summary["pairs"], summary["pairs_requested"], summary["error"]), (3, 3, None), out)
            keys = ["Candidate1", "Candidate2", "Candidate3"]
            self.assertEqual(history["scores"],
                             [round(cmaes.candidate_score(summary["pairings"][k]), 4) for k in keys])
            config = json.loads((gen_dir / "config.json").read_text())["meta"]
            self.assertEqual({p["book"]["seed"] for p in config["pairings"]}, {5 + gen})
            self.assertEqual({p["second"]["options"]["Strength"] for p in config["pairings"]}, {30})
            # every candidate played the same openings, with its own values, in jobs that recover
            games = [json.loads(line) for line in (gen_dir / "games.jsonl").read_text().splitlines()]
            played = {}
            for job, key in sorted({(g["job"], g["pairing"]) for g in games}):
                played.setdefault(key, []).extend(openings(gen_dir / "jobs" / f"job{job:05d}_book.pgn"))
            self.assertEqual(sorted(played), keys)
            self.assertEqual(len({tuple(sorted(v)) for v in played.values()}), 1, played)
            commands = [p.read_text().splitlines()[0] for p in (gen_dir / "jobs").glob("job*.log")]
            self.assertTrue(all("-recover" in c for c in commands))
            for key, values in zip(keys, history["candidates"]):
                self.assertTrue(any(f"name={key}" in c and f"option.Strength={values['Strength']}" in c
                                    for c in commands), (key, values))
        self.assertEqual(json.loads((tmp / "cmaes.json").read_text())["history"], final["history"])


if __name__ == "__main__":
    unittest.main()
