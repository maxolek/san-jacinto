"""SPSA tuner: schedules, stochastic rounding and convergence on a synthetic objective."""
import argparse
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tuning import spsa_tuner as spsa  # noqa: E402
from utils.match_runner import EngineSpec  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
