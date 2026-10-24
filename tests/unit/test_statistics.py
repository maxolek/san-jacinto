"""Pentanomial statistics (utils/pentanomial.py) and the Stockfish strength model."""
import contextlib
import io
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.strength_estimator import EstimatorController, StrengthModel, expected_score  # noqa: E402
from utils.match_runner import EngineSpec, Slot  # noqa: E402
from utils.pentanomial import (PairTally, PentanomialSPRT, compute_llr, elo_to_score, estimate,  # noqa: E402
                               pair_bin, pair_detail, score_to_elo)


def play_pair(sprt, number, candidate_first, second):
    """Candidate has White in game `number`; scores are the candidate's."""
    results = {1.0: ("1-0", "0-1"), 0.5: ("1/2-1/2", "1/2-1/2"), 0.0: ("0-1", "1-0")}
    first_result = results[candidate_first][0]
    second_result = results[second][1]
    return (sprt.add_game(number, "Candidate", "Baseline", first_result)
            + sprt.add_game(number + 1, "Baseline", "Candidate", second_result))


class Pentanomial(unittest.TestCase):
    def test_elo_score_round_trip(self):
        for elo in (-300, -10, 0, 7.5, 250):
            self.assertAlmostEqual(score_to_elo(elo_to_score(elo)), elo, places=6)

    def test_symmetric_results_are_neutral(self):
        e = estimate([5, 20, 50, 20, 5])
        self.assertAlmostEqual(e["score"], 0.5)
        self.assertAlmostEqual(e["elo_diff"], 0.0, places=6)
        self.assertAlmostEqual(e["los"], 50.0, places=6)
        self.assertLess(e["elo_low"], 0 < e["elo_high"])

    def test_interval_shrinks_with_more_pairs(self):
        small, large = estimate([2, 8, 20, 12, 3]), estimate([20, 80, 200, 120, 30])
        self.assertLess(large["elo_high"] - large["elo_low"], small["elo_high"] - small["elo_low"])
        self.assertAlmostEqual(small["score"], large["score"], places=2)

    def test_llr_prefers_the_hypothesis_the_data_supports(self):
        good = [2, 10, 40, 30, 8]
        self.assertGreater(compute_llr(good, 0, 10), 0)
        self.assertLess(compute_llr([8, 30, 40, 10, 2], 0, 10), 0)

    def test_pair_bins_and_detail(self):
        self.assertEqual(pair_bin(1, 0.5), 3)
        self.assertEqual(pair_detail(1, 0), "WL")
        self.assertEqual(pair_detail(0.5, 0.5), "DD")
        tally = PairTally()
        tally.add_pair(1, 0)
        tally.add_pair(0.5, 0.5)
        self.assertEqual(tally.counts, [0, 0, 2, 0, 0])
        self.assertEqual((tally.detail["WL"], tally.detail["DD"]), (1, 1))

    def test_pairs_are_tested_in_start_order(self):
        sprt = PentanomialSPRT(0, 5)
        self.assertEqual(play_pair(sprt, 3, 1.0, 1.0), [])          # pair 2 finished first: waits
        self.assertEqual(sprt.snapshot()["untested_games"], 2)
        updates = play_pair(sprt, 1, 0.5, 0.5)                      # pair 1 completes: both tested
        self.assertEqual([u["pairs"] for u in updates], [1, 2])
        self.assertEqual(sprt.counts, [0, 0, 1, 0, 1])

    def test_a_voided_pair_does_not_block_later_pairs(self):
        sprt = PentanomialSPRT(0, 5)
        play_pair(sprt, 3, 1.0, 0.5)
        sprt.add_game(1, "Candidate", "Baseline", "1-0")            # pair 1 will never finish
        updates = sprt.void_pair(0)
        self.assertEqual(len(updates), 1)
        self.assertEqual(sprt.snapshot()["voided_pairs"], 1)
        self.assertEqual(sprt.counts, [0, 0, 0, 1, 0])

    def test_sprt_stops_at_the_bound_and_freezes(self):
        sprt = PentanomialSPRT(0, 20)
        number = 1
        while sprt.result == "inconclusive":
            play_pair(sprt, number, 1.0, 0.5)
            number += 2
        self.assertEqual(sprt.result, "pass")
        frozen = sprt.snapshot()["pairs"]
        play_pair(sprt, number, 0.0, 0.0)
        self.assertEqual(sprt.snapshot()["pairs"], frozen)

    def test_pairing_variance_ratio_is_reported(self):
        sprt = PentanomialSPRT(0, 5)
        rng = random.Random(1)
        for pair in range(200):
            white_wins = rng.random() < 0.6          # openings favour White
            play_pair(sprt, 2 * pair + 1, 1.0 if white_wins else 0.5, 0.0 if white_wins else 0.5)
        ratio = sprt.snapshot()["variance_ratio"]
        self.assertLess(ratio, 1.0, "colour-reversed pairs cancel the opening bias")


class Strength(unittest.TestCase):
    def fit(self, rating, slope, levels, pairs=400, seed=2):
        rng = random.Random(seed)
        model = StrengthModel(prior_mean=2000)
        for i in range(pairs):
            level = levels[i % len(levels)]
            p = expected_score(rating, level, slope)
            scores = [1.0 if rng.random() < p else 0.0 for _ in range(2)]
            model.add(level, sum(scores) / 2)
        return model.fit_full()

    def test_expected_score_is_symmetric(self):
        self.assertAlmostEqual(expected_score(2000, 2000), 0.5)
        self.assertAlmostEqual(expected_score(2100, 2000) + expected_score(2000, 2100), 1.0)

    def test_recovers_rating_and_calibrated_scale(self):
        rating, se, slope, _ = self.fit(2150, 1.0, [2000, 2150, 2150, 2300])
        self.assertLess(abs(rating - 2150), 3 * se)
        self.assertAlmostEqual(slope, 1.0, delta=0.35)

    def test_recovers_a_compressed_scale(self):
        rating, se, slope, _ = self.fit(1900, 0.4, [1550, 1900, 1900, 2250], pairs=800)
        self.assertLess(abs(rating - 1900), 3 * se)
        self.assertAlmostEqual(slope, 0.4, delta=0.15)
        self.assertTrue(math.isfinite(se) and se > 0)

    def test_estimator_stops_when_jobs_keep_failing(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "job.log"
            log.write_text("Warning: cannot start engine\n")
            args = SimpleNamespace(start_elo=2000, seed=1, max_games=200, pairs_per_job=4, spread=50,
                                   min_games=20, quiet=True, status_every=60, target_ci=30)
            slots = [Slot(0, 0, "P"), Slot(1, 8, "E")]
            ctl = EstimatorController(args, EngineSpec("Candidate", "candidate"), EngineSpec("Stockfish", "sf"),
                                      SimpleNamespace(take=lambda n: [None] * n), slots, tmp, (1320, 3190))

            def end(spec, games):
                ctl.on_job_end(SimpleNamespace(id=1, spec=spec, pairs=len(spec.openings), pairs_started=0,
                                               killed=False, returncode=0 if games else 1, games=games,
                                               log_path=log))
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    end(ctl.next_job(slots[0]), [])
                    end(ctl.next_job(slots[1]), [])
                    end(ctl.next_job(slots[0]), ["game"])  # a job that played resets the count
                    for _ in range(3):
                        spec = ctl.next_job(slots[0])
                        self.assertIsNotNone(spec)
                        end(spec, [])
                    self.assertIsNone(ctl.next_job(slots[1]))
                    self.assertTrue(ctl.finished())
                self.assertIn("failing", ctl.decision)
                self.assertEqual(ctl.planned, 0)
            finally:
                ctl.close()


if __name__ == "__main__":
    unittest.main()
