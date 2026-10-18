"""Tests for utils.rating_fit. Run from the repo root:

    python -m unittest discover -s tests/unit -t .
"""

import math
import unittest
from itertools import combinations

import numpy as np

from utils.rating_fit import expected_score, fit_ratings

TRUE_RATINGS = [1500, 1550, 1600, 1680, 1700]


def score_to_elo(score):
    return 400 * math.log10(score / (1 - score))


def round_robin(rng, ratings, games_per_pair, draw_rate=0.0):
    """Simulated (i, j, wins_i, draws, losses_i) rows; draws keep the logistic expected score."""
    rows = []
    for i, j in combinations(range(len(ratings)), 2):
        p = expected_score(ratings[i] - ratings[j])
        wins, draws, losses = rng.multinomial(games_per_pair, [p - draw_rate / 2, draw_rate, 1 - p - draw_rate / 2])
        rows.append((i, j, int(wins), int(draws), int(losses)))
    return rows


def interval_coverage(rng, replications, draw_rate=0.0):
    """Fraction of 95% intervals of all pairwise differences that contain the true difference."""
    covered = total = 0
    for _ in range(replications):
        fit = fit_ratings(round_robin(rng, TRUE_RATINGS, 2000, draw_rate), anchor=0)
        for a, b in combinations(range(len(TRUE_RATINGS)), 2):
            diff, se = fit.diff(a, b)
            covered += abs(diff - (TRUE_RATINGS[a] - TRUE_RATINGS[b])) <= 1.96 * se
            total += 1
    return covered / total


class ExpectedScoreTests(unittest.TestCase):
    def test_values_and_symmetry(self):
        self.assertEqual(expected_score(0), 0.5)
        self.assertAlmostEqual(expected_score(400), 10 / 11)
        for diff in (-1234.5, -50, 3, 800):
            self.assertAlmostEqual(expected_score(diff) + expected_score(-diff), 1.0)

    def test_extreme_differences_do_not_overflow(self):
        self.assertEqual(expected_score(1e6), 1.0)
        self.assertEqual(expected_score(-1e6), 0.0)


class TwoPlayerTests(unittest.TestCase):
    ROWS = [("A", "B", 60, 20, 20)]  # A scores (60 + 10) / 100

    def test_diff_is_logistic_elo_of_score(self):
        fit = fit_ratings(self.ROWS, anchor="A", prior_draws=0)
        self.assertAlmostEqual(fit.diff("A", "B")[0], score_to_elo(0.7), delta=0.5)
        self.assertEqual(fit.ratings["A"], 1500.0)
        self.assertAlmostEqual(fit.ratings["B"], 1500 - score_to_elo(0.7), delta=0.5)

    def test_prior_shrinks_diff_slightly(self):
        exact = fit_ratings(self.ROWS, prior_draws=0).diff("A", "B")[0]
        shrunk = fit_ratings(self.ROWS, prior_draws=1).diff("A", "B")[0]
        self.assertLess(shrunk, exact)
        self.assertGreater(shrunk, exact - 3)
        self.assertAlmostEqual(shrunk, score_to_elo(70.5 / 101), delta=1e-6)  # one virtual draw

    def test_se_matches_analytic_value(self):
        fit = fit_ratings(self.ROWS, anchor="A", prior_draws=0)
        n, p = 100, 0.7
        analytic = math.sqrt(1 / (n * p * (1 - p))) * 400 / math.log(10)
        self.assertAlmostEqual(fit.se("B") / analytic, 1, delta=0.01)
        self.assertAlmostEqual(fit.diff("A", "B")[1] / analytic, 1, delta=0.01)
        self.assertEqual(fit.se("A"), 0.0)


class RecoveryTests(unittest.TestCase):
    def test_round_robin_recovers_true_ratings(self):
        rows = round_robin(np.random.default_rng(2026), TRUE_RATINGS, 2000, draw_rate=0.3)
        fit = fit_ratings(rows, anchor=0, anchor_elo=1500)
        self.assertEqual(fit.ratings[0], 1500.0)
        self.assertEqual(fit.se(0), 0.0)
        for player, true in enumerate(TRUE_RATINGS[1:], start=1):
            self.assertLess(abs(fit.ratings[player] - true), 3 * fit.se(player), player)
        self.assertEqual(fit.games, dict.fromkeys(range(5), 8000))
        self.assertTrue(1 <= fit.iterations < 20)

    def test_95_percent_intervals_cover_the_truth(self):
        coverage = interval_coverage(np.random.default_rng(7), 200)
        self.assertGreaterEqual(coverage, 0.92)
        self.assertLessEqual(coverage, 0.98)

    def test_draws_make_intervals_conservative(self):
        self.assertGreaterEqual(interval_coverage(np.random.default_rng(8), 100, draw_rate=0.4), 0.95)

    def test_expected_points_equal_actual_points_at_optimum(self):
        rows = round_robin(np.random.default_rng(1), TRUE_RATINGS, 500, draw_rate=0.2)
        fit = fit_ratings(rows, anchor=0, prior_draws=0)
        expected = dict.fromkeys(fit.ratings, 0.0)
        for a, b, wins, draws, losses in rows:
            p = expected_score(fit.ratings[a] - fit.ratings[b])
            expected[a] += (wins + draws + losses) * p
            expected[b] += (wins + draws + losses) * (1 - p)
        for player, points in fit.points.items():
            self.assertAlmostEqual(expected[player], points, delta=1e-6)


class GraphTests(unittest.TestCase):
    def test_chain_rates_players_who_never_met(self):
        fit = fit_ratings([("A", "B", 30, 40, 30), ("B", "C", 50, 30, 20)], anchor="A")
        self.assertEqual(fit.component, {"A": 0, "B": 0, "C": 0})
        ab, se_ab = fit.diff("A", "B")
        bc, se_bc = fit.diff("B", "C")
        ac, se_ac = fit.diff("A", "C")
        self.assertGreater(se_ac, se_ab)
        self.assertGreater(se_ac, se_bc)
        # A tree decouples: each pair is fitted on its own score and the errors add in quadrature.
        self.assertAlmostEqual(ab, 0.0, delta=1e-6)
        self.assertAlmostEqual(bc, score_to_elo(65.5 / 101), delta=1e-6)
        self.assertAlmostEqual(ac, ab + bc, delta=1e-9)
        self.assertAlmostEqual(se_ac ** 2, se_ab ** 2 + se_bc ** 2, delta=1e-6)

    def test_differences_do_not_depend_on_anchor(self):
        rows = round_robin(np.random.default_rng(3), TRUE_RATINGS, 300) + [(4, 5, 10, 5, 3)]
        by_0 = fit_ratings(rows, anchor=0)
        by_5 = fit_ratings(rows, anchor=5, anchor_elo=2000)
        self.assertEqual(by_5.ratings[5], 2000.0)
        for a, b in combinations(range(6), 2):
            for got, want in zip(by_5.diff(a, b), by_0.diff(a, b)):
                self.assertAlmostEqual(got, want, delta=1e-6)

    def test_disconnected_graph_has_separate_components(self):
        fit = fit_ratings([("A", "B", 6, 2, 2), ("C", "D", 3, 4, 3)], anchor="C", anchor_elo=2000)
        self.assertEqual(fit.component, {"A": 1, "B": 1, "C": 0, "D": 0})
        self.assertEqual(fit.anchors, {0: "C", 1: "A"})
        self.assertEqual(fit.ratings["C"], 2000.0)
        self.assertEqual(fit.ratings["A"], 2000.0)  # every component is anchored at anchor_elo
        self.assertEqual(fit.se("A"), 0.0)
        self.assertGreater(fit.diff("A", "B")[1], 0)
        with self.assertRaises(ValueError):
            fit.diff("A", "C")
        with self.assertRaises(ValueError):
            fit.covariance("B", "D")

    def test_without_anchor_largest_component_comes_first(self):
        fit = fit_ratings([("x", "y", 1, 1, 1), ("c", "d", 1, 0, 1), ("e", "d", 2, 0, 0)])
        self.assertEqual(fit.anchors, {0: "c", 1: "x"})
        self.assertEqual({key for key, c in fit.component.items() if c == 0}, {"c", "d", "e"})
        tie = fit_ratings([("q", "r", 1, 0, 0), ("b", "p", 0, 1, 0)])
        self.assertEqual(tie.anchors, {0: "b", 1: "q"})

    def test_smallest_key_orders_ints_numerically_before_strings(self):
        fit = fit_ratings([(10, 9, 1, 1, 1), ("a", 10, 1, 0, 1), (2, "b", 0, 2, 0)])
        self.assertEqual(fit.anchors, {0: 9, 1: 2})
        self.assertEqual(fit.ratings[9], 1500.0)


class PerfectScoreTests(unittest.TestCase):
    def test_prior_keeps_perfect_score_finite(self):
        fit = fit_ratings([("A", "B", 10, 0, 0)], anchor="B")
        self.assertTrue(math.isfinite(fit.ratings["A"]))
        self.assertTrue(math.isfinite(fit.se("A")))
        self.assertAlmostEqual(fit.ratings["A"] - 1500, score_to_elo(10.5 / 11), delta=1e-6)

    def test_perfect_score_without_prior_raises(self):
        for rows in ([("A", "B", 10, 0, 0)], [("B", "A", 0, 0, 10)],
                     [("A", "B", 3, 0, 0), ("A", "C", 2, 0, 0), ("B", "C", 1, 4, 1)]):
            with self.assertRaisesRegex(ValueError, r"\['A'\] scored 100%.*prior_draws"):
                fit_ratings(rows, prior_draws=0)

    def test_perfect_pairs_are_finite_when_every_split_has_points_both_ways(self):
        fit = fit_ratings([("A", "B", 5, 0, 0), ("B", "C", 5, 0, 0), ("C", "A", 5, 0, 0)], prior_draws=0)
        for player in "ABC":
            self.assertAlmostEqual(fit.ratings[player], 1500.0, delta=1e-6)


class AggregationTests(unittest.TestCase):
    def test_rows_are_summed_in_either_orientation(self):
        split = fit_ratings([("A", "B", 10, 5, 5), ("B", "A", 5, 5, 10)], anchor="A")
        single = fit_ratings([("A", "B", 20, 10, 10)], anchor="A")
        self.assertEqual(split.games, {"A": 40, "B": 40})
        self.assertEqual(split.points, {"A": 25.0, "B": 15.0})
        self.assertEqual((split.games, split.points), (single.games, single.points))
        self.assertAlmostEqual(split.ratings["B"], single.ratings["B"], delta=1e-9)
        self.assertAlmostEqual(split.se("B"), single.se("B"), delta=1e-9)

    def test_self_play_and_empty_rows_are_ignored(self):
        fit = fit_ratings([("A", "A", 5, 0, 0), ("A", "B", 0, 0, 0), ("A", "C", 1, 1, 1)])
        self.assertEqual(fit.games, {"A": 3, "C": 3})
        self.assertNotIn("B", fit.ratings)

    def test_empty_input_gives_empty_result(self):
        fit = fit_ratings([])
        self.assertEqual((fit.ratings, fit.games, fit.points, fit.component, fit.anchors), ({},) * 5)
        self.assertEqual(fit.iterations, 0)


class FitResultTests(unittest.TestCase):
    def test_covariance_is_symmetric_and_zero_for_the_anchor(self):
        fit = fit_ratings(round_robin(np.random.default_rng(5), TRUE_RATINGS, 200), anchor=2)
        for a, b in combinations(range(5), 2):
            self.assertAlmostEqual(fit.covariance(a, b), fit.covariance(b, a), delta=1e-9)
        for player in range(5):
            self.assertAlmostEqual(fit.se(player) ** 2, fit.covariance(player, player), delta=1e-9)
            self.assertEqual(fit.covariance(2, player), 0.0)
        self.assertEqual(fit.diff(1, 1), (0.0, 0.0))
        with self.assertRaises(KeyError):
            fit.se("missing")


class ValidationTests(unittest.TestCase):
    def test_negative_counts_raise(self):
        for row in (("A", "B", -1, 2, 3), ("A", "B", 1, -2, 3), ("A", "B", 1, 2, -3)):
            with self.assertRaises(ValueError):
                fit_ratings([row])

    def test_non_integer_counts_raise(self):
        for bad in (1.5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fit_ratings([("A", "B", bad, 0, 0)])
        self.assertEqual(fit_ratings([("A", "B", 2.0, np.int64(1), 0)]).games["A"], 3)

    def test_unknown_anchor_raises(self):
        rows = [(1, 2, 3, 4, 5)]
        for anchor in (3, "1"):
            with self.assertRaises(ValueError):
                fit_ratings(rows, anchor=anchor)
        with self.assertRaises(ValueError):  # present only in a row without games
            fit_ratings(rows + [(1, 3, 0, 0, 0)], anchor=3)

    def test_bad_parameters_raise(self):
        rows = [("A", "B", 1, 1, 1)]
        for kwargs in (dict(prior_draws=-1), dict(prior_draws=float("nan")), dict(tol=0), dict(max_iter=0)):
            with self.assertRaises(ValueError):
                fit_ratings(rows, **kwargs)

    def test_non_convergence_raises(self):
        with self.assertRaisesRegex(ValueError, "did not converge"):
            fit_ratings([("A", "B", 90, 5, 5)], max_iter=1)


if __name__ == "__main__":
    unittest.main()
