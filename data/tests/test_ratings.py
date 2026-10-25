"""Pairwise results in match_results (data/ratings.py): the pentanomial middle bin and the
interface the match runners call.

    python -m unittest data.tests.test_ratings
"""
import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

from data import ratings
from data.databases.init_raw_db import init_raw_db
from data.databases.migrate_schema import MIGRATIONS
from data.etl.schema import ensure_schema
from utils.pentanomial import PentanomialSPRT

PENTA_WL_MIGRATION = "2026_10_25_match_results_penta_wl"


def _snapshot(pairs):
    """A PentanomialSPRT snapshot after `pairs` (the candidate's results of the two games)."""
    sprt = PentanomialSPRT(0.0, 5.0, sequential=False)
    outcome = {(True, 1.0): "1-0", (True, 0.0): "0-1", (False, 1.0): "0-1", (False, 0.0): "1-0"}
    game = 0
    for scores in pairs:
        for candidate_white, score in zip((True, False), scores):
            game += 1
            result = "1/2-1/2" if score == 0.5 else outcome[(candidate_white, score)]
            white, black = ("Candidate", "Baseline") if candidate_white else ("Baseline", "Candidate")
            sprt.add_game(game, white, black, result)
    return sprt.snapshot()


class MatchResultsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        with contextlib.redirect_stdout(io.StringIO()):
            self.path = init_raw_db(Path(tmp.name) / "chess.db")
        self.cnxn = sqlite3.connect(self.path)
        self.addCleanup(self.cnxn.close)
        self.a = self.cnxn.execute("INSERT INTO engines (name, version) VALUES ('a', '1')").lastrowid
        self.b = self.cnxn.execute("INSERT INTO engines (name, version) VALUES ('b', '2')").lastrowid
        self.cnxn.commit()

    def stored(self, match_id):
        return self.cnxn.execute("SELECT penta_ll, penta_ld, penta_dd, penta_wl, penta_wd, penta_ww "
                                 "FROM match_results WHERE id = ?", (match_id,)).fetchone()

    def bins(self, match_id):
        return list(self.cnxn.execute(f"SELECT {ratings.PENTA_SQL} FROM match_results WHERE id = ?",
                                      (match_id,)).fetchone())

    def test_a_snapshot_stores_two_draws_and_win_loss_pairs_apart(self):
        # LL, LD, DD, DD, WL, LW, WD, WW: the middle bin holds 2 DD and 2 WL pairs
        snapshot = _snapshot([(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0.5), (1, 0), (0, 1), (1, 0.5), (1, 1)])
        self.assertEqual(snapshot["pentanomial"], [1, 1, 4, 1, 1])
        match_id = ratings.record_snapshot(self.cnxn, source="sprt", engine_a=self.a, engine_b=self.b,
                                           snapshot=snapshot, tc_category="ultra_fast", time_control="10+0.1",
                                           experiment_id=None)
        self.assertEqual(self.stored(match_id), (1, 1, 2, 2, 1, 1))
        self.assertEqual(self.bins(match_id), snapshot["pentanomial"])
        self.assertEqual(self.cnxn.execute("SELECT wins, draws, losses, source, tc_category FROM match_results")
                         .fetchone(), (5, 6, 5, "sprt", "ultra_fast"))

    def test_without_the_split_the_middle_bin_stays_combined(self):
        snapshot = _snapshot([(0.5, 0.5), (1, 0), (1, 1)])
        del snapshot["pentanomial_detail"]  # e.g. a runner that only reports the five bins
        match_id = ratings.record_snapshot(self.cnxn, source="sprt", engine_a=self.a, engine_b=self.b,
                                           snapshot=snapshot, tc_category="blitz")
        self.assertEqual(self.stored(match_id), (0, 0, 2, None, 0, 1))
        self.assertEqual(self.bins(match_id), [0, 0, 2, 0, 1])

        tournament = ratings.record_match(self.cnxn, source="tournament", engine_a=self.a, engine_b=self.b,
                                          wins=7, draws=6, losses=3, penta=[1, 2, 3, 1, 1], tc_category="blitz")
        self.assertEqual(self.stored(tournament), (1, 2, 3, None, 1, 1))
        self.assertEqual(self.bins(tournament), [1, 2, 3, 1, 1])

    def test_a_split_that_does_not_add_up_is_not_used(self):
        detail = {"LL": 1, "LD": 2, "DD": 1, "WL": 1, "WD": 1, "WW": 1}  # DD + WL = 2, the bin says 3
        match_id = ratings.record_match(self.cnxn, source="manual", engine_a=self.a, engine_b=self.b,
                                        wins=7, draws=6, losses=3, penta=[1, 2, 3, 1, 1], penta_detail=detail,
                                        tc_category="blitz")
        self.assertEqual(self.stored(match_id), (1, 2, 3, None, 1, 1))

    def test_results_without_pentanomial_counts_stay_null(self):
        match_id = ratings.record_match(self.cnxn, source="manual", engine_a=self.a, engine_b=self.b,
                                        wins=1, draws=0, losses=0, tc_category="blitz")
        self.assertEqual(self.stored(match_id), (None,) * 6)
        self.assertEqual(self.bins(match_id), [None] * 5)
        self.assertIsNone(ratings.record_snapshot(self.cnxn, source="sprt", engine_a=self.a, engine_b=self.b,
                                                  snapshot=_snapshot([]), tc_category="blitz"))

    def test_rows_recorded_before_the_split_keep_their_middle_bin(self):
        # match_results as the 2026_10_15 rating migration created it, with a combined middle bin
        old_table = next(m for m in MIGRATIONS if m.id == "2026_10_15_rating_tables").statements[0]
        self.cnxn.executescript(f"DROP TABLE match_results; {old_table};"
                                f"DELETE FROM schema_migrations WHERE id = '{PENTA_WL_MIGRATION}';")
        self.cnxn.execute("INSERT INTO match_results (source, engine_a_id, engine_b_id, tc_category, wins, draws, "
                          "losses, penta_ll, penta_ld, penta_dd, penta_wd, penta_ww) "
                          "VALUES ('tournament', ?, ?, 'blitz', 7, 6, 3, 1, 2, 3, 1, 1)", (self.a, self.b))
        self.cnxn.commit()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            ensure_schema(self.cnxn)
        self.assertIn(PENTA_WL_MIGRATION, out.getvalue())
        self.assertEqual(self.stored(1), (1, 2, 3, None, 1, 1))
        self.assertEqual(self.bins(1), [1, 2, 3, 1, 1])
        description = self.cnxn.execute("SELECT description FROM schema_migrations WHERE id = ?",
                                        (PENTA_WL_MIGRATION,)).fetchone()[0]
        self.assertIn("COALESCE(penta_wl, 0)", description)

    def test_commit_false_leaves_the_rows_in_the_callers_transaction(self):
        """tests/tournament.py stores a time control (experiment, games, results) as one transaction."""
        from data import etl
        experiment = etl.start_experiment(self.cnxn, "tournament", self.a, commit=False)
        ratings.record_match(self.cnxn, source="tournament", engine_a=self.a, engine_b=self.b, wins=1, draws=0,
                             losses=0, tc_category="blitz", experiment_id=experiment, commit=False)
        self.assertIsNone(ratings.record_snapshot(self.cnxn, source="sprt", engine_a=self.a, engine_b=self.b,
                                                  snapshot=_snapshot([]), tc_category="blitz", commit=False))
        ratings.record_snapshot(self.cnxn, source="sprt", engine_a=self.a, engine_b=self.b,
                                snapshot=_snapshot([(1, 0.5)]), tc_category="blitz", commit=False)
        etl.update_experiment(self.cnxn, experiment, {"end_time_utc": "2026-10-25T00:00:00+00:00"}, commit=False)
        self.assertTrue(self.cnxn.in_transaction)
        self.assertEqual(self.cnxn.execute("SELECT count(*) FROM match_results").fetchone(), (2,))
        self.cnxn.rollback()
        for table in ("experiments", "match_results"):
            self.assertEqual(self.cnxn.execute(f"SELECT count(*) FROM {table}").fetchone(), (0,), table)
        # by default each call commits, as before
        experiment = etl.start_experiment(self.cnxn, "SPRT", self.a)
        self.assertFalse(self.cnxn.in_transaction)
        ratings.record_snapshot(self.cnxn, source="sprt", engine_a=self.a, engine_b=self.b,
                                snapshot=_snapshot([(1, 0.5)]), tc_category="blitz", experiment_id=experiment)
        self.assertFalse(self.cnxn.in_transaction)
        etl.update_experiment(self.cnxn, experiment, {"end_time_utc": "2026-10-25T00:00:00+00:00"})
        self.assertFalse(self.cnxn.in_transaction)
        self.assertEqual(self.cnxn.execute("SELECT count(*) FROM match_results WHERE experiment_id = ?",
                                           (experiment,)).fetchone(), (1,))


class RunnerInterfaceTest(unittest.TestCase):
    """The calls tests/sprt.py, tests/tournament.py and nnue/registry.py make."""

    def test_category_of_every_limit(self):
        for limit, expected in ((dict(tc="10+0.1"), "ultra_fast"), (dict(tc="180+2"), "blitz"),
                                (dict(depth=8), "depth=8"), (dict(nodes=50000), "nodes=50000"),
                                (dict(time=0.1), "st=0.1")):
            with self.subTest(**limit):
                # tests/sprt.py passes every limit, the unused ones as None, and threads possibly None
                full = dict(dict(tc=None, depth=None, time=None, nodes=None), **limit)
                self.assertEqual(ratings.category(**full, threads=None), expected)
                self.assertEqual(ratings.category(**full, threads=1), expected)
                self.assertEqual(ratings.category(**full, threads=4), f"{expected}@T4")
        with self.assertRaises(ValueError):
            ratings.category(tc=None, depth=None, time=None, nodes=None, threads=1)

    def test_latest_fit_without_fits(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                path = init_raw_db(Path(tmp) / "chess.db")
            with contextlib.closing(sqlite3.connect(path)) as cnxn:
                self.assertIsNone(ratings.latest_fit(cnxn, "blitz"))


if __name__ == "__main__":
    unittest.main()
