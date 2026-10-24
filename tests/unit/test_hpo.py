"""nnue/hpo.py: search space, train.py flags, loss extraction, successive halving and resume."""
import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nnue"))

import hpo  # noqa: E402  (nnue/hpo.py)


def args_for(**kw):
    base = dict(min_superbatches=5, eta=3, rungs=3, initial=9, brackets=1, max_hours=None, seed=3,
                random=True, example="v3", score_cmd=None)
    base.update(kw)
    return argparse.Namespace(**base)


class Pieces(unittest.TestCase):
    def test_space_maps_the_unit_interval(self):
        space = hpo.Space({"a": {"type": "log", "low": 1e-4, "high": 1e-2},
                           "b": {"type": "int", "low": 1, "high": 3},
                           "c": {"type": "choice", "values": [8, 16]}})
        self.assertAlmostEqual(space.from_unit("a", 0.5), 1e-3)
        self.assertEqual([space.from_unit("b", u) for u in (0.0, 0.5, 0.999)], [1, 2, 3])
        self.assertEqual(space.from_unit("c", 0.9), 16)

    def test_flags_derive_the_final_lr(self):
        flags = hpo.train_flags({"lr_start": 0.002, "lr_final_ratio": 0.01, "batch_size": 16384, "wdl": 0.3})
        self.assertEqual(flags, ["--lr_start", "0.002", "--batch-size", "16384", "--wdl-start", "0.3",
                                 "--wdl-end", "0.3", "--lr_final", "2e-05"])

    def test_final_loss_is_the_mean_of_the_last_validations(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "t1-9"
            ckpt.mkdir()
            (ckpt / "val_log.txt").write_text("".join(f"{sb},63,{loss}\n" for sb, loss in
                                                      enumerate([0.9, 0.5, 0.4, 0.3, 0.2], 5)))
            self.assertAlmostEqual(hpo.final_val_loss(tmp, "t1", 9), 0.3)
            self.assertIsNone(hpo.final_val_loss(tmp, "t1", 10))

    def test_a_curriculum_is_scored_at_the_end_of_its_last_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "curriculum"
            data.mkdir()
            for name in ("01_a.binpack", "02_b.binpack", "03_c.binpack"):
                (data / name).write_bytes(b"")
            args = args_for(train_data=[str(data)], val_data=None, features=None, train_args=[], keep_runs=True)
            evaluator = hpo.TrainEvaluator(args, Path(tmp) / "study")
            self.assertEqual(evaluator.stages, 3)

            def fake_train(cmd, **kw):
                runs, net_id = Path(cmd[cmd.index("--output-dir") + 1]), cmd[cmd.index("--net-id") + 1]
                for sb, loss in ((5, 0.9), (15, 0.25)):  # end of stage 1, end of the curriculum
                    ckpt = runs / net_id / f"{net_id}-{sb}"
                    ckpt.mkdir(parents=True)
                    (ckpt / "val_log.txt").write_text(f"{sb},63,{loss}\n")
                return 0

            with mock.patch.object(hpo.subprocess, "call", side_effect=fake_train):
                loss = evaluator(1, 0, 5, {"lr_start": 1e-3, "lr_final_ratio": 0.01}, False)
            self.assertAlmostEqual(loss, 0.25)


class Search(unittest.TestCase):
    def test_successive_halving_promotes_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(hpo, "HPO_ROOT", Path(tmp)):
            study = hpo.Study(Path(tmp) / "s")
            calls = []

            def evaluator(trial, rung, budget, params, last):
                calls.append((trial, rung, budget))
                return (math.log(params["lr_start"] / 1e-3)) ** 2 + 1 / budget

            summary = hpo.run_search(args_for(), study, hpo.Space(hpo.DEFAULT_SPACE), evaluator)
            self.assertEqual([c[1] for c in calls].count(0), 9)
            self.assertEqual([c[1] for c in calls].count(1), 3)
            self.assertEqual([c[1] for c in calls].count(2), 1)
            self.assertEqual(summary["best"]["budget"], 45)
            # Resuming the finished study evaluates nothing again.
            calls.clear()
            hpo.run_search(args_for(), hpo.Study(Path(tmp) / "s"), hpo.Space(hpo.DEFAULT_SPACE), evaluator)
            self.assertEqual(calls, [])

    def test_failed_trials_are_not_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = hpo.Study(Path(tmp) / "s")

            def evaluator(trial, rung, budget, params, last):
                return None if trial % 2 else params["lr_start"]

            hpo.run_search(args_for(rungs=2), study, hpo.Space(hpo.DEFAULT_SPACE), evaluator)
            promoted = [r["trial"] for r in study.records if r["rung"] == 1]
            self.assertTrue(promoted and all(t % 2 == 0 for t in promoted))


if __name__ == "__main__":
    unittest.main()
