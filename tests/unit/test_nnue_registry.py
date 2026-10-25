"""nnue/registry.py on fake training-run folders laid out like the bullet fork and train.py leave them.

A run folder <root>/<net>/ holds checkpoints <net>-<N>/ (quantised.bin, and log.txt /
val_log.txt with the loss record of the process that saved them: one curriculum stage),
train_config.json, and train.py's best-validation folders (best/, best_<val set>/,
best*_previous_<time>/: hard links to a checkpoint's files plus best.json).
Every database is a temporary one.
"""
import contextlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nnue"))

from data.databases.init_raw_db import init_raw_db  # noqa: E402
import nnue.registry as registry  # noqa: E402

NET = "net"
# stage 1 = superbatches 1-2 (checkpoint net-2), stage 2 = superbatches 3-4 (net-3 pruned, net-4 kept)
TRAIN = {1: [0.9, 0.7], 2: [0.6, 0.5], 3: [0.45, 0.4], 4: [0.35, 0.3]}
VAL = {1: 0.95, 2: 0.62, 3: 0.48, 4: 0.5}


def _log(rows):
    return "".join(f"{sb},{batch},{loss}\n" for sb, batch, loss in rows)


def _train_rows(superbatches):
    return [(sb, 32 * i, loss) for sb in superbatches for i, loss in enumerate(TRAIN[sb])]


class FakeRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name) / "checkpoints"
        self.run = self.root / NET
        self.run.mkdir(parents=True)
        (self.run / "train_config.json").write_text(json.dumps({"net_id": NET, "wdl_start": 0.0}))
        self.checkpoint(2, stage=[1, 2])
        self.checkpoint(4, stage=[3, 4], extra_val=[(4, 31, 0.9)])  # an earlier validation in superbatch 4
        with contextlib.redirect_stdout(io.StringIO()):
            self.db = init_raw_db(Path(self.tmp.name) / "chess.db")
        self.cnxn = sqlite3.connect(self.db)

    def tearDown(self):
        self.cnxn.close()
        self.tmp.cleanup()

    def checkpoint(self, superbatch, stage, extra_val=()):
        """<net>-<superbatch>/ with the logs of its stage up to superbatch."""
        path = self.run / f"{NET}-{superbatch}"
        (path / "optimiser_state").mkdir(parents=True)
        (path / "quantised.bin").write_bytes(f"net {superbatch}".encode())
        covered = [sb for sb in stage if sb <= superbatch]
        (path / "log.txt").write_text(_log(_train_rows(covered)))
        val = sorted([*extra_val, *((sb, 63, VAL[sb]) for sb in covered)])
        (path / "val_log.txt").write_text(_log(val))
        return path

    def best_folder(self, name, superbatch, val_loss, source=None):
        """A best folder like train.py's BestTracker writes: the files of `source` and best.json."""
        path = self.run / name
        if source is not None:
            shutil.copytree(source, path, copy_function=os.link)
        else:
            path.mkdir()
            (path / "quantised.bin").write_bytes(f"earlier run, net {superbatch}".encode())
        (path / "best.json").write_text(json.dumps(dict(
            net_id=NET, superbatch=superbatch, val_loss=val_loss, val_data="val.binpack", wdl=[0.0, 0.0],
            checkpoint=f"{NET}-{superbatch}", saved="2026-10-25T10:00:00")))
        return path

    def nets(self):
        rows = self.cnxn.execute("SELECT run_id, superbatch, file_path, val_loss, train_loss, best_val_loss, "
                                 "best_val_superbatch, train_config FROM nets ORDER BY superbatch, id").fetchall()
        keys = ("run", "superbatch", "file", "val", "train", "best_val", "best_sb", "config")
        return [dict(zip(keys, row)) for row in rows]


class Losses(FakeRuns):
    def test_losses_come_from_the_checkpoint_logs_of_every_stage(self):
        losses = registry.load_losses(self.run)
        self.assertEqual(sorted(losses), [1, 2, 3, 4])
        for sb in losses:
            self.assertAlmostEqual(losses[sb]["train"], sum(TRAIN[sb]) / len(TRAIN[sb]))
            self.assertAlmostEqual(losses[sb]["val"], VAL[sb], msg="the superbatch's last validation")

    def test_a_run_without_checkpoints_has_no_losses(self):
        self.assertEqual(registry.load_losses(Path(self.tmp.name) / "empty"), {})


class Scan(FakeRuns):
    def test_checkpoints_are_registered_with_their_losses_and_settings(self):
        self.assertEqual(len(registry.scan(self.cnxn, self.root, NET)), 2)
        two, four = self.nets()
        self.assertEqual((two["run"], two["superbatch"], two["val"]), (NET, 2, VAL[2]))
        self.assertAlmostEqual(two["train"], 0.55)
        self.assertEqual((four["superbatch"], four["val"]), (4, VAL[4]))
        for net in (two, four):
            self.assertEqual((net["best_val"], net["best_sb"]), (VAL[3], 3))
            self.assertEqual(json.loads(net["config"])["net_id"], NET)
        # scanning again refreshes the same rows
        registry.scan(self.cnxn, self.root)
        self.assertEqual(len(self.nets()), 2)

    def test_a_best_folder_is_skipped_while_its_checkpoint_exists(self):
        self.best_folder("best", 4, VAL[4], source=self.run / f"{NET}-4")
        self.assertEqual(len(registry.scan(self.cnxn, self.root, NET)), 2)
        self.assertFalse(any("best" in Path(n["file"]).parent.name for n in self.nets()))

    def test_a_best_folder_of_a_pruned_checkpoint_is_registered_as_that_checkpoint(self):
        # best state at superbatch 3 of stage 2; checkpoint net-3 was pruned. The folder name
        # (best_<val set>) ends in -<digits> like a checkpoint's: best.json decides.
        source = self.checkpoint(3, stage=[3, 4])
        best = self.best_folder("best_holdout-7", 3, VAL[3], source=source)
        shutil.rmtree(source)
        self.assertEqual(len(registry.scan(self.cnxn, self.root, NET)), 3)
        three = next(n for n in self.nets() if n["superbatch"] == 3)
        self.assertEqual((three["run"], Path(three["file"]).parent), (NET, best))
        self.assertEqual((three["val"], three["best_val"], three["best_sb"]), (VAL[3], VAL[3], 3))
        self.assertAlmostEqual(three["train"], sum(TRAIN[3]) / 2)
        self.assertIsNotNone(three["config"])
        self.assertEqual(sorted(registry.load_losses(self.run)), [1, 2, 3, 4], "best folders are not checkpoints")

    def test_an_archived_best_of_an_earlier_run_is_described_by_its_best_json(self):
        # an earlier run with the same net id kept superbatch 2 at loss 0.4; this run's net-2 is another net
        self.best_folder("best_previous_20261024_120000", 2, 0.4)
        self.assertEqual(len(registry.scan(self.cnxn, self.root, NET)), 3)
        archived = next(n for n in self.nets() if "previous" in n["file"])
        self.assertEqual((archived["run"], archived["superbatch"], archived["val"]), (NET, 2, 0.4))
        self.assertEqual((archived["best_val"], archived["best_sb"]), (0.4, 2))
        self.assertIsNone(archived["train"], "this run's logs are not the earlier run's")
        self.assertIsNone(archived["config"], "this run's settings are not the earlier run's")

    def test_adding_a_best_folder_directly_maps_it_too(self):
        best = self.best_folder("best", 4, VAL[4], source=self.run / f"{NET}-4")
        net_id = registry.register_net(self.cnxn, best)
        row = self.cnxn.execute("SELECT run_id, superbatch, val_loss FROM nets WHERE id = ?", (net_id,)).fetchone()
        self.assertEqual(row, (NET, 4, VAL[4]))


class Report(FakeRuns):
    def test_report_uses_the_latest_rating_fit_of_the_category(self):
        registry.scan(self.cnxn, self.root, NET)
        digest = registry.sha256_file(self.run / f"{NET}-4" / "quantised.bin")
        engine = self.cnxn.execute("INSERT INTO engines (name, version, net_sha256) VALUES ('tomahawk', '2.0', ?)",
                                   (digest,)).lastrowid
        for elo in (1400.0, 1550.0):  # the second fit is the latest
            fit = self.cnxn.execute("INSERT INTO rating_fits (tc_category, sources) VALUES ('blitz', 'tournament')"
                                    ).lastrowid
            self.cnxn.execute("INSERT INTO ratings (fit_id, engine_id, elo, se, games, points, component) "
                              "VALUES (?, ?, ?, 10.0, 200, 100.0, 0)", (fit, engine, elo))
        self.cnxn.commit()
        rows, fit_id = registry.report_rows(self.cnxn, "blitz")
        self.assertEqual(fit_id, fit)
        rated = next(r for r in rows if r["superbatch"] == 4)
        self.assertEqual((rated["engines"], rated["elo"], rated["games"]), ([engine], 1550.0, 200))
        self.assertIn("NNUE nets vs Elo (blitz", registry.format_report(rows, "blitz", fit_id))
        self.assertEqual(registry.report_rows(self.cnxn, "rapid")[1], None)


class TrainHook(FakeRuns):
    def test_train_registers_the_run_in_the_configured_database(self):
        import registry as top_level  # the module train.py imports (nnue/ on sys.path)
        import nnue.train as train
        args = mock.Mock(no_register=False, output_dir=str(self.root), net_id=NET)
        with mock.patch.object(top_level, "RAW_DB", self.db), contextlib.redirect_stdout(io.StringIO()) as out:
            train.register_checkpoints(args)
        self.assertIn("registered 2 checkpoint(s)", out.getvalue())
        self.assertEqual(self.cnxn.execute("SELECT COUNT(*) FROM nets").fetchone()[0], 2)
        args.no_register = True
        with mock.patch.object(top_level, "RAW_DB", Path(self.tmp.name) / "unused.db"):
            train.register_checkpoints(args)
        self.assertFalse((Path(self.tmp.name) / "unused.db").exists())


# Run in a fresh interpreter where every matplotlib import fails, as on a machine without it.
WITHOUT_MATPLOTLIB = textwrap.dedent("""
    import json, sys
    from types import SimpleNamespace
    sys.modules["matplotlib"] = None
    project, root, net = sys.argv[1:4]
    sys.path[:0] = [project, project + "/nnue"]
    import nnue.train as train
    import nnue.registry as registry
    import plot
    try:
        plot.create_dashboard()
        drawing = "drew"
    except ImportError:
        drawing = "ImportError"
    train.register_checkpoints(SimpleNamespace(no_register=False, output_dir=root, net_id=net))
    registry.main(["scan", "--root", root, "--run-id", net])
    print(json.dumps(dict(drawing=drawing, losses=sorted(registry.load_losses(root + "/" + net)))))
""")


class WithoutMatplotlib(FakeRuns):
    def test_registration_and_scan_work_without_matplotlib(self):
        env = dict(os.environ, CHESS_RAW_DB=str(self.db), CHESS_ANALYTICS_DB=str(Path(self.tmp.name) / "a.duckdb"),
                   PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", WITHOUT_MATPLOTLIB, str(ROOT), str(self.root), NET],
                                capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("registered 2 checkpoint(s)", result.stdout)  # train.py's hook
        self.assertIn("registered 2 net(s)", result.stdout)  # python -m nnue.registry scan
        out = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(out["drawing"], "ImportError", "matplotlib must really be unavailable in the child")
        self.assertEqual(out["losses"], [1, 2, 3, 4])
        self.assertEqual(self.cnxn.execute("SELECT COUNT(*) FROM nets WHERE val_loss IS NOT NULL").fetchone()[0], 2)


if __name__ == "__main__":
    unittest.main()
