"""nnue/train.py curriculum orchestration and nnue/plot.py checkpoint-log loading.

cargo is replaced by a fake trainer that behaves like the bullet fork: it saves
<output_dir>/<net_id>-<superbatch>/ every save_rate superbatches and at the end
of the run, each with optimiser_state/weights.bin and the run's complete
log.txt / val_log.txt, and it records the environment it was started with.
"""
import json
import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nnue"))

import nnue.train as train  # noqa: E402
import plot  # noqa: E402  (nnue/plot.py)

FAKE_TRAINER = textwrap.dedent("""
    import json, os, sys
    env = os.environ
    start, count = int(env["superbatch_start"]), int(env["superbatches"])
    out, net = env["output_dir"], env["net_id"]
    save_rate = int(env.get("save_rate", "10"))
    end = start + count - 1
    record = {k: env.get(k) for k in ("superbatch_start", "superbatches", "lr_final_superbatch",
                                      "wdl_final_superbatch", "reset_optimiser", "train_data", "val_data",
                                      "save_rate")}
    if start > 1:
        previous = os.path.join(out, f"{net}-{start - 1}", "optimiser_state", "weights.bin")
        record["loaded"] = os.path.exists(previous)
    with open(env["FAKE_LOG"], "a") as f:
        f.write(json.dumps(record) + "\\n")
    if env.get("FAKE_SKIP_FINAL") == str(end):
        sys.exit(0)
    val_losses = [float(x) for x in env["FAKE_VAL"].split(",")] if env.get("FAKE_VAL") else None
    train_rows, val_rows = [], []
    for sb in range(start, end + 1):
        for batch in range(0, 64, 32):
            train_rows.append(f"{sb},{batch},{1.0 / sb:.6f}")
        val = val_losses[sb - 1] if val_losses else 1.1 / sb
        val_rows.append(f"{sb},63,{val:.6f}")
        print(f"[TRAINING] superbatch {sb} | running loss {1.0 / sb:.6f}", flush=True)
        if sb % save_rate == 0 or sb == end:
            path = os.path.join(out, f"{net}-{sb}")
            os.makedirs(os.path.join(path, "optimiser_state"), exist_ok=True)
            open(os.path.join(path, "optimiser_state", "weights.bin"), "w").write(f"weights {sb}")
            open(os.path.join(path, "quantised.bin"), "w").write(f"net {sb}")
            open(os.path.join(path, "log.txt"), "w").write("\\n".join(train_rows) + "\\n")
            open(os.path.join(path, "val_log.txt"), "w").write("\\n".join(val_rows) + "\\n")
            print(f"Saved [\\x1b[31m{net}-{sb}\\x1b[0m]", flush=True)
        if env.get("FAKE_CRASH_AT") == str(sb):
            sys.exit(1)
""")


class TrainOrchestration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.fake = self.dir / "fake_trainer.py"
        self.fake.write_text(FAKE_TRAINER)
        self.log = self.dir / "env.jsonl"
        for name in ("a.binpack", "b.binpack", "val.binpack"):
            (self.dir / name).write_bytes(b"")
        os.environ["FAKE_LOG"] = str(self.log)
        os.environ.pop("FAKE_SKIP_FINAL", None)

    def tearDown(self):
        os.environ.pop("FAKE_LOG", None)
        os.environ.pop("FAKE_SKIP_FINAL", None)
        self.tmp.cleanup()

    def run_main(self, *extra):
        argv = ["train.py", "--no-plot", "--output-dir", str(self.dir / "ckpt"), "--net-id", "net",
                "--save-rate", "2", *extra]
        with mock.patch.object(train, "build_command", lambda args: [sys.executable, str(self.fake)]), \
                mock.patch.object(sys, "argv", argv):
            return train.main()

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_curriculum_chains_checkpoints_and_global_schedules(self):
        code = self.run_main("--train-data", str(self.dir / "a.binpack"), str(self.dir / "b.binpack"),
                             "--superbatches", "3", "2", "--val-data", str(self.dir / "val.binpack"),
                             "--wdl-start", "0.1", "--wdl-end", "0.4")
        self.assertEqual(code, 0)
        first, second = self.records()
        self.assertEqual(first["superbatch_start"], "1")
        self.assertEqual(second["superbatch_start"], "4")
        self.assertTrue(second["loaded"], "stage 2 must continue from checkpoint net-3")
        for stage in (first, second):
            self.assertEqual(stage["lr_final_superbatch"], "5")
            self.assertEqual(stage["wdl_final_superbatch"], "5")
            self.assertTrue(os.path.isabs(stage["train_data"]) and os.path.isabs(stage["val_data"]))
        series = plot.load_metrics(str(self.dir / "ckpt" / "net"), batches_per_superbatch=64)
        superbatches = sorted({int(x) for x, _ in series["train"]})
        self.assertEqual(superbatches, [1, 2, 3, 4, 5])
        self.assertEqual(len(series["val"]), 5)

    def test_missing_final_checkpoint_stops_the_curriculum(self):
        os.environ["FAKE_SKIP_FINAL"] = "3"
        code = self.run_main("--train-data", str(self.dir / "a.binpack"), str(self.dir / "b.binpack"),
                             "--superbatches", "3", "2")
        self.assertEqual(code, 4)
        self.assertEqual(len(self.records()), 1, "stage 2 must not start without stage 1's checkpoint")

    def test_resume_requires_the_checkpoint(self):
        code = self.run_main("--train-data", str(self.dir / "a.binpack"), "--superbatch-start", "7",
                             "--superbatches", "2")
        self.assertEqual(code, 3)
        self.assertFalse(self.log.exists())

    def test_reset_optimiser_flag_is_forwarded(self):
        self.run_main("--train-data", str(self.dir / "a.binpack"), str(self.dir / "b.binpack"),
                      "--superbatches", "2", "--reset-optimiser")
        self.assertEqual([r["reset_optimiser"] for r in self.records()], ["1", "1"])

    def test_relative_dataset_paths_become_absolute(self):
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            self.assertEqual(train.resolve_dataset("a.binpack", "unused"), str(self.dir / "a.binpack"))
            with self.assertRaises(FileNotFoundError):
                train.resolve_dataset("missing.binpack", str(self.dir))
        finally:
            os.chdir(cwd)

    def test_best_state_is_kept_at_any_superbatch(self):
        # lowest validation at superbatch 4, although only every 5th checkpoint is kept
        os.environ["FAKE_VAL"] = "5.0,4.0,3.0,2.5,2.7,2.6,2.55"
        try:
            code = self.run_main("--train-data", str(self.dir / "a.binpack"), "--superbatches", "7",
                                 "--save-rate", "5")
        finally:
            os.environ.pop("FAKE_VAL", None)
        self.assertEqual(code, 0)
        self.assertEqual(self.records()[0]["save_rate"], "1")
        net = self.dir / "ckpt" / "net"
        info = json.loads((net / "best" / "best.json").read_text())
        self.assertEqual((info["superbatch"], info["val_loss"]), (4, 2.5))
        self.assertEqual((net / "best" / "optimiser_state" / "weights.bin").read_text(), "weights 4")
        kept = sorted(int(p.name.split("-")[1]) for p in net.glob("net-*"))
        self.assertEqual(kept, [5, 7], "only every 5th checkpoint and the stage end remain")

    def test_no_best_tracking_while_the_wdl_blend_changes(self):
        code = self.run_main("--train-data", str(self.dir / "a.binpack"), "--superbatches", "4",
                             "--save-rate", "2", "--wdl-start", "0.0", "--wdl-end", "0.5")
        self.assertEqual(code, 0)
        self.assertEqual(self.records()[0]["save_rate"], "2")
        self.assertFalse((self.dir / "ckpt" / "net" / "best").exists())

    def test_zero_wdl_is_constant_and_tracked(self):
        code = self.run_main("--train-data", str(self.dir / "a.binpack"), "--superbatches", "3",
                             "--wdl-start", "0", "--wdl-end", "0")
        self.assertEqual(code, 0)
        self.assertTrue((self.dir / "ckpt" / "net" / "best" / "best.json").exists())

    def test_temporary_checkpoints_are_removed_when_the_trainer_dies(self):
        os.environ["FAKE_VAL"] = "5,4,3,2,1,0.5,0.4,0.3,0.2"
        os.environ["FAKE_CRASH_AT"] = "7"
        try:
            code = self.run_main("--train-data", str(self.dir / "a.binpack"), "--superbatches", "9",
                                 "--save-rate", "5")
        finally:
            os.environ.pop("FAKE_VAL", None)
            os.environ.pop("FAKE_CRASH_AT", None)
        self.assertEqual(code, 1)
        net = self.dir / "ckpt" / "net"
        kept = sorted(int(p.name.split("-")[1]) for p in net.glob("net-*"))
        self.assertEqual(kept, [5], "only the save-rate checkpoint survives a crash")
        self.assertEqual(json.loads((net / "best" / "best.json").read_text())["superbatch"], 7)


class CheckpointLogs(unittest.TestCase):
    def test_stale_checkpoints_of_an_earlier_run_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            net = Path(tmp)
            for sb, loss in ((2, 9.0), (4, 9.0)):
                path = net / f"net-{sb}"
                path.mkdir()
                (path / "log.txt").write_text("".join(f"{s},0,{loss}\n" for s in range(1, sb + 1)))
            old = time.time() - 3600
            os.utime(net / "net-4" / "log.txt", (old, old))
            os.utime(net / "net-2" / "log.txt", (old, old))
            fresh = net / "net-2" / "log.txt"
            fresh.write_text("1,0,1.0\n2,0,0.5\n")
            series = plot.load_checkpoint_logs(str(net), batches_per_superbatch=1, since=time.time() - 60)
            self.assertEqual(series["train"], [(1.0, 1.0), (2.0, 0.5)])


if __name__ == "__main__":
    unittest.main()
