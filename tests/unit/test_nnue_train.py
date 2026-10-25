"""nnue/train.py curriculum orchestration and nnue/plot.py checkpoint-log loading.

cargo is replaced by a fake trainer that behaves like the bullet fork: it saves
<output_dir>/<net_id>-<superbatch>/ every save_rate superbatches and at the end
of the run, each with optimiser_state/weights.bin and the run's complete
log.txt / val_log.txt, and it records the environment it was started with.
"""
import argparse
import json
import os
import sys
import tempfile
import textwrap
import threading
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
                                      "save_rate", "wdl_start", "wdl_end", "net_id")}
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
    ENV_SETTINGS = ("net_id", "wdl_start", "wdl_end")

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
        # settings train.py would otherwise take from the environment
        self.saved_env = {k: os.environ.pop(k) for k in self.ENV_SETTINGS if k in os.environ}

    def tearDown(self):
        os.environ.pop("FAKE_LOG", None)
        os.environ.pop("FAKE_SKIP_FINAL", None)
        for k in self.ENV_SETTINGS:
            os.environ.pop(k, None)
        os.environ.update(self.saved_env)
        self.tmp.cleanup()

    def run_main(self, *extra, example="no_such_script", net_id="net"):
        # The default example has no nnue/<name>.rs, so train.py's own fallbacks apply (a constant
        # 0.25 WDL) and these tests do not depend on the defaults in nnue/v2.rs / v3.rs.
        # --no-register: the net registry is the real raw database unless CHESS_RAW_DB says otherwise.
        argv = ["train.py", "--no-plot", "--no-register", "--example", example,
                "--output-dir", str(self.dir / "ckpt"), "--save-rate", "2", *extra]
        if net_id is not None:
            argv += ["--net-id", net_id]
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

    @unittest.skipUnless(os.name == "nt", "Windows: a directory holding an open file cannot be removed or renamed")
    def test_a_file_held_open_for_a_moment_does_not_lose_a_new_best(self):
        # as an antivirus scanning the files just written does: the old best/ cannot be removed (nor the
        # new one renamed) until it lets go, which once failed test_best_state_is_kept_at_any_superbatch
        out = self.dir / "ckpt" / "net"
        for superbatch, loss in ((1, 2.0), (2, 1.0)):
            checkpoint = Path(train.checkpoint_path(str(out), "net", superbatch))
            checkpoint.mkdir(parents=True)
            (checkpoint / "quantised.bin").write_text(f"net {superbatch}")
            (checkpoint / "val_log.txt").write_text(f"{superbatch},63,{loss}\n")
        tracker = train.BestTracker(str(out), "net", 0, None, (0.25, 0.25))
        tracker.on_saved(1)
        held = open(out / "best" / "quantised.bin")
        timer = threading.Timer(0.3, held.close)
        timer.start()
        try:
            tracker.on_saved(2)
        finally:
            timer.join()
            held.close()
        self.assertEqual(json.loads((out / "best" / "best.json").read_text())["superbatch"], 2)
        self.assertEqual((out / "best" / "quantised.bin").read_text(), "net 2")
        self.assertFalse((out / "best.new").exists())

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

    def test_wdl_and_net_id_defaults_come_from_the_training_script(self):
        for example in ("v2", "v3"):
            with self.subTest(example=example):
                defaults = train.script_defaults(example)
                for key, kind in (("wdl_start", float), ("wdl_end", float), ("net_id", str)):
                    self.assertIsInstance(defaults.get(key), kind, f"nnue/{example}.rs: literal default of {key}")
                code = self.run_main("--train-data", str(self.dir / "a.binpack"), "--superbatches", "2",
                                     example=example, net_id=None)
                self.assertEqual(code, 0)
                record = self.records()[-1]
                self.assertEqual(record["net_id"], defaults["net_id"])
                self.assertEqual((float(record["wdl_start"]), float(record["wdl_end"])),
                                 (defaults["wdl_start"], defaults["wdl_end"]))

    def test_flags_and_environment_override_the_script_defaults(self):
        os.environ["wdl_start"], os.environ["net_id"] = "0.3", "from_env"
        code = self.run_main("--train-data", str(self.dir / "a.binpack"), "--superbatches", "2",
                             "--wdl-end", "0.6", example="v3", net_id=None)
        self.assertEqual(code, 0)
        record = self.records()[0]
        self.assertEqual((record["net_id"], record["wdl_start"], record["wdl_end"]), ("from_env", "0.3", "0.6"))

    def test_script_defaults_are_the_literal_ones(self):
        nnue = self.dir / "nnue"
        nnue.mkdir()
        (nnue / "vx.rs").write_text(textwrap.dedent("""
            let wdl_start = env_f32("wdl_start", 0.0);
            let wdl_end = env_f32("wdl_end", 0.25); // env_f32("wdl_end", 0.9)
            /* let lr = env_f32("lr_start", 9.0); */
            let final_lr = env_f32("lr_final", initial_lr * 0.3f32.powi(5));
            let batch_size = env_usize("batch_size", 16_384);
            let net_id = env_string("net_id", "1024x16x32_0wdl");
            let train_path = env_string("train_data", "F:/training_data/a.binpack");
        """))
        with mock.patch.object(train, "NNUE_SJ_ROOT", str(nnue)):
            self.assertEqual(train.script_defaults("vx"),
                             {"wdl_start": 0.0, "wdl_end": 0.25, "batch_size": 16384, "net_id": "1024x16x32_0wdl",
                              "train_data": "F:/training_data/a.binpack"})
            self.assertEqual(train.script_defaults("missing"), {})


class BuildRoute(unittest.TestCase):
    MANIFEST = ROOT / "libs" / "bullet" / "crates" / "bullet_lib" / "Cargo.toml"

    def test_scripts_run_as_cargo_examples_of_the_fork(self):
        args = argparse.Namespace(example="v3", features="cuda")
        self.assertEqual(train.build_command(args),
                         ["cargo", "run", "--release", "--example", "v3", "--features", "cuda"])

    @unittest.skipUnless(MANIFEST.is_file(), "libs/bullet submodule not checked out")
    def test_the_fork_registers_the_training_scripts(self):
        import tomllib
        with open(self.MANIFEST, "rb") as f:
            examples = {e["name"]: e["path"] for e in tomllib.load(f).get("example", [])}
        for name in ("v2", "v3"):
            self.assertIn(name, examples)
            target = (self.MANIFEST.parent / examples[name]).resolve()
            self.assertEqual(target, (ROOT / "nnue" / f"{name}.rs").resolve())


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
