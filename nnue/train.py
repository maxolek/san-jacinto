#!/usr/bin/env python3
"""
Training wrapper around the bullet Rust trainer (the fork in libs/bullet).

Primary purpose: make long training runs visual-friendly. It launches the
training script, tees its (colourful) output to your console and a log file,
and shows a live-updating loss plot fed by the loss logs bullet writes into
every checkpoint (<net>/<net_id>-<superbatch>/log.txt and val_log.txt), so the
plot advances each time a checkpoint is saved (--save-rate).

Training scripts: --example NAME runs `cargo run --release --example NAME` in
libs/bullet. The fork registers san-jacinto's scripts nnue/v2.rs and
nnue/v3.rs as its examples v2 and v3 (libs/bullet/crates/bullet_lib/Cargo.toml);
any other NAME is one of bullet's own examples. Tuning flags are forwarded to
the script as environment variables; settings not given are left to the
script's defaults, except the WDL blend and the net id, which train.py needs
itself: it reads their defaults from nnue/NAME.rs and always passes them, so a
run through train.py trains what `cargo run --example NAME` would.

Multi-dataset (curriculum) runs:
    Pass --train-data with more than one path (or a folder) and each dataset is
    run as its own sequential stage: stage 1 trains from scratch, stage 2+
    start at the next global superbatch and load the checkpoint the previous
    stage saved at its last superbatch (same --output-dir / --net-id), then
    continue on the new dataset. Before each stage the checkpoint is checked to
    exist, and after it the stage's own final checkpoint, so a failed stage
    stops the curriculum instead of silently training from random weights.
    --reset-optimiser loads only the weights at each stage boundary (fresh
    AdamW moments); by default the optimiser state carries over too.

    Schedules across stages: bullet indexes its cosine LR by the global
    superbatch number. Every stage receives the same `lr_start`, `lr_final`,
    and `lr_final_superbatch` (the final superbatch of the curriculum), and the
    WDL blend tapers over the same global range (`wdl_final_superbatch`), so
    both follow one curve across all stages, matching the dashboard.
    `superbatches` accepts one shared length or one length per dataset stage.
    --superbatch-start N resumes a run from checkpoint N-1.

Examples:
    python nnue/train.py --example v3 --features cuda
    python nnue/train.py --example v3 --superbatches 400 --lr_start 0.0008 \
        --train-data train.binpack --val-data val.binpack

    # curriculum: three stages, one dataset each, checkpoints chained
    python nnue/train.py --example v3 --net-id my_net \
        --train-data stage1.binpack stage2.binpack stage3.binpack --superbatches 200 400 800 \
        --val-data val.binpack

    python nnue/train.py --example v3 --no-plot        # headless
Dataset paths are taken as given, or else relative to libs/bullet/data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
# train.py lives in /san-jacinto/nnue/
SAN_JACINTO_ROOT = os.path.dirname(HERE)
NNUE_SJ_ROOT = os.path.join(SAN_JACINTO_ROOT, "nnue")
# Bullet lives in /san-jacinto/libs/bullet/
BULLET_ROOT = os.path.join(SAN_JACINTO_ROOT, "libs", "bullet")
DATA_ROOT = os.path.join(BULLET_ROOT, "data")

DATASET_EXTENSIONS = (".binpack", ".bin")
def resolve_dataset(entry: str, data_root: str) -> str:
    """Absolute path of a dataset file/folder: as given, or else under `data_root`.

    Paths are made absolute because cargo runs with libs/bullet as its working
    directory, where a path relative to wherever train.py was started would
    silently point somewhere else.
    """
    for candidate in (entry, os.path.join(data_root, entry)):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"dataset not found: {entry} (also looked in {data_root})")


def expand_train_data(entries: list[str], data_root: str) -> list[str]:
    """Resolve --train-data entries into a flat, ordered list of dataset
    file paths. Each entry can be:
      - a file path (used as-is, one stage)
      - a folder path -- every dataset file inside becomes its own stage,
        in sorted filename order (so e.g. 01_opening.binpack,
        02_midgame.binpack, ... gives you explicit control over order;
        otherwise plain alphabetical is used)

    Bare names (no existing relative/absolute path) are also tried
    relative to `data_root`, so `--train-data curriculum_v1` resolves to
    `<bullet>/data/curriculum_v1` without needing the full path.
    """
    expanded: list[str] = []

    for entry in entries:
        candidate = resolve_dataset(entry, data_root)

        if os.path.isdir(candidate):
            files = sorted(
                os.path.join(candidate, f)
                for f in os.listdir(candidate)
                if f.lower().endswith(DATASET_EXTENSIONS)
            )

            if not files:
                print(f"[train] warning: no dataset files "
                      f"({', '.join(DATASET_EXTENSIONS)}) found in {candidate}")

            expanded.extend(files)
        else:
            expanded.append(candidate)

    return expanded


def resolve_lr_bounds(args) -> tuple[float, float]:
    """The (initial_lr, final_lr) pair for the *whole* curriculum.

    Shared by the plot config and the trainer environment so
    both always agree on the same global endpoints.
    """
    initial_lr = args.lr_start if args.lr_start is not None else 0.001
    final_lr = args.lr_final if args.lr_final is not None else initial_lr * 0.3 ** 5
    return initial_lr, final_lr


def resolve_stage_lengths(superbatches, stage_count):
    """Broadcast a single stage length, or validate one per expanded dataset."""
    lengths = [superbatches] if isinstance(superbatches, int) else list(superbatches)
    if not lengths or any(n < 1 for n in lengths):
        raise ValueError("--superbatches values must be positive")
    if len(lengths) == 1:
        return lengths * stage_count
    if len(lengths) != stage_count:
        raise ValueError(f"--superbatches has {len(lengths)} values for {stage_count} "
                         "resolved dataset stages; pass one value or one per stage")
    return lengths


def build_stage_boundaries(train_stages, lengths, start=1, example="training"):
    boundaries = []
    for dataset, length in zip(train_stages, lengths):
        end = start + length - 1
        boundaries.append((start, end, os.path.basename(dataset) if dataset else example))
        start = end + 1
    return boundaries


def _cosine_lr(
    superbatch: float,
    initial_lr: float,
    final_lr: float,
    final_superbatch: int,
) -> float:
    """Value of the single, curriculum-wide cosine schedule at `superbatch`.

    Deliberately mirrors plot._cosine_lr (same formula, same clamping) so
    the per-stage diagnostic rates line up with the curve the dashboard
    draws. Kept as a plain-math duplicate rather than importing
    plot.py, so this part of train.py does not depend on the plotting module
    (which itself imports matplotlib only when drawing).
    """
    if final_superbatch <= 0:
        return final_lr

    progress = min(1.0, max(0.0, superbatch / final_superbatch))

    return final_lr + 0.5 * (initial_lr - final_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def build_plot_config(args, train_data, val_data, final_superbatch):
    import plot

    initial_lr, final_lr = resolve_lr_bounds(args)
    wdl_start, wdl_end = effective_wdl(args)

    return plot.PlotConfig(
        initial_lr=initial_lr,
        final_lr=final_lr,
        final_superbatch=final_superbatch,

        wdl_start=wdl_start,
        wdl_end=wdl_end,

        batch_size=args.batch_size if args.batch_size is not None else 16_384,
        batches_per_superbatch=args.batches if args.batches is not None else 6104,
        threads=args.threads if args.threads is not None else 4,

        net_id=args.net_id,
        train_data=train_data,
        val_data=val_data,

        l1=args.L1 if args.L1 is not None else 1024,
        l2=args.L2 if args.L2 is not None else 16,
        l3=args.L3 if args.L3 is not None else 32,

        qa=args.QA if args.QA is not None else 127,
        qb=args.QB if args.QB is not None else 64,
        qc=args.QC if args.QC is not None else 64,
    )

def local_script(example: str) -> str | None:
    """nnue/<example>.rs if the example is one of san-jacinto's training scripts."""
    path = os.path.join(NNUE_SJ_ROOT, f"{example}.rs")
    return path if os.path.isfile(path) else None


# `env_f32("wdl_start", 0.0)`, `env_string("net_id", "...")`: a setting's literal default in a script.
_ENV_DEFAULT = re.compile(
    r'\benv_\w+\(\s*"(?P<key>\w+)"\s*,\s*(?:"(?P<text>[^"]*)"|(?P<number>[-+]?[\d_]*\.?[\d_]+(?:[eE][-+]?\d+)?))\s*\)')


def script_defaults(example: str) -> dict:
    """Literal defaults of the env settings of nnue/<example>.rs, e.g. {"wdl_start": 0.0, "net_id": "..."}.

    The scripts are the source of truth for their defaults; settings given as an
    expression (lr_final) are left out, and so is everything for bullet's own examples.
    """
    path = local_script(example)
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
    except OSError:
        return {}
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"//[^\n]*", "", source)
    defaults = {}
    for match in _ENV_DEFAULT.finditer(source):
        if match["text"] is not None:
            value = match["text"]
        else:
            number = match["number"].replace("_", "")
            try:
                value = float(number) if any(c in number for c in ".eE") else int(number)
            except ValueError:
                continue
        defaults.setdefault(match["key"], value)
    return defaults


def build_command(args) -> list:
    # The fork registers nnue/v2.rs and nnue/v3.rs as bullet_lib examples, so every
    # script, ours or bullet's, is a cargo example run from libs/bullet.
    cmd = ["cargo", "run", "--release", "--example", args.example]
    if args.features:
        cmd += ["--features", args.features]
    return cmd


def checkpoint_path(out_dir: str, net_id: str, superbatch: int) -> str:
    """Directory bullet saves the checkpoint of `superbatch` to."""
    return os.path.join(out_dir, f"{net_id}-{superbatch}")


def checkpoint_ready(out_dir: str, net_id: str, superbatch: int) -> bool:
    return os.path.isfile(os.path.join(checkpoint_path(out_dir, net_id, superbatch),
                                       "optimiser_state", "weights.bin"))


def build_env(
        args, 
        train_data: str, 
        val_data: str | None, 
        start_superbatch: int,
        out_dir: str,
        final_superbatch: int,
    ) -> dict:
    env = os.environ.copy()
    initial_lr, final_lr = resolve_lr_bounds(args)
    # Set the resolved schedule explicitly; leave other unspecified settings
    # to the example's defaults.
    mapping = {
        "superbatch_start": start_superbatch,
        "superbatches": args.superbatches,
        # These global schedule parameters stay identical across stages.
        "lr_start": initial_lr,
        "lr_final": final_lr,
        "lr_final_superbatch": final_superbatch,
        "wdl_start": args.wdl_start,
        "wdl_end": args.wdl_end,
        "wdl_final_superbatch": final_superbatch,
        "reset_optimiser": 1 if args.reset_optimiser else None,

        "net_id": args.net_id,
        "output_dir": out_dir,
        "train_data": train_data,
        "val_data": val_data,

        "save_rate": args.save_rate,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "threads": args.threads,

        "L1": args.L1,
        "L2": args.L2,
        "L3": args.L3,

        "QA": args.QA,
        "QB": args.QB,
        "QC": args.QC,
    }

    for key, value in mapping.items():
        if value is not None:
            env[key] = str(value)

    # force ANSI colours through the pipe so the console still looks nice
    env.setdefault("CLICOLOR_FORCE", "1")
    return env


def stream_output(proc: subprocess.Popen, log_path: str, on_line=None) -> None:
    """Tee subprocess stdout to our stdout and a log file (and to `on_line`, if given)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log, proc.stdout:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            if on_line is not None:
                try:
                    on_line(line)
                except Exception as exc:  # never let bookkeeping kill the training output
                    print(f"[train] best-checkpoint tracking error: {exc}")


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SAVED = re.compile(r"Saved \[(.+)-(\d+)\]")


FALLBACK_WDL = 0.25  # for bullet's own examples, whose defaults train.py does not read
FALLBACK_NET_ID = "1024x16x32_25wdl"

# Windows refuses to rename or delete a directory while any process has a file in it open
# (WinError 5 "Access is denied"; 32, a sharing violation; 145, "directory not empty" for a file
# deleted while open), and os.replace cannot rename onto an existing directory. An antivirus or the
# search indexer opening the files just written (best.json, new hard links) does that for a moment,
# longer when the machine is busy, so the best-checkpoint renames are retried there.
_TRANSIENT_WINERRORS = (5, 32, 145)


def retry_on_windows(action, *args, attempts: int = 12, delay: float = 0.05):
    """action(*args), retried for a few seconds on Windows while it fails with a transient lock error."""
    for attempt in range(attempts):
        try:
            return action(*args)
        except OSError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) not in _TRANSIENT_WINERRORS \
                    or attempt == attempts - 1:
                raise
            time.sleep(min(delay * 2 ** attempt, 0.5))


def _remove_tree(path: str) -> None:
    if os.path.lexists(path):
        shutil.rmtree(path)


def effective_wdl(args) -> tuple[float, float]:
    """WDL start/end of the run: --wdl-start/--wdl-end, else the wdl_start/wdl_end environment
    variables (which the script would read too), else the script's own defaults (nnue/<example>.rs)."""
    defaults = script_defaults(args.example)

    def resolve(flag, key):
        if flag is not None:
            return float(flag)
        for value in (os.environ.get(key), defaults.get(key)):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return FALLBACK_WDL

    return resolve(args.wdl_start, "wdl_start"), resolve(args.wdl_end, "wdl_end")


class BestTracker:
    """Keep the state with the lowest validation loss, at any superbatch.

    bullet saves only every save_rate superbatches, and its run() has no hook,
    so the trainer is run with save_rate=1 and this class does the rest from the
    trainer's "Saved [<net>-<N>]" lines: it reads the validation loss measured
    at the end of superbatch N (on exactly the weights saved in checkpoint N),
    hard-links checkpoint N into <net>/best/ (with best.json) when it is the
    lowest so far, and deletes the checkpoints nobody asked for: every
    `keep_every`-th and each stage's last one are kept (curriculum chaining and
    resuming need them). Deletion lags one checkpoint so the newest one always
    exists for the live plot.

    Only meaningful when the WDL blend is constant: validation loss is computed
    against the current blend, so losses at different blends are not comparable.
    One tracker per validation set; a new set starts a new best.
    """

    def __init__(self, out_dir: str, net_id: str, keep_every: int, val_data: str | None,
                 wdl: tuple[float, float], best_dir: str | None = None):
        self.out_dir, self.net_id, self.keep_every = out_dir, net_id, keep_every
        self.val_data, self.wdl = val_data, wdl
        self.best_dir = best_dir or os.path.join(out_dir, "best")
        self.keep = set()          # superbatches that must survive (stage ends)
        self.pending = None        # last processed superbatch, deleted when the next one arrives
        self.best_loss, self.best_superbatch = math.inf, None
        self._load_previous()

    def _load_previous(self):
        info_path = os.path.join(self.best_dir, "best.json")
        if not os.path.isfile(info_path):
            return
        try:
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, ValueError):
            return
        if info.get("val_data") == self.val_data and info.get("wdl") == list(self.wdl) \
                and info.get("net_id") == self.net_id:
            self.best_loss, self.best_superbatch = info["val_loss"], info["superbatch"]
            print(f"[train] continuing best tracking: {self.best_loss:.6f} at superbatch {self.best_superbatch}")
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            retry_on_windows(os.replace, self.best_dir, f"{self.best_dir}_previous_{stamp}")
            print(f"[train] earlier best/ was for other settings; kept as best_previous_{stamp}")

    def on_line(self, line: str) -> None:
        match = _SAVED.search(_ANSI.sub("", line))
        if match and match.group(1) == self.net_id:
            self.on_saved(int(match.group(2)))

    def validation_loss(self, superbatch: int) -> float | None:
        path = os.path.join(checkpoint_path(self.out_dir, self.net_id, superbatch), "val_log.txt")
        loss, last_batch = None, -1
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for row in f:
                    parts = row.strip().split(",")
                    if len(parts) == 3 and parts[0] == str(superbatch) and int(parts[1]) > last_batch:
                        last_batch, loss = int(parts[1]), float(parts[2])
        except (OSError, ValueError):
            return None
        return loss if loss is not None and math.isfinite(loss) else None

    def on_saved(self, superbatch: int) -> None:
        loss = self.validation_loss(superbatch)
        if loss is not None and loss < self.best_loss:
            self._save_best(superbatch, loss)
        if self.pending is not None and self.pending != superbatch:
            self._prune(self.pending)
        self.pending = superbatch

    def _save_best(self, superbatch: int, loss: float) -> None:
        source = checkpoint_path(self.out_dir, self.net_id, superbatch)
        staging = self.best_dir + ".new"
        shutil.rmtree(staging, ignore_errors=True)
        for root, _, files in os.walk(source):
            target = os.path.join(staging, os.path.relpath(root, source))
            os.makedirs(target, exist_ok=True)
            for name in files:
                try:
                    os.link(os.path.join(root, name), os.path.join(target, name))  # instant, no extra space
                except OSError:
                    shutil.copy2(os.path.join(root, name), os.path.join(target, name))
        with open(os.path.join(staging, "best.json"), "w", encoding="utf-8") as f:
            json.dump(dict(net_id=self.net_id, superbatch=superbatch, val_loss=loss, val_data=self.val_data,
                           wdl=list(self.wdl), checkpoint=f"{self.net_id}-{superbatch}",
                           saved=datetime.now().isoformat(timespec="seconds")), f, indent=2)
        # the old best goes first (os.replace cannot overwrite a directory); a failure is raised, not
        # ignored, as the rename would then fail anyway
        retry_on_windows(_remove_tree, self.best_dir)
        retry_on_windows(os.replace, staging, self.best_dir)
        self.best_loss, self.best_superbatch = loss, superbatch

    def _prune(self, superbatch: int) -> None:
        if superbatch in self.keep or (self.keep_every and superbatch % self.keep_every == 0):
            return
        shutil.rmtree(checkpoint_path(self.out_dir, self.net_id, superbatch), ignore_errors=True)

    def finish_stage(self, stage_end: int) -> None:
        """Keep the stage's last checkpoint (the next stage continues from it)."""
        self.keep.add(stage_end)
        if self.pending is not None and self.pending != stage_end:
            self._prune(self.pending)
        self.pending = None

    def report(self) -> None:
        if self.best_superbatch is not None:
            print(f"[train] best validation loss {self.best_loss:.6f} at superbatch {self.best_superbatch} "
                  f"-> {self.best_dir}")


def net_output_dir(args) -> str:
    """Keep checkpoints, metrics, and logs together for each net."""
    return os.path.join(NNUE_SJ_ROOT, args.output_dir, args.net_id)


def write_train_config(args, train_stages, val_stages, stage_lengths, final_superbatch) -> None:
    """The run's settings next to its checkpoints, for the net registry (nnue/registry.py)."""
    out_dir = net_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    config = dict(vars(args), train_stages=train_stages, val_stages=val_stages, stage_lengths=stage_lengths,
                  final_superbatch=final_superbatch, written=datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)


def register_checkpoints(args) -> None:
    """Add the run's checkpoints to the net registry; training never fails on this."""
    if args.no_register:
        return
    try:
        import registry  # nnue/registry.py

        cnxn = registry.connect_raw(registry.RAW_DB)
        try:
            count = len(registry.scan(cnxn, os.path.join(NNUE_SJ_ROOT, args.output_dir), args.net_id))
        finally:
            cnxn.close()
        print(f"[train] registered {count} checkpoint(s) of {args.net_id} in the net registry")
    except Exception as exc:
        print(f"[train] could not register checkpoints in the net registry ({exc}); later run: "
              f"python -m nnue.registry scan --run-id {args.net_id}")


def archive_previous_plot(out_dir: str, history_dir: str) -> None:
    """Start a new plot history while retaining the previous run's files."""
    if not os.path.isdir(out_dir):
        return

    paths = []
    for name in os.listdir(out_dir):
        if any(
            name == f"{base}{suffix}"
            or (name.startswith(f"{base}_stage") and name.endswith(suffix))
            for base, suffix in (("metrics", ".csv"), ("loss", ".png"))
        ):
            path = os.path.join(out_dir, name)
            if os.path.isfile(path):
                paths.append(path)

    if not paths:
        return

    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    backup_dir = os.path.join(history_dir, timestamp)
    os.makedirs(backup_dir)
    for path in paths:
        shutil.move(path, os.path.join(backup_dir, os.path.basename(path)))
    print(f"[train] previous plot history saved to {backup_dir}")


def _wait_for_stage(args, proc, out_dir, out_png, train_data, val_data, final_superbatch,
                    stage_boundaries, fig, axes, run_started) -> None:
    """Block until the trainer exits, refreshing the live plot meanwhile (unless --no-plot)."""
    if args.no_plot:
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return
    try:
        import plot  # local module (nnue/plot.py)

        plot_config = build_plot_config(
            args,
            train_data,
            val_data,
            final_superbatch
        )

        # fig/axes is the one shared dashboard built once in main();
        # close_on_exit=False keeps it alive for the next stage.
        plot.watch(
            out_dir,
            out_png,
            smooth=args.smooth,
            log_y=args.log_y,
            interval=args.interval,
            stop=lambda: proc.poll() is not None,
            stages=stage_boundaries,
            config=plot_config,
            fig=fig,
            axes=axes,
            close_on_exit=False,
            since=run_started,
            start_superbatch=args.superbatch_start,
        )
    except ImportError as e:
        print(f"[train] matplotlib unavailable ({e}); running without live plot.")
        print("[train] install with: pip install -r requirements.txt")
    if proc.poll() is None:
        proc.wait()


def run_stage(
    args,
    train_data: str,
    val_data: str | None,
    stage_label: str,
    start_superbatch: int,
    final_superbatch: int,
    stage_boundaries: list[tuple[int, int, str]],
    fig=None,
    axes=None,
    run_started: float | None = None,
    tracker: BestTracker | None = None,
) -> int:
    """Launch one cargo training run and return its exit code."""
    out_dir = net_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    cmd = build_command(args)
    env = build_env(
        args,
        train_data,
        val_data,
        start_superbatch,
        out_dir,
        final_superbatch=final_superbatch,
    )
    if tracker is not None:
        env["save_rate"] = "1"  # every superbatch reaches disk; BestTracker keeps what is needed

    metrics_csv = os.path.join(out_dir, "metrics.csv")
    out_png = os.path.join(out_dir, "loss.png")
    log_path = os.path.join(out_dir, f"train{stage_label}.log")
    stage_end = start_superbatch + args.superbatches - 1

    if os.path.exists(metrics_csv):
        os.remove(metrics_csv)
    if start_superbatch > 1 and not checkpoint_ready(out_dir, args.net_id, start_superbatch - 1):
        print(f"[train] cannot start at superbatch {start_superbatch}: the checkpoint "
              f"{checkpoint_path(out_dir, args.net_id, start_superbatch - 1)} it continues from "
              "does not exist (did the previous stage finish?)")
        return 3

    print(f"\n{'=' * 60}")
    print(f"[train] {stage_label.lstrip('_') or 'single run'}: {train_data}")
    print(f"[train] superbatches {start_superbatch}-{stage_end}"
          + (f", continuing from {checkpoint_path(out_dir, args.net_id, start_superbatch - 1)}"
             + (" (weights only, fresh optimiser)" if args.reset_optimiser else "")
             if start_superbatch > 1 else ", from scratch"))
    print(f"[train] lr_start={env.get('lr_start')} lr_final={env.get('lr_final')}")
    print(f"[train] lr/wdl schedules end at superbatch {env['lr_final_superbatch']}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Working dir: {BULLET_ROOT}")
    print(f"Checkpoints and loss logs: {out_dir}\n")

    proc = subprocess.Popen(
        cmd,
        cwd=BULLET_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tee = threading.Thread(target=stream_output, args=(proc, log_path),
                           kwargs=dict(on_line=tracker.on_line if tracker else None), daemon=True)
    tee.start()

    try:
        _wait_for_stage(args, proc, out_dir, out_png, train_data, val_data, final_superbatch,
                        stage_boundaries, fig, axes, run_started)
    finally:
        # Whatever happened (normal end, failure, Ctrl+C): stop the trainer if it still runs,
        # let the tee thread see its last "Saved" lines, then delete the temporary
        # every-superbatch checkpoints so only the save-rate ones, the stage end and best/ remain.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
        tee.join(timeout=5)
        if tracker is not None:
            tracker.finish_stage(stage_end)

    # (No separate final-render step here anymore -- plot.watch() already
    # does a fresh reload + render + save right before it returns, using
    # the same shared figure. Calling plot.one_shot() here used to force
    # a matplotlib backend switch mid-run and silently kill the live plot
    # at every stage boundary.)

    code = proc.returncode or 0
    print(f"\n[train] stage{stage_label or 'run'} cargo exited with code {code}")

    # archive this stage's metrics/plot so the next stage's metrics.csv
    # (which the trainer will overwrite) doesn't clobber this one
    if stage_label:
        for src, suffix in ((metrics_csv, ".csv"), (out_png, ".png")):
            if os.path.exists(src):
                base = os.path.splitext(os.path.basename(src))[0]
                dst = os.path.join(out_dir, f"{base}{stage_label}{suffix}")
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    pass
                
    if code == 0 and not checkpoint_ready(out_dir, args.net_id, stage_end):
        print(f"[train] the stage exited normally but its final checkpoint "
              f"{checkpoint_path(out_dir, args.net_id, stage_end)} is missing")
        return 4

    # Keep a copy of the dashboard as it looked at the end of each stage.
    if stage_label and os.path.exists(out_png):
        try:
            shutil.copy2(out_png, os.path.join(out_dir, f"loss{stage_label}.png"))
        except OSError:
            pass

    return code


def main() -> int:
    p = argparse.ArgumentParser(description="Launch + visualise a bullet training run.")
    p.add_argument("--example", default="v3",
                   help="cargo example run in libs/bullet: v2/v3 are nnue/v2.rs and nnue/v3.rs "
                        "(registered by the fork); other names are bullet's own examples")
    p.add_argument("--features", default=None, help="cargo features, e.g. cuda / rocm / metal")

    # tuning control
    p.add_argument("--superbatch-start", type=int, default=1,
                   help="first superbatch; > 1 resumes from checkpoint <net-id>-<start-1>")
    p.add_argument("--superbatches", type=int, nargs="+", default=[800], metavar="N",
                   help="one length for all stages or one per resolved dataset, "
                        "e.g. 200 400 800 (default: 800 per stage)")
    p.add_argument("--lr_start", type=float, default=None)
    p.add_argument("--lr_final", type=float, default=None)
    p.add_argument("--wdl-start", type=float, default=None,
                   help="WDL blend at superbatch 1 (default: $wdl_start, else the script's own default)")
    p.add_argument("--wdl-end", type=float, default=None,
                   help="WDL blend at the last superbatch (default: $wdl_end, else the script's own default)")
    p.add_argument("--reset-optimiser", action="store_true",
                   help="at each stage boundary / resume, load only the weights (fresh AdamW moments)")
    p.add_argument("--no-best", action="store_true",
                   help="do not keep the lowest-validation state in <net>/best/ (tracked only with a constant WDL)")

    # data
    p.add_argument("--net-id", default=None,
                   help="run name and checkpoint prefix (default: $net_id, else the script's net_id)")
    p.add_argument("--output-dir", default="checkpoints",
                   help="output root; each net writes to <output-dir>/<net-id>/")
    p.add_argument(
        "--train-data",
        nargs="+",
        default=None,
        help="one or more dataset paths, OR a folder (path or bare name "
        "resolved under data/) whose dataset files are used as sequential "
        "stages in sorted-filename order. Each resulting stage is chained "
        "via checkpoints (global superbatch numbering). Prefix "
        "filenames like 01_, 02_ to control stage order explicitly."
        "Call folders via  /data/folder1/  and files via  /data/file1.binpack  "
        "with as many files as you want.",
    )
    p.add_argument(
        "--val-data",
        nargs="+",
        default=None,
        help="single path (used for every stage) or one path per --train-data entry",
    )

    # gpu params
    p.add_argument("--save-rate", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--batches", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)

    # model params
    p.add_argument("--L1", type=int, default=None)
    p.add_argument("--L2", type=int, default=None)
    p.add_argument("--L3", type=int, default=None)

    # quantisation params
    p.add_argument("--QA", type=int, default=None)
    p.add_argument("--QB", type=int, default=None)
    p.add_argument("--QC", type=int, default=None)

    # plotting
    p.add_argument("--no-plot", action="store_true", help="do not open a live plot window")
    p.add_argument("--interval", type=float, default=3.0, help="live plot refresh seconds")
    p.add_argument("--smooth", type=int, default=15, help="train-curve moving-average window")
    p.add_argument("--log-y", action="store_true", help="logarithmic loss axis")
    p.add_argument("--no-register", action="store_true",
                   help="do not add the run's checkpoints to the net registry when training finishes")
    args = p.parse_args()
    # Settings train.py needs itself are resolved here and always passed to the script,
    # with the script's own defaults (nnue/<example>.rs) as the defaults.
    if args.net_id is None:
        args.net_id = (os.environ.get("net_id") or script_defaults(args.example).get("net_id")
                       or FALLBACK_NET_ID)
    args.wdl_start, args.wdl_end = effective_wdl(args)
    if args.superbatch_start < 1 or any(n < 1 for n in args.superbatches):
        p.error("--superbatch-start and --superbatches must be positive")
    if (not args.net_id or args.net_id in (".", "..", "plot_history")
            or any(c in args.net_id for c in '/\\<>:"|?*')
            or args.net_id.endswith((" ", "."))):
        p.error("--net-id must be a single folder name other than plot_history")

    sys.path.insert(0, HERE)

    # plotting

    dashboard_fig = None
    dashboard_axes = None

    # Curriculum learning

    stage_boundaries = []
    try:
        train_stages = (
            expand_train_data(args.train_data, DATA_ROOT)
            if args.train_data
            else [None]
        )
        val_resolved = [resolve_dataset(v, DATA_ROOT) for v in args.val_data] if args.val_data else None
    except FileNotFoundError as exc:
        print(f"[train] {exc}")
        return 2

    if not train_stages:
        print("[train] no dataset files resolved from --train-data")
        return 2

    script = local_script(args.example)
    if script:
        print(f"[train] training script: {script} (cargo example '{args.example}' of libs/bullet)")
    else:
        print(f"[train] nnue/{args.example}.rs not found; running '{args.example}' as one of bullet's own "
              "examples (registered in libs/bullet/crates/bullet_lib/Cargo.toml); it ignores train.py's settings "
              "unless it reads them from the environment")

    # stage lengths

    start_superbatch = args.superbatch_start
    try:
        stage_lengths = resolve_stage_lengths(args.superbatches, len(train_stages))
    except ValueError as exc:
        p.error(str(exc))
    total_superbatches = sum(stage_lengths)
    final_superbatch = start_superbatch + total_superbatches - 1

    stage_boundaries = build_stage_boundaries(
        train_stages, stage_lengths, start_superbatch, args.example)
    for i, (start, end, label) in enumerate(stage_boundaries, 1):
        print(f"[train] stage {i}: {label} | superbatches {start}-{end} ({end - start + 1})")

    multi_stage = len(train_stages) > 1

    initial_lr, final_lr = resolve_lr_bounds(args)
    print(
        f"[train] global LR schedule: {initial_lr:g} -> {final_lr:g} "
        f"over superbatches 1-{final_superbatch} (continuous across stages)"
    )
    print(f"[train] global WDL blend: {args.wdl_start:g} -> {args.wdl_end:g} over superbatches 1-{final_superbatch}"
          + ("" if args.wdl_start != args.wdl_end else " (constant)"))

    val_stages: list
    if val_resolved is None:
        val_stages = [None] * len(train_stages)
    elif len(val_resolved) == 1:
        val_stages = val_resolved * len(train_stages)
    elif len(val_resolved) == len(train_stages):
        val_stages = val_resolved
    else:
        print(
            f"[train] --val-data has {len(val_resolved)} entries but "
            f"--train-data has {len(train_stages)}; pass one val set total "
            f"or one per training stage."
        )
        return 2

    out_dir = net_output_dir(args)
    if start_superbatch > 1 and not checkpoint_ready(out_dir, args.net_id, start_superbatch - 1):
        print(f"[train] --superbatch-start {start_superbatch} resumes from "
              f"{checkpoint_path(out_dir, args.net_id, start_superbatch - 1)}, which does not exist")
        return 3

    if not args.no_plot:
        try:
            import plot
            dashboard_fig, dashboard_axes = plot.create_dashboard()
        except ImportError as e:
            print(f"[train] matplotlib unavailable ({e}); running without live plot.")
            args.no_plot = True

    run_started = time.time()
    wdl = effective_wdl(args)
    trackers: dict = {}
    if args.no_best:
        print("[train] best-validation tracking off (--no-best)")
    elif wdl[0] != wdl[1]:
        print(f"[train] best-validation tracking off: the WDL blend changes ({wdl[0]} -> {wdl[1]}), so "
              "validation losses at different superbatches are measured against different targets")
    else:
        keep_every = args.save_rate or 10
        print(f"[train] keeping the lowest-validation state in {os.path.join(out_dir, 'best')} "
              f"(trainer saves every superbatch; checkpoints kept every {keep_every} and at stage ends)")
    distinct_val = len({v for v in val_stages})
    try:
        archive_previous_plot(
            net_output_dir(args),
            os.path.join(NNUE_SJ_ROOT, args.output_dir, "plot_history", args.net_id),
        )
        write_train_config(args, train_stages, val_stages, stage_lengths, final_superbatch)
        for i, (train_data, val_data) in enumerate(zip(train_stages, val_stages)):
            stage_label = f"_stage{i + 1}" if multi_stage else ""

            stage_start, stage_end, _ = stage_boundaries[i]
            stage_args = argparse.Namespace(**vars(args))
            stage_args.superbatches = stage_lengths[i]
            print(
                f"[train] stage {i + 1} scheduled LR: "
                f"{_cosine_lr(stage_start, initial_lr, final_lr, final_superbatch):.6g} -> "
                f"{_cosine_lr(stage_end, initial_lr, final_lr, final_superbatch):.6g}"
            )

            tracker = None
            if not args.no_best and wdl[0] == wdl[1]:
                if val_data not in trackers:
                    # One best per validation set: losses on different sets are not comparable.
                    name = "best" if distinct_val == 1 else \
                        f"best_{os.path.splitext(os.path.basename(val_data or 'default'))[0]}"
                    trackers[val_data] = BestTracker(out_dir, args.net_id, args.save_rate or 10, val_data, wdl,
                                                     best_dir=os.path.join(out_dir, name))
                tracker = trackers[val_data]

            code = run_stage(
                stage_args,
                train_data=train_data,
                val_data=val_data,
                stage_label=stage_label,
                start_superbatch=stage_start,
                final_superbatch=final_superbatch,
                stage_boundaries=stage_boundaries,
                fig=dashboard_fig,
                axes=dashboard_axes,
                run_started=run_started,
                tracker=tracker,
            )

            if code != 0:
                print(
                    f"[train] stage {i + 1}/{len(train_stages)} failed "
                    f"(exit {code}); stopping curriculum."
                )
                return code

        for tracker in trackers.values():
            tracker.report()
        register_checkpoints(args)
        return 0
    finally:
        if dashboard_fig is not None:
            import matplotlib.pyplot as plt

            plt.ioff()
            plt.close(dashboard_fig)


if __name__ == "__main__":
    raise SystemExit(main())
