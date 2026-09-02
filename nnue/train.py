#!/usr/bin/env python3
"""
Training wrapper around the bullet Rust trainer.

Primary purpose: make long training runs visual-friendly. It launches the
chosen cargo example, tees its (colourful) output to your console and a log
file, and shows a live-updating loss plot fed by the trainer's `metrics.csv`.

Secondary purpose: light tuning control. Any of the tuning flags below are
forwarded to the example via env variables (currently wired up in
`examples/halfka_deep.rs`).

Multi-dataset (curriculum) runs:
    Pass --train-data with more than one path and each one is run as its own
    sequential stage: stage 1 trains from scratch, stage 2+ set
    `is_later_run=1` in the environment so the Rust side loads the checkpoint
    left behind by the previous stage (same --output-dir / --net-id) and
    continues training on the new dataset. Per-stage logs/metrics/plots are
    archived as e.g. `train_stage2.log`, `metrics_stage2.csv`, `loss_stage2.png`
    so nothing gets clobbered when the next stage starts writing metrics.csv.

Examples:
    python python/train.py --example halfka_deep --features cuda
    python python/train.py --example halfka_deep --superbatches 400 --lr_start 0.0008 \
        --train-data data/train.binpack --val-data data/val.binpack

    # curriculum: three stages, one dataset each, checkpoints chained
    python python/train.py --example halfka_deep --net-id my_net \
        --train-data data/stage1.binpack data/stage2.binpack data/stage3.binpack \
        --val-data data/val.binpack

    python python/train.py --example halfka_deep --no-plot        # headless
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
# train.py lives in Projects/san-jacinto/nnue/
SAN_JACINTO_ROOT = os.path.dirname(HERE)
NNUE_SJ_ROOT = os.path.join(SAN_JACINTO_ROOT, "nnue")
# Bullet lives in Projects/libs/bullet/
PROJECTS_ROOT = os.path.dirname(SAN_JACINTO_ROOT)
BULLET_ROOT = os.path.join(PROJECTS_ROOT, "libs", "bullet")
DATA_ROOT = os.path.join(BULLET_ROOT, "data")

DATASET_EXTENSIONS = (".binpack", ".bin")
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
        candidate = entry

        if not os.path.exists(candidate):
            maybe = os.path.join(data_root, entry)
            if os.path.exists(maybe):
                candidate = maybe

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

def build_plot_config(args, train_data, val_data, final_superbatch):
    import plot

    initial_lr = args.lr_start
    final_lr = args.lr_final

    if initial_lr is None:
        initial_lr = 0.001

    if final_lr is None:
        final_lr = initial_lr * 0.3 ** 5

    return plot.PlotConfig(
        initial_lr=initial_lr,
        final_lr=final_lr,
        final_superbatch=final_superbatch,

        wdl_start=args.wdl_start if args.wdl_start is not None else 0.25,
        wdl_end=args.wdl_end if args.wdl_end is not None else 0.25,

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

def build_command(args) -> list:
    cmd = ["cargo", "run", "--release", "--example", args.example]
    if args.features:
        cmd += ["--features", args.features]
    return cmd


def build_env(
        args, 
        train_data: str, 
        val_data: str | None, 
        start_superbatch: int,
        out_dir: str,
    ) -> dict:
    env = os.environ.copy()
    # only set overrides the user actually passed, so example defaults remain
    mapping = {
        "superbatch_start": start_superbatch,
        "superbatches": args.superbatches,
        "lr_start": args.lr_start,
        "lr_final": args.lr_final,
        "wdl_start": args.wdl_start,
        "wdl_end": args.wdl_end,

        "net_id": args.net_id,
        "output_dir": out_dir,
        "train_data": train_data,
        "val_data": os.path.join(DATA_ROOT, val_data),

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


def stream_output(proc: subprocess.Popen, log_path: str) -> None:
    """Tee subprocess stdout to our stdout and a log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()


def run_stage(
    args,
    train_data: str,
    val_data: str | None,
    stage_label: str,
    start_superbatch: int,
    final_superbatch: int,
    stage_boundaries: list[tuple[int, int, str]],
) -> int:
    """Launch one cargo training run and return its exit code."""
    out_dir = os.path.join(NNUE_SJ_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    cmd = build_command(args)
    env = build_env(args, train_data, val_data, start_superbatch, out_dir)

    metrics_csv = os.path.join(out_dir, "metrics.csv")
    out_png = os.path.join(out_dir, "loss.png")
    log_path = os.path.join(out_dir, f"train{stage_label}.log")

    if os.path.exists(metrics_csv):
        os.remove(metrics_csv)

    print(f"\n{'=' * 60}")
    print(f"[train] stage{stage_label or ' (single run)'}: {train_data}")
    print(f"[train] superbatch_start={env['superbatch_start']}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"Working dir: {BULLET_ROOT}")
    print(f"Metrics: {metrics_csv}\n")

    proc = subprocess.Popen(
        cmd,
        cwd=BULLET_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tee = threading.Thread(target=stream_output, args=(proc, log_path), daemon=True)
    tee.start()

    if args.no_plot:
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    else:
        try:
            import plot  # local module (python/plot.py)

            plot_config = build_plot_config(
                args, 
                train_data, 
                val_data, 
                final_superbatch
            )

            plot.watch(
                metrics_csv,
                out_png,
                smooth=args.smooth,
                log_y=args.log_y,
                interval=args.interval,
                stop=lambda: proc.poll() is not None,
                stages=stage_boundaries,
                config=plot_config
            )
        except ImportError as e:
            print(f"[train] matplotlib unavailable ({e}); running without live plot.")
            print("[train] install with: pip install -r python/requirements.txt")
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.wait()

    tee.join(timeout=5)

    # one final static render for the record
    if not args.no_plot:
        try:
            import plot

            plot_config = build_plot_config(
                args, 
                train_data, 
                val_data, 
                final_superbatch
            )

            plot.one_shot(
                metrics_csv, 
                out_png, 
                smooth=args.smooth, 
                log_y=args.log_y, 
                stages=stage_boundaries,
                config=plot_config
            )
        except Exception:  # noqa: BLE001
            pass

    code = proc.returncode or 0
    print(f"\n[train] stage{stage_label or ''} cargo exited with code {code}")

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

    return code


def main() -> int:
    p = argparse.ArgumentParser(description="Launch + visualise a bullet training run.")
    p.add_argument("--example", default="halfka_deep", help="cargo example name to run")
    p.add_argument("--features", default=None, help="cargo features, e.g. cuda / rocm / metal")

    # tuning control
    p.add_argument("--superbatch-start", type=int, default=None)
    p.add_argument("--superbatches", type=int, default=None)
    p.add_argument("--lr_start", type=float, default=None)
    p.add_argument("--lr_final", type=float, default=None)
    p.add_argument("--wdl-start", type=float, default=None)
    p.add_argument("--wdl-end", type=float, default=None)

    # data
    p.add_argument("--net-id", default=None)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument(
        "--train-data",
        nargs="+",
        default=None,
        help="one or more dataset paths, OR a folder (path or bare name "
        "resolved under data/) whose dataset files are used as sequential "
        "stages in sorted-filename order. Each resulting stage is chained "
        "via checkpoints (is_later_run=1 from stage 2 onward). Prefix "
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
    args = p.parse_args()

    sys.path.insert(0, HERE)

    stage_boundaries = []
    train_stages = (
        expand_train_data(args.train_data, DATA_ROOT)
        if args.train_data
        else [None]
    )

    if not train_stages:
        print("[train] no dataset files resolved from --train-data")
        return 2

    start_superbatch = args.superbatch_start or 1
    total_superbatches = len(train_stages) * args.superbatches
    final_superbatch = start_superbatch + total_superbatches - 1

    if args.superbatches is not None:
        for i, train_data in enumerate(train_stages):
            stage_start = start_superbatch + i * args.superbatches
            stage_end = stage_start + args.superbatches - 1

            stage_boundaries.append(
                (
                    stage_start,
                    stage_end,
                    os.path.basename(train_data),
                )
            )

    val_stages: list
    if args.val_data is None:
        val_stages = [None] * len(train_stages)
    elif len(args.val_data) == 1:
        val_stages = args.val_data * len(train_stages)
    elif len(args.val_data) == len(train_stages):
        val_stages = args.val_data
    else:
        print(
            f"[train] --val-data has {len(args.val_data)} entries but "
            f"--train-data has {len(train_stages)}; pass one val set total "
            f"or one per training stage."
        )
        return 2

    multi_stage = len(train_stages) > 1

    for i, (train_data, val_data) in enumerate(zip(train_stages, val_stages)):
        stage_label = f"_stage{i + 1}" if multi_stage else ""

        code = run_stage(
            args,
            train_data=train_data,
            val_data=val_data,
            stage_label=stage_label,
            start_superbatch=start_superbatch,
            final_superbatch=final_superbatch,
            stage_boundaries=stage_boundaries,
        )

        if code != 0:
            print(
                f"[train] stage {i + 1}/{len(train_stages)} failed "
                f"(exit {code}); stopping curriculum."
            )
            return code

        if args.superbatches is not None:
            start_superbatch += args.superbatches  # +1 to avoid overlap

    return 0


if __name__ == "__main__":
    raise SystemExit(main())