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
    sequential stage: stage 1 trains from scratch, stage 2+ advance
    `superbatch_start` so the Rust side loads the checkpoint
    left behind by the previous stage (same --output-dir / --net-id) and
    continues training on the new dataset. Per-stage logs/metrics/plots are
    archived as e.g. `train_stage2.log`, `metrics_stage2.csv`, `loss_stage2.png`
    so nothing gets clobbered when the next stage starts writing metrics.csv.

    LR scheduling across stages: Bullet indexes its cosine by the global
    superbatch number. Every stage receives the same `lr_start`, `lr_final`,
    and `lr_final_superbatch` (the final superbatch of the curriculum).
    `superbatches` accepts one shared length or one length per dataset stage. This gives one
    global cosine matching the dashboard, without restarting or clamping
    the decay at stage boundaries. Supported by nnue/v2.rs and nnue/v3.rs.

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
import math
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime

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
    plot.py, since plot.py pulls in matplotlib at import time and this
    needs to work even in --no-plot / no-matplotlib setups.
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

        "net_id": args.net_id,
        "output_dir": out_dir,
        "train_data": train_data,
        "val_data": os.path.join(DATA_ROOT, val_data) if val_data is not None else None,

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


def net_output_dir(args) -> str:
    """Keep checkpoints, metrics, and logs together for each net."""
    return os.path.join(NNUE_SJ_ROOT, args.output_dir, args.net_id)


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

    metrics_csv = os.path.join(out_dir, "metrics.csv")
    out_png = os.path.join(out_dir, "loss.png")
    log_path = os.path.join(out_dir, f"train{stage_label}.log")

    if os.path.exists(metrics_csv):
        os.remove(metrics_csv)

    print(f"\n{'=' * 60}")
    print(f"[train] stage{stage_label or ' (single run)'}: {train_data}")
    print(f"[train] superbatch_start={env['superbatch_start']}")
    print(f"[train] lr_start={env.get('lr_start')} lr_final={env.get('lr_final')}")
    print(f"[train] lr_final_superbatch={env['lr_final_superbatch']}")
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

            # fig/axes is the one shared dashboard built once in main();
            # close_on_exit=False keeps it alive for the next stage.
            plot.watch(
                metrics_csv,
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
            )
        except ImportError as e:
            print(f"[train] matplotlib unavailable ({e}); running without live plot.")
            print("[train] install with: pip install -r python/requirements.txt")
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.wait()

    tee.join(timeout=5)

    # (No separate final-render step here anymore -- plot.watch() already
    # does a fresh reload + render + save right before it returns, using
    # the same shared figure. Calling plot.one_shot() here used to force
    # a matplotlib backend switch mid-run and silently kill the live plot
    # at every stage boundary.)

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
    p.add_argument("--superbatch-start", type=int, default=1)
    p.add_argument("--superbatches", type=int, nargs="+", default=[800], metavar="N",
                   help="one length for all stages or one per resolved dataset, "
                        "e.g. 200 400 800 (default: 800 per stage)")
    p.add_argument("--lr_start", type=float, default=None)
    p.add_argument("--lr_final", type=float, default=None)
    p.add_argument("--wdl-start", type=float, default=None)
    p.add_argument("--wdl-end", type=float, default=None)

    # data
    p.add_argument("--net-id", default=os.environ.get("net_id", "1024x16x32_25wdl"))
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
    args = p.parse_args()
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
    train_stages = (
        expand_train_data(args.train_data, DATA_ROOT)
        if args.train_data
        else [None]
    )

    if not train_stages:
        print("[train] no dataset files resolved from --train-data")
        return 2

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

    if not args.no_plot:
        try:
            import plot
            dashboard_fig, dashboard_axes = plot.create_dashboard()
        except ImportError as e:
            print(f"[train] matplotlib unavailable ({e}); running without live plot.")
            args.no_plot = True

    try:
        archive_previous_plot(
            net_output_dir(args),
            os.path.join(NNUE_SJ_ROOT, args.output_dir, "plot_history", args.net_id),
        )
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
            )

            if code != 0:
                print(
                    f"[train] stage {i + 1}/{len(train_stages)} failed "
                    f"(exit {code}); stopping curriculum."
                )
                return code

        return 0
    finally:
        if dashboard_fig is not None:
            import matplotlib.pyplot as plt

            plt.ioff()
            plt.close(dashboard_fig)


if __name__ == "__main__":
    raise SystemExit(main())
