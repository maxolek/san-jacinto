#!/usr/bin/env python3
"""
Plot Bullet NNUE training metrics.

metrics.csv:
    unix_time,superbatch,batch,split,loss

Dashboard:
    1. Train / validation loss
    2. Learning-rate schedule
    3. Validation loss / best validation
    4. Validation - training loss gap

The plot can be used standalone or through python/train.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
from matplotlib.ticker import MaxNLocator


Series = Dict[str, List[Tuple[float, float]]]


# Fixed qualitative palette (matplotlib's "tab10" hex values) so stage
# colors are stable and legible without needing to import a colormap
# module before the Agg backend is configured.
_STAGE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _stage_color(index: int) -> str:
    return _STAGE_PALETTE[index % len(_STAGE_PALETTE)]


@dataclass
class PlotConfig:
    initial_lr: float | None = None
    final_lr: float | None = None
    final_superbatch: int | None = None

    wdl_start: float | None = None
    wdl_end: float | None = None

    batch_size: int | None = None
    batches_per_superbatch: int | None = None
    threads: int | None = None

    net_id: str | None = None
    train_data: str | None = None
    val_data: str | None = None

    l1: int | None = None
    l2: int | None = None
    l3: int | None = None

    qa: int | None = None
    qb: int | None = None
    qc: int | None = None


def _moving_average(ys: List[float], k: int) -> List[float]:
    if k <= 1 or len(ys) < 2:
        return ys

    out: List[float] = []
    window: deque[float] = deque()
    acc = 0.0

    for y in ys:
        window.append(y)
        acc += y

        if len(window) > k:
            acc -= window.popleft()

        out.append(acc / len(window))

    return out


def load_metrics(path: str, include_archived: bool = False) -> Series:
    """Return {split: [(superbatch_fraction, loss), ...]}."""

    paths = []

    if include_archived:
        directory = os.path.dirname(os.path.abspath(path))
        base = os.path.splitext(os.path.basename(path))[0]

        archived = []

        for filename in os.listdir(directory):
            if (
                filename.startswith(base + "_stage")
                and filename.endswith(".csv")
            ):
                archived.append(filename)

        def stage_number(filename: str) -> int:
            try:
                return int(
                    filename[
                        len(base) + len("_stage"):
                        -len(".csv")
                    ]
                )
            except ValueError:
                return 10**9

        archived.sort(key=stage_number)

        paths.extend(
            os.path.join(directory, filename)
            for filename in archived
        )

    # Current metrics.csv goes last.
    paths.append(path)

    rows: List[Tuple[str, int, int, float]] = []
    offset = 0  # cumulative superbatch count from earlier stages

    for metrics_path in paths:
        file_rows: List[Tuple[str, int, int, float]] = []

        try:
            with open(
                metrics_path,
                newline="",
                encoding="utf-8",
            ) as f:
                for r in csv.DictReader(f):
                    try:
                        file_rows.append(
                            (
                                r["split"],
                                int(r["superbatch"]),
                                int(r["batch"]),
                                float(r["loss"]),
                            )
                        )
                    except (ValueError, KeyError):
                        continue

        except (FileNotFoundError, OSError):
            continue

        if not file_rows:
            continue

        # Shift this file's superbatch numbers so they continue on from
        # the previous stage instead of restarting at 1. Without this,
        # a new stage's x-values collide with the previous stage's and
        # get silently dropped as "duplicates" by _merge_series.
        max_sb_in_file = max(sb for _, sb, _, _ in file_rows)

        for split, sb, batch, loss in file_rows:
            rows.append((split, offset + sb, batch, loss))

        offset += max_sb_in_file

    if not rows:
        return {}

    max_batch = max(
        batch
        for _, _, batch, _ in rows
    ) or 1

    series: Series = {}

    for split, sb, batch, loss in rows:
        x = sb + batch / (max_batch + 1)

        series.setdefault(
            split,
            [],
        ).append((x, loss))

    for points in series.values():
        points.sort()

    return series


def _all_points(series: Series) -> List[Tuple[float, float]]:
    return [
        point
        for points in series.values()
        for point in points
    ]

def _merge_series(old: Series, new: Series) -> Series:
    """Merge newly loaded metrics into previously observed history."""

    merged: Series = {
        split: list(points)
        for split, points in old.items()
    }

    for split, points in new.items():
        existing_x = {
            x
            for x, _ in merged.get(split, [])
        }

        for x, y in points:
            if x in existing_x:
                continue

            merged.setdefault(split, []).append((x, y))
            existing_x.add(x)

        merged[split].sort()

    return merged

def _global_stage_end(stages, series: Series, config: PlotConfig | None) -> float:
    """Return the final superbatch of the entire curriculum."""

    if stages:
        return max(end for _, end, _ in stages)

    if config is not None and config.final_superbatch is not None:
        return float(config.final_superbatch)

    points = _all_points(series)
    if points:
        return max(x for x, _ in points)

    return 1.0


def _set_integer_x_axis(ax) -> None:
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )

def _set_global_x_scale(
    ax,
    stages,
    series: Series,
    config: PlotConfig | None,
) -> None:
    end = _global_stage_end(stages, series, config)

    ax.set_xlim(0.5, end + 0.5)
    _set_integer_x_axis(ax)


def _latest(series: Series, split: str) -> Tuple[float, float] | None:
    points = series.get(split)
    if not points:
        return None
    return points[-1]


def _best_validation(series: Series) -> Tuple[float, float] | None:
    points = series.get("val")
    if not points:
        return None
    return min(points, key=lambda p: p[1])


def _set_loss_scale(ax, series: Series) -> None:
    ys: List[float] = []

    for x, y in _all_points(series):
        if x >= 1.0:
            ys.append(y)

    if len(ys) < 20:
        ax.relim()
        ax.autoscale_view()
        return

    ys.sort()

    low = ys[max(0, len(ys) // 100)]
    high = ys[min(len(ys) - 1, len(ys) * 99 // 100)]

    if high <= low:
        ax.relim()
        ax.autoscale_view()
        return

    margin = (high - low) * 0.10

    ax.set_ylim(
        max(0.0, low - margin),
        high + margin,
    )


def _draw_stages(ax, stages) -> None:
    """Shade and label each training stage with a stable per-stage color.

    The on-chart label is kept short ("stage 1", "stage 2", ...) so it
    doesn't clutter the plot; the stage -> dataset mapping is shown once,
    separately, via _draw_stage_legend in the info panel.
    """

    if not stages:
        return

    for i, (start, end, _label) in enumerate(stages):
        color = _stage_color(i)

        # Stages are inclusive:
        # 1–5, 6–10, 11–15.
        left = start - 0.5
        right = end + 0.5

        ax.axvspan(left, right, color=color, alpha=0.08, zorder=0)
        ax.axvline(left, linestyle="--", alpha=0.35, color=color)
        ax.axvline(right, linestyle="--", alpha=0.35, color=color)

        midpoint = (start + end) / 2

        ymin, ymax = ax.get_ylim()

        ax.text(
            midpoint,
            ymax,
            f"stage {i + 1}",
            ha="center",
            va="top",
            fontsize=9,
            alpha=0.85,
            color=color,
            fontweight="bold",
        )


def _draw_stage_legend(ax, stages) -> None:
    """Render the "stage N -> dataset label" key into the info panel."""

    if not stages:
        return

    x = 0.76
    top = 0.92
    line_height = 0.11

    ax.text(
        x,
        top,
        "Stages:",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
        fontweight="bold",
    )

    for i, (_start, _end, label) in enumerate(stages):
        color = _stage_color(i)

        ax.text(
            x,
            top - (i + 1) * line_height,
            f"stage {i + 1}: {label}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            family="monospace",
            color=color,
        )


def _cosine_lr(
    superbatch: float,
    initial_lr: float,
    final_lr: float,
    final_superbatch: int,
) -> float:
    if final_superbatch <= 0:
        return final_lr

    progress = min(
        1.0,
        max(0.0, superbatch / final_superbatch),
    )

    return final_lr + 0.5 * (initial_lr - final_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def _draw_loss(
    ax,
    series: Series,
    smooth: int,
    log_y: bool,
    config: PlotConfig | None,
    stages,
) -> None:
    ax.clear()

    for split in ("train", "val"):
        points = series.get(split)

        if not points:
            continue

        xs = [x for x, _ in points]
        ys = [y for _, y in points]

        if split == "train":
            ax.plot(
                xs,
                ys,
                alpha=0.20,
                linewidth=0.8,
                label="train raw",
            )

            ax.plot(
                xs,
                _moving_average(ys, smooth),
                linewidth=1.8,
                label=f"train ({smooth}-point MA)",
            )

        else:
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label="validation",
            )

        last_x = xs[-1]
        last_y = ys[-1]

        ax.annotate(
            f"{last_y:.5f}",
            xy=(last_x, last_y),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
        )

    best = _best_validation(series)

    if best is not None:
        bx, by = best

        ax.scatter(
            [bx],
            [by],
            marker="*",
            s=100,
            zorder=5,
            label=f"best val ({by:.5f})",
        )

    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel("superbatch")
    ax.set_ylabel("loss")
    ax.set_title("Training / validation loss")
    ax.grid(True, alpha=0.3)

    if series:
        ax.legend(loc="best", fontsize=8)

    _set_loss_scale(ax, series)
    _draw_stages(ax, stages)
    _set_global_x_scale(ax, stages, series, config)


def _draw_lr(ax, series: Series, config: PlotConfig | None, stages) -> None:
    ax.clear()

    if config is None or config.initial_lr is None:
        ax.text(
            0.5,
            0.5,
            "Learning-rate configuration unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Learning rate")
        return

    latest_points = _all_points(series)

    if latest_points:
        max_x = max(x for x, _ in latest_points)
    else:
        max_x = 1.0

    final_sb = _global_stage_end(stages, series, config)

    xs = [
        final_sb * i / 300
        for i in range(301)
    ]

    final_lr = (
        config.final_lr
        if config.final_lr is not None
        else config.initial_lr * (0.3 ** 5)
    )

    ys = [
        _cosine_lr(
            x,
            config.initial_lr,
            final_lr,
            final_sb,
        )
        for x in xs
    ]

    ax.plot(xs, ys, linewidth=2, label="cosine LR")

    latest = max_x

    current_lr = _cosine_lr(
        latest,
        config.initial_lr,
        final_lr,
        final_sb,
    )

    ax.scatter(
        [latest],
        [current_lr],
        s=45,
        zorder=5,
        label=f"current: {current_lr:.3g}",
    )

    ax.set_xlim(0.5, final_sb + 0.5)
    _set_integer_x_axis(ax)

    ax.set_xlabel("superbatch")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning-rate schedule")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    _draw_stages(ax, stages)


def _draw_wdl(ax, series: Series, config: PlotConfig | None, stages) -> None:
    ax.clear()

    if config is None or config.wdl_start is None:
        ax.text(
            0.5,
            0.5,
            "WDL configuration unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("WDL target")
        return

    latest_points = _all_points(series)

    max_x = (
        max(x for x, _ in latest_points)
        if latest_points
        else 1.0
    )

    final_sb = _global_stage_end(stages, series, config)

    end = (
        config.wdl_end
        if config.wdl_end is not None
        else config.wdl_start
    )

    xs = [
        final_sb * i / 100
        for i in range(101)
    ]

    ys = [
        config.wdl_start
        + (end - config.wdl_start) * (x / final_sb)
        for x in xs
    ]

    ax.plot(
        xs,
        ys,
        linewidth=2,
        label="WDL schedule",
    )

    current = config.wdl_start + (
        end - config.wdl_start
    ) * min(1.0, max_x / final_sb)

    ax.scatter(
        [max_x],
        [current],
        s=45,
        zorder=5,
        label=f"current: {current:.4f}",
    )

    ax.set_xlim(0.5, final_sb + 0.5)
    _set_integer_x_axis(ax)

    ax.set_xlabel("superbatch")
    ax.set_ylabel("WDL target")
    ax.set_title("WDL schedule")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    _draw_stages(ax, stages)


def _draw_validation(ax, series: Series, config: PlotConfig | None, stages) -> None:
    ax.clear()

    points = series.get("val", [])

    if not points:
        ax.text(
            0.5,
            0.5,
            "No validation data yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Validation loss")
        return

    xs = [x for x, _ in points]
    ys = [y for _, y in points]

    ax.plot(
        xs,
        ys,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="validation",
    )

    best_x, best_y = min(points, key=lambda p: p[1])

    ax.scatter(
        [best_x],
        [best_y],
        marker="*",
        s=120,
        zorder=5,
        label=f"best: {best_y:.5f}",
    )

    ax.annotate(
        f"best {best_y:.5f}",
        xy=(best_x, best_y),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
    )

    ax.set_xlabel("superbatch")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    _draw_stages(ax, stages)
    _set_global_x_scale(ax, stages, series, config)


def _draw_gap(ax, series: Series, config: PlotConfig | None, stages) -> None:
    ax.clear()

    train = series.get("train", [])
    val = series.get("val", [])

    if not train or not val:
        ax.text(
            0.5,
            0.5,
            "Need both train and validation data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Validation − training gap")
        return

    xs: List[float] = []
    gaps: List[float] = []

    train_points = train
    val_points = val

    for i, (_, vy) in enumerate(val_points):
        if len(val_points) == 1:
            train_index = len(train_points) - 1
        else:
            train_index = round(
                i * (len(train_points) - 1)
                / (len(val_points) - 1)
            )

        tx, ty = train_points[train_index]

        xs.append(tx)
        gaps.append(vy - ty)

    ax.axhline(
        0.0,
        linestyle="--",
        alpha=0.5,
    )

    ax.plot(
        xs,
        gaps,
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="val - train",
    )

    ax.set_xlabel("superbatch")
    ax.set_ylabel("loss difference")
    ax.set_title("Validation − training gap")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="best",
        fontsize=8,
    )

    _draw_stages(ax, stages)
    _set_global_x_scale(ax, stages, series, config)


def _draw_info(
    ax,
    series: Series,
    config: PlotConfig | None,
    stages=None,
    started_at: float | None = None,
) -> None:
    ax.clear()
    ax.axis("off")

    latest_points = _all_points(series)

    if not latest_points:
        ax.text(
            0.5,
            0.5,
            "No metrics available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    current_sb = max(x for x, _ in latest_points)

    left_lines = [
        f"Current superbatch: {current_sb:.2f}",
    ]

    if config is not None:
        if config.final_superbatch is not None:
            progress = (
                current_sb / config.final_superbatch * 100
            )
            left_lines.append(
                f"Progress: {progress:.1f}%"
            )

        if config.initial_lr is not None:
            left_lines.append(
                f"Initial LR: {config.initial_lr:g}"
            )

        if config.final_superbatch is not None:
            left_lines.append(
                f"Final superbatch: {config.final_superbatch}"
            )

        if config.wdl_start is not None:
            left_lines.append(
                f"WDL start: {config.wdl_start:g}"
            )

        if config.wdl_end is not None:
            left_lines.append(
                f"WDL end: {config.wdl_end:g}"
            )

    right_lines = []

    if config is not None:
        if config.batch_size is not None:
            right_lines.append(
                f"Batch size: {config.batch_size:,}"
            )

        if config.batches_per_superbatch is not None:
            right_lines.append(
                f"Batches / superbatch: {config.batches_per_superbatch:,}"
            )

        if config.threads is not None:
            right_lines.append(
                f"Threads: {config.threads}"
            )

        if config.net_id is not None:
            right_lines.append(
                f"Net: {config.net_id}"
            )

        # Dataset paths are also captured per-stage in the stage legend
        # (when stages are provided), so they're kept here too for the
        # no-stages case.
        if config.train_data is not None:
            right_lines.append(
                f"Train: {config.train_data}"
            )

        if config.val_data is not None:
            right_lines.append(
                f"Val: {config.val_data}"
            )

    ax.text(
        0.01,
        0.92,
        "\n".join(left_lines),
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
    )

    ax.text(
        0.31,
        0.92,
        "\n".join(right_lines),
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
    )

    _draw_stage_legend(ax, stages)

    ax.set_title(
        "Training information",
        loc="left",
        fontsize=10,
        pad=4,
    )


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))

    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if days:
        return f"{days}d {hours}h {minutes}m"

    if hours:
        return f"{hours}h {minutes}m"

    if minutes:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"



def render(
    series: Series,
    fig,
    axes,
    title: str,
    smooth: int = 15,
    log_y: bool = False,
    stages=None,
    config: PlotConfig | None = None,
    started_at: float | None = None,
) -> None:
    if started_at is None:
        started_at = time.time()

    _draw_loss(
        axes["loss"],
        series,
        smooth,
        log_y,
        config,
        stages,
    )

    _draw_lr(
        axes["lr"],
        series,
        config,
        stages,
    )

    _draw_wdl(
        axes["wdl"],
        series,
        config,
        stages,
    )

    _draw_validation(
        axes["validation"],
        series,
        config,
        stages,
    )

    _draw_info(
        axes["info"],
        series,
        config,
        stages,
        started_at,
    )

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
    )


def _title_for(path: str) -> str:
    parent = (
        os.path.basename(
            os.path.dirname(
                os.path.abspath(path)
            )
        )
        or "training"
    )

    return f"Bullet training — {parent}"


def one_shot(
    path: str,
    out_png: str,
    smooth: int = 15,
    log_y: bool = False,
    stages=None,
    config: PlotConfig | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1, 1, 0.28],
        hspace=0.32,
        wspace=0.18,
    )

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_val = fig.add_subplot(gs[0, 1])
    ax_lr = fig.add_subplot(gs[1, 0])
    ax_wdl = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[2, :])

    axes = {
        "loss": ax_loss,
        "validation": ax_val,
        "lr": ax_lr,
        "wdl": ax_wdl,
        "info": ax_info,
    }

    started_at = time.time()

    series = load_metrics(path, include_archived=True)

    render(
        series,
        fig,
        axes,
        _title_for(path),
        smooth=smooth,
        log_y=log_y,
        stages=stages,
        config=config,
        started_at=started_at,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    os.makedirs(
        os.path.dirname(os.path.abspath(out_png)),
        exist_ok=True,
    )

    fig.savefig(
        out_png,
        dpi=120,
        bbox_inches="tight",
    )

    print(f"Wrote {out_png}")

    plt.close(fig)


def watch(
    path: str,
    out_png: str,
    smooth: int = 15,
    log_y: bool = False,
    interval: float = 3.0,
    stop=None,
    stages=None,
    config: PlotConfig | None = None,
) -> None:
    """Live-refresh the dashboard until stop() returns True."""

    import matplotlib.pyplot as plt

    plt.ion()

    fig = plt.figure(figsize=(16, 10))

    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1, 1, 0.28],
        hspace=0.32,
        wspace=0.18,
    )

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_val = fig.add_subplot(gs[0, 1])
    ax_lr = fig.add_subplot(gs[1, 0])
    ax_wdl = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[2, :])

    axes = {
        "loss": ax_loss,
        "validation": ax_val,
        "lr": ax_lr,
        "wdl": ax_wdl,
        "info": ax_info,
    }

    title = _title_for(path)
    started_at = time.time()

    history: Series = {}
    consecutive_errors = 0

    while True:
        try:
            loaded = load_metrics(
                path,
                include_archived=True,
            )

            # Never throw away history we've already observed.
            history = _merge_series(history, loaded)
            series = history

            render(
                series,
                fig,
                axes,
                title,
                smooth=smooth,
                log_y=log_y,
                stages=stages,
                config=config,
                started_at=started_at,
            )

            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.canvas.draw_idle()

            consecutive_errors = 0

        except KeyboardInterrupt:
            break

        except Exception as e:
            # A single bad read/render -- metrics.csv mid-write, a
            # transient matplotlib redraw glitch, a divide-by-zero from
            # a not-yet-fully-populated config -- must never kill the
            # whole live-plot loop. If it did, train.py exits entirely
            # and you lose the plot AND the log tee for a run that's
            # still training fine in the background.
            consecutive_errors += 1
            print(f"[plot] watch: render error ({e}); retrying "
                  f"(consecutive_errors={consecutive_errors})")

            if consecutive_errors >= 20:
                # Something is persistently broken, not transient --
                # surface it loudly instead of spinning silently forever.
                print("[plot] watch: too many consecutive errors, "
                      "giving up on the live plot (training continues).")
                break

        try:
            plt.pause(interval)
        except Exception as e:
            print(f"[plot] watch: pause/draw error ({e}); retrying")

        if stop is not None and stop():
            break

        if not plt.fignum_exists(fig.number):
            break

    try:
        fig.savefig(
            out_png,
            dpi=120,
            bbox_inches="tight",
        )
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"Could not save {out_png}: {e}")

    plt.ioff()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot Bullet training metrics."
    )

    p.add_argument(
        "metrics",
        help="path to metrics.csv",
    )

    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="output PNG",
    )

    p.add_argument(
        "--watch",
        action="store_true",
        help="live-refresh",
    )

    p.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="refresh interval",
    )

    p.add_argument(
        "--smooth",
        type=int,
        default=15,
        help="training moving-average window",
    )

    p.add_argument(
        "--log-y",
        action="store_true",
        help="logarithmic loss axis",
    )

    args = p.parse_args()

    out_png = (
        args.output
        or os.path.join(
            os.path.dirname(
                os.path.abspath(args.metrics)
            ),
            "loss.png",
        )
    )

    if args.watch:
        watch(
            args.metrics,
            out_png,
            args.smooth,
            args.log_y,
            args.interval,
        )
    else:
        one_shot(
            args.metrics,
            out_png,
            args.smooth,
            args.log_y,
        )


if __name__ == "__main__":
    main()