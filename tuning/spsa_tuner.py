"""
SPSA parameter tuner.

Uses standard 2-point simultaneous perturbation (Spall, 1992) with
per-parameter configuration (Fishtest-style). Wraps cutechess-cli 

Usage:
    python -m tests.spsa --engine <path> --baseline <path> --tc 0:1+0.01 --iterations 500
"""
import os
import sys
import argparse
import subprocess
import math
import json
import time
import re
import copy
import threading
import queue
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

# ============================================================================
# SPSA Configuration
# ============================================================================

# Default parameter definitions.
# Each entry: name -> {value, min, max, c_end, integer}
#   - value: starting value (current best)
#   - min/max: hard bounds
#   - c_end: perturbation size (should create a measurable score difference)
#             Rule of thumb: ~5-10% of the useful range, or enough to change
#             the rounded value for integer params (c_end >= 0.5 for integers)
#   - integer: whether to round before passing to engine via UCI
#
# The step size (a_end) is derived automatically:
#   a_end = r × c_end²
# where r is the global learning rate (default 0.002, same as Fishtest).
#
# This comes from Spall (1998): the optimal relationship between step size
# and perturbation size is a ∝ c². The constant r controls convergence speed:
#   - Too large: oscillation, divergence
#   - Too small: convergence is glacially slow
#   - r=0.002 is the empirical sweet spot from Fishtest/OpenBench
#
# games_per_iter affects noise (more = cleaner gradients = fewer iterations
# needed) AND allows slightly larger r (gradient is more trustworthy).
# Scaling: r = r_ref × √(N/N_ref) where N_ref=2 (Fishtest's game pair).
# This is because gradient noise std ∝ 1/√N, so safe step size ∝ √N.

R_REF = 0.002   # Fishtest's learning rate at 2 games (1 game pair)
N_REF = 2       # Fishtest's games per evaluation

# Adam step size as a fraction of each parameter's c_end (perturbation scale).
# Since Adam normalizes the gradient to ~unit magnitude, the step is taken in
# parameter units: peak step ≈ ADAM_LR × c_end. 0.15 means ~15% of the
# perturbation size per iteration at the start, decaying over time.
ADAM_LR = 0.15

DEFAULT_PARAMS = {
    "r_nmp": {
        "value": 3, "min": 2, "max": 5,
        "c_end": 1.0,
        "integer": True,
    },
    "r_lmr_const": {
        "value": 99, "min": 30, "max": 200,  # stored as int (×100)
        "c_end": 8,
        "integer": True,
    },
    "r_lmr_denom": {
        "value": 314, "min": 100, "max": 600,  # stored as int (×100)
        "c_end": 20,
        "integer": True,
    },
    "lmr_depth_threshold": {
        "value": 3, "min": 1, "max": 8,
        "c_end": 1.0,
        "integer": True,
    },
    "lmr_move_order_threshold": {
        "value": 3, "min": 1, "max": 8,
        "c_end": 1.0,
        "integer": True,
    },
    "aspiration_window": {
        "value": 50, "min": 10, "max": 200,
        "c_end": 10,
        "integer": True,
    },
    "aspiration_start_depth": {
        "value": 6, "min": 3, "max": 12,
        "c_end": 1.0,
        "integer": True,
    },
    "delta_prune_threshold": {
        "value": 1000, "min": 200, "max": 2000,
        "c_end": 80,
        "integer": True,
    },
    "see_prune_threshold": {
        "value": -50, "min": -500, "max": 200,
        "c_end": 30,
        "integer": True,
    },
}


def compute_r(games_per_iter):
    """Compute learning rate scaled by games per iteration.
    
    Gradient noise std ∝ 1/√N, so with more games per iteration the gradient
    is more trustworthy and we can take proportionally larger steps.
    Scaling: r = R_REF × √(games_per_iter / N_REF)
    
    Examples:
        2 games:  r = 0.002 (Fishtest baseline)
        16 games: r = 0.0057
        32 games: r = 0.008
        64 games: r = 0.0113
    """
    return R_REF * math.sqrt(games_per_iter / N_REF)


# ============================================================================
# SPSA Math
# ============================================================================

def spsa_coefficients(iteration, total_iterations, c_end, r):
    """
    Compute a_k and c_k using Spall's recommended decay schedule.
    
    c_k = c_end / (k + 1)^gamma        — perturbation, decays slowly
    a_k = a / (A + k + 1)^alpha         — step size, decays faster
    
    The step size is derived from the optimal relationship:
        a_end = r × c_end²   (Spall, 1998)
    where r scales with games_per_iter (more games → less noise → bigger steps).
    """
    alpha = 0.602
    gamma = 0.101
    A = total_iterations * 0.1  # stability constant

    # c_k: starts at c_end, decays very slowly (gamma=0.101)
    c_k = c_end / (iteration + 1) ** gamma

    # a_k: derived from r × c_end²
    a_end = r * c_end ** 2
    a = a_end * (A + 2) ** alpha
    a_k = a / (A + iteration + 1) ** alpha

    return a_k, c_k


def bernoulli_perturbation(n_params):
    """Generate symmetric Bernoulli ±1 perturbation vector."""
    return np.where(np.random.random(n_params) < 0.5, -1.0, 1.0)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# ============================================================================
# Cutechess integration
# ============================================================================

def build_option_string(params_dict, values):
    """Build cutechess option string for a parameter set."""
    options = []
    for name, val in zip(params_dict.keys(), values):
        cfg = params_dict[name]
        v = int(round(val)) if cfg["integer"] else val
        v = clamp(v, cfg["min"], cfg["max"])
        options.append(f"option.{name}={v}")
    return options


def run_match(engine_path, baseline_path, params_dict, theta_plus, theta_minus,
              games_per_iter, tc_args, book_args, concurrency, cutechess_cli,
              plotter=None, iteration=None, total_iterations=None):
    """
    Run a match: theta_plus vs theta_minus.
    Returns (score_plus, score_minus, W, D, L).
    Score is from each side's perspective (W + D/2) / N.

    Uses a non-blocking Popen + reader-thread instead of subprocess.run()
    so that, when a live plot window is open, we can keep pumping its GUI
    event loop while cutechess-cli runs. subprocess.run() blocks the main
    thread for the whole match, during which the window manager sees no
    paint/input events from the process and marks it "Not Responding" --
    the tuner isn't actually hung, the GUI just never gets serviced.

    Also prints a live, in-place progress line ("N/games W-D-L") as cutechess
    finishes each game, parsed from its stdout, so the terminal isn't blank
    for the several minutes a match can take.
    """
    opts_plus = build_option_string(params_dict, theta_plus)
    opts_minus = build_option_string(params_dict, theta_minus)

    cmd = [
        cutechess_cli,
        "-engine", "name=Plus", f"cmd={engine_path}",
        f"dir={os.path.dirname(engine_path)}",
    ] + opts_plus + [
        "-engine", "name=Minus", f"cmd={engine_path}",
        f"dir={os.path.dirname(engine_path)}",
    ] + opts_minus + [
        "-each", "proto=uci",
    ] + tc_args + [
        "-games", str(games_per_iter),
        "-repeat",
        "-concurrency", str(concurrency),
        "-maxmoves", "100",
    ] + book_args + [
        "-wait", "50",
    ]

    # Per-game progress lines, e.g.:
    #   Finished game 3 (Plus vs Minus): 1-0 {White wins}
    #   Score of Plus vs Minus: 2 - 1 - 0 [0.667] 3
    finished_re = re.compile(r"Finished game (\d+)")
    score_re = re.compile(
        r"Score of Plus vs Minus:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)"
    )

    iter_label = f"Iter {iteration}/{total_iterations} | " if iteration is not None else ""

    output_lines = []
    timed_out = False
    games_done = 0
    W = L = D = 0
    match_start = time.time()

    def _print_progress(final=False):
        n = W + L + D
        elapsed = time.time() - match_start
        rate = f"{elapsed / n:.1f}s/game" if n > 0 else "..."
        line = (
            f"\r  [running] {iter_label}{games_done}/{games_per_iter} games "
            f"(W:{W} D:{D} L:{L}) | {elapsed:.0f}s elapsed, {rate}"
        )
        sys.stdout.write(line + " " * 10)  # clear any trailing chars from a longer previous line
        sys.stdout.flush()
        if final:
            sys.stdout.write("\n")
            sys.stdout.flush()

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    ) as proc:

        # Read stdout in a background thread so the main thread stays free
        # to pump GUI events while waiting on cutechess-cli.
        line_queue = queue.Queue()

        def _reader():
            for ln in proc.stdout:
                line_queue.put(ln)
            line_queue.put(None)  # EOF sentinel

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        start = time.time()
        last_print = 0.0
        progress_dirty = False
        while True:
            done = False
            while True:
                try:
                    line = line_queue.get_nowait()
                except queue.Empty:
                    break
                if line is None:  # EOF sentinel
                    done = True
                    break
                output_lines.append(line)

                if finished_re.search(line):
                    games_done += 1
                    progress_dirty = True
                m = score_re.search(line)
                if m:
                    W, L, D = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    progress_dirty = True

            # Keep the plot window responsive while cutechess-cli runs.
            if plotter:
                plotter._pump_events()

            # Throttle terminal updates to ~4/sec so we don't spam the tty.
            now = time.time()
            if progress_dirty and (now - last_print) > 0.25:
                _print_progress()
                last_print = now
                progress_dirty = False

            if done:
                break

            if time.time() - start > 7200:
                timed_out = True
                proc.kill()
                break

            time.sleep(0.05)

        ret = proc.wait()

    _print_progress(final=True)

    if timed_out:
        print("[SPSA] WARNING: cutechess-cli timed out after 7200s and was killed.")

    result_stdout = "".join(output_lines)

    # Final score parse (authoritative — overwrite whatever the progress
    # loop last saw, in case the last line arrived after our final read).
    W, L, D = 0, 0, 0
    for line in result_stdout.splitlines():
        m = score_re.search(line)
        if m:
            W, L, D = int(m.group(1)), int(m.group(2)), int(m.group(3))

    N = W + L + D
    if N == 0:
        print("[SPSA] WARNING: No games completed!")
        print(f"  output tail: {result_stdout[-500:]}")
        return 0.5, 0.5, 0, 0, 0

    score_plus = (W + D / 2.0) / N
    score_minus = (L + D / 2.0) / N

    return score_plus, score_minus, W, D, L


# ============================================================================
# Live Plotter
# ============================================================================

class SPSAPlotter:
    def __init__(self, param_names, params_dict, total_iterations):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        self.plt = plt
        self.param_names = param_names
        self.params_dict = params_dict
        self.n_params = len(param_names)
        self.total_iterations = total_iterations

        # data series
        self.iters = []
        self.theta_history = {name: [] for name in param_names}
        self.gradient_history = {name: [] for name in param_names}
        self.score_deltas = []
        self.running_avg = []
        self.cumulative_wins = []  # running θ+ win rate
        self.total_plus_wins = 0
        self.total_plus_games = 0

        # colors for gradient lines
        cmap = plt.cm.get_cmap('tab10')
        self.param_colors = [cmap(i % 10) for i in range(self.n_params)]

        # layout: 2 columns
        # left: param traces (stacked, n_params rows)
        # right: 3 plots (score delta, gradient magnitudes, win rate + decay)
        self.plt.ion()
        n_left_rows = max(self.n_params, 3)
        n_right_rows = 3
        self.fig = self.plt.figure(figsize=(15, max(8, n_left_rows * 1.4)))
        gs = self.fig.add_gridspec(
            n_left_rows, 4, hspace=0.5, wspace=0.35,
            left=0.06, right=0.98, top=0.93, bottom=0.06
        )

        # ── LEFT: Parameter traces (columns 0-1) ──
        self.ax_params = []
        self.lines_params = []
        for i, name in enumerate(param_names):
            ax = self.fig.add_subplot(gs[i, 0:2])
            cfg = params_dict[name]
            ax.axhline(cfg["value"], color='grey', linestyle=':', alpha=0.5, linewidth=0.8)
            ax.axhline(cfg["min"], color='red', linestyle='--', alpha=0.2, linewidth=0.5)
            ax.axhline(cfg["max"], color='red', linestyle='--', alpha=0.2, linewidth=0.5)
            ax.set_ylabel(name, fontsize=7, rotation=0, labelpad=65, ha='left')
            ax.tick_params(labelsize=7)
            if i < len(param_names) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Iteration", fontsize=8)
            margin = max(cfg["c_end"] * 3, (cfg["max"] - cfg["min"]) * 0.1)
            ax.set_ylim(cfg["value"] - margin, cfg["value"] + margin)
            line, = ax.plot([], [], color=self.param_colors[i], linewidth=1.0)
            self.ax_params.append(ax)
            self.lines_params.append(line)

        # ── RIGHT TOP: Score delta (column 2-3, top third) ──
        r_rows = n_left_rows // 3 or 1
        self.ax_delta = self.fig.add_subplot(gs[0:r_rows, 2:4])
        self.ax_delta.set_ylabel("Score Δ", fontsize=8)
        self.ax_delta.axhline(0, color='grey', linestyle='-', alpha=0.3)
        self.ax_delta.tick_params(labelsize=7)
        self.scatter_delta = None
        self.line_avg, = self.ax_delta.plot([], [], 'r-', linewidth=1.5, label='Avg(20)')
        self.ax_delta.legend(fontsize=7, loc='upper right')
        self.ax_delta.set_ylim(-0.15, 0.15)
        self.ax_delta.set_title("Score Δ (S+ − S−)", fontsize=8)

        # ── RIGHT MIDDLE: Per-param smoothed gradient (column 2-3, middle third) ──
        self.ax_grad = self.fig.add_subplot(gs[r_rows:2*r_rows, 2:4])
        self.ax_grad.axhline(0, color='grey', linestyle='-', alpha=0.3)
        self.ax_grad.tick_params(labelsize=7)
        self.ax_grad.set_ylabel("Gradient (smoothed)", fontsize=8)
        self.ax_grad.set_title("Per-param gradient signal", fontsize=8)
        self.lines_grad = []
        for i, name in enumerate(param_names):
            line, = self.ax_grad.plot([], [], color=self.param_colors[i],
                                      linewidth=1.0, alpha=0.8, label=name)
            self.lines_grad.append(line)
        self.ax_grad.legend(fontsize=6, loc='upper right', ncol=max(1, self.n_params // 3))

        # ── RIGHT BOTTOM: Convergence (v/c) + win rate (column 2-3, bottom third) ──
        self.ax_conv = self.fig.add_subplot(gs[2*r_rows:, 2:4])
        self.ax_conv.set_xlabel("Iteration", fontsize=8)
        self.ax_conv.set_ylabel("v/c (convergence)", fontsize=8, color='purple')
        self.ax_conv.tick_params(labelsize=7, axis='y', colors='purple')
        self.ax_conv.set_title("Convergence & Win Rate", fontsize=8)
        self.line_conv, = self.ax_conv.plot([], [], 'purple', linewidth=1.5, label='v/c')
        # Convergence level bands
        self.ax_conv.axhspan(0.1, 1.0, alpha=0.06, color='red')
        self.ax_conv.axhspan(0.05, 0.1, alpha=0.06, color='orange')
        self.ax_conv.axhspan(0.01, 0.05, alpha=0.06, color='yellow')
        self.ax_conv.axhspan(0.0, 0.01, alpha=0.06, color='green')
        self.ax_conv.axhline(0.1, color='red', linestyle='--', alpha=0.5, linewidth=0.7)
        self.ax_conv.axhline(0.05, color='orange', linestyle='--', alpha=0.5, linewidth=0.7)
        self.ax_conv.axhline(0.01, color='green', linestyle='--', alpha=0.5, linewidth=0.7)
        # Labels on right edge
        self.ax_conv.text(1.0, 0.15, ' exploring', fontsize=6, color='red',
                         alpha=0.7, va='center', transform=self.ax_conv.get_yaxis_transform())
        self.ax_conv.text(1.0, 0.07, ' tuning', fontsize=6, color='orange',
                         alpha=0.7, va='center', transform=self.ax_conv.get_yaxis_transform())
        self.ax_conv.text(1.0, 0.03, ' settling', fontsize=6, color='#888800',
                         alpha=0.7, va='center', transform=self.ax_conv.get_yaxis_transform())
        self.ax_conv.text(1.0, 0.005, ' converged', fontsize=6, color='green',
                         alpha=0.7, va='center', transform=self.ax_conv.get_yaxis_transform())
        self.ax_conv.set_ylim(0, 0.2)
        self.ax_conv.legend(fontsize=7, loc='upper left')
        self.convergence_series = []

        # Secondary y-axis for win rate
        self.ax_wr = self.ax_conv.twinx()
        self.ax_wr.set_ylabel("θ+ Win Rate", fontsize=7, color='blue')
        self.ax_wr.tick_params(labelsize=6, colors='blue')
        self.ax_wr.axhline(0.5, color='blue', linestyle=':', alpha=0.3)
        self.line_wr, = self.ax_wr.plot([], [], 'b-', linewidth=0.8, alpha=0.6, label='Win rate')
        self.ax_wr.set_ylim(0.35, 0.65)

        self.fig.suptitle("SPSA Parameter Tuning", fontweight='bold', fontsize=11)
        self.plt.show(block=False)
        self.fig.canvas.draw()

        # grab the Tk root for direct event pumping (avoids TkAgg blocking issues)
        self._tk_root = None
        try:
            self._tk_root = self.fig.canvas.manager.window
        except Exception:
            pass

    def _pump_events(self):
        """non-blocking GUI event pump"""
        try:
            if self._tk_root is not None:
                self._tk_root.update()
            else:
                self.fig.canvas.flush_events()
        except Exception:
            pass

    def update(self, iteration, theta, score_plus, score_minus, gradient, convergence_ratio):
        self.iters.append(iteration)
        delta = score_plus - score_minus
        self.score_deltas.append(delta)

        # Cumulative win rate (S+ > 0.5 means θ+ won)
        self.total_plus_games += 1
        if score_plus > score_minus:
            self.total_plus_wins += 1
        wr = self.total_plus_wins / self.total_plus_games
        self.cumulative_wins.append(wr)

        # Running average of score delta (window = 20)
        window = min(20, len(self.score_deltas))
        avg = sum(self.score_deltas[-window:]) / window
        self.running_avg.append(avg)

        # Per-param gradient (smoothed)
        for i, name in enumerate(self.param_names):
            self.gradient_history[name].append(gradient[i])

        # ── Update param traces ──
        for i, name in enumerate(self.param_names):
            self.theta_history[name].append(theta[i])
            self.lines_params[i].set_data(self.iters, self.theta_history[name])
            ax = self.ax_params[i]
            ax.set_xlim(0, max(self.iters) * 1.05 + 1)
            vals = self.theta_history[name]
            cfg = self.params_dict[name]
            lo = min(min(vals), cfg["value"])
            hi = max(max(vals), cfg["value"])
            margin = max(cfg["c_end"], (hi - lo) * 0.15, 1)
            ax.set_ylim(lo - margin, hi + margin)

        x_max = max(self.iters) * 1.05 + 1

        # ── Update score delta ──
        self.ax_delta.set_xlim(0, x_max)
        if self.scatter_delta:
            self.scatter_delta.remove()
        self.scatter_delta = self.ax_delta.scatter(
            self.iters, self.score_deltas, s=8, alpha=0.4, color='blue', zorder=2
        )
        self.line_avg.set_data(self.iters, self.running_avg)
        d_max = max(abs(min(self.score_deltas)), abs(max(self.score_deltas)), 0.05)
        self.ax_delta.set_ylim(-d_max * 1.3, d_max * 1.3)

        # ── Update gradient lines (smoothed with EMA) ──
        self.ax_grad.set_xlim(0, x_max)
        g_max = 0.01
        for i, name in enumerate(self.param_names):
            raw = self.gradient_history[name]
            # EMA smoothing (alpha=0.2)
            smoothed = []
            ema = 0
            for j, g in enumerate(raw):
                ema = 0.2 * g + 0.8 * ema if j > 0 else g
                smoothed.append(ema)
            self.lines_grad[i].set_data(self.iters, smoothed)
            if smoothed:
                g_max = max(g_max, max(abs(s) for s in smoothed))
        self.ax_grad.set_ylim(-g_max * 1.3, g_max * 1.3)

        # ── Update convergence + win rate ──
        self.convergence_series.append(convergence_ratio)
        self.ax_conv.set_xlim(0, x_max)
        self.line_conv.set_data(self.iters, self.convergence_series)
        conv_max = max(max(self.convergence_series), 0.12)
        self.ax_conv.set_ylim(0, min(conv_max * 1.3, 0.5))

        self.ax_wr.set_xlim(0, x_max)
        self.line_wr.set_data(self.iters, self.cumulative_wins)
        wr_dev = max(abs(wr - 0.5), 0.05)
        self.ax_wr.set_ylim(0.5 - wr_dev * 1.5, 0.5 + wr_dev * 1.5)

        # Live status in suptitle
        if convergence_ratio > 0.1:
            phase = "EXPLORING"
        elif convergence_ratio > 0.05:
            phase = "TUNING"
        elif convergence_ratio > 0.01:
            phase = "SETTLING"
        else:
            phase = "CONVERGED"
        n_iters = len(self.iters)
        self.fig.suptitle(
            f"SPSA | Iter {iteration}/{self.total_iterations} | "
            f"v/c={convergence_ratio:.3f} ({phase}) | "
            f"WR={wr:.1%}",
            fontweight='bold', fontsize=10
        )

        self.fig.canvas.draw_idle()
        self._pump_events()

    def finalize(self):
        self.fig.suptitle("SPSA Parameter Tuning — COMPLETE", fontweight='bold', fontsize=11)
        self.plt.ioff()
        self.plt.show()


# ============================================================================
# Main SPSA loop
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="SPSA parameter tuner for Tomahawk")

    p.add_argument("--engine", required=True, help="Path to candidate engine")
    p.add_argument("--baseline", default=None,
                   help="Path to baseline engine (for validation only, not used in tuning)")
    p.add_argument("--cutechess-cli",
                   default=r"C:\Program Files (x86)\Cute Chess\cutechess-cli.exe")

    # Time control
    p.add_argument("--tc", type=str, default=None, help="Time control (e.g. 0:1+0.01)")
    p.add_argument("--time", type=float, default=None, help="Seconds per move")
    p.add_argument("--depth", type=int, default=None, help="Fixed depth")

    # SPSA settings
    p.add_argument("--iterations", type=int, default=500, help="Number of SPSA iterations")
    p.add_argument("--games-per-iter", type=int, default=4,
                   help="Games per iteration (split between +/- perturbations)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--r", type=float, default=None,
                   help="Override learning rate r (default: auto from games_per_iter)")

    # Opening book
    p.add_argument("--book", default= PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn", help="Opening book file")
    p.add_argument("--book-depth", type=int, default=16)

    # Params to tune (subset of DEFAULT_PARAMS keys)
    p.add_argument("--params", nargs="+", default=None,
                   help="Parameter names to tune (default: all)")

    # Output
    p.add_argument("--output", default="logs/dev_logs/spsa_results.json", help="Output file for results")
    p.add_argument("--resume", default=None, help="Resume from a previous output JSON")
    p.add_argument("--plot", action="store_true", help="Show live parameter plots")

    # Optimizer variant
    p.add_argument("--optimizer", choices=["vanilla", "adam"], default="adam",
                   help="Gradient update rule (default: adam)")
    p.add_argument("--no-polyak", action="store_true",
                   help="Disable Polyak-Ruppert averaging of the final theta")

    return p.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    # Select parameters to tune
    if args.params:
        params_dict = {k: copy.deepcopy(DEFAULT_PARAMS[k]) for k in args.params
                       if k in DEFAULT_PARAMS}
    else:
        params_dict = copy.deepcopy(DEFAULT_PARAMS)

    if not params_dict:
        print("[SPSA] ERROR: No valid parameters selected.")
        sys.exit(1)

    param_names = list(params_dict.keys())
    n_params = len(param_names)

    print(f"[SPSA] Tuning {n_params} parameters: {param_names}")
    print(f"[SPSA] Iterations: {args.iterations}, Games/iter: {args.games_per_iter}")

    # Initial theta
    theta = np.array([params_dict[name]["value"] for name in param_names], dtype=float)

    # Resume from checkpoint
    start_iter = 0
    history = []
    if args.resume and Path(args.resume).exists():
        with open(args.resume, "r") as f:
            checkpoint = json.load(f)
        theta = np.array(checkpoint["theta"])
        start_iter = checkpoint["iteration"] + 1
        history = checkpoint.get("history", [])
        total_games = checkpoint.get("total_games", 0)
        total_w = checkpoint.get("total_w", 0)
        total_d = checkpoint.get("total_d", 0)
        total_l = checkpoint.get("total_l", 0)
        # Restore params_config if available
        if "params_config" in checkpoint:
            saved_params = checkpoint["params_config"]
            for name in param_names:
                if name in saved_params:
                    params_dict[name] = saved_params[name]
        # Validate theta length matches
        if len(theta) != n_params:
            print(f"[SPSA] ERROR: Checkpoint has {len(theta)} params, expected {n_params}")
            sys.exit(1)
        print(f"[SPSA] Resuming from iteration {start_iter}, theta = {theta}")

    # Time control args for cutechess
    tc_args = []
    if args.depth is not None:
        tc_args = [f"depth={args.depth}"]
    elif args.time is not None:
        tc_args = [f"st={args.time}", "timemargin=30"]
    elif args.tc is not None:
        tc_args = [f"tc={args.tc}"]
    else:
        print("[SPSA] ERROR: Specify --tc, --time, or --depth")
        sys.exit(1)

    # Book args
    book_args = []
    if args.book:
        ext = os.path.splitext(args.book)[1][1:]
        book_args = [
            "-openings", f"file={os.path.abspath(args.book)}",
            f"format={ext}", "order=random", f"plies={args.book_depth}",
        ]

    engine_path = os.path.abspath(args.engine)

    # Compute global learning rate
    r = compute_r(args.games_per_iter)
    if args.r is not None:
        r = args.r
    print(f"[SPSA] Optimizer: {args.optimizer}, Polyak avg: {'off' if args.no_polyak else 'on'}")
    print(f"[SPSA] Learning rate r = {r:.6f} (R_REF={R_REF}, √({args.games_per_iter}/{N_REF}) scaling)")
    for name in param_names:
        c = params_dict[name]["c_end"]
        a = r * c ** 2
        print(f"  {name:30s}: c_end={c:>6.1f}, a_end={a:.4f}")

    # Live plotter
    plotter = None
    if args.plot:
        plotter = SPSAPlotter(param_names, params_dict, args.iterations)

    print(f"\n[SPSA] Initial theta: {dict(zip(param_names, theta))}")
    print(f"[SPSA] Starting tuning...")
    print()

    if start_iter == 0:
        total_games = 0
        total_w = 0
        total_d = 0
        total_l = 0

    # Adam optimizer state (per-parameter adaptive learning rates + momentum)
    adam_m = np.zeros(n_params)  # first moment (mean of gradients)
    adam_v = np.zeros(n_params)  # second moment (mean of squared gradients)
    adam_beta1 = 0.9             # momentum decay
    adam_beta2 = 0.999           # RMS decay
    adam_eps = 1e-8              # numerical stability

    # Polyak-Ruppert averaging (average θ over the last half of iterations)
    polyak_sum = np.zeros(n_params)
    polyak_count = 0
    polyak_start = args.iterations // 2  # start averaging at the halfway point

    # Restore Polyak state on resume
    if start_iter > 0 and args.resume and Path(args.resume).exists():
        with open(args.resume, "r") as f:
            _ckpt = json.load(f)
        _pt = _ckpt.get("polyak_theta")
        _pc = _ckpt.get("polyak_count", 0)
        if _pt is not None and _pc > 0:
            polyak_sum = np.array(_pt) * _pc  # reconstruct running sum
            polyak_count = _pc

    start_time = time.time()

    for k in range(start_iter, args.iterations):
        # Compute coefficients per parameter
        a_k = np.array([
            spsa_coefficients(k, args.iterations, params_dict[name]["c_end"], r)[0]
            for name in param_names
        ])
        c_k = np.array([
            spsa_coefficients(k, args.iterations, params_dict[name]["c_end"], r)[1]
            for name in param_names
        ])

        # For Adam, the gradient is normalized to ~unit magnitude, so the step
        # size must be in parameter units (proportional to c_end, not c_end²).
        # a_k_adam = adam_lr * c_end, with the same decay schedule as a_k.
        alpha = 0.602
        A = args.iterations * 0.1
        decay = (A + 2) ** alpha / (A + k + 1) ** alpha
        a_k_adam = np.array([
            ADAM_LR * params_dict[name]["c_end"] * decay
            for name in param_names
        ])

        # Generate perturbation
        delta = bernoulli_perturbation(n_params)

        # Perturbed parameter vectors
        theta_plus = np.array([
            clamp(theta[i] + c_k[i] * delta[i],
                  params_dict[param_names[i]]["min"],
                  params_dict[param_names[i]]["max"])
            for i in range(n_params)
        ])
        theta_minus = np.array([
            clamp(theta[i] - c_k[i] * delta[i],
                  params_dict[param_names[i]]["min"],
                  params_dict[param_names[i]]["max"])
            for i in range(n_params)
        ])

        # Run match
        score_plus, score_minus, W, D, L = run_match(
            engine_path, None, params_dict, theta_plus, theta_minus,
            args.games_per_iter, tc_args, book_args, args.concurrency,
            args.cutechess_cli, plotter=plotter,
            iteration=k, total_iterations=args.iterations
        )
        n_games = W + D + L
        total_games += n_games
        total_w += W
        total_d += D
        total_l += L

        if n_games == 0:
            print(f"[SPSA] Iter {k}: No games played, skipping update.")
            continue

        # Warn if all draws (no signal)
        if W == 0 and L == 0:
            print(f"[SPSA] WARNING: All {D} games drawn — perturbation may be too small")

        # Gradient estimate (per parameter)
        gradient = (score_plus - score_minus) / (2.0 * c_k * delta)

        # Update theta (gradient ASCENT — maximizing score)
        theta_old = theta.copy()
        if args.optimizer == "adam":
            # Adam: adaptive per-param learning rates + momentum.
            # Step index for bias correction (1-based).
            t = k - start_iter + 1
            adam_m = adam_beta1 * adam_m + (1 - adam_beta1) * gradient
            adam_v = adam_beta2 * adam_v + (1 - adam_beta2) * (gradient ** 2)
            m_hat = adam_m / (1 - adam_beta1 ** t)
            v_hat = adam_v / (1 - adam_beta2 ** t)
            # a_k_adam is in param units (scaled by c_end); Adam gives unit direction.
            theta_new = theta + a_k_adam * m_hat / (np.sqrt(v_hat) + adam_eps)
        else:
            theta_new = theta + a_k * gradient

        # Clamp to bounds
        for i, name in enumerate(param_names):
            theta_new[i] = clamp(theta_new[i], params_dict[name]["min"], params_dict[name]["max"])

        theta = theta_new

        # Actual applied step (after clamping) — used for the convergence metric.
        # Reflects whichever optimizer was used, not just the vanilla step.
        actual_step = theta - theta_old

        # Polyak-Ruppert averaging: accumulate theta over the second half
        if not args.no_polyak and k >= polyak_start:
            polyak_sum += theta
            polyak_count += 1

        # Log
        elapsed = time.time() - start_time
        current_values = {}
        for i, name in enumerate(param_names):
            v = int(round(theta[i])) if params_dict[name]["integer"] else round(theta[i], 4)
            current_values[name] = v

        step_record = {
            "iteration": k,
            "score_plus": round(score_plus, 4),
            "score_minus": round(score_minus, 4),
            "W": W, "D": D, "L": L,
            "games": n_games,
            "theta": {name: float(theta[i]) for i, name in enumerate(param_names)},
            "step_norm": float(np.linalg.norm(actual_step)),
        }
        history.append(step_record)

        # Convergence metric: normalized velocity (avg |step| / c_end over last 20 iters)
        window = min(20, len(history))
        recent_steps = [h["step_norm"] for h in history[-window:]]
        avg_velocity = sum(recent_steps) / window
        # Normalized: how much are params moving relative to their perturbation size?
        avg_c = np.mean([params_dict[name]["c_end"] for name in param_names])
        convergence_ratio = avg_velocity / avg_c if avg_c > 0 else 0

        # Progress output
        diff = score_plus - score_minus
        draw_pct = D / n_games * 100 if n_games > 0 else 0
        print(
            f"  Iter {k:4d}/{args.iterations} | "
            f"S+={score_plus:.3f} S-={score_minus:.3f} (Δ={diff:+.3f}) | "
            f"W/D/L: {W}/{D}/{L} ({draw_pct:.0f}%D) | "
            f"Games: {total_games} | "
            f"v/c={convergence_ratio:.3f} | "
            f"Time: {elapsed:.0f}s"
        )
        print(f"         θ = {current_values}")

        # Update live plot
        if plotter:
            plotter.update(k, theta, score_plus, score_minus, gradient, convergence_ratio)

        # Checkpoint every 10 iterations
        if k % 10 == 0 or k == args.iterations - 1:
            # Running Polyak average (of iterations seen so far in the averaging window)
            polyak_theta_now = (polyak_sum / polyak_count) if polyak_count > 0 else None
            polyak_values_now = ({
                name: (int(round(polyak_theta_now[i])) if params_dict[name]["integer"]
                       else round(polyak_theta_now[i], 4))
                for i, name in enumerate(param_names)
            } if polyak_theta_now is not None else None)

            checkpoint = {
                "iteration": k,
                "theta": theta.tolist(),
                "polyak_theta": polyak_theta_now.tolist() if polyak_theta_now is not None else None,
                "current_values": current_values,
                "final_values": current_values,
                "polyak_values": polyak_values_now,
                "start_values": {name: params_dict[name]["value"] for name in param_names},
                "params_config": params_dict,
                "params": param_names,
                "polyak_count": polyak_count,
                "total_games": total_games,
                "total_w": total_w,
                "total_d": total_d,
                "total_l": total_l,
                "elapsed": elapsed,
                "history": history,
                "optimizer": args.optimizer,
                "args": {
                    "engine": engine_path,
                    "games_per_iter": args.games_per_iter,
                    "iterations": args.iterations,
                    "tc_args": tc_args,
                    "concurrency": args.concurrency,
                    "r": r,
                },
            }
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(checkpoint, f, indent=2)

    # Final results
    print()
    print("=" * 60)
    print("[SPSA] TUNING COMPLETE")
    print("=" * 60)
    print(f"  Total games: {total_games}")
    print(f"  Total time:  {time.time() - start_time:.0f}s")
    print(f"  Optimizer:   {args.optimizer}")

    # Polyak-Ruppert averaged theta (lower variance than the final iterate)
    polyak_theta = None
    if not args.no_polyak and polyak_count > 0:
        polyak_theta = polyak_sum / polyak_count

    print(f"  Final parameters (last iterate):")
    for i, name in enumerate(param_names):
        start_val = params_dict[name]["value"]
        v = int(round(theta[i])) if params_dict[name]["integer"] else round(theta[i], 4)
        print(f"    {name:30s}: {start_val} -> {v}")

    if polyak_theta is not None:
        print(f"\n  Polyak-averaged parameters (RECOMMENDED, avg of last {polyak_count} iters):")
        for i, name in enumerate(param_names):
            start_val = params_dict[name]["value"]
            v = int(round(polyak_theta[i])) if params_dict[name]["integer"] else round(polyak_theta[i], 4)
            print(f"    {name:30s}: {start_val} -> {v}")

    # Save both to output
    final_checkpoint = {
        "iteration": args.iterations - 1,
        "theta": theta.tolist(),
        "polyak_theta": polyak_theta.tolist() if polyak_theta is not None else None,
        "final_values": {
            name: (int(round(theta[i])) if params_dict[name]["integer"] else round(theta[i], 4))
            for i, name in enumerate(param_names)
        },
        "polyak_values": ({
            name: (int(round(polyak_theta[i])) if params_dict[name]["integer"] else round(polyak_theta[i], 4))
            for i, name in enumerate(param_names)
        } if polyak_theta is not None else None),
        "start_values": {name: params_dict[name]["value"] for name in param_names},
        "params_config": params_dict,
        "params": param_names,
        "total_games": total_games,
        "total_w": total_w,
        "total_d": total_d,
        "total_l": total_l,
        "elapsed": time.time() - start_time,
        "history": history,
        "optimizer": args.optimizer,
        "args": {
            "engine": engine_path,
            "games_per_iter": args.games_per_iter,
            "iterations": args.iterations,
            "tc_args": tc_args,
            "concurrency": args.concurrency,
            "r": r,
        },
    }
    with open(Path(args.output), "w") as f:
        json.dump(final_checkpoint, f, indent=2)

    print(f"\n  Results saved to: {args.output}")

    if plotter:
        plotter.finalize()

    return final_checkpoint["final_values"]


if __name__ == "__main__":
    main()