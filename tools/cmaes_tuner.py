#!/usr/bin/env python3
"""

!!!!!!!!!!!!!!!!!!!!!!!!!
UNTESTED

Not the right call over SPSA for search param values
Better for eval tuning (irrelevant under nnue framework, except HP tuning of trainer)
!!!!!!!!!!!!!!!!!!!!!!!!!



CMA-ES parameter tuner .

Uses Covariance Matrix Adaptation Evolution Strategy for black-box
optimization of search parameters. Wraps cutechess-cli as the game oracle.

Advantages over SPSA:
  - No learning rate to tune (adapts automatically)
  - Handles parameter correlations (learns covariance structure)
  - Population-based (naturally parallel)
  - Better convergence for <20 parameters

Requires: pip install cma

Usage:
    python -m tests.cmaes --engine <path> --baseline <path> --depth 8 --plot
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
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

try:
    import cma
except ImportError:
    print("ERROR: CMA-ES requires the 'cma' package. Install with: pip install cma")
    sys.exit(1)


# ============================================================================
# Parameter Configuration
# ============================================================================

DEFAULT_PARAMS = {
    "r_nmp": {
        "value": 3, "min": 2, "max": 5,
        "sigma0": 0.5,
        "integer": True,
    },
    "r_lmr_const": {
        "value": 99, "min": 30, "max": 200,
        "sigma0": 8,
        "integer": True,
    },
    "r_lmr_denom": {
        "value": 314, "min": 100, "max": 600,
        "sigma0": 20,
        "integer": True,
    },
    "lmr_depth_threshold": {
        "value": 3, "min": 1, "max": 8,
        "sigma0": 1.0,
        "integer": True,
    },
    "lmr_move_order_threshold": {
        "value": 3, "min": 1, "max": 8,
        "sigma0": 1.0,
        "integer": True,
    },
    "aspiration_window": {
        "value": 50, "min": 10, "max": 200,
        "sigma0": 10,
        "integer": True,
    },
    "aspiration_start_depth": {
        "value": 6, "min": 3, "max": 12,
        "sigma0": 1.0,
        "integer": True,
    },
    "delta_prune_threshold": {
        "value": 1000, "min": 200, "max": 2000,
        "sigma0": 80,
        "integer": True,
    },
    "see_prune_threshold": {
        "value": -50, "min": -500, "max": 200,
        "sigma0": 30,
        "integer": True,
    },
}


# ============================================================================
# Cutechess integration
# ============================================================================

def build_option_string(params_dict, param_names, values):
    """Build cutechess option string for a parameter vector."""
    options = []
    for name, val in zip(param_names, values):
        cfg = params_dict[name]
        v = int(round(val)) if cfg["integer"] else val
        v = max(cfg["min"], min(cfg["max"], v))
        options.append(f"option.{name}={v}")
    return options


def evaluate_candidate(engine_path, baseline_path, params_dict, param_names, candidate,
                       games, tc_args, book_args, concurrency, cutechess_cli,
                       baseline_opts=None):
    """
    Evaluate a candidate by playing against the baseline.
    The baseline runs with `baseline_opts` (the current parameter values) so it
    represents the engine as currently configured. Returns score (higher = better).
    """
    opts_candidate = build_option_string(params_dict, param_names, candidate)
    if baseline_opts is None:
        baseline_opts = []

    cmd = [
        cutechess_cli,
        "-engine", "name=Candidate", f"cmd={engine_path}",
        f"dir={os.path.dirname(engine_path)}",
    ] + opts_candidate + [
        "-engine", "name=Baseline", f"cmd={baseline_path}",
        f"dir={os.path.dirname(baseline_path)}",
    ] + baseline_opts + [
        "-each", "proto=uci",
    ] + tc_args + [
        "-games", str(games),
        "-repeat",
        "-concurrency", str(concurrency),
        "-maxmoves", "100",
    ] + book_args + [
        "-wait", "50",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    score_re = re.compile(
        r"Score of Candidate vs Baseline:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)"
    )

    W, L, D = 0, 0, 0
    for line in result.stdout.splitlines():
        m = score_re.search(line)
        if m:
            W, L, D = int(m.group(1)), int(m.group(2)), int(m.group(3))

    N = W + L + D
    if N == 0:
        print(f"    [WARN] No games completed")
        return 0.5

    score = (W + D / 2.0) / N
    return score


# ============================================================================
# Live Plotter
# ============================================================================

class CMAPlotter:
    def __init__(self, param_names, params_dict):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.param_names = param_names
        self.params_dict = params_dict
        self.n_params = len(param_names)

        self.generations = []
        self.best_scores = []
        self.mean_scores = []
        self.sigma_history = []
        self.mean_history = {name: [] for name in param_names}

        cmap = plt.cm.get_cmap('tab10')
        self.param_colors = [cmap(i % 10) for i in range(self.n_params)]

        self.plt.ion()
        self.fig = self.plt.figure(figsize=(13, 8))
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3,
                                   left=0.07, right=0.97, top=0.92, bottom=0.08)

        # Top-left: parameter means (normalized)
        self.ax_params = self.fig.add_subplot(gs[0, 0])
        self.ax_params.set_xlabel("Generation", fontsize=8)
        self.ax_params.set_ylabel("Normalized value", fontsize=8)
        self.ax_params.set_title("Distribution Mean", fontsize=9)
        self.ax_params.tick_params(labelsize=7)
        self.lines_params = []
        for i, name in enumerate(param_names):
            line, = self.ax_params.plot([], [], color=self.param_colors[i],
                                        linewidth=1.2, label=name)
            self.lines_params.append(line)
        self.ax_params.legend(fontsize=6, loc='upper right', ncol=max(1, self.n_params // 4))
        self.ax_params.set_ylim(-0.1, 1.1)

        # Top-right: scores
        self.ax_score = self.fig.add_subplot(gs[0, 1])
        self.ax_score.set_xlabel("Generation", fontsize=8)
        self.ax_score.set_ylabel("Score vs Baseline", fontsize=8)
        self.ax_score.set_title("Population Fitness", fontsize=9)
        self.ax_score.axhline(0.5, color='grey', linestyle='-', alpha=0.3)
        self.ax_score.tick_params(labelsize=7)
        self.line_best, = self.ax_score.plot([], [], 'g-', linewidth=1.5, label='Gen best')
        self.line_mean, = self.ax_score.plot([], [], 'b-', linewidth=1.0, label='Gen mean')
        self.ax_score.legend(fontsize=7)
        self.ax_score.set_ylim(0.4, 0.6)

        # Bottom-left: sigma
        self.ax_sigma = self.fig.add_subplot(gs[1, 0])
        self.ax_sigma.set_xlabel("Generation", fontsize=8)
        self.ax_sigma.set_ylabel("σ (step size)", fontsize=8)
        self.ax_sigma.set_title("CMA-ES Adaptation", fontsize=9)
        self.ax_sigma.tick_params(labelsize=7)
        self.line_sigma, = self.ax_sigma.plot([], [], 'r-', linewidth=1.5)
        self.ax_sigma.axhspan(0, 0.001, alpha=0.08, color='green')
        self.ax_sigma.axhline(0.001, color='green', linestyle=':', alpha=0.5)

        # Bottom-right: status
        self.ax_status = self.fig.add_subplot(gs[1, 1])
        self.ax_status.set_xticks([])
        self.ax_status.set_yticks([])
        self.ax_status.set_facecolor('#f8f8f8')
        for spine in self.ax_status.spines.values():
            spine.set_color('#cccccc')
        self.status_text = self.ax_status.text(
            0.05, 0.95, '', fontsize=8, fontfamily='monospace',
            verticalalignment='top', transform=self.ax_status.transAxes
        )

        self.fig.suptitle("CMA-ES Parameter Tuning", fontweight='bold', fontsize=11)
        self.plt.show(block=False)
        self.plt.pause(0.1)

    def update(self, gen, best_score, mean_score, sigma, mean_theta, best_theta, total_games):
        self.generations.append(gen)
        self.best_scores.append(best_score)
        self.mean_scores.append(mean_score)
        self.sigma_history.append(sigma)

        x_max = max(self.generations) * 1.05 + 1

        # Params (normalized)
        for i, name in enumerate(self.param_names):
            cfg = self.params_dict[name]
            rng = cfg["max"] - cfg["min"]
            normalized = (mean_theta[i] - cfg["min"]) / rng if rng > 0 else 0.5
            self.mean_history[name].append(normalized)
            self.lines_params[i].set_data(self.generations, self.mean_history[name])
        self.ax_params.set_xlim(0, x_max)

        # Score
        self.line_best.set_data(self.generations, self.best_scores)
        self.line_mean.set_data(self.generations, self.mean_scores)
        self.ax_score.set_xlim(0, x_max)
        s_min = min(min(self.mean_scores), 0.45)
        s_max = max(max(self.best_scores), 0.55)
        self.ax_score.set_ylim(s_min - 0.02, s_max + 0.02)

        # Sigma
        self.line_sigma.set_data(self.generations, self.sigma_history)
        self.ax_sigma.set_xlim(0, x_max)
        self.ax_sigma.set_ylim(0, max(max(self.sigma_history), 0.01) * 1.3)

        # Status text — show RECOMMENDED distribution mean (noise-unbiased)
        lines = []
        lines.append(f"Generation: {gen}")
        lines.append(f"Total games: {total_games}")
        lines.append(f"σ: {sigma:.5f}")
        lines.append(f"Best score: {best_score:.3f}")
        lines.append("")
        lines.append("Mean θ (RECOMMENDED):")
        for i, name in enumerate(self.param_names):
            cfg = self.params_dict[name]
            v = int(round(mean_theta[i])) if cfg["integer"] else round(mean_theta[i], 3)
            start = cfg["value"]
            lines.append(f"  {name}: {start}→{v}")
        self.status_text.set_text('\n'.join(lines))

        # Title
        self.fig.suptitle(
            f"CMA-ES | Gen {gen} | σ={sigma:.4f} | Best={best_score:.3f}",
            fontweight='bold', fontsize=10
        )

        self.fig.canvas.draw_idle()
        try:
            self.plt.pause(0.001)
        except Exception:
            pass

    def finalize(self):
        self.fig.suptitle("CMA-ES — COMPLETE", fontweight='bold', fontsize=11)
        self.plt.ioff()
        self.plt.show()


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="CMA-ES parameter tuner for Tomahawk")

    p.add_argument("--engine", required=True, help="Path to candidate engine")
    p.add_argument("--baseline", default=None,
                   help="Path to baseline engine (default: same as --engine, "
                        "running the current parameter values)")
    p.add_argument("--cutechess-cli",
                   default=r"C:\Program Files (x86)\Cute Chess\cutechess-cli.exe")

    # Time control
    p.add_argument("--tc", type=str, default=None)
    p.add_argument("--time", type=float, default=None)
    p.add_argument("--depth", type=int, default=None)

    # CMA-ES settings
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--popsize", type=int, default=None,
                   help="Population size (default: 4 + 3*ln(n_params))")
    p.add_argument("--games-per-eval", type=int, default=16)
    p.add_argument("--sigma0", type=float, default=None,
                   help="Initial global step size (default: auto)")
    p.add_argument("--concurrency", type=int, default=2)

    # Opening book
    p.add_argument("--book", default=None)
    p.add_argument("--book-depth", type=int, default=16)

    # Params
    p.add_argument("--params", nargs="+", default=None)

    # Output
    p.add_argument("--output", default="logs/dev_logs/cmaes_results.json")
    p.add_argument("--plot", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # Select parameters
    if args.params:
        params_dict = {k: copy.deepcopy(DEFAULT_PARAMS[k]) for k in args.params
                       if k in DEFAULT_PARAMS}
    else:
        params_dict = copy.deepcopy(DEFAULT_PARAMS)

    if not params_dict:
        print("[CMA-ES] ERROR: No valid parameters selected.")
        sys.exit(1)

    param_names = list(params_dict.keys())
    n_params = len(param_names)

    # Initial point and bounds
    x0 = [float(params_dict[name]["value"]) for name in param_names]
    lower_bounds = [float(params_dict[name]["min"]) for name in param_names]
    upper_bounds = [float(params_dict[name]["max"]) for name in param_names]

    # Initial sigma
    if args.sigma0:
        sigma0 = args.sigma0
    else:
        # Mean of per-param sigma0 relative to range
        ranges = [params_dict[name]["max"] - params_dict[name]["min"] for name in param_names]
        sigmas_rel = [params_dict[name]["sigma0"] / r for name, r in zip(param_names, ranges)]
        sigma0 = float(np.mean(sigmas_rel))

    # Population size
    popsize = args.popsize or int(4 + 3 * math.log(n_params))

    # CMA-ES options
    cma_opts = {
        'popsize': popsize,
        'bounds': [lower_bounds, upper_bounds],
        'maxiter': args.generations,
        'seed': 42,
        'verbose': -1,
        'CMA_stds': [params_dict[name]["sigma0"] for name in param_names],
    }

    print(f"[CMA-ES] Tuning {n_params} parameters: {param_names}")
    print(f"[CMA-ES] Population: {popsize}, Generations: {args.generations}")
    print(f"[CMA-ES] Games/eval: {args.games_per_eval}")
    print(f"[CMA-ES] Budget: ~{popsize * args.generations * args.games_per_eval} games")
    print(f"[CMA-ES] σ0: {sigma0:.4f}")
    print(f"[CMA-ES] x0: {dict(zip(param_names, x0))}")

    # Time control
    tc_args = []
    if args.depth is not None:
        tc_args = [f"depth={args.depth}"]
    elif args.time is not None:
        tc_args = [f"st={args.time}", "timemargin=30"]
    elif args.tc is not None:
        tc_args = [f"tc={args.tc}"]
    else:
        print("[CMA-ES] ERROR: Specify --tc, --time, or --depth")
        sys.exit(1)

    # Book
    book_args = []
    if args.book:
        ext = os.path.splitext(args.book)[1][1:]
        book_args = [
            "-openings", f"file={os.path.abspath(args.book)}",
            f"format={ext}", "order=random", f"plies={args.book_depth}",
        ]

    engine_path = os.path.abspath(args.engine)
    baseline_path = os.path.abspath(args.baseline) if args.baseline else engine_path

    # Baseline plays the current (start) parameter values so it represents the
    # engine as currently configured. Candidates are scored against this.
    start_theta = [params_dict[name]["value"] for name in param_names]
    baseline_opts = build_option_string(params_dict, param_names, start_theta)

    if args.baseline:
        print(f"[CMA-ES] Baseline: external engine {baseline_path}")
    else:
        print(f"[CMA-ES] Baseline: same engine at current values {dict(zip(param_names, start_theta))}")

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Plotter
    plotter = None
    if args.plot:
        plotter = CMAPlotter(param_names, params_dict)

    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)

    print(f"\n[CMA-ES] Starting optimization...")
    print()

    total_games = 0
    start_time = time.time()
    best_ever_score = 0.0
    best_ever_theta = np.array(x0)
    history = []

    gen = 0
    while not es.stop():
        gen += 1
        candidates = es.ask()

        # Evaluate each candidate
        scores = []
        for i, x in enumerate(candidates):
            score = evaluate_candidate(
                engine_path, baseline_path, params_dict, param_names, x,
                args.games_per_eval, tc_args, book_args, args.concurrency,
                args.cutechess_cli, baseline_opts=baseline_opts
            )
            scores.append(score)
            total_games += args.games_per_eval

        # CMA-ES minimizes → negate scores
        fitnesses = [-s for s in scores]
        es.tell(candidates, fitnesses)

        # Stats
        best_idx = int(np.argmax(scores))
        gen_best_score = scores[best_idx]
        gen_best_theta = candidates[best_idx]
        gen_mean_score = float(np.mean(scores))
        sigma = es.sigma

        if gen_best_score > best_ever_score:
            best_ever_score = gen_best_score
            best_ever_theta = np.array(gen_best_theta)

        # Format
        current_best = {}
        for i, name in enumerate(param_names):
            cfg = params_dict[name]
            v = int(round(best_ever_theta[i])) if cfg["integer"] else round(float(best_ever_theta[i]), 3)
            current_best[name] = v

        elapsed = time.time() - start_time
        print(
            f"  Gen {gen:3d}/{args.generations} | "
            f"Best={gen_best_score:.3f} Mean={gen_mean_score:.3f} | "
            f"σ={sigma:.4f} | "
            f"Games: {total_games} | "
            f"Time: {elapsed:.0f}s"
        )
        print(f"       Best ever: {current_best} (score={best_ever_score:.3f})")

        # History
        history.append({
            "generation": gen,
            "best_score": round(gen_best_score, 4),
            "mean_score": round(gen_mean_score, 4),
            "sigma": round(float(sigma), 6),
            "best_theta": {name: float(best_ever_theta[i]) for i, name in enumerate(param_names)},
            "mean_theta": {name: float(es.mean[i]) for i, name in enumerate(param_names)},
            "games": total_games,
        })

        # Plot
        if plotter:
            plotter.update(gen, gen_best_score, gen_mean_score, sigma,
                          es.mean, best_ever_theta, total_games)

        # Checkpoint
        if gen % 5 == 0 or es.stop():
            # Distribution mean (xfavorite) — the recommended estimate of the
            # optimum. Unlike best-ever, it is not biased by evaluation noise.
            mean_values = {}
            for i, name in enumerate(param_names):
                cfg = params_dict[name]
                v = int(round(es.mean[i])) if cfg["integer"] else round(float(es.mean[i]), 3)
                mean_values[name] = v

            checkpoint = {
                "generation": gen,
                "best_ever_score": round(best_ever_score, 4),
                "best_ever_theta": best_ever_theta.tolist(),
                "best_values": current_best,
                "mean_values": mean_values,          # RECOMMENDED (noise-unbiased)
                "cma_mean": es.mean.tolist(),
                "start_values": {name: params_dict[name]["value"] for name in param_names},
                "sigma": float(sigma),
                "params_config": params_dict,
                "params": param_names,
                "total_games": total_games,
                "elapsed": elapsed,
                "history": history,
                "args": {
                    "engine": engine_path,
                    "baseline": baseline_path,
                    "baseline_is_self": args.baseline is None,
                    "games_per_eval": args.games_per_eval,
                    "generations": args.generations,
                    "popsize": popsize,
                    "tc_args": tc_args,
                    "concurrency": args.concurrency,
                },
            }
            with open(Path(args.output), "w") as f:
                json.dump(checkpoint, f, indent=2)

    # Final
    print()
    print("=" * 60)
    print("[CMA-ES] OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"  Generations: {gen}")
    print(f"  Total games: {total_games}")
    print(f"  Total time:  {time.time() - start_time:.0f}s")
    print(f"  Stop reason: {es.stop()}")
    print(f"  Best score:  {best_ever_score:.3f}")

    print(f"\n  Best-ever candidate (noisy — may be lucky sample):")
    for i, name in enumerate(param_names):
        start_val = params_dict[name]["value"]
        v = int(round(best_ever_theta[i])) if params_dict[name]["integer"] else round(float(best_ever_theta[i]), 3)
        print(f"    {name:30s}: {start_val} -> {v}")

    print(f"\n  Distribution mean (RECOMMENDED — noise-unbiased, SPRT this):")
    for i, name in enumerate(param_names):
        start_val = params_dict[name]["value"]
        v = int(round(es.mean[i])) if params_dict[name]["integer"] else round(float(es.mean[i]), 3)
        print(f"    {name:30s}: {start_val} -> {v}")

    print(f"\n  Results saved to: {args.output}")

    if plotter:
        plotter.finalize()


if __name__ == "__main__":
    main()