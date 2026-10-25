#!/usr/bin/env python3
"""

!!!!!!!!!!!!!!!!!!!!!!!!!
UNTESTED

Not the right call over SPSA for search param values
Better for eval tuning (irrelevant under nnue framework, except HP tuning of trainer)
!!!!!!!!!!!!!!!!!!!!!!!!!

Exercised since that note (after the move to the match runner): the game plumbing
and the CMA-ES loop, with tests/unit/mock_uci_engine.py tuning its Strength option
(tests/unit/test_cmaes.py: a synthetic objective with the match faked, and a real
two-generation tune with fastchess at depth 1, which raised Strength from its start).
Still not exercised: a real tune of Tomahawk's search parameters, i.e. whether
CMA-ES finds better values than SPSA within a sensible number of games.


CMA-ES parameter tuner .

Uses Covariance Matrix Adaptation Evolution Strategy for black-box
optimization of search parameters. Wraps fastchess as the game oracle.

Advantages over SPSA:
  - No learning rate to tune (adapts automatically)
  - Handles parameter correlations (learns covariance structure)
  - Population-based (naturally parallel)
  - Better convergence for <20 parameters

Each generation is one fixed match (utils/fixed_match.py on utils/match_runner.py:
one game per core, pinned to it unless --no-pin; jobs of --pairs-per-job opening
pairs; a crashed or stalled engine is restarted, -recover; every pair counted).
Candidate i ("Candidate<i>", its parameter values as UCI options) plays the
baseline ("Baseline": --baseline, default the same engine, with the start values)
--games-per-eval games as opening pairs, colours reversed. Every candidate of a
generation plays the same openings (the book shuffled with seed + generation:
common random numbers, so the ranking compares parameters, not openings), and the
whole generation is scheduled at once. A candidate's score is its points per game,
(W + D/2) / N; CMA-ES minimizes -score. The CPUs are planned once per tune.

Parameters are UCI options the engine must declare, with [min, max] inside its
bounds (checked before the run, as are the baseline's start values and --nodes;
fastchess would skip anything else with only a warning). Each one starts at
`value` with standard deviation --sigma0 x its own sigma0 and stays in [min, max];
integer parameters are sent rounded. Parameters: DEFAULT_PARAMS or --params-file.

Results: the checkpoint (every 5 generations, at the end, and when the tune stops on an
error or Ctrl-C, with its text in `stopped`; `cma_stop` holds pycma's termination criteria
met, {"maxiter": --generations} for a tune that ran its course)
in --output and <results-dir>/<run>/cmaes_results.json; each generation's match in
<results-dir>/<run>/genNNN/ (config.json, games.jsonl, games.pgn, jobs/,
summary.json). The distribution mean (mean_values) is the recommended result: it is
not biased by lucky samples as the best-ever candidate is. Confirm it with an SPRT.

Requires: pip install cma

Usage:
    python -m tuning.cmaes_tuner --engine tomahawk.exe --tc 10+0.1 --generations 50
    python -m tuning.cmaes_tuner --engine <path> --baseline <path> --depth 8 --plot
    python -m tuning.cmaes_tuner --engine <path> --nodes 20000 --params-file params.json --games-per-eval 32
"""
import os
import sys
import argparse
import math
import json
import random
import time
import copy
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.fastchess_match import FASTCHESS  # noqa: E402
from utils.fixed_match import (MatchStopped, Pairing, require_node_limit, require_options,  # noqa: E402
                               run_fixed_match)
from utils.match_runner import (EngineSpec, find_fastchess, plan_slots, print_line, release_slots,  # noqa: E402
                                tc_arguments)
from utils.openings import OpeningBook  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "logs" / "cmaes_results"
DEFAULT_BOOK = PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn"

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
# Parameters and engines
# ============================================================================

def load_params(names=None, path=None):
    """The parameters to tune: DEFAULT_PARAMS or a JSON file of the same form, optionally only `names`."""
    params = copy.deepcopy(DEFAULT_PARAMS)
    if path:
        params = json.loads(Path(path).read_text(encoding="utf-8"))
    if names:
        unknown = [n for n in names if n not in params]
        if unknown:
            raise ValueError(f"unknown parameter(s): {unknown}; known: {list(params)}")
        params = {n: params[n] for n in names}
    if not params:
        raise ValueError("no parameters to tune")
    for name, cfg in params.items():
        missing = [key for key in ("value", "min", "max", "sigma0") if key not in cfg]
        if missing:
            raise ValueError(f"parameter {name} needs {missing}")
        cfg.setdefault("integer", False)
        if not cfg["min"] <= cfg["value"] <= cfg["max"] or cfg["min"] == cfg["max"] or cfg["sigma0"] <= 0:
            raise ValueError(f"parameter {name}: need min <= value <= max, min < max and sigma0 > 0")
    return params


def option_values(params_dict, param_names, values):
    """{name: value sent to the engine} for a parameter vector: integers rounded, all clamped to [min, max]."""
    out = {}
    for name, val in zip(param_names, values):
        cfg = params_dict[name]
        v = int(round(float(val))) if cfg["integer"] else float(val)
        out[name] = max(cfg["min"], min(cfg["max"], v))
    return out


def display_values(params_dict, param_names, theta):
    """{name: value} as reported: integers rounded, floats to 3 decimals."""
    return {name: int(round(float(theta[i]))) if params_dict[name]["integer"] else round(float(theta[i]), 3)
            for i, name in enumerate(param_names)}


def candidate_engine(engine, name, values):
    """`engine` playing as `name` with `values` set on top of its own options."""
    return EngineSpec(name, engine.cmd, engine.args, {**engine.options, **values},
                      workdir=engine.workdir, pin_option=engine.pin_option)


def check_options(engine, baseline, params_dict, start_values):
    """Refuse values fastchess would skip with only a warning (the games would then be played with the
    engine's default): every tuned parameter must be an option the engine declares, with [min, max]
    inside its bounds, and the baseline must take the start values."""
    require_options(engine, {name: (cfg["min"], cfg["max"]) for name, cfg in params_dict.items()})
    try:
        require_options(baseline, start_values)
    except ValueError as e:
        raise ValueError(f"the baseline plays the start values of the tuned parameters: {e}") from None


# ============================================================================
# A generation: one fixed match, a pairing per candidate
# ============================================================================

def generation_pairings(engine, baseline, params_dict, param_names, candidates, pairs, book):
    """One Pairing per candidate: "Candidate<i>" (its values, White first) vs `baseline`, `pairs` opening pairs.

    Each pairing gets its own copy of `book` (the same shuffle, its own cursor), so every
    candidate plays the same openings: common random numbers for the generation's ranking."""
    pairings = []
    for i, x in enumerate(candidates, 1):
        values = option_values(params_dict, param_names, x)
        first = candidate_engine(engine, f"Candidate{i}", values)
        pairings.append(Pairing(first, baseline, pairs, key=first.name, book=copy.copy(book),
                                meta=dict(values=values)))
    return pairings


def candidate_score(snapshot):
    """A pairing's points per game for its candidate, (W + D/2) / N; None if no pair was counted."""
    w, d, l = snapshot["candidate_wins"], snapshot["candidate_draws"], snapshot["candidate_losses"]
    n = w + d + l
    return (w + d / 2.0) / n if n else None


def evaluate_generation(pairings, gen, *, tc_args, slots, fastchess, results_dir, plies, extra_args=(),
                        pairs_per_job=4, quiet=False, meta=None, show_setup=True):
    """Play a generation's pairings as one fixed match on `slots`; returns (scores in pairing order, result).

    show_setup=False leaves the machine, slot and fastchess lines out of the match header (they
    are the same for every generation of a tune). Raises fixed_match.MatchStopped /
    MatchInterrupted (with .result) if the match could not be completed."""
    result = run_fixed_match(pairings, tc_args=tc_args, results_dir=results_dir, run_name=f"gen{gen:03d}",
                             plies=plies, slots=slots, fastchess=fastchess, extra_args=extra_args,
                             pairs_per_job=pairs_per_job, quiet=quiet, label=f"CMA-ES gen {gen}", meta=meta,
                             show_setup=show_setup)
    scores = []
    for p in pairings:
        score = candidate_score(result["pairings"][p.key])
        if score is None:
            print_line(f"    [WARN] {p.key}: no games completed; scored 0.5")
            score = 0.5
        scores.append(score)
    return scores, result


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

        cmap = plt.get_cmap('tab10')  # matplotlib.cm.get_cmap is gone (3.9+)
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

def build_parser():
    p = argparse.ArgumentParser(description="CMA-ES parameter tuner for Tomahawk")

    p.add_argument("--engine", required=True, help="Path to candidate engine")
    p.add_argument("--baseline", default=None,
                   help="Path to baseline engine (default: same as --engine, "
                        "running the current parameter values)")
    p.add_argument("--fastchess", default=str(FASTCHESS),
                   help="Path to the fastchess executable (default: $FASTCHESS, libs/fastchess, or fastchess on PATH)")

    # Time control (exactly one)
    p.add_argument("--tc", type=str, default=None, help="Time control (e.g. 10+0.1)")
    p.add_argument("--time", type=float, default=None, help="Seconds per move")
    p.add_argument("--depth", type=int, default=None, help="Depth per move")
    p.add_argument("--nodes", type=int, default=None, help="Fixed nodes per move (the engine must honour go nodes)")

    # CMA-ES settings
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--popsize", type=int, default=None,
                   help="Population size (default: 4 + 3*ln(n_params))")
    p.add_argument("--games-per-eval", type=int, default=16,
                   help="Games per candidate and generation, played as opening pairs (odd: one more)")
    p.add_argument("--sigma0", type=float, default=1.0,
                   help="Initial global step size: each parameter starts with standard deviation "
                        "sigma0 x its own sigma0 (default 1)")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed of the CMA-ES samples and the openings (generation g plays the book shuffled "
                        "with seed + g), 1 to 2**32-1; default random, recorded in the results")
    p.add_argument("--concurrency", type=int, default=None,
                   help="Concurrent games (default: every physical core but one); each game gets its own core")
    p.add_argument("--no-pin", action="store_true", help="Leave the placement of games to the OS")
    p.add_argument("--pairs-per-job", type=int, default=4, help="Opening pairs per fastchess process")
    p.add_argument("--maxmoves", type=int, default=100,
                   help="Adjudicate a draw after this many moves (fastchess: 'Draw by adjudication')")

    # Opening book
    p.add_argument("--book", default=str(DEFAULT_BOOK), help="Opening book (.pgn or .epd)")
    p.add_argument("--book-depth", type=int, default=16, help="Plies used from each book line")

    # Params
    p.add_argument("--params", nargs="+", default=None, help="Parameters to tune (default: all)")
    p.add_argument("--params-file", default=None, help="JSON with parameter definitions (see DEFAULT_PARAMS)")

    # Output
    p.add_argument("--output", default="logs/dev_logs/cmaes_results.json",
                   help="Checkpoint / result JSON (also written to the run folder)")
    p.add_argument("--results-dir", default=str(RESULTS_DIR),
                   help="Where run folders go (default logs/cmaes_results): a match folder per generation")
    p.add_argument("--quiet", action="store_true", help="No line per game")
    p.add_argument("--plot", action="store_true")

    return p


def parse_args(argv=None):
    """The tuner's arguments from argv (default sys.argv)."""
    return build_parser().parse_args(argv)


def main(argv=None):
    """Tune; returns the final checkpoint (mean_values: the recommended values). argv: default sys.argv."""
    args = parse_args(argv)
    params_dict = load_params(args.params, args.params_file)
    param_names = list(params_dict)
    if args.generations < 1 or args.games_per_eval < 1 or args.pairs_per_job < 1:
        raise ValueError("--generations, --games-per-eval and --pairs-per-job must be positive")
    if args.seed is not None and not 1 <= args.seed <= 2 ** 32 - 1:
        # pycma seeds numpy with it, and takes 0 for "from the clock": the tune could not be repeated
        raise ValueError(f"--seed must be between 1 and {2 ** 32 - 1}, not {args.seed}")

    # Everything that can be refused is checked before cpus are claimed or folders created
    tc_args = tc_arguments(depth=args.depth, movetime=args.time, tc=args.tc, nodes=args.nodes)
    fastchess = find_fastchess(args.fastchess)
    # Baseline plays the current (start) parameter values so it represents the
    # engine as currently configured. Candidates are scored against this.
    start_values = option_values(params_dict, param_names, [params_dict[n]["value"] for n in param_names])
    engine = EngineSpec.from_path("Engine", args.engine).resolve()
    if args.baseline:
        baseline = EngineSpec.from_path("Baseline", args.baseline, start_values).resolve()
    else:
        baseline = candidate_engine(engine, "Baseline", start_values)
    check_options(engine, baseline, params_dict, start_values)
    if args.nodes is not None:
        require_node_limit([engine, baseline])
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 2 ** 31)
    book = OpeningBook(args.book, plies=args.book_depth, seed=seed + 1)  # generation 1's openings

    # Each game its own cpu (core), for the whole tune
    slots = plan_slots(concurrency=args.concurrency, pin=not args.no_pin)
    try:
        return _tune(args, params_dict, engine, baseline, start_values, seed, book, slots, fastchess, tc_args)
    finally:
        release_slots(slots)


def _tune(args, params_dict, engine, baseline, start_values, seed, book, slots, fastchess, tc_args):
    param_names = list(params_dict)
    n_params = len(param_names)

    # Initial point and bounds
    x0 = [float(params_dict[name]["value"]) for name in param_names]
    lower_bounds = [float(params_dict[name]["min"]) for name in param_names]
    upper_bounds = [float(params_dict[name]["max"]) for name in param_names]

    # Initial sigma: cma multiplies it by CMA_stds, so each parameter starts with sigma0 x its own sigma0
    sigma0 = args.sigma0

    # Population size
    popsize = args.popsize or int(4 + 3 * math.log(n_params))
    pairs = math.ceil(args.games_per_eval / 2)

    # CMA-ES options
    cma_opts = {
        'popsize': popsize,
        'bounds': [lower_bounds, upper_bounds],
        'maxiter': args.generations,
        'seed': seed,
        'verbose': -1,
        'CMA_stds': [params_dict[name]["sigma0"] for name in param_names],
        # The fitness is a noisy game score in steps of 1/games: a generation whose candidates
        # tie (e.g. all draws) is not convergence, so no function-value tolerance (as pycma
        # does for integer problems) and no limit on generations of flat fitness (pycma's
        # default stops after two such generations, e.g. on the first two of a tune at a short TC).
        'tolfun': 0,
        'tolfunhist': 0,
        'tolflatfitness': math.inf,
    }
    if n_params == 1:
        # pycma supports 1-D poorly: its step-size cap (a third of the bounds' range) fails
        # there (IndexError in _stds_into_limits), so it is lifted
        cma_opts['maxstd'] = math.inf

    engine_path = os.path.abspath(args.engine)
    baseline_path = os.path.abspath(args.baseline) if args.baseline else engine_path
    run_root = Path(args.results_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    extra_args = ["-maxmoves", str(args.maxmoves)]

    print_line(f"[CMA-ES] Tuning {n_params} parameters: {param_names}")
    print_line(f"[CMA-ES] Population: {popsize}, Generations: {args.generations}")
    print_line(f"[CMA-ES] Games/eval: {2 * pairs} ({pairs} opening pairs; the candidates of a generation "
               "play the same openings)")
    print_line(f"[CMA-ES] Budget: ~{popsize * args.generations * 2 * pairs} games")
    print_line(f"[CMA-ES] σ0: {sigma0:.4f} (x each parameter's sigma0)")
    print_line(f"[CMA-ES] x0: {dict(zip(param_names, x0))}")
    if args.baseline:
        print_line(f"[CMA-ES] Baseline: external engine {baseline_path} at the start values {start_values}")
    else:
        print_line(f"[CMA-ES] Baseline: same engine at current values {start_values}")
    print_line(f"[CMA-ES] {len(slots)} concurrent games, seed {seed}, results {run_root}")

    # Plotter
    plotter = None
    if args.plot:
        plotter = CMAPlotter(param_names, params_dict)

    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)

    print_line("\n[CMA-ES] Starting optimization...\n")

    total_games = 0
    start_time = time.time()
    best_ever_score = 0.0
    best_ever_theta = np.array(x0)
    history = []

    def cma_stop():
        """pycma's termination criteria that are met (e.g. {"maxiter": 100}), JSON-ready."""
        return {str(k): (v.item() if hasattr(v, "item") else v) for k, v in es.stop().items()}

    def checkpoint(stopped=None):
        # Distribution mean (xfavorite: es.mean mapped into the bounds) — the recommended
        # estimate of the optimum. Unlike best-ever, it is not biased by evaluation noise.
        data = {
            "generation": len(history),
            "best_ever_score": round(best_ever_score, 4),
            "best_ever_theta": best_ever_theta.tolist(),
            "best_values": display_values(params_dict, param_names, best_ever_theta),
            "mean_values": display_values(params_dict, param_names, es.result.xfavorite),  # RECOMMENDED
            "cma_mean": es.mean.tolist(),
            "start_values": {name: params_dict[name]["value"] for name in param_names},
            "sigma": float(es.sigma),
            "params_config": params_dict,
            "params": param_names,
            "total_games": total_games,
            "elapsed": time.time() - start_time,
            "history": history,
            "seed": seed,
            "run_dir": str(run_root),
            "stopped": stopped,
            "cma_stop": cma_stop(),
            "args": {
                "engine": engine_path,
                "baseline": baseline_path,
                "baseline_is_self": args.baseline is None,
                "games_per_eval": 2 * pairs,
                "generations": args.generations,
                "popsize": popsize,
                "sigma0": sigma0,
                "tc_args": tc_args,
                "concurrency": len(slots),
                "pin": not args.no_pin,
                "book": str(args.book),
                "book_depth": args.book_depth,
                "maxmoves": args.maxmoves,
                "fastchess": fastchess,
            },
        }
        for path in (Path(args.output), run_root / "cmaes_results.json"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data

    gen = 0
    try:
        while not es.stop():
            gen += 1
            candidates = es.ask()
            if gen > 1:
                book = OpeningBook(args.book, plies=args.book_depth, seed=seed + gen)

            # Evaluate the whole generation in one match: every candidate against the baseline
            pairings = generation_pairings(engine, baseline, params_dict, param_names, candidates, pairs, book)
            scores, result = evaluate_generation(
                pairings, gen, tc_args=tc_args, slots=slots, fastchess=fastchess, results_dir=run_root,
                plies=args.book_depth, extra_args=extra_args, pairs_per_job=args.pairs_per_job, quiet=args.quiet,
                meta=dict(arguments=vars(args), tuner="cmaes", generation=gen, seed=seed, openings_seed=seed + gen,
                          sigma=float(es.sigma), mean=dict(zip(param_names, map(float, es.result.xfavorite)))),
                show_setup=gen == 1)  # the machine, slots and fastchess are the same for every generation
            total_games += result["games"]

            # CMA-ES minimizes → negate scores
            fitnesses = [-s for s in scores]
            es.tell(candidates, fitnesses)

            # Stats
            best_idx = int(np.argmax(scores))
            gen_best_score = scores[best_idx]
            gen_best_theta = candidates[best_idx]
            gen_mean_score = float(np.mean(scores))
            sigma = es.sigma
            mean_theta = es.result.xfavorite

            if gen_best_score > best_ever_score:
                best_ever_score = gen_best_score
                best_ever_theta = np.array(gen_best_theta)

            current_best = display_values(params_dict, param_names, best_ever_theta)

            elapsed = time.time() - start_time
            print_line(
                f"  Gen {gen:3d}/{args.generations} | "
                f"Best={gen_best_score:.3f} Mean={gen_mean_score:.3f} | "
                f"σ={sigma:.4f} | "
                f"Games: {total_games} | "
                f"Time: {elapsed:.0f}s"
            )
            print_line(f"       Best ever: {current_best} (score={best_ever_score:.3f})")
            print_line(f"       Mean:      {display_values(params_dict, param_names, mean_theta)}")

            # History
            history.append({
                "generation": gen,
                "best_score": round(gen_best_score, 4),
                "mean_score": round(gen_mean_score, 4),
                "sigma": round(float(sigma), 6),
                "best_theta": {name: float(best_ever_theta[i]) for i, name in enumerate(param_names)},
                "mean_theta": {name: float(mean_theta[i]) for i, name in enumerate(param_names)},
                "games": total_games,
                "scores": [round(s, 4) for s in scores],
                "candidates": [p.meta["values"] for p in pairings],
                "run_dir": result["run_dir"],
            })

            # Plot
            if plotter:
                plotter.update(gen, gen_best_score, gen_mean_score, sigma,
                               mean_theta, best_ever_theta, total_games)

            # Checkpoint
            if gen % 5 == 0:
                checkpoint()
    except BaseException as e:
        # A generation that could not be completed (a stopped match, Ctrl-C, or any error: fastchess
        # failing to start, a game that did not reverse colours, a full disk, ...) is not told to
        # CMA-ES: the checkpoint keeps the complete generations (and counts the games played).
        result = getattr(e, "result", None)
        total_games += result.get("games", 0) if isinstance(result, dict) else 0
        interrupted = isinstance(e, KeyboardInterrupt)
        reason = ("interrupted" if interrupted else str(e) if isinstance(e, MatchStopped)
                  else f"{type(e).__name__}: {e}")
        try:
            checkpoint(stopped=reason)
            saved = f"{len(history)} complete generation(s) saved to {args.output}"
        except Exception as save_error:  # the original error is the one raised
            saved = f"the checkpoint could not be saved ({type(save_error).__name__}: {save_error})"
        print_line(f"\n[CMA-ES] {'interrupted' if interrupted else 'stopped'} in generation {gen}"
                   + ("" if interrupted else f" ({reason})") + f"; {saved}")
        raise

    final = checkpoint()
    reasons = final["cma_stop"]
    complete = gen >= args.generations

    # Final
    print_line("")
    print_line("=" * 60)
    if complete:
        print_line("[CMA-ES] OPTIMIZATION COMPLETE")
    else:  # one of pycma's own termination criteria (tolx, tolstagnation, ...) ended the tune
        print_line(f"[CMA-ES] STOPPED EARLY after {gen} of {args.generations} generations (pycma: "
                   + ", ".join(f"{k}={v}" for k, v in reasons.items()) + ")")
    print_line("=" * 60)
    print_line(f"  Generations: {gen}")
    print_line(f"  Total games: {total_games}")
    print_line(f"  Total time:  {time.time() - start_time:.0f}s")
    print_line(f"  Stop reason: {reasons}")
    print_line(f"  Best score:  {best_ever_score:.3f}")

    print_line("\n  Best-ever candidate (noisy — may be lucky sample):")
    for name, v in final["best_values"].items():
        print_line(f"    {name:30s}: {params_dict[name]['value']} -> {v}")

    print_line("\n  Distribution mean (RECOMMENDED — noise-unbiased, SPRT this):")
    for name, v in final["mean_values"].items():
        print_line(f"    {name:30s}: {params_dict[name]['value']} -> {v}")

    print_line(f"\n  Results saved to: {args.output}")
    print_line(f"  Games:            {run_root}")

    if plotter:
        plotter.finalize()
    return final


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"[CMA-ES] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
