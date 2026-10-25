"""
SPSA tuner for engine UCI parameters (Fishtest / OpenBench conventions).

Every iteration is one opening pair: the engine with theta + c*delta ("Plus")
plays the engine with theta - c*delta ("Minus") on one opening, once with each
colour, where delta is a random +-1 vector. The pair's result R = (Plus points -
Minus points) in {-2..2} moves every parameter along its own perturbation:

    theta_i += a_k,i * R * delta_i / c_k,i

Schedules, with N = total pairs, k = pair number (1-based), A = 0.1 N,
alpha = 0.602, gamma = 0.101 (Spall; as Fishtest and OpenBench):

    c_k = c_end * (N / k)^gamma                 perturbation, reaches c_end at the end
    a_k = r_end * c_end^2 * ((A + N) / (A + k))^alpha   step, so a_N / c_N^2 = r_end

c_end is the perturbation at the END of the tune and r_end the learning rate
there (Fishtest's defaults: r_end = 0.002). Pick c_end so that +-c_end changes
the engine by a few Elo; ~range/20 is a common start.

Games are played with fastchess (utils/match_runner): pairs run asynchronously
on every core, each game pinned to one core, so no batch waits for its slowest
game. fastchess sets an engine's options once per process, so every job (one
fastchess process of --pairs-per-job opening pairs) plays one perturbation:
Plus and Minus get their values as option.NAME=VALUE and the job's pairs update
theta together.

  * integer parameters use stochastic rounding (E[sent value] = theta +- c), so
    small perturbations are unbiased instead of rounding to the same integer;
  * per-parameter signal: t_i = sum(R delta_i) / sqrt(sum R^2) is ~N(0,1) when a
    parameter has no effect, so |t| > 2 shows which parameters are being tuned
    and which are noise;
  * iterate (Polyak) averaging of the second half, checkpoint/resume after
    every job, and an optional verification match (tuned vs start, pentanomial
    Elo with 95% CI). A tune is only a proposal until such a match (or an SPRT
    with tests/sprt.py --param ...) confirms it.

Per-move limit: exactly one of --tc, --time, --depth or --nodes (--nodes needs an
engine that honours `go nodes`; checked before the run). Parameters must be UCI
options the engine declares, within its bounds (fastchess skips anything else).

Results go to <results-dir>/<run>/ (default logs/spsa_results): state.json (resume
with --resume and the same engine / limit options), history.jsonl (one line per
job), config.json, result.json, games.pgn and the jobs' fastchess logs.

Usage:
    python -m tuning.spsa_tuner --engine tomahawk.exe --tc 10+0.1 --pairs 8000
    python -m tuning.spsa_tuner --engine tomahawk.exe --nodes 20000 --params r_nmp aspiration_window --verify-pairs 2000
    python -m tuning.spsa_tuner --engine tomahawk.exe --tc 10+0.1 --resume logs/spsa_results/<run>/state.json
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

TUNING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TUNING_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import uci_probe  # noqa: E402
from utils.fastchess_match import limit_label  # noqa: E402
from utils.match_runner import (EngineSpec, JobSpec, MatchRunner, describe_slots, find_fastchess,  # noqa: E402
                                plan_slots, print_line, probe_uci, release_slots, tc_arguments)
from utils.openings import OpeningBook  # noqa: E402
from utils.pentanomial import PairTally  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "logs" / "spsa_results"

# ============================================================================
# Parameters
# ============================================================================
# name -> value (start), min, max, c_end (perturbation at the end of the tune),
# integer (sent rounded), optional r_end (learning rate at the end; default --r-end).
DEFAULT_PARAMS = {
    "r_nmp":                    {"value": 3,    "min": 2,    "max": 5,    "c_end": 0.5, "integer": True},
    "r_lmr_const":              {"value": 99,   "min": 30,   "max": 200,  "c_end": 8,   "integer": True},  # x100
    "r_lmr_denom":              {"value": 314,  "min": 100,  "max": 600,  "c_end": 25,  "integer": True},  # x100
    "lmr_depth_threshold":      {"value": 3,    "min": 1,    "max": 8,    "c_end": 0.5, "integer": True},
    "lmr_move_order_threshold": {"value": 3,    "min": 1,    "max": 8,    "c_end": 0.5, "integer": True},
    "aspiration_window":        {"value": 50,   "min": 10,   "max": 200,  "c_end": 10,  "integer": True},
    "aspiration_start_depth":   {"value": 6,    "min": 3,    "max": 12,   "c_end": 0.5, "integer": True},
    "delta_prune_threshold":    {"value": 1000, "min": 200,  "max": 2000, "c_end": 90,  "integer": True},
    "see_prune_threshold":      {"value": -50,  "min": -500, "max": 200,  "c_end": 35,  "integer": True},
}

ALPHA, GAMMA = 0.602, 0.101
ADAM_LR = 0.15  # experimental Adam variant: peak step ~ ADAM_LR * c_k per update


def load_params(names=None, path=None):
    params = copy.deepcopy(DEFAULT_PARAMS)
    if path:
        params = json.loads(Path(path).read_text(encoding="utf-8"))
    if names:
        unknown = [n for n in names if n not in params]
        if unknown:
            raise SystemExit(f"unknown parameter(s): {unknown}; known: {list(params)}")
        params = {n: params[n] for n in names}
    for name, cfg in params.items():
        for key in ("value", "min", "max", "c_end"):
            if key not in cfg:
                raise SystemExit(f"parameter {name} needs '{key}'")
        cfg.setdefault("integer", False)
        if not cfg["min"] <= cfg["value"] <= cfg["max"] or cfg["c_end"] <= 0:
            raise SystemExit(f"parameter {name}: need min <= value <= max and c_end > 0")
    return params


class Schedule:
    """Fishtest/OpenBench gain sequences for N pairs."""

    def __init__(self, pairs, A=None, alpha=ALPHA, gamma=GAMMA, r_end=0.002):
        self.N = pairs
        self.A = 0.1 * pairs if A is None else A
        self.alpha, self.gamma, self.r_end = alpha, gamma, r_end

    def c(self, cfg, k):
        return cfg["c_end"] * (self.N / k) ** self.gamma

    def a(self, cfg, k):
        r_end = cfg.get("r_end", self.r_end)
        return r_end * cfg["c_end"] ** 2 * ((self.A + self.N) / (self.A + k)) ** self.alpha


def stochastic_round(x, rng):
    """floor(x) or ceil(x) with probabilities that make the expectation x."""
    low = math.floor(x)
    return int(low + (rng.random() < x - low))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# ============================================================================
# Tuning state (checkpointed after every job)
# ============================================================================

class SpsaState:
    def __init__(self, params, pairs, seed):
        self.params = params
        self.names = list(params)
        self.pairs = pairs
        self.theta = {n: float(cfg["value"]) for n, cfg in params.items()}
        self.planned = 0          # pairs handed out (k of the next job = planned + 1)
        self.done = 0             # pairs whose results were applied
        self.signal = {n: 0.0 for n in self.names}   # sum R * delta_i
        self.r_squared = 0.0                          # sum R^2
        self.polyak_sum = {n: 0.0 for n in self.names}
        self.polyak_weight = 0.0
        self.adam = {"m": {n: 0.0 for n in self.names}, "v": {n: 0.0 for n in self.names}, "t": 0}
        self.tally = PairTally()  # pairs from Plus's point of view
        self.seed = seed
        self.rng_state = None
        self.elapsed = 0.0

    def to_json(self, rng):
        data = {k: v for k, v in vars(self).items() if k not in ("tally",)}
        data["tally"] = dict(counts=self.tally.counts, detail=self.tally.detail, wdl=self.tally.wdl)
        data["rng_state"] = list(rng.getstate()[1]) + [rng.getstate()[2]]
        data["rng_version"] = rng.getstate()[0]
        return data

    @classmethod
    def from_json(cls, data):
        state = cls(data["params"], data["pairs"], data.get("seed"))
        for key, value in data.items():
            if key not in ("tally", "rng_state", "rng_version") and hasattr(state, key):
                setattr(state, key, value)
        tally = data.get("tally") or {}
        state.tally.counts = tally.get("counts", state.tally.counts)
        state.tally.detail = tally.get("detail", state.tally.detail)
        state.tally.wdl = tally.get("wdl", state.tally.wdl)
        if data.get("rng_state"):
            state.rng_state = (data.get("rng_version", 3), tuple(data["rng_state"][:-1]), data["rng_state"][-1])
        return state

    def polyak(self):
        if not self.polyak_weight:
            return None
        return {n: self.polyak_sum[n] / self.polyak_weight for n in self.names}

    def t_values(self):
        if not self.r_squared:
            return {n: 0.0 for n in self.names}
        root = math.sqrt(self.r_squared)
        return {n: self.signal[n] / root for n in self.names}


def engine_value(cfg, x, rng):
    x = clamp(x, cfg["min"], cfg["max"])
    return clamp(stochastic_round(x, rng), math.ceil(cfg["min"]), math.floor(cfg["max"])) if cfg["integer"] \
        else round(x, 6)


def rounded(params, theta):
    return {n: (int(round(theta[n])) if params[n]["integer"] else round(theta[n], 4)) for n in params}


def check_options(engine, params):
    """Refuse parameters fastchess would not send: options the engine does not declare
    (names are case-sensitive) and ranges beyond the engine's own bounds (fastchess
    skips such a value with a warning and the game is played with the default)."""
    declared = probe_uci(engine.cmd, engine.args)["options"]
    problems = uci_probe.option_problems(declared, {name: (cfg["min"], cfg["max"]) for name, cfg in params.items()})
    if problems:
        raise ValueError(f"{engine.cmd} cannot be tuned on: " + "; ".join(problems))


def require_node_limit(engine):
    """Refuse an engine that ignores `go nodes` (uci_probe); a script run by an interpreter is not checked."""
    if engine.args:
        print_line(f"[SPSA] warning: could not check that {engine.args[-1]} honours go nodes")
    else:
        uci_probe.require_node_limit([engine.cmd])


# ============================================================================
# Controller
# ============================================================================

class SpsaController:
    def __init__(self, args, state, schedule, engine, book, slots, run_dir, plotter=None):
        self.args, self.state, self.schedule = args, state, schedule
        self.engine, self.book, self.slots = engine, book, slots
        self.run_dir = Path(run_dir)
        self.plotter = plotter
        self.rng = random.Random(state.seed)
        if state.rng_state:
            self.rng.setstate(state.rng_state)
        self.results = {}   # job id -> {pair_no: [plus points game 1, game 2]}
        self.started = time.time() - state.elapsed
        self.last_status = 0.0
        self.polyak_from = int(args.polyak_from * state.pairs)
        self.failures = 0   # jobs in a row that applied no pair
        self.stop_reason = None
        self.history = open(self.run_dir / "history.jsonl", "a", encoding="utf-8")

    def next_job(self, slot):
        state = self.state
        if state.planned >= state.pairs or self.stop_reason:
            return None
        size = min(self.args.pairs_per_job, state.pairs - state.planned)
        k = state.planned + 1
        state.planned += size
        delta = {n: self.rng.choice((-1, 1)) for n in state.names}
        c = {n: self.schedule.c(state.params[n], k) for n in state.names}
        a = {n: self.schedule.a(state.params[n], k) for n in state.names}
        plus = {n: engine_value(state.params[n], state.theta[n] + c[n] * delta[n], self.rng) for n in state.names}
        minus = {n: engine_value(state.params[n], state.theta[n] - c[n] * delta[n], self.rng) for n in state.names}
        # one fastchess process sets each engine's options once: the whole job plays this perturbation
        engines = (self.variant("Plus", plus), self.variant("Minus", minus))
        return JobSpec(engines, self.book.take(size), self.book,
                       meta=dict(k=k, delta=delta, c=c, a=a, plus=plus, minus=minus))

    def variant(self, name, values):
        return EngineSpec(name, self.engine.cmd, self.engine.args, {**self.engine.options, **values},
                          workdir=self.engine.workdir, pin_option=self.engine.pin_option)

    def on_pair_start(self, job, pair_no):
        return True

    def on_game(self, game):
        if game.env.get("core_mixed"):
            self.results.setdefault(game.job.id, {}).setdefault(game.pair_no, []).append(None)
            return
        score = game.score_for("Plus")
        if score is not None:
            self.results.setdefault(game.job.id, {}).setdefault(game.pair_no, []).append(score)

    def on_job_end(self, job):
        state, meta = self.state, job.spec.meta
        pairs = [scores for scores in self.results.pop(job.id, {}).values()
                 if len(scores) == 2 and None not in scores]
        # pairs the job did not complete (stopped, crashed, unfinished or excluded games) are played again
        state.planned -= len(job.spec.openings) - len(pairs)
        if not pairs:
            if not job.killed:
                self.failures += 1
                if job.returncode or not job.games:
                    tail = job.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
                    print_line(f"[SPSA] job {job.id} failed (code {job.returncode}):\n    " + "\n    ".join(tail))
                if self.failures >= max(3, len(self.slots)) and not self.stop_reason:
                    self.stop_reason = f"{self.failures} jobs in a row completed no pair (see the job logs)"
                    print_line(f"[SPSA] stopping: {self.stop_reason}")
            return
        self.failures = 0
        result = int(round(sum(2 * (s1 + s2) - 2 for s1, s2 in pairs)))   # Plus points - Minus points
        for s1, s2 in pairs:
            state.tally.add_pair(s1, s2)
        old = dict(state.theta)
        if self.args.optimizer == "adam":
            adam = state.adam
            adam["t"] += 1
            decay = ((self.schedule.A + 1) / (self.schedule.A + meta["k"])) ** self.schedule.alpha
        for n in state.names:
            cfg, delta, c = state.params[n], meta["delta"][n], meta["c"][n]
            gradient = result * delta / c
            if self.args.optimizer == "adam":
                adam["m"][n] = 0.9 * adam["m"][n] + 0.1 * gradient
                adam["v"][n] = 0.999 * adam["v"][n] + 0.001 * gradient ** 2
                m_hat = adam["m"][n] / (1 - 0.9 ** adam["t"])
                v_hat = adam["v"][n] / (1 - 0.999 ** adam["t"])
                step = ADAM_LR * c * decay * m_hat / (math.sqrt(v_hat) + 1e-8)
            else:
                step = meta["a"][n] * gradient
            state.theta[n] = clamp(state.theta[n] + step, cfg["min"], cfg["max"])
            state.signal[n] += result * delta
        state.r_squared += result ** 2
        state.done += len(pairs)
        if not self.args.no_polyak and state.done > self.polyak_from:
            for n in state.names:
                state.polyak_sum[n] += state.theta[n] * len(pairs)
            state.polyak_weight += len(pairs)
        state.elapsed = time.time() - self.started
        self.history.write(json.dumps(dict(k=meta["k"], pairs=len(pairs), result=result, delta=meta["delta"],
                                           c=meta["c"], plus=meta["plus"], minus=meta["minus"],
                                           theta=state.theta, core=job.slot.core_type)) + "\n")
        self.history.flush()
        self.save(force=True)  # in step with history.jsonl: a resume replays exactly the pairs not in either
        if self.plotter:
            self.plotter.update(state, {n: state.theta[n] - old[n] for n in state.names})
        if not self.args.quiet:
            movement = "  ".join(f"{n}={state.theta[n]:.2f}" for n in state.names[:6])
            print_line(f"  pair {meta['k']:>6}/{state.pairs}  R {result:+d}  {job.slot.core_type}  {movement}"
                       + (" ..." if len(state.names) > 6 else ""))

    def on_warning(self, job, text):
        if "doesn't have option" in text or "Invalid value for option" in text:
            print_line(f"[SPSA] ERROR from fastchess: {text} (the value was not sent to the engine)")
        else:
            print_line(f"[fastchess] {text}")

    def finished(self):
        return self.stop_reason is not None

    def tick(self):
        if self.plotter:
            self.plotter.pump()
        now = time.monotonic()
        if now - self.last_status >= self.args.status_every and self.state.done:
            self.last_status = now
            self.print_status()

    def print_status(self):
        s = self.state
        elapsed = time.time() - self.started
        rate = s.done / elapsed * 3600 if elapsed else 0
        eta = (s.pairs - s.done) / rate if rate else float("inf")
        t = s.t_values()
        strongest = sorted(t.items(), key=lambda kv: -abs(kv[1]))[:3]
        summary = s.tally.summary()
        print_line(f"[SPSA] {s.done}/{s.pairs} pairs ({rate:.0f}/h, ~{eta:.1f} h left) | Plus score "
                   f"{summary['raw_score']:.3f} | strongest signal: " +
                   ", ".join(f"{n} t={v:+.1f}" for n, v in strongest))

    def save(self, force=False):
        """Checkpoint the tuning state (at most every few seconds unless forced)."""
        now = time.monotonic()
        if not force and now - getattr(self, "_last_save", -1e9) < 5:
            return
        self._last_save = now
        path = self.run_dir / "state.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state.to_json(self.rng), indent=1), encoding="utf-8")
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # Windows: a scanner/indexer briefly holds the file
                time.sleep(0.05 * (attempt + 1))
        raise PermissionError(f"could not replace {path}")

    def close(self):
        self.state.elapsed = time.time() - self.started
        self.save(force=True)
        self.history.close()


class VerifyController:
    """Fixed number of pairs: tuned values (Candidate) against the start values (Baseline)."""

    def __init__(self, engines, book, pairs, pairs_per_job):
        self.engines, self.book, self.pairs, self.per_job = engines, book, pairs, pairs_per_job
        self.planned = 0
        self.scores = {}
        self.tally = PairTally()

    def next_job(self, slot):
        if self.planned >= self.pairs:
            return None
        size = min(self.per_job, self.pairs - self.planned)
        self.planned += size
        return JobSpec(self.engines, self.book.take(size), self.book)

    def on_pair_start(self, job, pair_no):
        return True

    def on_game(self, game):
        score = game.score_for("Tuned")
        if score is None or game.env.get("core_mixed"):
            return
        pair = self.scores.setdefault((game.job.id, game.pair_no), [])
        pair.append(score)
        if len(pair) == 2:
            self.tally.add_pair(*pair)

    def on_job_end(self, job):
        pass

    def on_warning(self, job, text):
        print_line(f"[fastchess] {text}")

    def finished(self):
        return False

    def tick(self):
        pass


# ============================================================================
# Live plot
# ============================================================================

class SPSAPlotter:
    """Parameter traces plus Plus's running score and the per-parameter signal."""

    def __init__(self, state):
        import matplotlib.pyplot as plt
        from utils.plotting import PALETTE, style_dashboard
        self.plt, self.style, self.palette = plt, style_dashboard, PALETTE
        self.names = state.names
        self.params = state.params
        self.xs, self.traces = [], {n: [] for n in self.names}
        self.scores = []
        plt.ion()
        rows = max(3, len(self.names))
        self.fig = plt.figure(figsize=(14, max(7, 1.2 * rows)), layout="constrained")
        grid = self.fig.add_gridspec(rows, 2)
        self.ax_params = [self.fig.add_subplot(grid[i, 0]) for i in range(len(self.names))]
        third = max(1, rows // 3)
        self.ax_score = self.fig.add_subplot(grid[:third, 1])
        self.ax_signal = self.fig.add_subplot(grid[third:, 1])
        if self.fig.canvas.required_interactive_framework is not None:
            plt.show(block=False)

    def update(self, state, step):
        self.xs.append(state.done)
        for n in self.names:
            self.traces[n].append(state.theta[n])
        self.scores.append(state.tally.summary()["raw_score"])
        for i, (ax, n) in enumerate(zip(self.ax_params, self.names)):
            ax.clear()
            cfg = self.params[n]
            ax.axhline(cfg["value"], color="#9aa5ab", linestyle=":", linewidth=1)
            ax.plot(self.xs, self.traces[n], color=self.palette[i % len(self.palette)], linewidth=1.4)
            ax.set_title(n, fontsize=9, loc="left")
            ax.tick_params(labelsize=7)
        self.ax_score.clear()
        self.ax_score.axhline(0.5, color="#9aa5ab", linewidth=1)
        self.ax_score.plot(self.xs, self.scores, linewidth=1.4)
        self.ax_score.set(title="Plus score so far (should stay near 50%)", xlabel="pairs")
        self.ax_signal.clear()
        t = state.t_values()
        self.ax_signal.barh(self.names, [t[n] for n in self.names],
                            color=[self.palette[i % len(self.palette)] for i in range(len(self.names))])
        for x in (-2, 2):
            self.ax_signal.axvline(x, color="#c46b61", linestyle="--", linewidth=1)
        self.ax_signal.set(title="Signal t per parameter (|t| > 2: the parameter clearly matters)")
        self.fig.suptitle(f"SPSA  {state.done:,}/{state.pairs:,} pairs")
        self.style(self.fig, self.ax_params + [self.ax_score, self.ax_signal])
        self.fig.canvas.draw_idle()

    def pump(self):
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def finalize(self, path):
        self.fig.savefig(path, dpi=130)
        self.plt.ioff()
        self.plt.show()


# ============================================================================
# Main
# ============================================================================

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True, help="Engine to tune (both sides of every game)")
    p.add_argument("--fastchess", default=None,
                   help="fastchess executable (default: $FASTCHESS, libs/fastchess, or fastchess on PATH)")
    tc = p.add_argument_group("per-move limit (choose one)")
    tc.add_argument("--tc", default=None, help="Time control (e.g. 10+0.1)")
    tc.add_argument("--time", type=float, default=None, help="Seconds per move")
    tc.add_argument("--depth", type=int, default=None, help="Depth per move")
    tc.add_argument("--nodes", type=int, default=None,
                    help="Nodes per move: hardware-independent (the engine must honour go nodes; checked first)")
    s = p.add_argument_group("SPSA")
    s.add_argument("--pairs", type=int, default=None, help="Total opening pairs (iterations); default 4000")
    s.add_argument("--r-end", type=float, default=0.002, help="Learning rate at the end (Fishtest default 0.002)")
    s.add_argument("--A", type=float, default=None, help="Stability constant (default 0.1 * pairs)")
    s.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd",
                   help="sgd: the standard SPSA update (default); adam: experimental")
    s.add_argument("--no-polyak", action="store_true", help="Do not average the second half of the tune")
    s.add_argument("--polyak-from", type=float, default=0.5, help="Fraction of the tune after which to average")
    s.add_argument("--params", nargs="+", default=None, help="Parameters to tune (default: all)")
    s.add_argument("--params-file", default=None, help="JSON with parameter definitions (see DEFAULT_PARAMS)")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--verify-pairs", type=int, default=0,
                   help="After tuning, play this many pairs tuned vs start values (pentanomial Elo)")
    # Accepted for older callers: old iteration = games_per_iter games.
    s.add_argument("--iterations", type=int, default=None, help=argparse.SUPPRESS)
    s.add_argument("--games-per-iter", type=int, default=None, help=argparse.SUPPRESS)
    s.add_argument("--r", type=float, default=None, help=argparse.SUPPRESS)
    # Accepted for older callers and ignored: the old tuner's baseline engine played no part in tuning.
    s.add_argument("--baseline", default=None, help=argparse.SUPPRESS)
    c = p.add_argument_group("run")
    c.add_argument("--concurrency", type=int, default=None,
                   help="Concurrent games (default: every physical core but one); each game gets its own core")
    c.add_argument("--no-pin", action="store_true", help="Leave the placement of games to the OS")
    c.add_argument("--pairs-per-job", type=int, default=2,
                   help="Opening pairs per fastchess process; they share one perturbation")
    c.add_argument("--book", default=PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn",
                   help="Opening book (.pgn or .epd)")
    c.add_argument("--book-depth", type=int, default=16, help="Plies used from each book line")
    c.add_argument("--maxmoves", type=int, default=100,
                   help="Adjudicate a draw after this many moves (fastchess: 'Draw by adjudication')")
    c.add_argument("--results-dir", default=RESULTS_DIR, help="Where run folders go (default logs/spsa_results)")
    c.add_argument("--output", default=None, help="Also write the final result JSON here")
    c.add_argument("--resume", default=None, help="state.json of an earlier run (pass the same engine and limit)")
    c.add_argument("--plot", action="store_true", help="Live plots of the parameters and their signal")
    c.add_argument("--status-every", type=float, default=30, help="Seconds between status lines")
    c.add_argument("--quiet", action="store_true", help="No per-job lines")
    return p


def resolve_args(args=None):
    """Accept argv, None (sys.argv) or a Namespace (missing fields get their defaults)."""
    parser = build_parser()
    if isinstance(args, argparse.Namespace):
        defaults = vars(parser.parse_args(["--engine", "x"]))
        args = argparse.Namespace(**{**defaults, **{k: v for k, v in vars(args).items() if v is not None}})
    else:
        args = parser.parse_args(args)
    if args.pairs is None:
        if args.iterations is not None:
            args.pairs = max(1, int(args.iterations) * int(args.games_per_iter or 2) // 2)
        else:
            args.pairs = 4000
    if args.r is not None:
        args.r_end = args.r
    return args


def parse_args(argv=None):
    """The tuner's arguments from argv (default sys.argv), with every default filled in."""
    return resolve_args(list(argv) if argv is not None else None)


def limit_flag(args):
    """The per-move limit as the flag tests/sprt.py takes."""
    if args.tc is not None:
        return f"--tc {args.tc}"
    if args.time is not None:
        return f"--time {args.time:g}"
    if args.depth is not None:
        return f"--depth {args.depth}"
    return f"--nodes {args.nodes}"


def main(args=None):
    """Tune; returns the recommended values {name: value}. `args`: argv, a Namespace or None (sys.argv)."""
    args = resolve_args(args)
    if args.pairs < 1 or args.pairs_per_job < 1:
        raise ValueError("--pairs and --pairs-per-job must be positive")
    # everything that can be refused is checked before cpus are claimed or folders created
    tc_args = tc_arguments(depth=args.depth, movetime=args.time, tc=args.tc, nodes=args.nodes)
    extra_args = ["-maxmoves", str(args.maxmoves)]
    fastchess = find_fastchess(args.fastchess)

    run_dir = None
    if args.resume:
        data = json.loads(Path(args.resume).read_text(encoding="utf-8"))
        state = SpsaState.from_json(data)
        # pairs that were handed out but never applied (jobs running when the tune stopped) are played again
        state.planned = state.done
        run_dir = Path(args.resume).resolve().parent
    else:
        params = load_params(args.params, args.params_file)
        seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 31)
        state = SpsaState(params, args.pairs, seed)
    schedule = Schedule(state.pairs, A=args.A, r_end=args.r_end)

    engine = EngineSpec.from_path("Engine", args.engine).resolve()
    check_options(engine, state.params)
    if args.nodes is not None:
        require_node_limit(engine)
    book = OpeningBook(args.book, plies=args.book_depth, seed=state.seed)
    book.cursor = state.planned % len(book)

    slots = plan_slots(concurrency=args.concurrency, pin=not args.no_pin)
    try:
        return _run(args, state, schedule, engine, book, slots, fastchess, tc_args, extra_args, run_dir)
    finally:
        release_slots(slots)


def _run(args, state, schedule, engine, book, slots, fastchess, tc_args, extra_args, run_dir):
    resumed = run_dir is not None
    if resumed:
        print_line(f"[SPSA] resuming {run_dir.name} at pair {state.done}/{state.pairs}")
        config_name = f"config_resume_{datetime.now():%Y%m%d_%H%M%S}.json"
    else:
        run_dir = Path(args.results_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True)
        config_name = "config.json"
    limit = limit_label(tc=args.tc, depth=args.depth, time=args.time, nodes=args.nodes)
    pinned = slots[0].cpu is not None
    (run_dir / config_name).write_text(json.dumps(dict(
        arguments=vars(args), params=state.params, schedule=vars(schedule), time_control=limit, fastchess=fastchess,
        slots=[vars(s) | {"label": s.label} for s in slots],
        pinning=("Affinity option + process mask" if engine.pin_option else "process mask") if pinned else "none"),
        indent=2, default=str), encoding="utf-8")
    print_line(f"[SPSA] tuning {len(state.names)} parameter(s) over {state.pairs:,} pairs "
               f"({2 * state.pairs:,} games), r_end {schedule.r_end}, A {schedule.A:g}, optimizer {args.optimizer}")
    for n in state.names:
        cfg = state.params[n]
        print_line(f"  {n:28s} start {cfg['value']:>8}  [{cfg['min']}, {cfg['max']}]  "
                   f"c {schedule.c(cfg, 1):.2f} -> {cfg['c_end']}  step/pair at end ~{schedule.a(cfg, state.pairs) / cfg['c_end']:.4f}*R")
    print_line(f"[SPSA] engine {args.engine}, {limit}, {args.pairs_per_job} pair(s) per job")
    print_line(f"[SPSA] {describe_slots(slots)}")
    if pinned and not engine.pin_option:
        print_line("[SPSA] note: the engine has no Affinity option; its process mask keeps it on the core")
    print_line(f"[SPSA] fastchess {fastchess}")
    print_line(f"[SPSA] results {run_dir}")

    plotter = SPSAPlotter(state) if args.plot else None
    controller = SpsaController(args, state, schedule, engine, book, slots, run_dir, plotter)
    runner = MatchRunner(fastchess, slots, run_dir, tc_args, plies=args.book_depth, extra_args=extra_args)
    interrupted = False
    try:
        runner.run(controller)
    except KeyboardInterrupt:
        interrupted = True
        print_line("\n[SPSA] interrupted; state saved, resume with --resume " + str(run_dir / "state.json"))
    finally:
        controller.close()

    final = rounded(state.params, state.theta)
    polyak = state.polyak()
    recommended = rounded(state.params, polyak) if polyak else final
    t = state.t_values()
    summary = state.tally.summary()
    lines = ["", "=" * 72, f"SPSA done: {state.done:,}/{state.pairs:,} pairs, {state.elapsed / 3600:.2f} h", "=" * 72,
             f"  {'parameter':28s} {'start':>8s} {'final':>9s} {'averaged':>9s} {'moved':>7s} {'signal t':>9s}"]
    for n in state.names:
        cfg = state.params[n]
        moved = (state.theta[n] - cfg["value"]) / cfg["c_end"]
        avg = f"{polyak[n]:9.2f}" if polyak else f"{'-':>9s}"
        lines.append(f"  {n:28s} {cfg['value']:>8} {state.theta[n]:9.2f} {avg} {moved:+6.1f}c {t[n]:+9.2f}")
    lines += ["", "  moved = (final - start) / c_end; |signal t| > 2: the games clearly preferred one direction,",
              "  |t| < 1: the parameter barely matters at this time control (its value is mostly noise).",
              f"  Plus scored {summary['raw_score']:.3f} over {summary['pairs']} pairs (penta {summary['pentanomial']}).",
              "", f"  recommended ({'averaged over the second half' if polyak else 'final'}): {recommended}"]
    print_line("\n".join(lines))
    if runner.repinned_games or runner.mixed_games:
        print_line(f"[SPSA] {runner.repinned_games} game(s) had an engine moved back onto its core at the start, "
                   f"{runner.mixed_games} game(s) were excluded for leaving it")
    result = dict(final_values=final, polyak_values=rounded(state.params, polyak) if polyak else None,
                  recommended=recommended, start_values={n: state.params[n]["value"] for n in state.names},
                  t_values=t, pairs_done=state.done, pairs=state.pairs, params_config=state.params,
                  plus_pentanomial=summary["pentanomial"], run_dir=str(run_dir), time_control=limit,
                  stopped=("interrupted" if interrupted else controller.stop_reason),
                  repinned_games=runner.repinned_games, mixed_core_games=runner.mixed_games)

    def write_result():
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if interrupted or controller.stop_reason:
        write_result()
        if interrupted:
            raise KeyboardInterrupt
        raise RuntimeError(f"SPSA stopped after {state.done}/{state.pairs} pairs: {controller.stop_reason}; "
                           f"resume with --resume {run_dir / 'state.json'}")

    if args.verify_pairs and state.done:
        print_line(f"\n[SPSA] verification: {args.verify_pairs} pairs, tuned vs start values")
        start = {n: state.params[n]["value"] for n in state.names}
        engines = (EngineSpec("Tuned", engine.cmd, engine.args, {**engine.options, **recommended},
                              workdir=engine.workdir, pin_option=engine.pin_option),
                   EngineSpec("Start", engine.cmd, engine.args, {**engine.options, **start},
                              workdir=engine.workdir, pin_option=engine.pin_option))
        verify = VerifyController(engines, book, args.verify_pairs, args.pairs_per_job)
        verify_runner = MatchRunner(fastchess, slots, run_dir / "verify", tc_args, plies=args.book_depth,
                                    extra_args=extra_args)
        try:
            verify_runner.run(verify)
        except KeyboardInterrupt:
            print_line("[SPSA] verification interrupted")
        v = verify.tally.summary()
        if v["pairs"]:
            print_line(f"[SPSA] tuned vs start: Elo {v['elo_diff']:+.1f} [{v['elo_low']:+.1f}, {v['elo_high']:+.1f}] "
                       f"(95%), LOS {v['los']:.1f}%, {v['pairs']} pairs, penta {v['pentanomial']}")
            result["verification"] = v
    params = " ".join(f"--param {n}={v}" for n, v in recommended.items())
    print_line(f"\n  Confirm with an SPRT:\n    python -m tests.sprt --engine-a {args.engine} --engine-b {args.engine} "
               f"{params} {limit_flag(args)} --elo0 0 --elo1 3")
    write_result()
    if plotter:
        plotter.finalize(run_dir / "spsa.png")
    return recommended


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"[SPSA] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
