#!/usr/bin/env python3
"""Lazy SMP strength test: what more search threads are worth in games.

Thread-scaling ladder: the same build plays itself, T_hi threads against T_lo
(e.g. --threads 1 2 4 plays 2 vs 1, then 4 vs 2), for a fixed number of opening
pairs at one time control, and reports each step's Elo and normalized Elo with a
95% interval and the total over the ladder. Games run on the match runner
(utils/match_runner.py through utils/fixed_match.py): every game is pinned to its
own slot of T_hi cpus of one core type (both engines take turns on them, the
engine places its threads inside with its Affinity code; on Windows the slot is
also the job's hard affinity limit), so steps measure threads, not scheduling
noise, and every game records its core type, cpus and clock. By default as many
games run at once as fit on the physical cores (every core but one); the steps
share one shuffled book, so each step plays its own openings.

benchmarks/smp.py measures the speed side (NPS, time-to-depth); this measures
what that speed is worth. Scaling depends on the time control: run it short and
long. To check that a search patch also helps with threads, run the SPRT itself
multi-threaded: python -m tests.sprt ... --threads 4 (release.py: --sprt_threads 1 4).

    python -m tests.smp --engine tomahawk.exe --threads 1 2 4 --tc 10+0.1 --pairs 200

Results: <results-dir>/<run>/summary.json (default logs/smp_results), one match
folder per step (games.jsonl, games.pgn, summary.json, jobs/).
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.fastchess_match import FASTCHESS, limit_args, limit_label, tb_args  # noqa: E402
from utils.fixed_match import (MatchInterrupted, Pairing, require_options, require_threads,  # noqa: E402
                               run_fixed_match)
from utils.match_runner import EngineSpec, plan_slots, release_slots  # noqa: E402
from utils.openings import OpeningBook  # noqa: E402

LOGS_DIR = PROJECT_ROOT / "logs"
Z95 = 1.959964


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lazy SMP thread-scaling ladder (games)")
    p.add_argument("--engine", required=True, help="Engine build to test")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4],
                   help="Thread counts; consecutive ones play each other (default: 1 2 4)")
    p.add_argument("--tc", default=None, help="Time control (e.g. 10+0.1); or --time / --depth")
    p.add_argument("--time", type=float, default=None, help="Seconds per move")
    p.add_argument("--depth", type=int, default=None, help="Depth per move (threads help less visibly here)")
    p.add_argument("--pairs", type=int, default=200, help="Opening pairs per step")
    p.add_argument("--concurrency", type=int, default=None,
                   help="Games at once per step (default: every physical core but one, divided by the step's "
                        "higher thread count; 1 with --no-pin)")
    p.add_argument("--book", default=PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn")
    p.add_argument("--book-depth", type=int, default=16)
    p.add_argument("--fastchess", default=str(FASTCHESS))
    p.add_argument("--no-pin", action="store_true", help="Don't pin games (threads then share cores; not recommended)")
    p.add_argument("--tb", default=None, help="Syzygy tablebase dir(s) for adjudication")
    p.add_argument("--option", action="append", default=[], metavar="NAME=VALUE",
                   help="Extra UCI option for both sides, e.g. --option Hash=64")
    p.add_argument("--pairs-per-job", type=int, default=4,
                   help="Opening pairs per fastchess process (amortizes engine start-up)")
    p.add_argument("--quiet", action="store_true", help="No line per game")
    p.add_argument("--results-dir", default=LOGS_DIR / "smp_results",
                   help="Where run folders go (default logs/smp_results)")
    args = p.parse_args(argv)
    if args.tc is None and args.time is None and args.depth is None:
        args.tc = "10+0.1"
    counts = sorted(set(args.threads))
    if len(counts) < 2 or counts[0] < 1:
        p.error("--threads needs at least two positive thread counts")
    if args.pairs < 1 or args.pairs_per_job < 1:
        p.error("--pairs and --pairs-per-job must be positive")
    if args.concurrency is not None and args.concurrency < 1:
        p.error("--concurrency must be positive")
    args.threads = counts
    return args


def parse_options(items):
    options = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise ValueError(f"--option expects NAME=VALUE, got {item!r}")
        options[name] = value
    return options


def step_result(lo, hi, snapshot, run_dir, interrupted=False):
    elo, low, high = snapshot["elo_diff"], snapshot["elo_low"], snapshot["elo_high"]
    se = (high - low) / (2 * Z95) if elo is not None else None
    step = dict(threads_hi=hi, threads_lo=lo, pairs=snapshot["pairs"], elo=elo, elo_low=low, elo_high=high, se=se,
                nelo=snapshot["nelo"], nelo_low=snapshot["nelo_low"], nelo_high=snapshot["nelo_high"],
                wins=snapshot["candidate_wins"], draws=snapshot["candidate_draws"],
                losses=snapshot["candidate_losses"], pentanomial=snapshot["pentanomial"],
                by_core={core: dict(pairs=t["pairs"], elo=t["elo_diff"], elo_low=t["elo_low"], elo_high=t["elo_high"])
                         for core, t in snapshot.get("by_core", {}).items()},
                run_dir=str(run_dir))
    if interrupted:
        step["interrupted"] = True
    return step


def format_report(steps, label):
    lines = [f"SMP thread scaling, {label} (Elo of more threads, 95% interval)",
             f"{'step':>10} {'Elo':>8} {'interval':>20} {'pairs':>6} {'W-D-L':>14} {'nElo':>7}"]
    total, variance = 0.0, 0.0
    for s in steps:
        note = " (interrupted)" if s.get("interrupted") else ""
        if s["elo"] is None:
            lines.append(f"{s['threads_hi']:>4} vs {s['threads_lo']:<3} {'n/a':>8}{note}")
            continue
        total += s["elo"]
        variance += s["se"] ** 2
        lines.append(f"{s['threads_hi']:>4} vs {s['threads_lo']:<3} {s['elo']:+8.1f} "
                     f"[{s['elo_low']:+7.1f}, {s['elo_high']:+7.1f}] {s['pairs']:6d} "
                     f"{s['wins']:>4}-{s['draws']}-{s['losses']} {s['nelo']:+7.1f}{note}")
    if len(steps) > 1:
        lines.append(f"{steps[-1]['threads_hi']:>4} vs {steps[0]['threads_lo']:<3} {total:+8.1f} "
                     f"+- {Z95 * math.sqrt(variance):.1f} (sum of the steps)")
    return "\n".join(lines)


def play_step(args, engine, options, book, lo, hi, tc_args, label, extra, run_root):
    """One ladder step, T_hi threads (first, White in each pair's first game) against T_lo; its pairing result."""
    def side(threads):
        return EngineSpec(f"T{threads}", engine.cmd, engine.args, {**options, "Threads": threads},
                          workdir=engine.workdir, pin_option=engine.pin_option)

    # slots of T_hi cpus of one core type: both engines of a game take turns on them
    slots = plan_slots(concurrency=args.concurrency or (1 if args.no_pin else None), threads=hi,
                       pin=not args.no_pin)
    try:
        print(f"\n[SMP] {hi} threads vs {lo}: {args.pairs} pairs, {label}, {len(slots)} game(s) at once")
        result = run_fixed_match([Pairing(side(hi), side(lo), args.pairs)], tc_args=tc_args, book=book,
                                 plies=args.book_depth, slots=slots, threads=hi, fastchess=args.fastchess,
                                 results_dir=run_root, run_name=f"T{hi}_vs_T{lo}", extra_args=extra,
                                 pairs_per_job=args.pairs_per_job, quiet=args.quiet, label="SMP",
                                 time_control=label, meta=dict(arguments=vars(args), step=[hi, lo]))
    finally:
        release_slots(slots)
    return step_result(lo, hi, next(iter(result["pairings"].values())), result["run_dir"])


def main(argv=None):
    args = parse_args(argv)
    tc_args = limit_args(args.depth, args.time, args.tc, None)
    label = limit_label(args.depth, args.time, args.tc, None)
    extra = ["-maxmoves", "100"] + tb_args(args.tb)
    options = parse_options(args.option)
    engine = EngineSpec.from_path("Engine", args.engine).resolve()
    require_options(engine, options)  # fastchess would skip a misspelt or out-of-range option with a warning
    require_threads(engine, args.threads[-1])  # before the first step, not when the ladder gets there
    book = OpeningBook(args.book, plies=args.book_depth)  # one shuffle: each step plays other openings
    run_root = Path(args.results_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    steps = []

    def save():
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "summary.json").write_text(json.dumps(dict(
            engine=os.path.abspath(args.engine), limit=label, book_seed=book.seed, steps=steps), indent=2,
            default=str), encoding="utf-8")

    interrupted = False
    for lo, hi in zip(args.threads, args.threads[1:]):
        try:
            steps.append(play_step(args, engine, options, book, lo, hi, tc_args, label, extra, run_root))
        except MatchInterrupted as stop:  # Ctrl-C: keep the steps so far and the interrupted one's pairs
            steps.append(step_result(lo, hi, next(iter(stop.result["pairings"].values())), stop.result["run_dir"],
                                     interrupted=True))
            interrupted = True
        save()
        print(format_report(steps, label))
        if interrupted:
            break
    report = format_report(steps, label)
    print(f"\n{report}\n\n[SMP] Results: {run_root}")
    if interrupted:
        raise KeyboardInterrupt
    return steps


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"[SMP] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
