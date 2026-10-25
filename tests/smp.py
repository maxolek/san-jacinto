#!/usr/bin/env python3
"""Lazy SMP strength test: what more search threads are worth in games.

Thread-scaling ladder: the same build plays itself, T_hi threads against T_lo
(e.g. --threads 1 2 4 plays 2 vs 1, then 4 vs 2), for a fixed number of opening
pairs at one time control, and reports each step's Elo with a 95% interval and
the total over the ladder. Every game is pinned to its own T_hi cpus (both
engines take turns on them, the engine places its threads inside with its
Affinity code), so steps measure threads, not scheduling noise. By default as
many games run at once as fit on the physical cores.

benchmarks/smp.py measures the speed side (NPS, time-to-depth); this measures
what that speed is worth. Scaling depends on the time control: run it short and
long. To check that a search patch also helps with threads, run the SPRT itself
multi-threaded: python -m tests.sprt ... --threads 4 (release.py: --sprt_threads 1 4).

    python -m tests.smp --engine tomahawk.exe --threads 1 2 4 --tc 10+0.1 --pairs 200

Results: logs/smp_results/<run>/summary.json, one match folder per step.
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
from utils import cpu_slots, uci_probe  # noqa: E402
from utils.fastchess_match import FASTCHESS, Side, limit_args, limit_label, prepare_match, run_match, tb_args  # noqa: E402
from utils.pentanomial import PentanomialSPRT  # noqa: E402

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
                   help="Games at once per step (default: as many as fit on the physical cores)")
    p.add_argument("--book", default=PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn")
    p.add_argument("--book-depth", type=int, default=16)
    p.add_argument("--fastchess", default=str(FASTCHESS))
    p.add_argument("--no-pin", action="store_true", help="Don't pin games (threads then share cores; not recommended)")
    p.add_argument("--tb", default=None, help="Syzygy tablebase dir(s) for adjudication")
    p.add_argument("--option", action="append", default=[], metavar="NAME=VALUE",
                   help="Extra UCI option for both sides, e.g. --option Hash=64")
    args = p.parse_args(argv)
    if args.tc is None and args.time is None and args.depth is None:
        args.tc = "10+0.1"
    counts = sorted(set(args.threads))
    if len(counts) < 2 or counts[0] < 1:
        p.error("--threads needs at least two positive thread counts")
    args.threads = counts
    return args


def step_result(lo, hi, snapshot, run_dir):
    elo, low, high = snapshot["elo_diff"], snapshot["elo_low"], snapshot["elo_high"]
    se = (high - low) / (2 * Z95) if elo is not None else None
    return dict(threads_hi=hi, threads_lo=lo, pairs=snapshot["pairs"], elo=elo, elo_low=low, elo_high=high, se=se,
                wins=snapshot["candidate_wins"], draws=snapshot["candidate_draws"],
                losses=snapshot["candidate_losses"], pentanomial=snapshot["pentanomial"], run_dir=str(run_dir))


def format_report(steps, label):
    lines = [f"SMP thread scaling, {label} (Elo of more threads, 95% interval)",
             f"{'step':>10} {'Elo':>8} {'interval':>20} {'pairs':>6} {'W-D-L':>14}"]
    total, variance = 0.0, 0.0
    for s in steps:
        if s["elo"] is None:
            lines.append(f"{s['threads_hi']:>4} vs {s['threads_lo']:<3} {'n/a':>8}")
            continue
        total += s["elo"]
        variance += s["se"] ** 2
        lines.append(f"{s['threads_hi']:>4} vs {s['threads_lo']:<3} {s['elo']:+8.1f} "
                     f"[{s['elo_low']:+7.1f}, {s['elo_high']:+7.1f}] {s['pairs']:6d} "
                     f"{s['wins']:>4}-{s['draws']}-{s['losses']}")
    if len(steps) > 1:
        lines.append(f"{steps[-1]['threads_hi']:>4} vs {steps[0]['threads_lo']:<3} {total:+8.1f} "
                     f"+- {Z95 * math.sqrt(variance):.1f} (sum of the steps)")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    engine = os.path.abspath(args.engine)
    limits = limit_args(args.depth, args.time, args.tc, None)
    label = limit_label(args.depth, args.time, args.tc, None)
    options = [f"option.{o}" for o in args.option]
    uci_probe.require_threads(engine, args.threads[-1])  # before the first step, not when the ladder gets there
    run_root = LOGS_DIR / "smp_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    steps = []
    for lo, hi in zip(args.threads, args.threads[1:]):
        concurrency = args.concurrency or (1 if args.no_pin else cpu_slots.physical_slots(engine, hi))
        print(f"\n[SMP] {hi} threads vs {lo}: {args.pairs} pairs, {label}, {concurrency} game(s) at once")
        run_dir = run_root / f"T{hi}_vs_T{lo}"
        match = prepare_match(Side(engine, options, hi), Side(engine, options, lo), limits=limits,
                              pairs=args.pairs, concurrency=concurrency, run_dir=run_dir,
                              fastchess=args.fastchess, book=args.book, book_depth=args.book_depth,
                              pin=not args.no_pin, extra=tb_args(args.tb),
                              meta=dict(arguments=vars(args), step=[hi, lo]))
        state = PentanomialSPRT(0, 10, sequential=False)  # fixed length: every pair counts
        steps.append(step_result(lo, hi, run_match(match, state), run_dir))
        print(format_report(steps, label))
    report = format_report(steps, label)
    (run_root / "summary.json").write_text(json.dumps(dict(engine=engine, limit=label, steps=steps), indent=2,
                                                      default=str), encoding="utf-8")
    print(f"\n{report}\n\n[SMP] Results: {run_root}")
    return steps


if __name__ == "__main__":
    main()
