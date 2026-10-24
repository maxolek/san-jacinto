"""A/B comparison of two engine builds: search equivalence + paired timing.

Both builds search the same position at the same time on two different physical
P-cores, so they share the machine's power/frequency state (a laptop's clock can
move by 2x with background load); the cores are swapped on every repetition.

For implementation-only optimizations, the searches must be identical: nodes,
best move, eval and PV are compared per position (single-threaded, fixed depth).
Use --allow-diff for changes that intentionally alter the search; the timing is
then only an NPS comparison, not a strength measurement.

Run an A/A test (same exe twice) to see the noise floor on your machine.

usage: python compare.py <base.exe> <new.exe> [--reps 4] [--cpus auto|a,b] [--depth-delta D] [--only a,b] [--allow-diff]
"""
import argparse
import math
import random
import statistics
import sys
import threading

import common


def run_pair(base, new, fen, depth, cpus, threads):
    res = [None, None]

    def go(i, exe, cpu):
        res[i] = common.search(exe, fen, depth=depth, threads=threads, cpus=[cpu])

    ts = [threading.Thread(target=go, args=(0, base, cpus[0])), threading.Thread(target=go, args=(1, new, cpus[1]))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--reps", type=int, default=4, help="paired runs per position (cores swap each run)")
    ap.add_argument("--cpus", default="auto", help="two logical CPUs on different physical P-cores, e.g. 2,4")
    ap.add_argument("--depth-delta", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--allow-diff", action="store_true", help="do not fail when the searches differ")
    args = ap.parse_args()

    cpus = common.parse_cpus(args.cpus)
    if cpus is None:
        fast = common.performance_cpus()
        cpus = fast[1:3] if len(fast) >= 3 else fast[:2]
    if len(cpus) < 2:
        sys.exit("need two P-cores (use --cpus)")
    c0, c1 = cpus[0], cpus[1]
    print(f"base {args.base}\nnew  {args.new}\ncpus {c0},{c1}")

    rows = []
    mismatches = 0
    for name, fen, depth in common.select_positions(args.only):
        d = depth + args.depth_delta
        tb, tn, ratios = [], [], []
        rb = rn = None
        for rep in range(args.reps):
            pair = (c0, c1) if rep % 2 == 0 else (c1, c0)
            rb, rn = run_pair(args.base, args.new, fen, d, pair, 1)
            tb.append(rb["time_ms"])
            tn.append(rn["time_ms"])
            ratios.append(rb["time_ms"] / max(rn["time_ms"], 1))
        key = ("nodes", "bestmove", "eval", "pv")
        same = all(rb[k] == rn[k] for k in key)
        mismatches += not same
        ratio = statistics.median(ratios)
        rows.append((rb, rn, statistics.median(tb), statistics.median(tn), ratio))
        print(f"{'OK  ' if same else 'DIFF'} {name:12s} d={d:2d} nodes {rb['nodes']:>9d}/{rn['nodes']:>9d} "
              f"best {rb['bestmove']}/{rn['bestmove']} eval {rb['eval']}/{rn['eval']} "
              f"ms {statistics.median(tb):>7.0f} -> {statistics.median(tn):>7.0f}  x{ratio:.3f} "
              f"[{min(ratios):.2f}-{max(ratios):.2f}]", flush=True)
        if not same and not args.allow_diff:
            print(f"   base pv: {rb['pv']}\n   new  pv: {rn['pv']}")

    nb = sum(r[0]["nodes"] for r in rows)
    nn = sum(r[1]["nodes"] for r in rows)
    tb = sum(r[2] for r in rows)
    tn = sum(r[3] for r in rows)
    logs = [math.log(r[4]) for r in rows]
    gm = math.exp(statistics.mean(logs))
    rnd = random.Random(1)
    boots = sorted(math.exp(statistics.mean(rnd.choices(logs, k=len(logs)))) for _ in range(4000))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
    print(f"\nnodes base={nb} new={nn}  time base={tb:.0f}ms new={tn:.0f}ms")
    print(f"NPS base={1000 * nb / tb:.0f} new={1000 * nn / tn:.0f}  total-time speedup x{tb / tn:.3f}")
    print(f"per-position geomean speedup x{gm:.3f}  (95% bootstrap CI over positions x{lo:.3f}..x{hi:.3f})")
    print(f"search mismatches: {mismatches}/{len(rows)}")
    sys.exit(1 if mismatches and not args.allow_diff else 0)


if __name__ == "__main__":
    main()
