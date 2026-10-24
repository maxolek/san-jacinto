"""Fixed-depth benchmark of one engine: nodes, time and NPS per position.

Every search runs in a fresh process pinned to one P-core. With a single thread the
node counts are deterministic, so they double as a signature of the search itself.

usage: python bench.py <engine.exe> [--reps 3] [--cpu N] [--depth-delta D] [--only a,b] [--threads T] [--out result.json]
"""
import argparse
import json
import statistics

import common


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("--reps", type=int, default=1, help="searches per position (median time is used)")
    ap.add_argument("--cpu", type=int, default=None, help="logical CPU to pin to (default: a P-core)")
    ap.add_argument("--depth-delta", type=int, default=0, help="add to every position's depth")
    ap.add_argument("--only", default="", help="comma-separated position names")
    ap.add_argument("--threads", type=int, default=1,
                    help="engine Threads option (pins to the P-cores while they suffice, else unpinned)")
    ap.add_argument("--out", default="", help="write the results as JSON")
    args = ap.parse_args()

    fast = common.performance_cpus()
    if args.threads > 1:
        cpus = common.cpus_for_threads(args.threads)
    else:
        cpus = [args.cpu] if args.cpu is not None else [fast[1] if len(fast) > 1 else fast[0]]
    print(f"engine {args.engine}  cpus {cpus if cpus else 'unpinned (engine placement)'}  threads {args.threads}")

    results = {}
    total_nodes = total_ms = 0
    for name, fen, depth in common.select_positions(args.only):
        d = depth + args.depth_delta
        runs = [common.search(args.engine, fen, depth=d, threads=args.threads, cpus=cpus) for _ in range(args.reps)]
        times = [r["time_ms"] for r in runs]
        first = runs[0]
        if first["threads"] != args.threads:
            print(f"  WARNING: the engine ran {first['threads']} threads (fewer cpus available)")
        if args.threads == 1 and any(r["nodes"] != first["nodes"] for r in runs):
            print(f"  WARNING: node count changed between repetitions of {name}")
        med = statistics.median(times)
        results[name] = {"depth": d, "nodes": first["nodes"], "bestmove": first["bestmove"], "eval": first["eval"],
                         "pv": first["pv"], "times_ms": times, "median_ms": med}
        total_nodes += first["nodes"]
        total_ms += med
        print(f"{name:12s} d={d:2d} nodes={first['nodes']:>10d} ms={med:>7.0f} "
              f"nps={1000.0 * first['nodes'] / max(med, 1):>9.0f} best={first['bestmove']} eval={first['eval']}",
              flush=True)
    print(f"TOTAL nodes={total_nodes} ms={total_ms:.0f} nps={1000.0 * total_nodes / max(total_ms, 1):.0f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"engine": args.engine, "cpus": cpus, "threads": args.threads, "results": results,
                       "total_nodes": total_nodes, "total_ms": total_ms}, f, indent=1)


if __name__ == "__main__":
    main()
