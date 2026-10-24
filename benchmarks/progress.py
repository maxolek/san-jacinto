"""Progress table: several engine configurations measured under identical conditions.

Each config is NAME=path[@threads]. Every position runs every config back to back
(order alternates between repetitions to cancel drift), each in a fresh process
pinned to the same P-cores (configs with more threads than P-cores run unpinned and
the engine places the threads). Speedups are relative to the first config.

example:
  python progress.py original=out/baseline.exe current=out/current.exe current-2T=out/current.exe@2 current-4T=out/current.exe@4
"""
import argparse

import common


def parse_config(text):
    name, _, rest = text.partition("=")
    path, _, threads = rest.partition("@")
    return name, path, int(threads or 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="NAME=path[@threads]")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--depth-delta", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    configs = [parse_config(c) for c in args.configs]
    print(f"P-cores {common.performance_cpus()}")
    totals = {name: [0, 0] for name, _, _ in configs}  # nodes, ms
    used = {name: set() for name, _, _ in configs}     # thread counts the engine confirmed
    for rep in range(args.reps):
        order = configs if rep % 2 == 0 else list(reversed(configs))
        for _, fen, depth in common.select_positions(args.only):
            for name, exe, threads in order:
                r = common.search(exe, fen, depth=depth + args.depth_delta, threads=threads,
                                  cpus=common.cpus_for_threads(threads))
                totals[name][0] += r["nodes"]
                totals[name][1] += r["time_ms"]
                used[name].add(r["threads"])
        print(f"rep {rep + 1}/{args.reps} done", flush=True)

    base_nodes, base_ms = totals[configs[0][0]]
    base_nps = 1000.0 * base_nodes / base_ms
    print(f"\n{'config':20s} {'suite ms':>10s} {'time-to-depth':>14s} {'NPS':>11s} {'NPS ratio':>10s}")
    for name, _, threads in configs:
        nodes, ms = totals[name]
        nps = 1000.0 * nodes / ms
        note = "" if used[name] == {threads} else f"  (engine ran {'/'.join(map(str, sorted(used[name])))} threads)"
        print(f"{name:20s} {ms:>10d} {base_ms / ms:>13.2f}x {nps:>11.0f} {nps / base_nps:>9.2f}x{note}")


if __name__ == "__main__":
    main()
