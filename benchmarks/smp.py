"""Lazy SMP checks: stability of multi-threaded searches, then scaling.

Stability: many short fixed-depth and timed searches with several thread counts;
every search must end with a bestmove and a clean exit.
Scaling: for each thread count, NPS at fixed time and time-to-depth at fixed depth.
Time-to-depth is the more meaningful number: Lazy SMP threads duplicate work, so
NPS overstates the benefit. Neither replaces game testing for strength.

usage: python smp.py <engine.exe> [--threads 1,2,4] [--reps 2] [--movetime 2000] [--skip-stability]
"""
import argparse
import statistics
import time

import common

SCALING_POSITIONS = ("startpos", "kiwipete", "sf_mid1", "sf_mid7", "sf_end2")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("--threads", default="1,2,4", help="thread counts to measure")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--movetime", type=int, default=2000, help="ms per timed search")
    ap.add_argument("--skip-stability", action="store_true")
    args = ap.parse_args()

    counts = [int(x) for x in args.threads.split(",")]
    print(f"engine {args.engine}  P-cores {common.performance_cpus()} "
          f"(larger thread counts run unpinned, placed by the engine)")

    if not args.skip_stability:
        t0 = time.time()
        failures = 0
        for threads in sorted({max(2, c) for c in counts} | {2 * max(counts)}):
            for name, fen, depth in common.POSITIONS[::2]:
                out = common.run_engine(args.engine, [
                    f"setoption name Threads value {threads}", "isready",
                    f"position fen {fen}", f"go depth {max(1, depth - 2)}",
                    f"position fen {fen}", "go movetime 200", "dumpstats"], common.cpus_for_threads(threads))
                if out.count("bestmove") != 2:
                    failures += 1
                    print(f"FAIL threads={threads} {name}\n{out[-400:]}")
        print(f"stability: {failures} failures ({time.time() - t0:.0f}s)")

    positions = [p for p in common.POSITIONS if p[0] in SCALING_POSITIONS]
    base_ttd = None
    for threads in counts:
        cpus = common.cpus_for_threads(threads)
        nps, ttd_ms, used = [], 0, set()
        for name, fen, depth in positions:
            for _ in range(args.reps):
                r = common.search(args.engine, fen, movetime=args.movetime, threads=threads, cpus=cpus)
                nps.append(1000.0 * r["nodes"] / max(r["time_ms"], 1))
                used.add(r["threads"])
                r = common.search(args.engine, fen, depth=depth + 1, threads=threads, cpus=cpus)
                ttd_ms += r["time_ms"]
        ttd_ms /= args.reps
        base_ttd = base_ttd or ttd_ms
        note = "" if used == {threads} else f"  (engine ran {'/'.join(map(str, sorted(used)))} threads: fewer cpus)"
        print(f"threads={threads}: mean NPS {statistics.mean(nps):>10.0f}   time-to-depth {ttd_ms:>8.0f} ms "
              f"(x{base_ttd / max(ttd_ms, 1):.2f}){note}", flush=True)


if __name__ == "__main__":
    main()
