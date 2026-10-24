"""Run a verification build over the benchmark positions.

Build the engine with -DTOMAHAWK_VERIFY_NNUE (see README). That build checks, at
every move made in search, the incrementally updated accumulators against a full
rebuild, and every evaluation against the scalar reference NNUE::evaluate(); the
first mismatch prints "NNUE VERIFY FAILED ..." and exits. It is slow, so the
default depth is reduced.

usage: python verify.py <verify-build.exe> [--depth-delta -2] [--threads 1]
"""
import argparse
import sys

import common


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("--depth-delta", type=int, default=-2)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    cpus = common.performance_cpus()
    failures = 0
    for name, fen, depth in common.select_positions(args.only):
        cmds = []
        if args.threads > 1:
            cmds.append(f"setoption name Threads value {args.threads}")
        cmds += [f"position fen {fen}", f"go depth {max(1, depth + args.depth_delta)} eval"]
        out = common.run_engine(args.engine, cmds, cpus)
        failed = [line for line in out.splitlines() if "VERIFY FAILED" in line]
        best = [line for line in out.splitlines() if line.startswith("eval ")]
        status = failed[0] if failed else ("ok " + best[-1] if best else "NO RESULT")
        failures += bool(failed) or not best
        print(f"{name:12s} {status}", flush=True)
    print(f"verification failures: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
