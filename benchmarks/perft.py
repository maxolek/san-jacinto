"""Perft regression suite (move generation + make/unmake), known counts from the
Chess Programming Wiki. Run after any board or move-generator change.

usage: python perft.py <engine.exe> [--quick]
"""
import argparse
import re
import sys
import time

import common

CASES = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603),
    ("pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083),
    ("pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333),
    ("pos4m", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 4, 422333),
    ("pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487),
    ("pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594),
]
QUICK_DEPTH = {"startpos": 4, "kiwipete": 3, "pos3": 5, "pos4": 3, "pos4m": 3, "pos5": 3, "pos6": 3}
QUICK_COUNT = {"startpos": 197281, "kiwipete": 97862, "pos3": 674624, "pos4": 9467, "pos4m": 9467,
               "pos5": 62379, "pos6": 89890}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("--quick", action="store_true", help="one depth less per position")
    args = ap.parse_args()

    cpus = common.performance_cpus()[:1]
    ok = True
    for name, fen, depth, expected in CASES:
        if args.quick:
            depth, expected = QUICK_DEPTH[name], QUICK_COUNT[name]
        t0 = time.time()
        out = common.run_engine(args.engine, [f"position fen {fen}", f"perft {depth}"], cpus, timeout=1800)
        m = re.search(r"Nodes searched:\s*(\d+)", out)
        got = int(m.group(1)) if m else -1
        good = got == expected
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} {name:9s} d={depth} got={got} expected={expected} ({time.time() - t0:.1f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
