"""
compare.py — full stat comparison between two engines

using --game enable persistent TT across moves (warm) to emulate a game 
not using --game runs each position in isolation (cold) and clears the TT between positions

Usage:
    py -m tests.compare --engines engines/0.1.0.exe engines/0.2.2.exe --time 1
    py -m tests.compare --engines engines/0.1.0.exe engines/0.2.2.exe --time 1 --game bin/test_positions/bench_game.txt --side 0
"""

import subprocess
import time
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

# --- Config ---
ENGINES = []

POSITIONS = [
    ("startpos",                                                                  "Starting position"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",      "Ruy Lopez"),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  "Kiwipete"),
    ("3r1rk1/pppqb1pp/1nn2p2/4p3/1P4b1/P1NPBNP1/2Q1PPBP/2R2RK1 b - - 3 13",   "Middlegame"),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                              "Endgame"),
]

MOVETIME_MS  = 1000
DEPTH = None
RUNS_PER_POS = 1

# All stats from dumpstats and their display config
# (key, label, format, higher_is_better)
STAT_DEFS = [
    ("time_ms",                 "Time (ms)",                ",.0f", None),
    ("depth",                   "Completed Depth",          ".1f",  True),
    ("nps",                     "NPS",                      ",.0f", True),
    ("nodes",                   "Nodes",                    ",.0f", True),
    ("qnodes",                  "QNodes",                   ",.0f", True),
    ("q_ratio",                 "Q Ratio %",                ".1f",  False),
    ("tt_hit_rate",             "TT Hit Rate %",            ".1f",  True),
    ("tt_return_rate",          "TT Return Rate %",         ".1f",  True),
    ("tt_fill",                 "TT Fill %",                ".2f",  False),
    ("fh_first_pct",            "FH First %",               ".1f",  True),
    ("fail_highs",              "Fail Highs",               ",.0f", True),
    ("fail_lows",               "Fail Lows",                ",.0f", True),
    ("see_pruned",              "SEE Pruned",               ",.0f", True),
    ("delta_pruned",            "Delta Pruned",             ",.0f", True),
    ("nmp_attempts",            "NMP Attempts",             ",.0f", True),
    ("nmp_success_pct",         "NMP Success %",            ".1f",  True),
    ("pvs_researches_full",     "PVS Re-searches (FULL)",   ",.1f", False),
    ("pvs_researches_lmr",      "PVS Re-searches (LMR)",    ",.1f", False),
    ("pvs_researches_root",     "PVS Re-searches (ROOT)",   ",.1f", False),
    ("asp_failhigh",            "Asp Fail High",            ",.1f", False),
    ("asp_faillow",             "Asp Fail Low",             ",.1f", False),
]

STAT_PATTERNS = {
    "depth":                r"Completed Depth\s+(\d+)",
    "nps":                  r"NPS\s+(\d+)",
    "nodes":                r"Total\s+(\d+)",
    "qnodes":               r"QNodes\s+(\d+)",
    "time_ms":              r"Time \(ms\)\s+(\d+)",
    "tt_return_rate":       r"Return Rate\s+([\d.]+)%",
    "tt_hit_rate":          r"Hit Rate\s+([\d.]+)%",
    "tt_fill":              r"Fill Ratio\s+([\d.]+)%",
    "fh_first_pct":         r"FH % at \[0\]\s+([\d.]+)%",
    "fail_highs":           r"Fail Highs\s+(\d+)",
    "fail_lows":            r"Fail Lows\s+(\d+)",
    "see_pruned":           r"SEE Prunes\s+(\d+)",
    "delta_pruned":         r"Delta Prunes\s+(\d+)",
    "nmp_attempts":         r"NMP Attempts\s+(\d+)",
    "nmp_success_pct":      r"NMP FH %\s+([\d.]+)%",
    "pvs_researches_full":  r"PVS full\s+(\d+)",
    "pvs_researches_lmr":   r"PVS w/ LMR\s+(\d+)",
    "pvs_researches_root":  r"PVS @ root\s+(\d+)",
    "asp_failhigh":         r"Aspiration FH Re\s+(\d+)",
    "asp_faillow":          r"Aspiration FL Re\s+(\d+)",
}


def parse_dumpstats(output: str) -> dict:
    stats = {}
    for key, pattern in STAT_PATTERNS.items():
        m = re.search(pattern, output)
        if m:
            stats[key] = float(m.group(1))
    # derived
    if "nodes" in stats and "qnodes" in stats and stats["nodes"] > 0:
        stats["q_ratio"] = 100.0 * stats["qnodes"] / stats["nodes"]
    return stats


def collect_dumpstats(proc, movetime: int) -> dict:
    """Wait for bestmove then send dumpstats and collect output."""
    start = time.time()
    wait_time = movetime if movetime else 5000 # 5s
    while time.time() - start < (wait_time / 1000) + 5:
        line = proc.stdout.readline().strip()
        if line.startswith("bestmove"):
            break

    proc.stdin.write("dumpstats\nisready\n")
    proc.stdin.flush()

    dump_output = []
    start = time.time()
    while time.time() - start < 3:
        line = proc.stdout.readline()
        if not line:
            break
        if line.strip() == "readyok":
            break
        dump_output.append(line)

    return parse_dumpstats("".join(dump_output))


def run_cold(engine_path: str, position: str, movetime: int, depth: int, runs: int) -> list[dict]:
    """Fresh engine per search — cold TT."""
    results = []
    for _ in range(runs):
        try:
            proc = subprocess.Popen(
                [engine_path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True,
            )
            proc.stdin.write("uci\nisready\n")
            proc.stdin.flush()
            # wait for readyok
            start = time.time()
            while time.time() - start < 5:
                if "readyok" in proc.stdout.readline():
                    break

            pos_cmd = "position startpos" if position == "startpos" else f"position fen {position}"
            proc.stdin.write(f"{pos_cmd}\ngo {"movetime" if movetime else "depth"} {movetime if movetime else depth}\n")
            proc.stdin.flush()

            stats = collect_dumpstats(proc, movetime)
            results.append(stats)

            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.wait(timeout=3)
        except Exception as e:
            print(f"  Error: {e}")
    return results


def run_warm(engine_path: str, moves: list[str], movetime: int, depth: int, side: int) -> list[dict]:
    """Single engine process, TT persists across positions."""
    results = []
    try:
        proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
        )

        def send(cmd):
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()

        def wait_for(token, timeout=10):
            start = time.time()
            while time.time() - start < timeout:
                if token in proc.stdout.readline():
                    return
        send("uci"); wait_for("uciok")
        send("isready"); wait_for("readyok")

        played = []
        for i, move in enumerate(moves):
            current_side = i % 2
            if current_side == side:
                pos_cmd = f"position startpos moves {' '.join(played)}" if played else "position startpos"
                send(pos_cmd)
                send(f"go {"movetime" if movetime else "depth"} {movetime if movetime else depth}")
                stats = collect_dumpstats(proc, movetime)
                stats["ply"] = i
                stats["move_num"] = len(played)
                results.append(stats)
            played.append(move)

        send("quit")
        proc.wait(timeout=3)
    except Exception as e:
        print(f"  Error: {e}")
    return results


def avg(stats_list: list[dict], key: str) -> Optional[float]:
    vals = [s[key] for s in stats_list if key in s]
    return sum(vals) / len(vals) if vals else None


def delta_str(base_val, cand_val, higher_is_better, fmt):
    if base_val is None or cand_val is None or base_val == 0:
        return "--"
    
    diff = cand_val - base_val
    pct  = (diff / base_val) * 100

    # color/symbol
    if higher_is_better is None:
        symbol = ""
    elif abs(pct) < 0.1: # .1% threshold for "no change"
        symbol = "="
    elif (higher_is_better and diff > 0) or (not higher_is_better and diff < 0):
        symbol = "▲"
    else:
        symbol = "▼"

    return f"{diff:+{fmt}} ({pct:+.1f}%) {symbol}"


def print_comparison(results: dict, engines: list[str], label: str):
    names = [e.split("/")[-1].replace(".exe", "") for e in engines]

    col = 20

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    print(f"  {'Stat':<25}", end="")
    for name in names:
        print(f"{name:>{col}}", end="")
    print()

    print(f"  {'-'*25}", end="")
    for _ in engines:
        print(f" {'-'*col}", end="")
    print()

    baseline = engines[0]

    for key, label_str, fmt, hib in STAT_DEFS:
        base_val = avg(results[baseline], key)
        if base_val is None:
            continue

        print(f"  {label_str:<25}", end="")

        for engine in engines:
            val = avg(results[engine], key)
            val_str = format(val, fmt) if val is not None else "--"
            print(f"{val_str:>{col}}", end="")

        print()


def load_game_moves(path: str) -> list[str]:
    with open(path) as f:
        content = f.read().strip()
    if not content.startswith("["):
        return content.replace("\n", " ").split()
    try:
        import chess.pgn, io
        game = chess.pgn.read_game(io.StringIO(content))
        board = chess.Board()
        moves = []
        for move in game.mainline_moves():
            moves.append(move.uci())
            board.push(move)
        return moves
    except ImportError:
        print("pip install chess for PGN support")
        sys.exit(1)


if __name__ == "__main__":
    engines     = ENGINES
    movetime_ms = MOVETIME_MS
    depth = DEPTH
    runs        = RUNS_PER_POS
    game_path   = None
    side        = 0
    positions   = POSITIONS

    if "--engines" in sys.argv:
        idx = sys.argv.index("--engines")
        engines = []
        for arg in sys.argv[idx+1:]:
            if arg.startswith("--"): break
            engines.append(arg)

    if "--time" in sys.argv:
        idx = sys.argv.index("--time")
        movetime_ms = int(float(sys.argv[idx+1]) * 1000)

    if "--depth" in sys.argv:
        idx = sys.argv.index("--depth")
        depth = int(sys.argv[idx+1])
        movetime_ms = None

    if "--runs" in sys.argv:
        idx = sys.argv.index("--runs")
        runs = int(sys.argv[idx+1])

    if "--game" in sys.argv:
        idx = sys.argv.index("--game")
        game_path = "bin/test_positions/bench_game.txt"

    if "--side" in sys.argv:
        idx = sys.argv.index("--side")
        side = int(sys.argv[idx+1])

    if "--positions" in sys.argv:
        idx = sys.argv.index("--positions")
        path = sys.argv[idx+1]
        positions = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if "|" in line:
                    fen, name = line.split("|", 1)
                    positions.append((fen.strip(), name.strip()))
                else:
                    positions.append((line, line[:40]))

    print(f"Engines: {[e.split('/')[-1] for e in engines]} | Time: {movetime_ms}{"ms" if movetime_ms else ""} | Depth: {depth}")

    if game_path: # warm TT
        moves = load_game_moves(game_path)
        print(f"Game: {len(moves)} moves, side={'white' if side==0 else 'black'}\n")

        all_results = {}
        for engine in engines:
            name = engine.split("/")[-1].replace(".exe", "")
            print(f"Running {name}...")
            all_results[engine] = run_warm(engine, moves, movetime_ms, depth, side)

        print_comparison(all_results, engines, f"Warm TT — avg over {len(all_results[engines[0]])} positions")

    else:  # cold
        for pos_fen, pos_name in positions:
            all_results = {}
            for engine in engines:
                name = engine.split("/")[-1].replace(".exe", "")
                print(f"  {name} | {pos_name}...", end="\r")
                all_results[engine] = run_cold(engine, pos_fen, movetime_ms, depth, runs)
            print_comparison(all_results, engines, f"Cold TT — {pos_name}")

        # aggregate cold summary
        print(f"\n{'='*80}")
        print(f"  AGGREGATE SUMMARY (avg across all positions)")
        print(f"{'='*80}")
        agg = {e: [] for e in engines}
        for pos_fen, pos_name in positions:
            # re-run isn't ideal but we'd need to cache — for now just note it
            pass
        print("  (run with --runs N for per-position averages)")