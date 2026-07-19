import subprocess
import time
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

# --- Config ---
ENGINES = []

POSITIONS = [
    ("startpos",                    "Starting position"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Ruy Lopez"),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "Kiwipete"),
    ("3r1rk1/pppqb1pp/1nn2p2/4p3/1P4b1/P1NPBNP1/2Q1PPBP/2R2RK1 b - - 3 13", "Middlegame"),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "Endgame"),
]

GAME_SIM_PATH = "bin/test_positions/bench_game.txt"

MOVETIME_MS  = 2000
RUNS_PER_POS = 1  # average over multiple runs


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


@dataclass
class SearchStats:
    nps:            list[float] = field(default_factory=list)
    nodes:          list[float] = field(default_factory=list)
    qnodes:         list[float] = field(default_factory=list)
    depth:          list[float] = field(default_factory=list)
    eval:           list[float] = field(default_factory=list)
    tt_return_rate: list[float] = field(default_factory=list)
    tt_hit_rate:    list[float] = field(default_factory=list)
    fh_first_pct:   list[float] = field(default_factory=list)
    see_pruned:     list[float] = field(default_factory=list)
    nmp_success:    list[float] = field(default_factory=list)

    def avg(self, key):
        vals = getattr(self, key)
        return sum(vals) / len(vals) if vals else 0


def print_results(results: dict, engines: list[str], positions: list[tuple]):
    baseline = engines[0]

    for _, pos_name in positions:
        print(f"\n{'='*70}")
        print(f"  {pos_name}")
        print(f"{'='*70}")
        print(f"  {'Engine':<20} {'NPS':>10} {'Nodes':>10} {'QNodes':>10} {'Depth':>6} {'TT%':>6} {'FH1%':>6} {'Q%':>6}  {'vs baseline':>11}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*11}")

        base_nps = results[baseline][pos_name].avg("nps")

        for engine in engines:
            s = results[engine][pos_name]
            pct = ((s.avg("nps") / base_nps) - 1) * 100 if base_nps and engine != baseline else None
            pct_str = f"{pct:+.1f}%" if pct is not None else "baseline"
            name = engine.split("/")[-1].replace(".exe", "")
            print(f"  {name:<20} {s.avg('nps'):>10,.0f} {s.avg('nodes'):>10,.0f} {s.avg('qnodes'):>10,.0f}"
                  f"{s.avg('depth'):>6.1f} {s.avg('tt_hit_rate'):>6.1f} "
                  f"{s.avg('fh_first_pct'):>6.1f} {s.avg('qnodes') / s.avg('nodes') * 100 if s.avg('nodes') else 0:>6.1f}  {pct_str:>11}")


def parse_dumpstats(output: str) -> dict:
    stats = {}
    for key, pattern in STAT_PATTERNS.items():
        m = re.search(pattern, output)
        if m:
            stats[key] = float(m.group(1))
    # derived
    if "nodes" in stats and "qnodes" in stats and stats["nodes"] > 0:
        stats["q_ratio"] = 100.0 * stats["qnodes"] / stats["nodes"]

    # parse per-depth table
    # format: D  Eval  Move   Time  Nodes    FH   FL   NMP  SEE  PVS
    depth_rows = []
    in_table = False
    for line in output.splitlines():
        if line.strip().startswith("D  Eval"):
            in_table = True
            continue
        if in_table:
            if line.startswith("----") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                depth_rows.append({
                    "depth": int(parts[0]),
                    "eval":  int(parts[1]),
                    "time":  int(parts[3]),
                    "nodes": int(parts[4]),
                })

    if depth_rows:
        # drop last depth (incomplete)
        completed = depth_rows[:-1]
        stats["depth_rows"] = completed

        # cumulative time
        cumulative = 0
        stats["depth_cumtime"] = {}
        for row in completed:
            cumulative += row["time"]
            stats["depth_cumtime"][row["depth"]] = cumulative

        # EBF per depth using sqrt(nodes_d / nodes_d-2)
        stats["depth_ebf"] = {}
        for i in range(2, len(completed)):
            d     = completed[i]["depth"]
            n_d   = completed[i]["nodes"]
            n_d2  = completed[i-2]["nodes"]
            if n_d2 > 0:
                stats["depth_ebf"][d] = (n_d / n_d2) ** 0.5

    return stats

def print_depth_table(raw: dict, engines: list[str], positions: list[tuple]):
    for _, pos_name in positions:
        names = [e.split("/")[-1].replace(".exe", "") for e in engines]

        all_depths = sorted(set(
            d
            for e in engines
            for s in raw[e][pos_name]
            for d in s.get("depth_cumtime", {}).keys()
        ))
        if not all_depths:
            continue

        print(f"\n  Depth table — {pos_name}")
        header = f"  {'D':>3}"
        for name in names:
            header += f"  {name+' ms':>12} {name+' nodes':>12} {name+' EBF':>8}"
        print(header)
        print("  " + "-" * (5 + len(engines) * 34))

        for d in all_depths:
            row = f"  {d:>3}"
            for engine in engines:
                stats_list = raw[engine][pos_name]
                times = [s["depth_cumtime"][d] for s in stats_list if d in s.get("depth_cumtime", {})]
                ebfs  = [s["depth_ebf"][d]     for s in stats_list if d in s.get("depth_ebf", {})]
                nodes = [s["depth_rows"][i]["nodes"] for s in stats_list 
                         for i, r in enumerate(s.get("depth_rows", [])) if r["depth"] == d]
                t_str    = f"{sum(times)/len(times):>12.0f}" if times else f"{'--':>12}"
                n_str    = f"{sum(nodes)/len(nodes):>12,.0f}" if nodes else f"{'--':>12}"
                ebf_str  = f"{sum(ebfs)/len(ebfs):>8.2f}"    if ebfs  else f"{'--':>8}"
                row += f"  {t_str} {n_str} {ebf_str}"
            print(row)

def run_search(engine_path: str, position: str, movetime: int) -> Optional[dict]:
    try:
        proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        if position == "startpos":
            pos_cmd = "position startpos\n"
        else:
            pos_cmd = f"position fen {position}\n"

        commands = f"uci\nisready\n{pos_cmd}go movetime {movetime}\n"
        proc.stdin.write(commands)
        proc.stdin.flush()

        # wait for bestmove
        start = time.time()
        while time.time() - start < (movetime / 1000) + 5:
            line = proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                break

        # request dumpstats and collect output
        proc.stdin.write("dumpstats\n")
        proc.stdin.flush()

        dump_output = []
        start = time.time()
        while time.time() - start < 3:
            line = proc.stdout.readline()
            if not line:
                break
            dump_output.append(line)

            W = 76
            rule = lambda c: c * W  # Returns the string instead of printing it
            if rule('=') in line:
                break

        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.wait(timeout=3)

        return parse_dumpstats("".join(dump_output))

    except Exception as e:
        print(f"  Error running {engine_path}: {e}")
        return None


def benchmark(engines: list[str], positions: list[tuple], movetime: int, runs: int) -> dict:
    # results[engine][pos_name] = SearchStats
    results = {e: {name: SearchStats() for _, name in positions} for e in engines}
    raw     = {e: {name: [] for _, name in positions} for e in engines}

    total = len(engines) * len(positions) * runs
    done = 0

    for pos_fen, pos_name in positions:
        for engine in engines:
            for run in range(runs):
                done += 1
                print(f"[{done}/{total}] {engine} | {pos_name} | run {run+1}/{runs}", end="\r")
                stats = run_search(engine, pos_fen, movetime)
                if stats:
                    raw[engine][pos_name].append(stats)
                    s = results[engine][pos_name]
                    if "nps"          in stats: s.nps.append(stats["nps"])
                    if "nodes"        in stats: s.nodes.append(stats["nodes"])
                    if "qnodes"       in stats: s.qnodes.append(stats["qnodes"])
                    if "depth"        in stats: s.depth.append(stats["depth"])
                    if "eval"         in stats: s.eval.append(stats["eval"])
                    if "tt_return_rate" in stats: s.tt_return_rate.append(stats['tt_return_rate'])
                    if "tt_hit_rate"  in stats: s.tt_hit_rate.append(stats["tt_hit_rate"])
                    if "fh_first_pct" in stats: s.fh_first_pct.append(stats["fh_first_pct"])
                    if "see_pruned"   in stats: s.see_pruned.append(stats["see_pruned"])
                    if "nmp_success"  in stats: s.nmp_success.append(stats["nmp_success"])

    print()
    return results, raw

def load_game_moves(pgn_or_uci_path: str) -> list[str]:
    """Load a game as a list of UCI moves from a simple UCI move file or PGN."""
    moves = []
    with open(pgn_or_uci_path) as f:
        content = f.read().strip()
    
    # simple UCI move list file (one move per line or space separated)
    if not content.startswith("["):
        moves = content.replace("\n", " ").split()
        return moves
    
    # PGN - use python-chess if available
    try:
        import chess.pgn
        import io
        game = chess.pgn.read_game(io.StringIO(content))
        board = chess.Board()
        for move in game.mainline_moves():
            moves.append(move.uci())
            board.push(move)
    except ImportError:
        print("python-chess required for PGN parsing: pip install chess")
        sys.exit(1)
    
    return moves


def run_game_sim(engine_path: str, moves: list[str], movetime: int, side: int = 0) -> list[dict]:
    """
    Feed moves to engine one at a time, recording search stats at each position.
    side: 0=white, 1=black — only search positions where it's this side's turn.
    TT persists across searches since we never quit the engine.
    """
    try:
        proc = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        def send(cmd):
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()

        def wait_for(token, timeout=10):
            start = time.time()
            while time.time() - start < timeout:
                line = proc.stdout.readline().strip()
                if token in line:
                    return line
            return None

        send("uci")
        wait_for("uciok")
        send("isready")
        wait_for("readyok")

        results = []
        played = []  # moves played so far

        for i, move in enumerate(moves):
            current_side = i % 2  # 0=white, 1=black

            if current_side == side:
                # search BEFORE this side's move is played
                pos_cmd = f"position startpos moves {' '.join(played)}" if played else "position startpos"
                send(pos_cmd)
                send(f"go movetime {movetime}")

                # wait for bestmove
                start = time.time()
                while time.time() - start < (movetime / 1000) + 5:
                    line = proc.stdout.readline().strip()
                    if line.startswith("bestmove"):
                        break

                # get stats
                send("dumpstats\nisready\n")
                dump_output = []
                start = time.time()
                while time.time() - start < 3:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    if line.strip() == "readyok":
                        break
                    dump_output.append(line)

                stats = parse_dumpstats("".join(dump_output))
                stats["move_num"] = len(played)
                stats["ply"] = i
                results.append(stats)

            played.append(move)  # always append AFTER search

        send("quit")
        proc.wait(timeout=3)
        return results

    except Exception as e:
        print(f"  Error in game sim for {engine_path}: {e}")
        return []


def print_game_sim_results(results: dict, engines: list[str]):
    baseline = engines[0]

    def avg(stats_list, key):
        vals = [s[key] for s in stats_list if key in s]
        return sum(vals) / len(vals) if vals else 0

    names = [e.split("/")[-1].replace(".exe", "") for e in engines]

    # --- Summary row ---
    print(f"\n{'='*70}")
    print(f"  Game Simulation Results (TT warm)")
    print(f"{'='*70}")
    base_nps = avg(results[baseline], "nps")
    print(f"  {'Engine':<20} {'NPS':>10} {'Nodes':>10} {'QNodes':>10} {'Depth':>6} {'TT%':>6} {'FH1%':>6}  {'vs baseline':>11}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*6}  {'-'*11}")
    for engine, name in zip(engines, names):
        s = results[engine]
        enps = avg(s, "nps")
        pct = ((enps / base_nps) - 1) * 100 if base_nps and engine != baseline else None
        pct_str = f"{pct:+.1f}%" if pct is not None else "baseline"
        print(f"  {name:<20} {enps:>10,.0f} {avg(s,'nodes'):>10,.0f} {avg(s,'qnodes'):>10,.0f}"
              f"{avg(s,'depth'):>6.1f} {avg(s,'tt_hit_rate'):>6.1f} "
              f"{avg(s,'fh_first_pct'):>6.1f}  {pct_str:>11}")

    # --- Per-ply table: depth and NPS per engine ---
    # align by move_num
    all_plies = sorted(set(s["move_num"] for e in engines for s in results[e]))

    # index results by move_num
    indexed = {
        engine: {s["move_num"]: s for s in results[engine]}
        for engine in engines
    }

    col_w = 14  # width per engine column pair (depth + nps)

    print(f"\n  Per-ply breakdown")

    # header
    header = f"  {'Ply':>4} "
    for name in names:
        header += f"  {name+' depth':>{col_w}} {name+' NPS':>{col_w}}"
    header += f"  {'Δdepth':>{col_w}} {'ΔNPS%':>{col_w}}"
    print(header)

    for ply in all_plies:
            row = f"  {ply:>4} "
            base_s = indexed[baseline].get(ply)
            
            for engine in engines:
                s = indexed[engine].get(ply)
                if s:
                    depth = int(s.get("depth", 0))
                    nps   = int(s.get("nps", 0))
                    row += f"  {depth:>{col_w}} {nps:>{col_w},}"
                else:
                    row += f"  {'--':>{col_w}} {'--':>{col_w}}"

            # delta columns (candidate vs baseline)
            if len(engines) > 1:
                cand_s = indexed[engines[-1]].get(ply)
                if base_s and cand_s:
                    d_depth = int(cand_s.get("depth", 0)) - int(base_s.get("depth", 0))
                    base_nps_ply = base_s.get("nps", 0)
                    cand_nps_ply = cand_s.get("nps", 0)
                    pct = ((cand_nps_ply / base_nps_ply) - 1) * 100 if base_nps_ply else 0
                    d_depth_str = f"{d_depth:+d}" if d_depth != 0 else "="
                    row += f"  {d_depth_str:>{col_w}} {pct:>{col_w}.1f}%"
                else:
                    row += f"  {'--':>{col_w}} {'--':>{col_w}}"

            print(row)


if __name__ == "__main__":
    engines = ENGINES
    move_time_ms = MOVETIME_MS
    runs = RUNS_PER_POS
    positions = POSITIONS

    if "--engines" in sys.argv:
        idx = sys.argv.index("--engines")
        # collect all args after --engines until next flag
        engines = []
        for arg in sys.argv[idx+1:]:
            if arg.startswith("--"): break
            engines.append(arg)

    if "--time" in sys.argv:
        idx = sys.argv.index("--time")
        move_time_ms = int(float(sys.argv[idx+1]) * 1000)  # accept seconds like sprt

    if "--runs" in sys.argv:
        idx = sys.argv.index("--runs")
        runs = int(sys.argv[idx+1])

    if "--positions" in sys.argv:
        idx = sys.argv.index("--positions")
        path = sys.argv[idx+1]
        positions = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                # format: "fen | name" or just "fen"
                if "|" in line:
                    fen, name = line.split("|", 1)
                    positions.append((fen.strip(), name.strip()))
                else:
                    positions.append((line, line[:40]))

    if "--game_sim" in sys.argv:
        idx = sys.argv.index("--game_sim")
        game_path = GAME_SIM_PATH
        side_arg = 0
        if "--side" in sys.argv:
            side_arg = int(sys.argv[sys.argv.index("--side")+1])
        
        moves = load_game_moves(game_path)
        print(f"Game sim: {len(moves)} moves, searching as {'white' if side_arg==0 else 'black'}\n")
        
        game_results = {}
        for engine in engines:
            print(f"Running {engine}...")
            game_results[engine] = run_game_sim(engine, moves, move_time_ms, side=side_arg)
        
        print_game_sim_results(game_results, engines)
    else:
        results, raw = benchmark(engines, positions, move_time_ms, runs)
        print_results(results, engines, positions)
        print_depth_table(raw, engines, positions)