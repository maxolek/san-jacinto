import argparse
import subprocess
import os
import json
import chess
import sqlite3
from data import etl
from pathlib import Path
from datetime import datetime, timezone
import platform

system = platform.system()

# paths
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
# log paths
LOGS_DIR = PROJECT_ROOT / "logs"
STS_LOGS_DIR = LOGS_DIR / "sts_logs"
GAME_JSON = STS_LOGS_DIR / "game.jsonl"
SEARCH_JSON = STS_LOGS_DIR / "search.jsonl"
TIMING_JSON = STS_LOGS_DIR / "timing.jsonl"
ROOT_MOVES_JSON = STS_LOGS_DIR / "root_moves.jsonl"
STS_JSON = STS_LOGS_DIR / "sts_suite.jsonl"

# --------------------------
#       logging
# --------------------------

def upload_logs(args):
    if system == "Windows": cnxn = sqlite3.connect('F:/databases/chess.db')
    elif system == "Darwin": cnxn = sqlite3.connect(Path.home() / "Documents/databases/chess.db")

    cnxn.row_factory = sqlite3.Row 

    # probe metadata
    meta = etl.probe_engine_metadata(args.engine)
    engine_id = etl.get_engine_id(cnxn, version=meta["version"])

    # auto-register if not found
    if engine_id is None:
        print(f"[STS] Engine {meta['version']} not registered, registering now...")
        engine_id = etl.register_engine(cnxn, {"engine_path": args.engine})

    # log
    sts_id = etl.start_experiment(
        cnxn, 
        "STS",
        engine_id
    )

    try:
        etl.bulk_log_sts(cnxn, STS_JSON, sts_id, **vars(args))

        # no games to map search to, so no game_map like in SPRT
        etl.bulk_log_search_and_timing(
            cnxn,
            SEARCH_JSON,
            {},
            timing_path = TIMING_JSON,
            sts_id = sts_id,
            engine_id = engine_id,
            root_moves_path = ROOT_MOVES_JSON
        )
        ingestion_ok = True
    except Exception as e:
        print(f"[DATA] STS ingestion failed: {e}")
        ingestion_ok = False

    etl.update_experiment(
        cnxn, 
        sts_id, 
        {"end_time_utc": datetime.now(timezone.utc).isoformat()}
    )

    # Only clear log directory if ingestion succeeded
    if ingestion_ok:
        etl.clear_log_dir(STS_LOGS_DIR)
    else:
        print("[DATA] Log files preserved due to ingestion failure.")
    print(f"[DATA] Logging completed for STS {sts_id}")

# --------------------------
# Run engine and get eval/bestmove
# --------------------------
def run_eval(engine, fen, depth=8, time=None):
    """Send position and go eval command, return (score, bestmove)."""
    # position + eval
    engine.stdin.write(f"position fen {fen}\n".encode())
    if time is not None: 
        engine.stdin.write(f"go eval movetime {time}\n".encode())
    else: 
        engine.stdin.write(f"go eval depth {depth}\n".encode())
    engine.stdin.flush()

    score, bestmove = None, None
    while True:
        line = engine.stdout.readline().decode().strip()
        if line.startswith("eval"):
            parts = line.split()
            try:
                score_idx = parts.index("eval") + 1
                bestmove_idx = parts.index("bestmove") + 1
                score = int(parts[score_idx])
                bestmove = parts[bestmove_idx]
            except (ValueError, IndexError):
                pass
            break
    return score, bestmove

# --------------------------
# Parse EPD line
# --------------------------
def parse_epd_line(line):
    """
    Parse an EPD line into (fen, ops dict, expected_moves, avoid_moves)
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None, {}, [], []

    # Split on semicolon for EPD operations
    parts = line.split(";")
    main_part = parts[0].strip()  # FEN + optional bm/am
    ops_parts = parts[1:]

    tokens = main_part.split()
    if len(tokens) < 4:
        return None, {}, [], []

    # Standard FEN fields
    pieces = tokens[0]
    turn = tokens[1]
    castling = tokens[2]
    en_passant = tokens[3]
    halfmove = "0"
    fullmove = "1"
    fen = f"{pieces} {turn} {castling} {en_passant} {halfmove} {fullmove}"

    # Initialize moves
    expected_moves = []
    avoid_moves = []

    # Extract expected moves after 'bm'
    if "bm" in tokens:
        bm_index = tokens.index("bm")
        # Take all tokens until the next token starts with ';' or end of list
        for t in tokens[bm_index + 1:]:
            if t.endswith(";"):
                expected_moves.append(t.rstrip(";"))
                break
            expected_moves.append(t)

    # Extract avoid moves after 'am'
    if "am" in tokens:
        am_index = tokens.index("am")
        for t in tokens[am_index + 1:]:
            if t.endswith(";"):
                avoid_moves.append(t.rstrip(";"))
                break
            avoid_moves.append(t)

    # Parse ops (id, ce, c0, etc.)
    ops = {}
    for token in ops_parts:
        token = token.strip()
        if not token:
            continue
        if token.startswith("id"):
            ops["id"] = token.split(" ", 1)[1].strip('" ')
        elif token.startswith("ce"):
            try:
                ops["ce"] = int(token.split()[1])
            except:
                ops["ce"] = None
        elif token.startswith("c0"):
            ops["c0"] = token.split(" ", 1)[1].strip('" ')

    return fen, ops, expected_moves, avoid_moves


def collect_epd_files(paths):
    epd_files = set()

    for p in paths:
        path = PROJECT_ROOT / Path(p)

        if path.is_file() and path.suffix.lower() == ".epd":
            epd_files.add(path.resolve())

        elif path.is_dir():
            for f in path.rglob("*.epd"):
                epd_files.add(f.resolve())

        else:
            print(f"⚠ Skipping unknown path: {path}")

    # return as strings
    return sorted(str(p) for p in epd_files)

# --------------------------
# Convert expected moves to UCI
# --------------------------
import re
SAN_CLEAN_RE = re.compile(r"[!?+#]")

def moves_to_uci(board, expected_moves):
    """
    Convert expected moves (SAN, capture SAN, or square-only hints) into UCI moves.
    """
    uci_moves = []

    for raw in expected_moves:
        m =raw.strip().rstrip(",;").strip()
        m = SAN_CLEAN_RE.sub("", m)  # remove + # ! ?

        # 1️⃣ Try SAN (handles captures, promotions, castles, etc.)
        try:
            move = board.parse_san(m)
            uci_moves.append(move.uci())
            continue
        except ValueError:
            pass

        # 2️⃣ Square-only fallback (e.g. "f5", "e4")
        if re.fullmatch(r"[a-h][1-8]", m):
            for legal in board.legal_moves:
                if chess.square_name(legal.to_square) == m:
                    uci_moves.append(legal.uci())
                    break
            else:
                print(f"⚠ Could not match square move '{raw}' on {board.fen()}")
            continue

        # 3️⃣ UCI fallback (some EPDs give e2e4 directly)
        if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", m):
            move = chess.Move.from_uci(m)
            if move in board.legal_moves:
                uci_moves.append(m)
                continue

        # ❌ Failed
        print(f"⚠ Could not parse expected move '{raw}' on board {board.fen()}")

    return uci_moves

# --------------------------
# Run STS for a single file
# --------------------------
def run_sts_file(engine_path, epd_file, depth=8, time = None, log_file = None):
    engine_dir = os.path.dirname(os.path.abspath(engine_path))
    engine = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    total, correct = 0, 0
    diffs = []
    category_stats = {}

    # logging options
    engine.stdin.write(f"setoption name log_dir value {STS_LOGS_DIR}\n".encode())
    engine.stdin.write(f"setoption name timer_logging value true\n".encode())
    engine.stdin.write(f"setoption name stats_logging value true\n".encode())
    engine.stdin.write(f"setoption name uci_logging value true\n".encode())
    engine.stdin.write(f"setoption name game_logging value false\n".encode())

    with open(epd_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    for i, line in enumerate(lines, 1):
        fen, ops, expected_moves_raw, avoid_moves_raw = parse_epd_line(line)
        if not fen:
            print(f"[{i}/{len(lines)}] Skipping malformed line: {line}")
            continue

        cat = ops.get("id", "Unknown")

        try:
            board = chess.Board(fen)
        except ValueError as e:
            print(f"[{i}/{len(lines)}] Invalid FEN: {fen} | Error: {e}")
            continue

        # Convert expected moves to UCI
        expected_moves = moves_to_uci(board, expected_moves_raw)
        avoid_moves = moves_to_uci(board, avoid_moves_raw)

        print(f"[STS] {epd_file}: position {i}")
        score, bestmove = run_eval(engine, fen, depth, time)
        total += 1

        move_ok = (expected_moves and bestmove in expected_moves) or (avoid_moves and bestmove not in avoid_moves)
        if move_ok:
            correct += 1
        category_stats.setdefault(cat, {"total": 0, "correct": 0})
        category_stats[cat]["total"] += 1
        if move_ok:
            category_stats[cat]["correct"] += 1

        expected_score = ops.get("ce", None)
        score_diff = None
        if expected_score is not None and score is not None:
            score_diff = score - expected_score
            diffs.append(abs(score_diff))

        print(f"[{i}/{len(lines)}] {cat}")
        print(f"  FEN:      {fen}")
        if expected_moves:
            print(f"  Expected: moves={expected_moves}, score={expected_score}")
        elif avoid_moves:
            print(f"  Avoid:    moves={avoid_moves}, score={expected_score}")
        print(f"  Engine:   move={bestmove}, score={score}")
        print("  ✅ Move correct" if move_ok else "  ❌ Move incorrect")
        if score_diff is not None:
            print(f"  Score diff: {score_diff}")
        print("-" * 50)

        record = {
            "epd_file": epd_file, 
            "index": i, 
            "category": cat, 
            "fen": fen, 
            "expected_moves": expected_moves, 
            "expected_score": expected_score,
            "engine_move": bestmove,
            "engine_score": score, 
            "avoid_moves": avoid_moves,
            "move_ok": move_ok, 
            "score_diff": score_diff
        }
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()

    engine.terminate()
    engine.wait(timeout=1)

    file_summary = {
        "file": epd_file,
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct / total if total else 0,
        "avg_diff": sum(diffs)/len(diffs) if diffs else None,
        "per_category": category_stats
    }

    return file_summary

# --------------------------
# Main
# --------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="STS (test positions) runner")

    parser.add_argument("--engine", default="./src/tomahawk.exe", help="Path to UCI engine")
    parser.add_argument("--time", type=int, default=None, help = "Move time in MS")
    parser.add_argument("--depth", type=int, default=None, help = "Search depth")
    parser.add_argument("--sts", nargs="+", default=["./bin/STS"], help="One or more EPD files and/or folders")
    #parser.add_argument("--log_dir", default="./logs/sts_logs/", help="Output log directory")

    return parser.parse_args()


def main(args=None):
    if args is None: 
        args = parse_args()

    #log_dir = os.path.abspath(args.log_dir)
    #os.makedirs(log_dir, exist_ok=True)
    #sts_log_path = os.path.join(log_dir, "sts_suite.jsonl")

    log_f = open(STS_JSON, "w", encoding="utf-8")
    print(f"Results logged to {STS_JSON}")

    epd_files = collect_epd_files(args.sts)
    if not epd_files:
        print("❌ No EPD files found.")
        return
    
    global_total, global_correct = 0, 0
    all_summaries = []

    for epd_file in epd_files:
        print(f"\n=== Running STS file: {epd_file} ===")
        summary = run_sts_file(args.engine, epd_file, args.depth, args.time, log_f)
        all_summaries.append(summary)
        print(f"File summary: {summary['correct']}/{summary['total']} correct ({summary['accuracy']:.1f}%)")
        global_total += summary['total']
        global_correct += summary['correct']

    global_acc = 100.0 * global_correct / global_total if global_total else 0
    print("\n=== Global STS Summary ===")
    print(f"Total tests: {global_total}")
    print(f"Total correct: {global_correct} ({global_acc:.1f}%)")

    # upload to db
    log_f.close()
    upload_logs(args)
    

if __name__ == "__main__":
    main()
