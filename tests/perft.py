#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import time
import sqlite3
from pathlib import Path
from data import etl
from datetime import datetime, timezone
import platform

system = platform.system()

# paths
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if system == "Windows": STOCKFISH = r"C:\Users\maxol\chess\engines\stockfish\stockfish-windows-x86-64-avx2.exe"
elif system == "Darwin": STOCKFISH = "engines/stockfish/stockfish-macos-m1-apple-silicon"

# -----------------------------
# Load positions
# -----------------------------
def load_positions(path):
    """
    Load perft positions from an epd file.

    Ignores any depth counts in the line (D1..Dn) and just returns the FEN.
    """
    positions = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Take only the first part before the first semicolon
            fen = line.split(";")[0].strip()
            positions.append(fen)

    return positions

# -----------------------------
# Stockfish perft
# -----------------------------
def sf_perft(stockfish_path, fen, depth):
    commands = (
        "uci\n"
        "isready\n"
        f"position fen {fen}\n"
        f"go perft {depth}\n"
        "quit\n"
    )

    sf = subprocess.Popen(
        [stockfish_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    )

    stdout, _ = sf.communicate(commands, timeout=30)

    for line in stdout.splitlines():
        if line.startswith("Nodes searched:"):
            return int(line.split(":")[1].strip())

    raise RuntimeError("Stockfish did not report perft node count")


# -----------------------------
# Engine perft
# -----------------------------
def my_perft(engine_proc, fen, depth):
    start = time.perf_counter()

    engine_proc.stdin.write(f"position fen {fen}\n")
    engine_proc.stdin.write(f"perft {depth}\n")
    engine_proc.stdin.flush()

    nodes = None

    while True:
        line = engine_proc.stdout.readline()
        if not line:
            break

        line = line.strip()

        if line.startswith("Nodes searched:"):
            nodes = int(line.split(":")[1].strip())
            break

    if nodes is None:
        raise RuntimeError("Engine did not return perft node count")

    time_ms = int((time.perf_counter() - start) * 1000)
    return nodes, time_ms


# -----------------------------
# Main
# -----------------------------
def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="Perft verifier + DB logger")
        parser.add_argument("--engine", required=True, help="Path to engine binary")
        parser.add_argument("--stockfish", default=STOCKFISH, help="Path to Stockfish binary")
        parser.add_argument("--positions", default=PROJECT_ROOT / "bin" / "test_positions" / "perft.epd", help="File containing FEN positions")
        parser.add_argument("--depth", type=int, required=True, help="Perft depth")

        args = parser.parse_args()

    positions = load_positions(args.positions)
    if not positions:
        print("❌ No positions loaded")
        sys.exit(1)

    # -----------------------------
    # DB setup
    # -----------------------------
    system = platform.system()
    if system == "Windows": cnxn = sqlite3.connect('F:/databases/chess.db')
    elif system == "Darwin": cnxn = sqlite3.connect(Path.home() / "Documents/databases/chess.db")
    cnxn.row_factory = sqlite3.Row

    engine_meta = etl.probe_engine_metadata(args.engine)
    engine_id = etl.get_engine_id(cnxn, version=engine_meta["version"])

    # auto-register if not found
    if engine_id is None:
        print(f"[PERFT] Engine {engine_meta['version']} not registered, registering now...")
        engine_id = etl.register_engine(cnxn, {"engine_path": args.engine})

    experiment_id = etl.start_experiment(
        cnxn,
        "PERFT",
        engine_id
    )

    print(f"[PERFT] Experiment ID: {experiment_id}")

    # -----------------------------
    # Launch engine
    # -----------------------------
    engine = subprocess.Popen(
        [args.engine],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    engine.stdin.write("uci\n")
    engine.stdin.flush()
    while True:
        line = engine.stdout.readline()
        if line.strip() == "uciok":
            break

    engine.stdin.write("isready\n")
    engine.stdin.flush()
    while True:
        line = engine.stdout.readline()
        if line.strip() == "readyok":
            break

    time.sleep(.5)  # Give engine time to initialize

    # -----------------------------
    # Run tests
    # -----------------------------
    for idx, fen in enumerate(positions):
        sf_nodes = sf_perft(args.stockfish, fen, args.depth)
        my_nodes, time_ms = my_perft(engine, fen, args.depth)

        correct = (sf_nodes == my_nodes)
        status = "OK" if correct else "MISMATCH"

        print(f"[{idx}] {status}")
        print(f"SF: {sf_nodes} | Mine: {my_nodes} | {time_ms} ms")

        etl.log_perft(
            cnxn,
            {
                'experiment_id':experiment_id,
                'fen':fen,
                'depth':args.depth,
                'nodes':my_nodes,
                'expected_nodes':sf_nodes,
                'correct':correct,
                'time_ms':time_ms
            }
        )

        if not correct:
            print(f"  FEN: {fen}")
            cnxn.commit()
            #engine.kill()
            #sys.exit(1)

    cnxn.commit()

    etl.update_experiment(
        cnxn, 
        experiment_id,
        {"end_time_utc": datetime.now(timezone.utc).isoformat()}
    )

    engine.stdin.write("quit\n")
    engine.stdin.flush()
    engine.wait(timeout=2)

    #print("✅ All perft tests passed")


if __name__ == "__main__":
    main()
