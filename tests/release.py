#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
import platform

sys.path.insert(0, str(Path(__file__).parent.parent))
from data import etl
from . import perft, sprt, sts, tournament
from tuning import spsa_tuner
from utils import uci_probe

system = platform.system()

# ============================================================
# Project layout: run from this repo's root (python -m tests.release);
# the engine is built from the tomahawk checkout next to it
# ============================================================

# paths
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
TOMAHAWK_ROOT = PROJECT_ROOT.parent / "tomahawk"

ENGINES_DIR = os.path.join(TOMAHAWK_ROOT, "engines")  # tomahawk's CMake writes <version>.exe here

LOGS_DIR    = os.path.join(PROJECT_ROOT, "logs")
TESTS_DIR   = os.path.join(PROJECT_ROOT, "tests")
BIN_DIR     = os.path.join(PROJECT_ROOT, "bin")

DB_PATH = str(etl.paths.RAW_DB)  # CHESS_RAW_DB overrides the default

DEV_DIR     = os.path.join(ENGINES_DIR, "dev")
PROD_DIR    = os.path.join(ENGINES_DIR, "prod")

SPRT_LOG_DIR  = os.path.join(LOGS_DIR, "sprt_logs")
STS_LOG_DIR   = os.path.join(LOGS_DIR, "sts_logs")
PERFT_LOG_DIR = os.path.join(LOGS_DIR, "perft")

STOCKFISH = os.path.join(
    ENGINES_DIR,
    "stockfish",
    "stockfish-windows-x86-64-avx2.exe"
)

# ============================================================
# Build
# ============================================================

def run_make(args, build_type):
    """configure + build with the given cmake_build_type ('DEV' or 'Release')"""

    print(f"[BUILD] Compiling engine VERSION={args.version}")

    build_dir = os.path.join(TOMAHAWK_ROOT, f"build_{build_type.lower()}")

    configure_cmd = [
        "cmake",
        "-S", str(TOMAHAWK_ROOT),  # engine source dir
        "-B", build_dir,           # build dir
        "-G", "MinGW Makefiles",   # generator
        "-DCMAKE_C_COMPILER=g++",
        "-DCMAKE_CXX_COMPILER=g++",
        f"-DVERSION={args.version}", # version
        f"-DCMAKE_BUILD_TYPE={build_type}", # release build type
    ]

    build_cmd = [
        "cmake",
        "--build", build_dir,
        #"--config", "Release",     #  MSVC
        "--parallel",
    ]

    try:
        shutil.rmtree(build_dir, ignore_errors=True)  # clean build
        subprocess.check_call(configure_cmd)
        subprocess.check_call(build_cmd)
    except subprocess.CalledProcessError:
        sys.exit("[BUILD] ❌ build failed")

    dest_dir = DEV_DIR if build_type == "DEV" else PROD_DIR
    dest_exe = os.path.join(dest_dir, f"{args.version}.exe")
    if not os.path.exists(dest_exe):
        sys.exit(f"[BUILD] ❌ expected binary not found at {dest_exe} after {build_type} build")

    # how this binary was built (commit, dirty state, compiler, flags), next to it
    info = etl.write_build_info(dest_exe, TOMAHAWK_ROOT, build_type, build_dir, version=args.version)
    git = info.get("git") or {}
    print(f"[BUILD] provenance: commit {git.get('commit', '?')[:12]}{' (dirty)' if git.get('dirty') else ''}, "
          f"sha256 {info['binary_sha256'][:12]}")
 
    print(f"[BUILD] ✅ {build_type} build -> {dest_exe}")
    return dest_exe

# ============================================================
# SPRT
# ============================================================

def _sprt_namespace(args, tc, threads, params=None, nodes=None):
    return argparse.Namespace(
        engine_a=args.prod_engine,
        engine_b=args.base_engine,
        concurrency=args.concurrency,
        depth=args.sprt_depth,
        time=args.sprt_time,
        tc=tc,
        nodes=nodes,
        elo0=args.elo0,
        elo1=args.elo1,
        alpha=args.alpha,
        beta=args.beta,
        max_games=args.sprt_games,
        book=args.opening_book,
        book_depth=args.sprt_book_depth,
        logroot=SPRT_LOG_DIR,
        fastchess=args.fastchess,
        log=True,
        plot=True,
        params=params or {},
        threads=threads,
        no_pin=args.no_pin,
        tb=args.tb,
        tbpieces=args.tbpieces,
    )


def run_sprt(args):
    """One SPRT per --sprt_threads count; passes only if every one passes."""
    print("[SPRT] Running SPRT")
    accepted = True
    for threads in args.sprt_threads:
        print(f"[SPRT] {threads} thread(s)")
        result = sprt.main(_sprt_namespace(args, args.sprt_tc, threads, nodes=args.sprt_nodes))
        accepted = accepted and result["accepted"]
    return accepted


def run_sprt_gate(args, tc, params=None, threads=1):
    print("\n============================")
    print(f"[SPRT] Gate TC: {tc}, {threads} thread(s)")
    print("============================\n")

    return sprt.main(_sprt_namespace(args, tc, threads, params))

def run_gated_sprt(args, params=None):

    print("\n################################")
    print("# SPRT GATING START")
    print("################################\n")


    gates = [(tc, threads) for tc in args.sprt_gates for threads in args.sprt_threads]
    for i, (tc, threads) in enumerate(gates):
        print(f"================================")
        print(f"SPRT GATE {i+1}/{len(gates)}")
        print(f"TC: {tc}, threads: {threads}")
        print(f"================================")

        result = run_sprt_gate(args, tc, params, threads)
        if not result["accepted"]:
            print(
                f"[SPRT] ❌ Failed gate {tc} ({threads} thread(s)). "
                "Candidate rejected."
            )
            return False
        print( f"[SPRT] ✅ Passed gate {tc} ({threads} thread(s))")

    print("[SPRT] 🎉 Candidate passed all gates")
    return True

# ============================================================
# TOURNAMENT
# ============================================================

def run_tournament(args, cnxn, engine_id):
    print("[TOURNAMENT] Running tournament")

    tournament_args = argparse.Namespace(
        engine=args.dev_engine,
        fastchess=args.fastchess,
        opening_book=args.opening_book,
        tournament_games=args.tournament_games,
        tournament_tc=args.tournament_tc,
        tournament_engines=args.tournament_engines,
        first_engine=False,
        concurrency=args.concurrency,
        no_pin=args.no_pin,
        round_robin=args.tournament_round_robin,
        tb=args.tb,
        tbpieces=args.tbpieces,
    )

    tournament.run_tournament(tournament_args, cnxn, engine_id)

# ============================================================
# STS
# ============================================================

def run_sts(args):
    print("[STS] Running STS")

    sts_args = argparse.Namespace(
        engine=args.dev_engine,
        time=args.sts_time,
        depth=args.sts_depth,
        sts=args.sts_files,
        log_dir=STS_LOG_DIR
    )

    sts.main(sts_args)

# ============================================================
# PERFT
# ============================================================

def run_perft(args):
    print("[PERFT] Running PERFT")

    perft_args = argparse.Namespace(
        engine=args.prod_engine, 
        stockfish=STOCKFISH, 
        positions=args.perft_positions,
        depth=args.perft_depth
    )

    perft.main(perft_args)

# ============================================================
# TUNING
# ============================================================

def run_tune(args):
    print("[TUNING] Running SPSA")   

    # through the tuner's own parser, so every option it reads gets its default
    tune_args = spsa_tuner.parse_args([
        "--engine", args.dev_engine,
        "--baseline", args.dev_engine,
        "--fastchess", args.fastchess,
        "--tc", "0:20+.2",
        "--iterations", "5000",
        "--games-per-iter", "4",
        "--concurrency", "4",
        "--output", os.path.join(LOGS_DIR, "dev_logs", "spsa_results.json"),
        "--plot",
    ] + (["--no-pin"] if args.no_pin else []))

    return spsa_tuner.main(tune_args)

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("Release pipeline")

    parser.add_argument("--version", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", required=True)
    
    parser.add_argument("--fastchess", default=str(sprt.FASTCHESS), help="Path to the fastchess executable\nUsed for game testing")
    parser.add_argument("--no-pin", action="store_true", help="Don't give each game its own cpus in the engines' Affinity order")

    # SPRT
    parser.add_argument("--sprt", action="store_true", help="Run SPRT")
    parser.add_argument("--base_engine", default=os.path.join(PROD_DIR, "0.0.0.exe"))
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent games for SPRT")
    parser.add_argument("--elo0", type=int, default=0)
    parser.add_argument("--elo1", type=float, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--sprt_tc", type=str)
    parser.add_argument("--sprt_time", type=float)
    parser.add_argument("--sprt_depth", type=int)
    parser.add_argument("--sprt_nodes", type=int,
                        help="Nodes per move for --sprt (gates use time controls); the engine must honour go nodes")
    parser.add_argument("--sprt_threads", type=int, nargs="+", default=[1],
                        help="Run every SPRT (and gate) at each of these thread counts, e.g. 1 4")
    parser.add_argument("--sprt_games", type=int, default=1000)
    parser.add_argument("--opening_book", default=os.path.join(BIN_DIR, "opening_books", "8moves_v3.pgn"))
    parser.add_argument("--sprt_book_depth", type=int, default=8)
    parser.add_argument("--tb", default=None,
                        help="Syzygy tablebase dir(s) for adjudication in SPRT and tournament (WDL files are enough)")
    parser.add_argument("--tbpieces", type=int, default=None)
    # SPRT gates
    parser.add_argument(
        "--sprt_gate",
        action="store_true",
        help="Run multi-stage SPRT gating"
    )

    parser.add_argument(
        "--sprt_gates",
        nargs="+",
        default=[
            "10+.1",
            "60+.6"
        ],
        help="SPRT gate time controls"
    )

    # STS
    parser.add_argument("--sts_files", nargs="+")
    parser.add_argument("--sts_time", type=float, default=5000)
    parser.add_argument("--sts_depth", type=int)

    # PERFT
    parser.add_argument("--perft", action="store_true")
    parser.add_argument("--perft_positions", default=os.path.join(BIN_DIR, "test_positions", "perft.epd"))
    parser.add_argument("--perft_depth", type=int, default=5)

    # TOURNAMENT (Elo rating)
    parser.add_argument("--tournament", action="store_true", help="Run rating tournament against all engines in DB")
    parser.add_argument("--tournament_tc", nargs="+", default=["blitz"],
                        choices=["ultra_fast", "bullet", "blitz", "rapid", "classical"],
                        help="Time control categories for rating tournament")
    parser.add_argument("--tournament_games", type=int, default=100,
                        help="Games per opponent per time control")
    parser.add_argument("--tournament_engines", type=int, default=1,
                        help="Number of earlier versions to play (one build each, Prod preferred)")
    parser.add_argument("--tournament_round_robin", action="store_true",
                        help="Also play the opponents against each other (default: gauntlet)")

    # TUNING
    parser.add_argument("--tune", action="store_true", help="Run SPSA tuning on search parameters after release. Re-run SPRT to see if improvement.")

    args = parser.parse_args()

    #args.engine = os.path.join(ENGINES_DIR, f"{args.version}.exe")

    cnxn = etl.connect_raw(DB_PATH)

    dev_exe = run_make(args, "DEV")
    prod_exe = run_make(args, "Prod")

    args.dev_engine = dev_exe 
    args.prod_engine = prod_exe
    args.engine = args.dev_engine # tournament.py expects args.engine as the candidate
                                  # all other aspects of this run are explicit

    # Register both builds: each is its own engine (binary + net hash), with the
    # provenance written next to it by run_make and the options probed via UCI
    engine_id = etl.resolve_engine(cnxn, dev_exe, name=args.name, description=f"{args.description} (DEV build)")
    etl.resolve_engine(cnxn, prod_exe, name=args.name, description=f"{args.description} (Prod build)")

    # Checked before anything runs, not when the first multi-threaded SPRT starts
    if (args.sprt or args.tune) and max(args.sprt_threads) > 1:
        for exe in (prod_exe, args.base_engine):
            uci_probe.require_threads(exe, max(args.sprt_threads))

    print('\n============================\n')

    if (args.perft):        run_perft(args)         # prod
    if (args.sprt):                                 # prod   
        if args.sprt_gate:
            passed = run_gated_sprt(args)
            if not passed: 
                print("[PIPELINE] Release aborted")
                return 
        elif not run_sprt(args):
            print("[PIPELINE] SPRT did not pass")
    if (args.sts_files):    run_sts(args)           # dev
    if (args.tournament):   run_tournament(args, cnxn, engine_id)    # dev
    if (args.tune):                                 # dev
        new_params = run_tune(args)
        passed = run_gated_sprt(args, new_params)
        if not passed: print("[PIPELINE] Tuning failed to prove improvement over release (which showed improvement over prior version).")

    print(f"[PIPELINE] ✅ release complete")
    print(f"[PIPELINE]     dev:   {dev_exe}")
    print(f"[PIPELINE]     prod:  {prod_exe}")

if __name__ == "__main__":
    main()
