#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import sqlite3
import re
from pathlib import Path
import time
from datetime import datetime, timezone
import platform
import threading
import json
import signal
import queue

system = platform.system()

# paths
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from utils.pentanomial import FINISHED_GAME, PentanomialSPRT
# log paths
LOGS_DIR = PROJECT_ROOT / "logs"
SPRT_LOG_DIR = LOGS_DIR / "sprt_logs"
GAME_JSON = SPRT_LOG_DIR / "game.jsonl"
SEARCH_JSON = SPRT_LOG_DIR / "search.jsonl"
TIMING_JSON = SPRT_LOG_DIR / "timing.jsonl"
ROOT_MOVES_JSON = SPRT_LOG_DIR / "root_moves.jsonl"

def parse_cutechess_output(output, candidate_name="Candidate", elo0=0, elo1=10,
                          alpha=0.05, beta=0.05):
    """Rebuild pentanomial evidence from game results, never cutechess summaries."""
    state = PentanomialSPRT(elo0, elo1, alpha, beta)
    for line in output.splitlines():
        match = FINISHED_GAME.search(line)
        if match:
            number, white, black, result = match.groups()
            white = "Candidate" if white == candidate_name else white
            black = "Candidate" if black == candidate_name else black
            state.add_game(int(number), white, black, result)
    return state.snapshot()


def stop_cutechess(proc, force=False):
    """Stop only this run's process tree, including the engine children.

    Cutechess handles SIGINT on POSIX. Windows has no equivalent targeted
    SIGINT delivery; taskkill /T stops the dedicated tree there. Our result
    journal is flushed before shutdown, independent of PGN finalization.
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode and proc.poll() is None:
            raise RuntimeError(f"Could not stop cutechess process tree: {result.stderr}")
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGINT)
        except ProcessLookupError:
            pass


def run_cutechess(cmd, state, run_dir, plotter=None):
    """Journal raw results and ordered pair snapshots, with or without a GUI."""
    run_dir = Path(run_dir)
    process_options = (dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                       if os.name == "nt" else dict(start_new_session=True))
    stopping_at = None
    failure = None
    with (open(run_dir / "cutechess_stdout.log", "w", encoding="utf-8") as transcript,
          open(run_dir / "games.jsonl", "w", encoding="utf-8") as games,
          open(run_dir / "stats.jsonl", "w", encoding="utf-8") as history,
          subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1, **process_options) as proc):
        line_queue = queue.Queue()

        def reader():
            try:
                for line in proc.stdout:
                    line_queue.put(line)
            finally:
                line_queue.put(None)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        try:
            done = False
            while not done:
                # Bound work per pass so high concurrency cannot starve the GUI.
                for _ in range(128):
                    try:
                        line = line_queue.get(timeout=0.05)
                    except queue.Empty:
                        break
                    if line is None:
                        done = True
                        break
                    print(line, end="")
                    transcript.write(line)
                    transcript.flush()
                    match = FINISHED_GAME.search(line)
                    if not match:
                        continue
                    number, white, black, outcome = match.groups()
                    games.write(json.dumps(dict(game=int(number), white=white, black=black,
                                                result=outcome)) + "\n")
                    games.flush()
                    updates = state.add_game(int(number), white, black, outcome)
                    for snapshot in updates:
                        history.write(json.dumps(snapshot, allow_nan=False) + "\n")
                        history.flush()
                    if updates:
                        snapshot = updates[-1]
                        print(f"[PENTA] pairs={snapshot['pairs']} bins={snapshot['pentanomial']} "
                              f"LLR={snapshot['llr']:+.4f} result={snapshot['result']}")
                    if state.result != "inconclusive" and stopping_at is None:
                        stopping_at = time.monotonic()
                        stop_cutechess(proc)
                    if plotter:
                        for snapshot in updates:
                            plotter.update(snapshot)
                if plotter:
                    plotter._pump_events()
                if stopping_at is not None and time.monotonic() - stopping_at > 10:
                    stop_cutechess(proc, force=True)
            ret = proc.wait(timeout=10)
            if ret != 0 and stopping_at is None:
                raise subprocess.CalledProcessError(ret, cmd)
        except BaseException as exc:
            failure = str(exc) or type(exc).__name__
            stop_cutechess(proc, force=True)
            proc.wait(timeout=10)
            raise
        finally:
            summary = state.snapshot()
            summary["error"] = failure
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
            thread.join(timeout=2)
    return state.snapshot()


def upload_logs(args, cute_chess_stats, runtime=None):
    from data import etl
    print("[DATA] Preparing to upload logs to database...")

    if system == "Windows": cnxn = sqlite3.connect('F:/databases/chess.db')
    elif system == "Darwin": cnxn = sqlite3.connect(Path.home() / "Documents/databases/chess.db")

    print("[DATA] Probing engine metadata...")
    candidate_engine_version = etl.probe_engine_metadata(args.engine_a)['version']
    baseline_engine_version = etl.probe_engine_metadata(args.engine_b)['version']

    # get engine_id by probing db.engines via version
    print("[DATA] Retrieving engine ids...")
    candidate_engine_id = etl.get_engine_id(cnxn, version=candidate_engine_version)
    baseline_engine_id = etl.get_engine_id(cnxn, version=baseline_engine_version)

    # auto-register if not found
    if candidate_engine_id is None:
        print(f"[SPRT] Candidate engine {candidate_engine_version} not registered, registering now...")
        candidate_engine_id = etl.register_engine(cnxn, {"engine_path": args.engine_a})
    if baseline_engine_id is None:
        print(f"[SPRT] Baseline engine {baseline_engine_version} not registered, registering now...")
        baseline_engine_id = etl.register_engine(cnxn, {"engine_path": args.engine_b})

    # log sprt experiment
    sprt_id = etl.start_experiment(
        cnxn, 
        "SPRT",
        candidate_engine_id,
        comparison_engine_id = baseline_engine_id
    )

    # Only consolidate and ingest JSONL data if logging was enabled
    ingestion_ok = False
    if args.log:
        # consolidate per-instance log files from concurrent engine processes
        print("[DATA] Consolidating per-instance log files...")
        etl.consolidate_instance_logs(args.logroot)

        try:
            # map search --> game
            print("[DATA] Building game map ...")
            game_map = etl.bulk_log_game(
                cnxn, 
                GAME_JSON, 
                sprt_id,
            )

            # log search+timing with game mapping
            print("[DATA] Logging all search data ...")
            etl.bulk_log_search_and_timing(
                cnxn, 
                SEARCH_JSON,
                game_map, 
                timing_path=TIMING_JSON,
                root_moves_path=ROOT_MOVES_JSON
            )
            ingestion_ok = True
        except Exception as e:
            print(f"[DATA] JSONL ingestion failed: {e}")
            print("[DATA] Log files preserved for retry.")
    else:
        print("[DATA] Logging was disabled, skipping JSONL ingestion.")

    # log sprt experiment details in db.sprt (always, regardless of --log)
    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    etl.log_sprt(
        cnxn,
        sprt_id,  # experiment_id
        candidate_engine_id,
        baseline_engine_id,
        **{**args_dict, **cute_chess_stats},
        runtime=runtime
    )
    etl.update_experiment(
        cnxn, 
        sprt_id, 
        {"end_time_utc": datetime.now(timezone.utc).isoformat()}
    )

    # Only clear log directory if ingestion succeeded
    if args.log and ingestion_ok:
        print("[DATA] Clearing log directory...")
        etl.clear_log_dir(args.logroot)
        print(f"[DATA] Logging completed for SPRT {sprt_id}.")


def parse_args():
    p = argparse.ArgumentParser(description="SPRT runner using cutechess-cli")

    # Engines
    p.add_argument("--engine-a", required=True, help="Candidate engine path")
    p.add_argument("--engine-b", required=True, help="Baseline engine path")
    p.add_argument("--params", type=dict, help="Dictionary of parameter and their values to test against the current version of the engine")

    # cutechess
    p.add_argument(
        "--cutechess-cli",
        default=r"C:\Program Files (x86)\Cute Chess\cutechess-cli.exe",
        help="Path to cutechess-cli.exe"
    )

    # Time control (choose ONE)
    p.add_argument("--depth", type=int, default=None, help="Depth per move")
    p.add_argument("--time", type=float, default=None, help="Seconds per move")
    p.add_argument("--tc", type=str, default=None, help="Time control (e.g. 0+1)")

    # SPRT parameters
    p.add_argument("--elo0", type=int, default=0)
    p.add_argument("--elo1", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--max-games", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=2, help="Number of concurrent games")

    # Opening book
    p.add_argument("--book", default= PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn" , help="Opening book file")
    p.add_argument("--book-depth", type=int, default=16) # 8 full moves

    # Logging
    p.add_argument('--log', action="store_true", help="Flag to turn on logging for candidate engine")
    p.add_argument('--plot', action='store_true', help='Show live SPRT plots (LLR, Elo)')
    p.add_argument(
        "--logroot",
        default=SPRT_LOG_DIR,
        help="Root directory for SPRT logs"
    )

    return p.parse_args()



def main(args=None):
    if args is None:
        args = parse_args()
    state = PentanomialSPRT(args.elo0, args.elo1, args.alpha, args.beta)
    if args.max_games < 2 or args.max_games % 2:
        raise ValueError("--max-games must be a positive even number (complete opening pairs)")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")

    # Validate TC
    if sum(x is not None for x in (args.depth, args.time, args.tc)) != 1:
        raise ValueError("Specify exactly one of --depth, --time, or --tc")

    engine_a = os.path.abspath(args.engine_a)
    engine_b = os.path.abspath(args.engine_b)

    should_log = args.log
    print(f"[SPRT] Engine logging: {'enabled' if should_log else 'disabled'}")
    Path(args.logroot).mkdir(parents=True, exist_ok=True)
    run_dir = LOGS_DIR / "sprt_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    print(f"[SPRT] Results: {run_dir}")

    each_block = [
        "-each",
        "proto=uci"
    ]
    log_a_block = [
        f"option.log_dir={args.logroot}",
        f"option.uci_logging={str(should_log).lower()}",
    ]
    if args.params:
        for name, value in args.params.items(): # add tuning result param values
            log_a_block.append(
                f"option.{name}={value}"
            )
    log_b_block = [
        f"option.log_dir={args.logroot}",
        f"option.uci_logging={str(should_log).lower()}",
    ]

    # Time control
    if args.depth is not None:
        each_block += [f"depth={args.depth}", "tc=inf"]
    elif args.time is not None:
        each_block += [f"st={args.time}", "timemargin=30"]
    else:
        each_block.append(f"tc={args.tc}")

    # opening book
    book_block = []
    if args.book is not None:
        book_block.append("-openings")
        book_block.append(f"file={os.path.abspath(args.book)}")
        book_block.append(f"format={os.path.splitext(args.book)[1][1:]}")
        book_block.append("order=random")
        book_block.append(f"plies={args.book_depth}")

    cmd = [
        args.cutechess_cli,

        # Candidate engine
        "-engine",
        "name=Candidate",
        f"cmd={engine_a}",
        f"dir={os.path.dirname(engine_a)}",
    ] + log_a_block + [

        # Baseline engine
        "-engine",
        "name=Baseline",
        f"cmd={engine_b}",
        f"dir={os.path.dirname(engine_b)}",
    ] + log_b_block + each_block + [

        # SPRT
        "-maxmoves", "100",
        "-games", "2",
        "-rounds", str(args.max_games // 2),
    ] + book_block + [

        # Runtime
        "-repeat",
        "-concurrency", str(args.concurrency),
        "-pgnout", str(run_dir / "games.pgn"),
    ]

    print("[SPRT] Launching cutechess:")
    print(" ".join(cmd))

    (run_dir / "config.json").write_text(
        json.dumps(dict(arguments=vars(args), command=cmd, model="pentanomial-logistic"),
                   indent=2, default=str), encoding="utf-8")
    start_time = time.time()
    plotter = None
    if args.plot:
        from data import etl
        from utils.sprt_plot import LivePlotter
        plotter = LivePlotter(args.elo0, args.elo1, args.alpha, args.beta,
                              etl.probe_engine_metadata(args.engine_a),
                              etl.probe_engine_metadata(args.engine_b),
                              args.time, args.depth, args.tc)
    stats = run_cutechess(cmd, state, run_dir, plotter)
    run_time = time.time() - start_time
    if plotter:
        plotter.finalize(stats, run_dir / "sprt.png")

    upload_logs(args, cute_chess_stats=stats, runtime=run_time)

    print(f"[SPRT] {stats['result']}: {stats['pairs']} pairs, LLR {stats['llr']:+.4f}")
    if plotter:
        plotter.show_final()

    return {
        "accepted": stats['result'] == "pass",
        "elo": stats['elo_diff'],
        "games": stats['games_played']
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[SPRT] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
