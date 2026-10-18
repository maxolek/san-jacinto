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
import shutil
import signal
import queue

from utils.pentanomial import FINISHED_GAME, PentanomialSPRT
from utils.fastchess_match import (
    FASTCHESS, Side, limit_args, limit_label, prepare_match, run_fastchess, run_match, stop_fastchess, tb_args
)

system = platform.system()

# paths
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# log paths
LOGS_DIR = PROJECT_ROOT / "logs"
SPRT_LOG_DIR = LOGS_DIR / "sprt_logs"

def parse_fastchess_output(output, candidate_name="Candidate", elo0=0, elo1=10,
                           alpha=0.05, beta=0.05):
    """Rebuild pentanomial evidence from game results, never fastchess summaries."""
    state = PentanomialSPRT(elo0, elo1, alpha, beta)
    for line in output.splitlines():
        match = FINISHED_GAME.search(line)
        if match:
            number, white, black, result = match.groups()
            white = "Candidate" if white == candidate_name else white
            black = "Candidate" if black == candidate_name else black
            state.add_game(int(number), white, black, result)
    return state.snapshot()


def ingest_telemetry(cnxn, run_logs, ids):
    """Load one run's engine logs (<run_logs>/<side>/), each side attributed to its own build.

    `ids`: the run's engine and experiment ids, as in its engines.json. Every run has
    its own log folder, so logs of other, aborted or concurrent runs are never mixed in.
    Each side is one transaction, loaded completely or not at all (even if the process
    is killed). It records the load in telemetry_loads, whose key refuses a second load
    of the same side, and the side's logs are then removed, so a retry loads each side
    exactly once. Returns True if everything was loaded.
    """
    from data import etl
    run_logs = Path(run_logs)
    etl.ensure_schema(cnxn)  # e.g. a migration added since the run (a retry): no transaction is open yet
    ok = True
    for side, opponent in (("candidate", "baseline"), ("baseline", "candidate")):
        log_dir = run_logs / side
        if side in ids.setdefault("loaded", []) or not log_dir.is_dir():
            continue
        print(f"[DATA] Logging {side} telemetry ...")
        try:
            # first, in the side's transaction: a side loaded before (e.g. the process stopped
            # right after that commit) fails here, before any of its rows are added again
            cnxn.execute("INSERT INTO telemetry_loads (experiment_id, side) VALUES (?, ?)", (ids["experiment"], side))
            etl.consolidate_instance_logs(log_dir)
            paths = etl.get_jsonl_paths(log_dir)
            game_map = {}
            if paths["game"].exists():
                game_map = etl.bulk_log_game(cnxn, paths["game"], ids["experiment"], second_engine_id=ids[opponent],
                                             engine_id=ids[side], commit=False)
            if paths["search"].exists():
                etl.bulk_log_search_and_timing(
                    cnxn,
                    paths["search"],
                    game_map,
                    engine_id=ids[side],
                    timing_path=paths["timing"] if paths["timing"].exists() else None,
                    root_moves_path=paths["root_moves"] if paths["root_moves"].exists() else None,
                    commit=False,
                )
            cnxn.commit()
        except sqlite3.IntegrityError:
            cnxn.rollback()
            print(f"[DATA] {side} telemetry of experiment {ids['experiment']} was loaded already")
        except Exception as e:
            cnxn.rollback()
            print(f"[DATA] JSONL ingestion failed ({side}): {e}")
            ok = False
            continue
        ids["loaded"].append(side)
        (run_logs / "engines.json").write_text(json.dumps(ids), encoding="utf-8")
        shutil.rmtree(log_dir, ignore_errors=True)
    return ok


def retry_ingest(run_logs):
    """Load a run's preserved telemetry with the engine ids recorded for it (engines.json)."""
    from data import etl
    run_logs = Path(run_logs)
    ids = json.loads((run_logs / "engines.json").read_text(encoding="utf-8"))
    cnxn = etl.connect_raw()
    try:
        if ingest_telemetry(cnxn, run_logs, ids):
            shutil.rmtree(run_logs, ignore_errors=True)
            print(f"[DATA] Telemetry of experiment {ids['experiment']} loaded.")
    finally:
        cnxn.close()



def upload_logs(args, fastchess_stats, runtime=None, run_logs=None):
    from data import etl, ratings
    print("[DATA] Preparing to upload logs to database...")

    cnxn = etl.connect_raw()  # waits for another run's write instead of failing after 5 s
    try:
        # engines are identified by their build (binary + net hash), not their version string;
        # the baseline first: it is usually the canonical build of its version (legacy rows)
        print("[DATA] Identifying engine builds...")
        baseline_engine_id = etl.resolve_engine(cnxn, args.engine_b)
        candidate_engine_id = etl.resolve_engine(cnxn, args.engine_a)

        # log sprt experiment
        sprt_id = etl.start_experiment(
            cnxn,
            "SPRT",
            candidate_engine_id,
            comparison_engine_id = baseline_engine_id
        )

        # The results first: telemetry below can fail and be retried, they cannot.
        # log sprt experiment details in db.sprt (always, regardless of --log)
        args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        etl.log_sprt(
            cnxn,
            sprt_id,  # experiment_id
            candidate_engine_id,
            baseline_engine_id,
            **{**args_dict, **fastchess_stats},
            runtime=runtime
        )
        # pairwise result; rating fits leave SPRT results out unless asked (early stopping biases them)
        limit = dict(tc=args.tc, depth=args.depth, time=args.time, nodes=getattr(args, "nodes", None))
        ratings.record_snapshot(cnxn, source="sprt", engine_a=candidate_engine_id, engine_b=baseline_engine_id,
                                snapshot=fastchess_stats,
                                tc_category=ratings.category(**limit, threads=getattr(args, "threads", 1)),
                                time_control=limit_label(**limit), experiment_id=sprt_id)
        etl.update_experiment(
            cnxn,
            sprt_id,
            {"end_time_utc": datetime.now(timezone.utc).isoformat()}
        )

        # This run's engine logs only (engines log games even with --log off)
        run_logs = Path(run_logs) if run_logs else None
        if not args.log:
            print("[DATA] Logging was disabled, skipping JSONL ingestion.")
            if run_logs:
                shutil.rmtree(run_logs, ignore_errors=True)
        elif run_logs and run_logs.is_dir():
            ids = dict(candidate=candidate_engine_id, baseline=baseline_engine_id, experiment=sprt_id, loaded=[])
            (run_logs / "engines.json").write_text(json.dumps(ids), encoding="utf-8")  # a retry attributes the logs
            if ingest_telemetry(cnxn, run_logs, ids):
                shutil.rmtree(run_logs, ignore_errors=True)
                print(f"[DATA] Logging completed for SPRT {sprt_id}.")
            else:
                print(f"[DATA] Log files preserved; retry with: python -m tests.sprt --retry-ingest {run_logs}")
    finally:
        cnxn.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SPRT runner using cutechess-cli")

    # Engines
    p.add_argument("--engine-a", required=True, help="Candidate engine path")
    p.add_argument("--engine-b", required=True, help="Baseline engine path")
    p.add_argument("--params", type=dict, help="Dictionary of parameter and their values to test against the current version of the engine")
    p.add_argument("--retry-ingest", metavar="RUN_LOG_DIR", 
        help="Only load the preserved telemetry of an earlier run (<logroot>/<run>) into the database")

    # fastchess-cli path
    p.add_argument(
        "--fastchess",
        default=str(FASTCHESS),
        help="Path to fastchess.exe"
    )

    # Time control (choose ONE)
    p.add_argument("--nodes", type=int, default=None, help="Nodes per move")
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
    p.add_argument("--threads", type=int, default=1, help="Threads per engine (option.Threads); each pinned game gets this many cpu cores")
    p.add_argument("--no-pin", action="store_true", 
                   help="Do not pin each game to its own CPU cores (default: pin)")

    # Opening book
    p.add_argument("--book", default= PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn" , help="Opening book file")
    p.add_argument("--book-depth", type=int, default=16) # 8 full moves

    # adjudication
    p.add_argument("--tb", default=None,
                   help="Syzygy tablebase dir(s) for adjudication (';'-separated on Windows); WDL files are enough")
    p.add_argument("--tbpieces", type=int, default=None, help="Only adjudicate positions with at most this many pieces")

    # Logging
    p.add_argument('--log', action="store_true", help="Flag to turn on logging for candidate engine")
    p.add_argument('--plot', action='store_true', help='Show live SPRT plots (LLR, Elo)')
    p.add_argument(
        "--logroot",
        default=SPRT_LOG_DIR,
        help="Root directory for SPRT logs (each run logs into <logroot>/<run>/<side>/)"
    )

    args = p.parse_args(argv)
    if not args.retry_ingest and not (args.engine_a and args.engine_b):
        p.error("--engine-a and --engine-b are required")
    return args



def main(args=None):
    if args is None:
        args = parse_args()
    if getattr(args, "retry_ingest", None):
        return retry_ingest(args.retry_ingest)
    state = PentanomialSPRT(args.elo0, args.elo1, args.alpha, args.beta)
    if args.max_games < 2 or args.max_games % 2:
        raise ValueError("--max-games must be a positive even number (complete opening pairs)")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")

    # Validate TC
    nodes = getattr(args, "nodes", None)
    limits = limit_args(depth=args.depth, time=args.time, tc=args.tc, nodes=nodes)

    engine_a = os.path.abspath(args.engine_a)
    engine_b = os.path.abspath(args.engine_b)

    # Threads: --threads for both engines; tuned params may give the candidate its own
    params = dict(args.params or {})
    threads_b = getattr(args, "threads", 1)
    threads_a = int(params.pop("Threads", threads_b))
    if min(threads_a, threads_b) < 1:
        raise ValueError("--threads must be positive")

    # logging // options
    print(f"[SPRT] Engine logging: {'enabled' if args.log else 'disabled'}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = LOGS_DIR / "sprt_results" / run_id
    run_logs = Path(args.logroot).resolve() / run_id  # this run's engine logs only (engines run in their own dir)

    def side_options(side):
        return [
            f"option.log_dir={run_logs / side}",
            f"option.uci_logging={str(args.log).lower()}",
        ]

    options_a = side_options("candidate") + [f"options.{name}={value}" for name, value in params.items()]
    match = prepare_match(
        Side(engine_a, options_a, threads_a),
        Side(engine_b, side_options("baseline"), threads_b),
        limits=limits, pairs=args.max_games // 2, concurrency=args.concurrency,
        run_dir=run_dir, fastchess=args.fastchess, book=args.book, book_depth=args.book_depth,
        pin=not getattr(args, "no_pin", False), maxmoves=100,
        extra=tb_args(getattr(args, "tb", None), getattr(args, "tbpieces", None)),
        meta=dict(arguments=vars(args), model="pentanomial-logistic")
    )
    print(f"[SPRT] Results: {run_dir}")
    print("[SPRT] Launching fastchess:")

    # launch
    start_time = time.time()
    plotter = None 
    if args.plot:
        from data import etl
        from utils.sprt_plot import LivePlotter
        plotter = LivePlotter(args.elo0, args.elo1, args.alpha, args.beta, 
                              etl.probe_engine_metadata(args.engine_a),
                              etl.probe_engine_metadata(args.engine_b),
                              args.time, args.depth, args.tc or (f"{nodes} nodes / move" if nodes else None)
                            )
    stats = run_match(match, state, plotter)
    run_time = time.time() - start_time 
    if plotter: plotter.finalize(stats, run_dir / "sprt.png")

    # finalize
    upload_logs(args, fastchess_stats=stats, runtime=run_time, run_logs=run_logs)

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
