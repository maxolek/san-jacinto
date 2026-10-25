#!/usr/bin/env python3
"""Run the analytics pipeline: raw SQLite -> analytics DuckDB (tables, views, aggregates).

Steps (each runs as `python -m <module>` from the repository root):
  1. data.databases.load_analytics         raw tables -> analytics tables (incremental or --full)
  2. data.transforms.transform_positions   Stockfish's top lines (optional) + position_features
  3. data.transforms.transform_search      SQL views
  4. data.transforms.aggregates            agg_* tables + search_sample for the dashboard
  5. data.transforms.validate              data quality checks
  6. data.databases.test_schema            schema check against the dashboard contract

Usage (from the repository root):
  python -m data.databases.run_analytics_pipeline
  python -m data.databases.run_analytics_pipeline --full --stockfish /path/to/stockfish
  python -m data.databases.run_analytics_pipeline --skip-load --skip-stockfish
  python -m data.databases.run_analytics_pipeline --raw-db chess.db --analytics-db chess_analytics.duckdb

The databases default to CHESS_RAW_DB / CHESS_ANALYTICS_DB (see data/etl/paths.py).
Stockfish is optional: without --stockfish, $STOCKFISH or `stockfish` on PATH no
new positions are analysed, the results stored earlier still fill the sf_*
columns, and everything else is built.
"""
import argparse
import subprocess
import sys
from pathlib import Path

MODULES = [
    "data.databases.load_analytics",
    "data.transforms.transform_positions",
    "data.transforms.transform_search",
    "data.transforms.aggregates",
    "data.transforms.validate",
    "data.databases.test_schema",
]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_module(module: str, extra_args: list = None) -> None:
    print(f"\n=== Running module: {module} ===\n", flush=True)
    cmd = [sys.executable, "-m", module]
    if extra_args:
        cmd.extend(extra_args)
    try:
        # run subprocess from repository root so package imports like 'data.*' resolve
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as e:
        print(f"Module {module} failed with exit code {e.returncode}")
        raise


def module_arguments(args) -> dict:
    """{module: extra command-line arguments} for the parsed pipeline options."""
    extra = {module: [] for module in MODULES}
    if args.raw_db:
        extra["data.databases.load_analytics"] += ["--raw-db", str(args.raw_db)]
    if args.analytics_db:
        extra["data.databases.load_analytics"] += ["--analytics-db", str(args.analytics_db)]
        for module in MODULES[1:]:
            extra[module] += ["--db", str(args.analytics_db)]
    if args.full:
        # Only the fact loader has a full/incremental distinction; views are live.
        extra["data.databases.load_analytics"].append("--full")
    positions = extra["data.transforms.transform_positions"]
    if args.skip_stockfish:
        positions.append("--skip-stockfish")
    if args.stockfish:
        positions += ["--stockfish", str(args.stockfish)]
    if args.stockfish_depth:
        positions += ["--depth", str(args.stockfish_depth)]
    if args.stockfish_multipv:
        positions += ["--multipv", str(args.stockfish_multipv)]
    if args.stockfish_workers:
        positions += ["--workers", str(args.stockfish_workers)]
    return extra


def main(argv=None, runner=run_module):
    parser = argparse.ArgumentParser(description="Run the analytics pipeline")
    parser.add_argument('--skip-load', action='store_true', help='Skip data.databases.load_analytics')
    parser.add_argument('--full', action='store_true', help='Full refresh (drop + reload all tables)')
    parser.add_argument('--raw-db', type=Path, help='SQLite raw database (default: CHESS_RAW_DB)')
    parser.add_argument('--analytics-db', type=Path, help='DuckDB analytics database (default: CHESS_ANALYTICS_DB)')
    parser.add_argument('--stockfish', help='Stockfish binary (default: $STOCKFISH, then stockfish on PATH)')
    parser.add_argument('--skip-stockfish', action='store_true',
                        help='No new Stockfish analysis (results stored earlier are still applied)')
    parser.add_argument('--stockfish-depth', type=int, help='Stockfish search depth (default: 10)')
    parser.add_argument('--stockfish-multipv', type=int, help="Stockfish lines per position, the N of 'top N' (default: 5)")
    parser.add_argument('--stockfish-workers', type=int,
                        help='Stockfish processes side by side (default: half the logical CPUs)')
    args = parser.parse_args(argv)

    print(f"Running pipeline from: {REPO_ROOT}")
    modules_to_run = MODULES[1:] if args.skip_load else MODULES
    if args.skip_load:
        print("Skipping data.databases.load_analytics (--skip-load)")
    extra = module_arguments(args)
    for module in modules_to_run:
        runner(module, extra_args=extra[module] or None)
    print("\nPipeline completed successfully.")


if __name__ == '__main__':
    main()
