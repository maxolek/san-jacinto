#!/usr/bin/env python3
"""Run the full ETL pipeline from the data package.

Usage:
  python3 -m data.databases.run_analytics_pipeline
  or
  python3 data/databases/run_analytics_pipeline.py
"""
import subprocess
import sys
from pathlib import Path
import argparse


MODULES = [
    "data.databases.load_analytics",
    "data.transforms.transform_positions",
    "data.transforms.transform_search",
    "data.transforms.validate",
    "data.databases.test_schema",
]



def run_module(module: str, extra_args: list = None) -> None:
    print(f"\n=== Running module: {module} ===\n")
    cmd = [sys.executable, "-m", module]
    if extra_args:
        cmd.extend(extra_args)
    try:
        # run subprocess from repository root so package imports like 'data.*' resolve
        repo_root = Path(__file__).resolve().parent.parent.parent
        subprocess.run(cmd, check=True, cwd=str(repo_root))
    except subprocess.CalledProcessError as e:
        print(f"Module {module} failed with exit code {e.returncode}")
        raise


def main():
    cwd = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Run ETL pipeline modules")
    parser.add_argument('--skip-load', action='store_true', help='Skip data.load_analytics step')
    parser.add_argument('--full', action='store_true', help='Full refresh (drop + reload all tables)')
    args = parser.parse_args()

    print(f"Running pipeline from: {cwd}")
    modules_to_run = MODULES[1:] if args.skip_load else MODULES
    #if args.full: modules_to_run.append("data.databases.migrate_schema")
    if args.skip_load:
        print("Skipping data.load_analytics (use --skip-load to enable)")
    for m in modules_to_run:
        # Pass --full to load_analytics and transform_search if requested
        extra = ['--full'] if args.full and m in ('data.databases.load_analytics', 'data.transforms.transform_search') else None
        run_module(m, extra_args=extra)
    print("\nPipeline completed successfully.")


if __name__ == '__main__':
    main()
