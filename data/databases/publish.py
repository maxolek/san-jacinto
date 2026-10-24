#!/usr/bin/env python3
"""Publish the analytics dashboard in one command.

    python -m data.databases.publish                 # export, release, deploy
    python -m data.databases.publish --refresh       # rebuild the analytics DB first
    python -m data.databases.publish --watch         # wait until GitHub Pages is live
    python -m data.databases.publish --dry-run       # export only; print the gh commands

Steps:
  1. (--refresh) run the analytics pipeline: raw DB -> analytics DB, views, aggregates
  2. export the public snapshot and its release bundle (export_public)
  3. create a GitHub release analytics-<UTC time> holding the bundle (not marked latest)
  4. start the "Publish analytics dashboard" workflow for that release
  5. (--watch) follow the workflow run until the site is deployed

Needs the GitHub CLI (`gh`, https://cli.github.com) logged in with access to the
repository (`gh auth login`), and, once, Settings -> Pages -> Source = "GitHub
Actions". The workflow runs from --ref (default: the repository's default
branch), which must contain .github/workflows/dashboard-pages.yml.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..etl.paths import ANALYTICS_DB, PROJECT_ROOT
from .export_public import DEFAULT_KEEP, DEFAULT_OUTPUT, export_public

WORKFLOW = "dashboard-pages.yml"


def run(cmd, dry_run=False, capture=False):
    printable = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print(f"$ {printable}")
    if dry_run:
        return ""
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=capture)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        raise SystemExit(f"command failed ({result.returncode}): {printable}\n{detail}")
    return result.stdout if capture else ""


def repository(explicit=None):
    """owner/name from --repo or the origin remote."""
    if explicit:
        return explicit
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=PROJECT_ROOT, capture_output=True,
                         text=True).stdout.strip()
    match = re.search(r"github\.com[:/]([^/]+)/([^/.]+?)(?:\.git)?$", url)
    if not match:
        raise SystemExit(f"cannot tell the GitHub repository from origin '{url}'; pass --repo owner/name")
    return f"{match.group(1)}/{match.group(2)}"


def check_gh(dry_run):
    if shutil.which("gh") is None:
        message = "GitHub CLI (gh) not found: install it from https://cli.github.com and run `gh auth login`"
        if dry_run:
            print(f"note: {message}")
            return
        raise SystemExit(message)
    if not dry_run and subprocess.run(["gh", "auth", "status"], capture_output=True).returncode:
        raise SystemExit("gh is not logged in: run `gh auth login`")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=ANALYTICS_DB, help="Analytics DuckDB (default: CHESS_ANALYTICS_DB)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Local public data folder")
    p.add_argument("--keep", type=int, default=DEFAULT_KEEP, help="Snapshots kept for open browsers")
    p.add_argument("--full", action="store_true", help="Also publish per-search tables (much larger)")
    p.add_argument("--refresh", action="store_true", help="Run the analytics pipeline before exporting")
    p.add_argument("--skip-stockfish", action="store_true", help="With --refresh: skip Stockfish analysis")
    p.add_argument("--repo", default=None, help="owner/name (default: from the origin remote)")
    p.add_argument("--ref", default=None, help="Branch the workflow runs from (default: the default branch)")
    p.add_argument("--tag", default=None, help="Release tag (default: analytics-<UTC time>)")
    p.add_argument("--watch", action="store_true", help="Wait for the deployment and print the site URL")
    p.add_argument("--dry-run", action="store_true", help="Export only and print the gh commands")
    args = p.parse_args(argv)

    check_gh(args.dry_run)
    repo = repository(args.repo)
    if args.refresh:
        cmd = [sys.executable, "-m", "data.databases.run_analytics_pipeline", "--full",
               "--analytics-db", str(args.source)]
        if args.skip_stockfish:
            cmd.append("--skip-stockfish")
        run(cmd)

    stamp = datetime.now(timezone.utc)
    tag = args.tag or f"analytics-{stamp:%Y%m%d-%H%M%S}"
    with tempfile.TemporaryDirectory(prefix="sj-publish-") as tmp:
        archive = Path(tmp) / "analytics-public.tar.gz"
        manifest = export_public(args.source, args.output, archive, keep=args.keep, full=args.full)
        notes = (f"Dashboard data exported {manifest['exported_at']}: {manifest['searches']:,} searches, "
                 f"{manifest['row_counts'].get('game_stats', 0):,} games, profile {manifest['profile']}.")
        run(["gh", "release", "create", tag, str(archive), "--repo", repo, "--title", f"Analytics data {stamp:%Y-%m-%d %H:%M} UTC",
             "--notes", notes, "--latest=false"], args.dry_run)

    workflow = ["gh", "workflow", "run", WORKFLOW, "--repo", repo, "-f", f"data_release={tag}"]
    if args.ref:
        workflow += ["--ref", args.ref]
    run(workflow, args.dry_run)
    if args.dry_run:
        print("dry run: nothing was released or deployed")
        return
    print(f"Release {tag} created; the Pages workflow is starting: https://github.com/{repo}/actions/workflows/{WORKFLOW}")
    if args.watch:
        import time
        time.sleep(5)  # let GitHub register the dispatched run
        listed = run(["gh", "run", "list", "--workflow", WORKFLOW, "--repo", repo, "--limit", "1",
                      "--json", "databaseId,url"], capture=True)
        runs = json.loads(listed or "[]")
        if runs:
            run(["gh", "run", "watch", str(runs[0]["databaseId"]), "--repo", repo, "--exit-status"])
            owner, name = repo.split("/")
            print(f"Deployed: https://{owner}.github.io/{name}/")


if __name__ == "__main__":
    main()
