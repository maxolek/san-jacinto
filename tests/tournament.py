#!/usr/bin/env python3
"""Progression test: rate a new engine against earlier versions.

The candidate plays a fixed number of opening pairs (no early stopping, so the
Elo is unbiased) against each opponent at each time control: a gauntlet by
default (candidate vs each opponent), --round-robin also plays the opponents
against each other. Every pairing's result (W/D/L and pentanomial counts) is
stored in match_results, and the ratings of all engines are refitted from every
stored tournament game (data/ratings.py: one maximum-likelihood fit with 95%
intervals, anchored at the oldest engine). This replaces per-run K-factor
updates, whose results depended on the order games were played in.

The report shows this run's pairings (pentanomial Elo with 95% interval) and the
version history: each engine's rating and its gain over the previous version.

Opponents: one build (Prod preferred) of each of the n latest earlier versions in
the database (+ the first version with --first_engine), or --opps VERSION ...; the
candidate's own version and temporary builds (dev, test) are skipped. Each must
still be the registered build (binary and net hashes are checked), so a rebuilt
exe is never mistaken for an old one.

Time control categories:
  - ultra_fast: < 15 s/game      (default: 0:10+0.1)
  - bullet:     < 2 min/game     (default: 1:00+0.6)
  - blitz:      < 10 min/game    (default: 3:00+1.8)
  - rapid:      < 30 min/game    (default: 10:00+5)
  - classical:  >= 30 min/game   (default: 30:00+10)
Short and long time controls in one run: --tc bullet rapid.

Usage (standalone):
  python -m tests.tournament --engine engines/1.0.exe --tc blitz --games 100

Called from release.py as:
  tournament.run_tournament(args, cnxn, engine_id)
"""
import argparse
import os
import subprocess
import sqlite3
import platform
from pathlib import Path
from datetime import datetime, timezone
from data import etl, ratings
from utils import cpu_slots
from utils.fastchess_match import FASTCHESS, book_args, tb_args
from utils.pentanomial import FINISHED_GAME, estimate

system = platform.system()

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
TOMAHAWK_ROOT = PROJECT_ROOT.parent / "tomahawk"
ENGINES_DIR = TOMAHAWK_ROOT / "engines"
ENGINES_DEV_DIR = ENGINES_DIR / "dev"
ENGINES_PROD_DIR = ENGINES_DIR / "prod"
LOGS_DIR = PROJECT_ROOT / "logs"
TOURNAMENT_LOG_DIR = LOGS_DIR / "tournament_logs"

# Default time controls per category
TC_DEFAULTS = {
    "ultra_fast": "0:10+0.1",
    "bullet":     "1:00+0.6",
    "blitz":      "3:00+1.8",
    "rapid":      "10:00+5",
    "classical":  "30:00+10",
}

classify_tc = ratings.classify_tc
RESULTS = ("1-0", "0-1", "1/2-1/2")
SKIP_VERSIONS = {"dev", "test", "debug", "tomahawk"}  # temporary builds, never "earlier versions"


def _engine_rows(cnxn, where, params=()):
    previous, cnxn.row_factory = cnxn.row_factory, sqlite3.Row
    try:
        return cnxn.execute("SELECT id, name, version, binary_path, binary_sha256, net_path, net_sha256, build_type "
                            f"FROM engines {where}", params).fetchall()
    finally:
        cnxn.row_factory = previous


def _locate(row):
    """The registered build's exe: its recorded path, else engines/{prod,dev}/<version>.exe if the hash matches."""
    path = etl.locate_engine(row, [ENGINES_PROD_DIR / f"{row['version']}.exe",
                                   ENGINES_DEV_DIR / f"{row['version']}.exe"])
    if path is None:
        print(f"[TOURNAMENT] skipping engine {row['id']} ({row['version']}): its build was not found "
              f"(no exe with the registered hash)")
    return path


def _is_prod(row):
    build_type = (row["build_type"] or "").lower()
    return build_type in ("prod", "release") or (
        bool(row["binary_path"]) and Path(row["binary_path"]).parent.name.lower() == "prod")


def _pick_build(rows):
    """One build of a version: Prod builds first, then the newest; the first that can be found."""
    for row in sorted(rows, key=lambda r: (not _is_prod(r), -r["id"])):
        path = _locate(row)
        if path:
            return {"id": row["id"], "version": row["version"], "path": path}
    return None


def get_all_engines(cnxn, n=3, first=False, exclude_version=None):
    """One build of each of the n latest versions (+ the first version), other than `exclude_version`.

    Versions are ordered by their first registration, so a later rebuild of an old
    version keeps its place; several builds of a version (DEV/Prod, rebuilds) count
    once, and temporary builds (dev, test, ...) are skipped.
    """
    etl.ensure_schema(cnxn)
    by_version = {}
    for row in _engine_rows(cnxn, "ORDER BY id DESC"):
        version = row["version"]
        if version != exclude_version and (version or "").lower() not in SKIP_VERSIONS:
            by_version.setdefault(version, []).append(row)
    newest_first = sorted(by_version, key=lambda v: -min(r["id"] for r in by_version[v]))
    engines = []
    for version in newest_first:
        if len(engines) >= int(n):
            break
        engine = _pick_build(by_version[version])
        if engine:
            engines.append(engine)
    if first and newest_first:
        oldest = newest_first[-1]
        if all(e["version"] != oldest for e in engines):
            engine = _pick_build(by_version[oldest])
            if engine:
                engines.append(engine)
    return engines


def get_engines_by_names(cnxn, names):
    """Engines for explicitly named versions: one build of each (Prod preferred, then the newest).

    A version that is not registered is registered from engines/prod/<version>.exe.
    """
    etl.ensure_schema(cnxn)
    engines = []
    for version in names:
        rows = _engine_rows(cnxn, "WHERE version = ? ORDER BY id DESC", (version,))
        if not rows:
            exe = ENGINES_PROD_DIR / f"{version}.exe"
            if not exe.exists():
                print(f"[TOURNAMENT] WARNING: engine {version} is not registered and {exe} does not exist")
                continue
            print(f"[TOURNAMENT] Engine {version} not registered, registering now...")
            etl.resolve_engine(cnxn, str(exe))
            rows = _engine_rows(cnxn, "WHERE version = ? ORDER BY id DESC", (version,))
        engine = _pick_build(rows)
        if engine:
            engines.append(engine)
    return engines


def engine_names(opponents):
    """fastchess names: Candidate, then v<version> (with #<id> when two builds share a version)."""
    versions = [o["version"] for o in opponents]
    return ["Candidate"] + [f"v{o['version']}" + (f"#{o['id']}" if versions.count(o["version"]) > 1 else "")
                            for o in opponents]


def pairing_results(output, names):
    """Per pairing (a, b) with a listed before b in `names`: a's W/D/L and pentanomial counts.

    fastchess numbers a pairing's two games of one opening 2k+1 and 2k+2 (colours
    reversed); only complete pairs enter the pentanomial counts.
    """
    order = {name: i for i, name in enumerate(names)}
    games = {}
    for line in output.splitlines():
        match = FINISHED_GAME.search(line)
        if match and match.group(4) in RESULTS and match.group(2) in order and match.group(3) in order:
            games[int(match.group(1))] = (match.group(2), match.group(3), match.group(4))
    stats = {}

    def score(game, player):
        white, black, result = game
        points = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
        return points if player == white else 1.0 - points

    for white, black, result in games.values():
        a, b = sorted((white, black), key=order.get)
        entry = stats.setdefault((a, b), dict(wins=0, draws=0, losses=0, penta=[0] * 5))
        s = score((white, black, result), a)
        entry["wins" if s == 1 else "draws" if s == 0.5 else "losses"] += 1
    for number in games:
        if number % 2 == 1 and number + 1 in games:
            first, second = games[number], games[number + 1]
            if {first[0], first[1]} == {second[0], second[1]} and first[0] != second[0]:
                a, b = sorted((first[0], first[1]), key=order.get)
                stats[(a, b)]["penta"][round(2 * (score(first, a) + score(second, a)))] += 1
    return stats


def parse_tournament_output(output, engine_names):
    """Pairwise results in both orientations: {nameA: {nameB: {"wins", "losses", "draws"}}}."""
    pairwise = {name: {} for name in engine_names}
    for (a, b), r in pairing_results(output, engine_names).items():
        pairwise[a][b] = dict(wins=r["wins"], losses=r["losses"], draws=r["draws"])
        pairwise[b][a] = dict(wins=r["losses"], losses=r["wins"], draws=r["draws"])
    return pairwise


def run_fastchess_tournament(candidate_path, opponents, tc, games_per_pair, fastchess, concurrency = 2, book=None,
                             affinity=(), round_robin=False, extra=(), pgn_path=None):
    """Run candidate vs all opponents using fastchess.

    `affinity`: fastchess options pinning each game to its own cpu (cpu_slots.fastchess_args).
    `pgn_path`: where the games go (default: the log folder, which is cleared after ingestion).
    Returns (raw stdout text, engine names).
    """
    TOURNAMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    names = engine_names(opponents)
    engine_blocks = []
    for name, path in zip(names, [candidate_path] + [o["path"] for o in opponents]):
        engine_blocks += ["-engine", f"name={name}", f"cmd={os.path.abspath(path)}",
                          f"dir={os.path.dirname(os.path.abspath(path))}"]

    # fastchess plays every opening twice with colors reversed (-games 2), so the
    # per-pair game count is set through -rounds; it cannot exceed 2 per round.
    rounds = max(1, -(-games_per_pair // 2))
    if 2 * rounds != games_per_pair:
        print(f"[TOURNAMENT] Playing {2 * rounds} games per opponent (complete opening pairs), "
              f"not {games_per_pair}")

    schedule = ["-tournament", "roundrobin"] if round_robin else ["-tournament", "gauntlet", "-seeds", "1"]
    cmd = [
        str(fastchess),
        # fastchess saves a resume file on exit; keep it out of the cwd
        "-config", f"outname={TOURNAMENT_LOG_DIR / 'fastchess.json'}",
    ] + engine_blocks + [
        "-each", "proto=uci", f"tc={tc}",
        *schedule,
        "-games", "2",
        "-rounds", str(rounds),
        "-maxmoves", "200",
        "-concurrency", f"{concurrency}", "-force-concurrency",
        # an engine that crashes or stalls is resetarted: the tournament goes on
        "-recover",
        "-pgnout", f"file={pgn_path or TOURNAMENT_LOG_DIR / 'tournament.pgn'}",
    ] + book_args(book, 16) + list(extra) + list(affinity)

    print(f"[TOURNAMENT] Running: tc={tc}, {len(opponents)} opponents, {2 * rounds} games per pairing, "
          f"{'round robin' if round_robin else 'gauntlet'}")
    print(f"[TOURNAMENT] cmd: {' '.join(cmd[:6])}...")

    output_lines = []
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        for line in proc.stdout:
            print(line, end="")
            output_lines.append(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"[TOURNAMENT] WARNING: fastchess exited with code {proc.returncode}")

    return "".join(output_lines), names


def format_pairings(results, names_to_ids):
    """This run's pairings: W/D/L and pentanomial Elo with a 95% interval."""
    lines = []
    for (a, b), r in results.items():
        pairs = sum(r["penta"])
        est = estimate(r["penta"]) if pairs else {}
        elo = (f"Elo {est['elo_diff']:+7.1f} [{est['elo_low']:+.1f}, {est['elo_high']:+.1f}] ({pairs} pairs)"
               if est.get("elo_diff") is not None else "Elo n/a (no complete pairs)")
        lines.append(f"  {a:>12} vs {b:<12} +{r['wins']} -{r['losses']} ={r['draws']}  {elo}")
    return "\n".join(lines)


def run_tournament(args, cnxn, engine_id):
    """Main entry point from release.py.

    Plays the pairings at each requested time control, stores them and refits ratings.
    """
    etl.ensure_schema(cnxn)
    candidate_path = args.engine
    fastchess = args.fastchess
    book = getattr(args, "opening_book", None)
    games_per_pair = getattr(args, "tournament_games", 20)
    tc_categories = getattr(args, "tournament_tc", ["blitz"])
    n_engines = getattr(args, "tournament_engines", 3)
    listed_opponents = getattr(args, "opps", [])
    concurrency = getattr(args, "concurrency")
    round_robin = getattr(args, "round_robin", False)
    extra = tb_args(getattr(args, "tb", None), getattr(args, "tbpieces", None))

    if len(listed_opponents) == 0:
        # one build of each of the latest earlier versions (never the candidate's own version)
        candidate_version = cnxn.execute("SELECT version FROM engines WHERE id = ?", (engine_id,)).fetchone()
        opponents = get_all_engines(cnxn, n_engines, getattr(args, "first_engine", False),
                                    exclude_version=candidate_version[0] if candidate_version else None)
        opponents = [e for e in opponents if e["id"] != engine_id]
    else:
        # Explicit opponent list. Accept either version-name strings
        # (resolved against the DB) or pre-built engine dicts.
        if all(isinstance(o, dict) for o in listed_opponents):
            resolved = listed_opponents
        else:
            resolved = get_engines_by_names(cnxn, listed_opponents)
        opponents = [e for e in resolved if e["id"] != engine_id]

    if not opponents:
        print("[TOURNAMENT] No opponents in database, skipping tournament.")
        return

    print(f"[TOURNAMENT] Found {len(opponents)} opponents: {[e['version'] for e in opponents]}")
    names_to_ids = dict(zip(engine_names(opponents), [engine_id] + [o["id"] for o in opponents]))

    # Each game's own cpu, in the engines' Affinity order (if every engine reports it)
    affinity = cpu_slots.fastchess_args(cpu_slots.plan(
        [candidate_path] + [o["path"] for o in opponents], 1, concurrency,
        enabled=not getattr(args, "no_pin", False)))

    # Start experiment
    experiment_id = etl.start_experiment(cnxn, "tournament", engine_id)
    # each run's games are kept (tests/mistakes.py mines them); the log folder is cleared
    results_dir = LOGS_DIR / "tournament_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        for tc_cat in tc_categories:
            tc = TC_DEFAULTS.get(tc_cat, tc_cat)
            actual_cat = classify_tc(tc)

            print(f"\n[TOURNAMENT] === {actual_cat.upper()} (tc={tc}) ===")

            output, names = run_fastchess_tournament(
                candidate_path, opponents, tc, games_per_pair, fastchess, concurrency, book, affinity,
                round_robin, extra, pgn_path=results_dir / f"{actual_cat}.pgn"
            )
            results = pairing_results(output, names)
            for (a, b), r in results.items():
                ratings.record_match(cnxn, source="tournament", engine_a=names_to_ids[a], engine_b=names_to_ids[b],
                                     wins=r["wins"], draws=r["draws"], losses=r["losses"], penta=r["penta"],
                                     tc_category=actual_cat, time_control=tc, experiment_id=experiment_id)

            print(f"\n[TOURNAMENT] {actual_cat} pairings this run (pentanomial Elo, 95% interval):")
            print(format_pairings(results, names_to_ids))

            fit_id, _ = ratings.fit(cnxn, actual_cat)
            if fit_id:
                print()
                print(ratings.format_history(cnxn, fit_id, highlight={engine_id}))
    finally:
        cpu_slots.release()

    # Consolidate per-instance logs and upload, like SPRT
    etl.consolidate_instance_logs(TOURNAMENT_LOG_DIR)

    game_json = TOURNAMENT_LOG_DIR / "game.jsonl"
    search_json = TOURNAMENT_LOG_DIR / "search.jsonl"
    timing_json = TOURNAMENT_LOG_DIR / "timing.jsonl"
    root_moves_json = TOURNAMENT_LOG_DIR / "root_moves.jsonl"

    # every engine logs its version string; within a run each version is one build, so this
    # attributes the logs exactly (a release's DEV and Prod builds share their version)
    version_ids = {o["version"]: o["id"] for o in opponents if o.get("version")}
    candidate_version = cnxn.execute("SELECT version FROM engines WHERE id = ?", (engine_id,)).fetchone()
    if candidate_version and candidate_version[0]:
        version_ids[candidate_version[0]] = engine_id

    ingestion_ok = True
    try:  # games and searches together, or neither: the kept logs can then be loaded again
        game_map = {}
        if game_json.exists():
            game_map = etl.bulk_log_game(cnxn, game_json, experiment_id, commit=False,
                                         engine_ids_by_version=version_ids)
        if search_json.exists():
            etl.bulk_log_search_and_timing(
                cnxn,
                search_json,
                game_map,
                timing_path=timing_json if timing_json.exists() else None,
                root_moves_path=root_moves_json if root_moves_json.exists() else None,
                commit=False,
                engine_ids_by_version=version_ids,
            )
        cnxn.commit()
    except Exception as e:
        cnxn.rollback()
        print(f"[DATA] Telemetry ingestion failed: {e}")
        ingestion_ok = False

    # finalize experiment
    etl.update_experiment(cnxn, experiment_id, {"end_time_utc": datetime.now(timezone.utc).isoformat()})

    # Only clear log directory if ingestion succeeded
    if ingestion_ok:
        etl.clear_log_dir(TOURNAMENT_LOG_DIR)
    else:
        print("[DATA] Log files preserved due to ingestion failure.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Progression test: rate a new engine against earlier versions")
    parser.add_argument("--engine", required=True, help="Candidate engine path")
    parser.add_argument("--opps", nargs="+", default=[],
                        help="Explicit list of opponent engine version names "
                             "(e.g. --opps 1.0 0.9.3). Overrides --n_engines auto-selection.")
    parser.add_argument("--n_engines", type=int, default=3,
                        help="Number of earlier versions to play (one build each; the candidate's version is skipped)")
    parser.add_argument("--first_engine", action="store_true", help="Also play the first registered version")
    parser.add_argument("--tc", nargs="+", default=["blitz"],
                        help="Time controls to run. Accepts category names "
                             "(ultra_fast, bullet, blitz, rapid, classical), which "
                             "map to their default tc string, and/or explicit tc "
                             "strings (e.g. 15+0.15, 1:00+0.6) which are used as-is "
                             "and auto-classified into a category for rating purposes.")
    parser.add_argument("--games", type=int, default=100, help="Games per pairing per TC (rounded up to whole pairs)")
    parser.add_argument("--round-robin", action="store_true",
                        help="Also play the opponents against each other (default: candidate vs each opponent)")
    parser.add_argument("--book", default=str(PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn"))
    parser.add_argument("--concurrency", default=2, type=int, help="Number of games to run simultaneously (uses 2*concurrency threads)")
    parser.add_argument("--fastchess",
                        default=str(FASTCHESS), help="Path to the fastchess executable")
    parser.add_argument("--no-pin", action="store_true",
                        help="Don't give each game its own cpu in the engines' Affinity order")
    parser.add_argument("--tb", default=None, help="Syzygy tablebase dir(s) for adjudication (WDL files are enough)")
    parser.add_argument("--tbpieces", type=int, default=None)
 
    # Accept a pre-built Namespace (programmatic callers like release.py)
    # as well as a raw argv list / sys.argv (CLI callers).
    if isinstance(argv, argparse.Namespace):
        args = argv
    else:
        args = parser.parse_args(argv)
 
    # Validate --tc entries up front: either a known category name, or a
    # parseable explicit tc string (e.g. "15+0.15", "1:00+0.6").
    for tc_entry in args.tc:
        if tc_entry in TC_DEFAULTS:
            continue
        try:
            classify_tc(tc_entry)
        except (ValueError, IndexError):
            parser.error(
                f"invalid --tc value '{tc_entry}': must be one of "
                f"{list(TC_DEFAULTS.keys())} or a parseable tc string like '15+0.15'"
            )
 
    cnxn = etl.connect_raw()
 
    # The candidate's build (binary + net hash), registered if new
    engine_id = etl.resolve_engine(cnxn, args.engine)
 
    # Build namespace for run_tournament
    ns = argparse.Namespace(
        engine=args.engine,
        fastchess=getattr(args, "fastchess"),
        opening_book=args.book,
        tournament_games=args.games,
        tournament_tc=args.tc,
        tournament_engines=args.n_engines,
        first_engine=args.first_engine,
        opps=args.opps,
        concurrency=args.concurrency,
        no_pin=getattr(args, "no_pin", False),
        round_robin=getattr(args, "round_robin", False),
        tb=getattr(args, "tb", None),
        tbpieces=getattr(args, "tbpieces", None),
    )
 
    run_tournament(ns, cnxn, engine_id)
    cnxn.close()
 
 
if __name__ == "__main__":
    main()
