#!/usr/bin/env python3
"""Progression test: rate a new engine against earlier versions.

The candidate plays a fixed number of opening pairs (no early stopping, so the
Elo is unbiased) against each opponent at each time control: a gauntlet by
default (candidate vs each opponent), --round-robin also plays the opponents
against each other. Every pairing's result (W/D/L and pentanomial counts) is
stored in match_results, and the ratings of all engines are refitted from every
stored tournament game (data/ratings.py: one maximum-likelihood fit with 95%
intervals, anchored at the previous fit's anchor engine and Elo, so the scale
stays put from fit to fit; the first fit, or one whose anchor has no games, starts
from the oldest engine of the largest connected group). This replaces per-run K-factor
updates, whose results depended on the order games were played in.

The games are played on the match runner (utils/fixed_match.py): one game per
physical core at a time (--concurrency, default every physical core but one), the
game's fastchess process and engines pinned to its core (--no-pin leaves placement
to the OS), every game recorded with its core type, cpu and measured clock. The
pairings of a time control are played together, interleaved, so a drift of the
machine favours none of them.

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

Each time control is one run, stored as one experiment (type 'tournament') in one
transaction (all of it or, if storing fails, none): every
played game (games, source 'match', with its core type, cpu and clock), each
pairing's result (match_results, source 'tournament', the middle pentanomial bin
split into DD and WL), then the rating refit. With --log also the engines'
telemetry: each engine logs into its own folder <logroot>/<run>/<engine name>/,
is attributed to its own build and linked to the games it played
(data/etl/telemetry.py); the run's ids (<logroot>/<run>/engines.json) are written as
soon as it is stored, so telemetry that fails to load (or is not loaded because the
refit or the process failed) is kept for `--retry-ingest <logroot>/<run>`.

Results go to <results-dir>/<time>/ (default logs/tournament_results): the games of
each time control as <category>.pgn (tests/mistakes.py mines them) and its run
folder <time>_<category>/ (games.jsonl, games.pgn, config.json, summary.json, jobs/).
A time control that cannot be completed (failing jobs, too many excluded pairs,
Ctrl-C) stores nothing and ends the tournament with an error
(utils.fixed_match.MatchStopped, a RuntimeError, or MatchInterrupted, a
KeyboardInterrupt); the time controls before it stay stored.

Usage (standalone):
  python -m tests.tournament --engine engines/1.0.exe --tc blitz --games 100
  python -m tests.tournament --retry-ingest logs/tournament_logs/20261025_101129_354256_blitz

Called from release.py as:
  tournament.run_tournament(args, cnxn, engine_id)
"""
import argparse
import json
import math
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from data import etl, ratings
from utils.fastchess_match import FASTCHESS, tb_args
from utils.fixed_match import Pairing, run_fixed_match
from utils.match_runner import EngineSpec, tc_arguments
from utils.openings import OpeningBook

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
TOMAHAWK_ROOT = PROJECT_ROOT.parent / "tomahawk"
ENGINES_DIR = TOMAHAWK_ROOT / "engines"
ENGINES_DEV_DIR = ENGINES_DIR / "dev"
ENGINES_PROD_DIR = ENGINES_DIR / "prod"
LOGS_DIR = PROJECT_ROOT / "logs"
TOURNAMENT_LOG_DIR = LOGS_DIR / "tournament_logs"
RESULTS_DIR = LOGS_DIR / "tournament_results"
DEFAULT_BOOK = PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn"
BOOK_PLIES = 16
MAX_MOVES = 200  # fastchess -maxmoves: a longer game is adjudicated a draw

# Default time controls per category
TC_DEFAULTS = {
    "ultra_fast": "0:10+0.1",
    "bullet":     "1:00+0.6",
    "blitz":      "3:00+1.8",
    "rapid":      "10:00+5",
    "classical":  "30:00+10",
}

classify_tc = ratings.classify_tc
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


def pairs_per_pairing(games):
    """Opening pairs each pairing plays for `games` games: whole pairs (rounded up), at least one."""
    pairs = max(1, math.ceil(games / 2))
    if 2 * pairs != games:
        print(f"[TOURNAMENT] Playing {2 * pairs} games per pairing (complete opening pairs), not {games}")
    return pairs


def make_pairings(engines, pairs, round_robin=False):
    """Gauntlet: the first engine (the candidate) against each other one; round robin: every two engines.

    The engine listed first is a pairing's first engine: White in the first game of each
    pair, and engine a of its match_results row (its wins, draws and losses).
    """
    if round_robin:
        return [Pairing(a, b, pairs) for a, b in combinations(engines, 2)]
    return [Pairing(engines[0], other, pairs) for other in engines[1:]]


def opponents_in(pairings):
    """{engine name: the name of its only opponent in `pairings`, or None if it has several}."""
    seen = {}
    for p in pairings:
        seen.setdefault(p.first.name, set()).add(p.second.name)
        seen.setdefault(p.second.name, set()).add(p.first.name)
    return {name: next(iter(others)) if len(others) == 1 else None for name, others in seen.items()}


def format_pairings(pairings):
    """This run's pairings (run_fixed_match's snapshots by key): the first engine's W/D/L and
    pentanomial Elo with a 95% interval."""
    lines = []
    for s in pairings.values():
        elo = (f"Elo {s['elo_diff']:+7.1f} [{s['elo_low']:+.1f}, {s['elo_high']:+.1f}] ({s['pairs']} pairs)"
               if s.get("elo_diff") is not None else "Elo n/a (no complete pairs)")
        lines.append(f"  {s['first']:>12} vs {s['second']:<12} +{s['candidate_wins']} -{s['candidate_losses']} "
                     f"={s['candidate_draws']}  {elo}")
    return "\n".join(lines)


def keep_pgn(run_dir, path):
    """Append a run's games (<run_dir>/games.pgn) to the time control's PGN `path`."""
    source = Path(run_dir) / "games.pgn"
    if source.exists():
        with open(source, "rb") as src, open(path, "ab") as dst:
            shutil.copyfileobj(src, dst)


def store_run(cnxn, result, engine_ids, candidate_id, category, tc, meta):
    """Store one time control's completed run: an experiment, every played game, each pairing's result.

    The runner's games are authoritative: every one becomes a games row (source 'match')
    with its engines' ids (`engine_ids`: runner name -> engine id). One transaction: if
    anything fails (e.g. the database stays locked) none of it is stored, so a rating fit
    never sees part of a gauntlet. Returns the experiment id.
    """
    etl.ensure_schema(cnxn)  # before the transaction: applying a migration commits an open one
    try:
        experiment_id = etl.start_experiment(cnxn, "tournament", candidate_id, info=json.dumps(meta, default=str),
                                             commit=False)
        games = etl.bulk_log_match_games(cnxn, result["run_dir"], experiment_id, engine_ids, commit=False)
        recorded = 0
        for s in result["pairings"].values():
            if not s["pairs"]:
                continue
            ratings.record_match(cnxn, source="tournament", engine_a=engine_ids[s["first"]],
                                 engine_b=engine_ids[s["second"]], wins=s["candidate_wins"],
                                 draws=s["candidate_draws"], losses=s["candidate_losses"], penta=s["pentanomial"],
                                 penta_detail=s.get("pentanomial_detail"), tc_category=category, time_control=tc,
                                 experiment_id=experiment_id, commit=False)
            recorded += 1
        etl.update_experiment(cnxn, experiment_id, {"end_time_utc": datetime.now(timezone.utc).isoformat()},
                              commit=False)
        cnxn.commit()
    except BaseException:
        cnxn.rollback()
        print(f"[DATA] {category} (tc={tc}) could not be stored: none of its results are stored")
        raise
    print(f"[DATA] Logged {len(games)} games with their core type, cpu and clock, and {recorded} pairing "
          f"result(s): experiment {experiment_id}")
    return experiment_id


def keep_telemetry_ids(run_logs, experiment_id, engine_ids, opponents, run_dir):
    """Write the stored run's ids (<run_logs>/engines.json), so its engine logs can be loaded by
    --retry-ingest whatever happens next. Returns them, or None if the engines left no logs.

    `opponents`: {engine name: its only opponent's name, or None}.
    """
    run_logs = Path(run_logs)
    if not run_logs.is_dir():
        return None
    ids = etl.telemetry_ids(experiment_id, engine_ids,
                            {name: engine_ids.get(other) for name, other in opponents.items()}, run_dir)
    (run_logs / "engines.json").write_text(json.dumps(ids), encoding="utf-8")
    return ids


def load_telemetry(cnxn, run_logs, ids):
    """Load a run's engine logs (<run_logs>/<engine name>/), linked to its played games.

    `ids`: keep_telemetry_ids's (None: no logs). Logs that fail to load are kept with
    engines.json for --retry-ingest. Returns True if all were loaded.
    """
    run_logs = Path(run_logs)
    if ids is None or not run_logs.is_dir():
        print(f"[DATA] No engine logs in {run_logs}")
        return True
    if etl.ingest_telemetry(cnxn, run_logs, ids):
        shutil.rmtree(run_logs, ignore_errors=True)
        print(f"[DATA] Telemetry loaded for experiment {ids['experiment']}.")
        return True
    print(f"[DATA] Log files preserved; retry with: python -m tests.tournament --retry-ingest {run_logs}")
    return False


def run_tournament(args, cnxn, engine_id):
    """Main entry point from release.py.

    Plays the pairings at each requested time control, stores them and refits ratings.
    Returns [dict(category, time_control, experiment_id, run_dir, pairings)] of the time
    controls played. A time control that cannot be completed raises (see the module doc).
    """
    etl.ensure_schema(cnxn)
    candidate_path = args.engine
    fastchess = getattr(args, "fastchess", None) or FASTCHESS
    book_path = getattr(args, "opening_book", None) or DEFAULT_BOOK
    games_per_pair = getattr(args, "tournament_games", 20)
    tc_categories = getattr(args, "tournament_tc", ["blitz"])
    n_engines = getattr(args, "tournament_engines", 3)
    listed_opponents = getattr(args, "opps", None) or []
    concurrency = getattr(args, "concurrency", None)
    round_robin = getattr(args, "round_robin", False)
    log = bool(getattr(args, "log", False))
    extra = ["-maxmoves", str(MAX_MOVES)] + tb_args(getattr(args, "tb", None), getattr(args, "tbpieces", None))

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
    unique = {}
    for e in opponents:
        unique.setdefault(e["id"], e)  # a build plays once, under one name
    opponents = list(unique.values())

    if not opponents:
        print("[TOURNAMENT] No opponents in database, skipping tournament.")
        return []

    print(f"[TOURNAMENT] Found {len(opponents)} opponents: {[e['version'] for e in opponents]}")
    names = engine_names(opponents)
    paths = dict(zip(names, [candidate_path] + [o["path"] for o in opponents]))
    engine_ids = dict(zip(names, [engine_id] + [o["id"] for o in opponents]))
    pairs = pairs_per_pairing(games_per_pair)
    book = OpeningBook(book_path, plies=BOOK_PLIES)  # one shuffle: each pairing and time control its own openings
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # each run's games are kept (tests/mistakes.py mines <results_dir>/<category>.pgn)
    results_dir = Path(getattr(args, "results_dir", None) or RESULTS_DIR) / stamp
    logroot = Path(getattr(args, "logroot", None) or TOURNAMENT_LOG_DIR).resolve()

    played, runs = [], {}
    for tc_cat in tc_categories:
        tc = TC_DEFAULTS.get(tc_cat, tc_cat)
        actual_cat = classify_tc(tc)
        runs[actual_cat] = runs.get(actual_cat, 0) + 1
        run_name = f"{stamp}_{actual_cat}" + (f"_{runs[actual_cat]}" if runs[actual_cat] > 1 else "")
        run_logs = logroot / run_name  # this run's engine logs only, one folder per engine

        print(f"\n[TOURNAMENT] === {actual_cat.upper()} (tc={tc}) ===")
        print(f"[TOURNAMENT] Running: tc={tc}, {len(opponents)} opponents, {2 * pairs} games per pairing, "
              f"{'round robin' if round_robin else 'gauntlet'}")
        engines = [EngineSpec.from_path(name, path, {"log_dir": str(run_logs / name),
                                                     "uci_logging": str(log).lower()})
                   for name, path in paths.items()]
        pairings = make_pairings(engines, pairs, round_robin)
        meta = dict(tournament=stamp, category=actual_cat, time_control=tc, round_robin=round_robin,
                    games_per_pairing=2 * pairs, book_seed=book.seed, engine_logs=str(run_logs) if log else None,
                    engines={name: dict(id=engine_ids[name], path=str(path)) for name, path in paths.items()})
        completed = False
        try:
            result = run_fixed_match(pairings, tc_args=tc_arguments(tc=tc), results_dir=results_dir,
                                     run_name=run_name, book=book, plies=BOOK_PLIES, concurrency=concurrency,
                                     pin=not getattr(args, "no_pin", False), fastchess=fastchess, extra_args=extra,
                                     pairs_per_job=getattr(args, "pairs_per_job", None) or 4,
                                     quiet=getattr(args, "quiet", False), label="TOURNAMENT", time_control=tc,
                                     meta=meta)
            completed = True
        finally:
            keep_pgn(results_dir / run_name, results_dir / f"{actual_cat}.pgn")
            if not log:
                shutil.rmtree(run_logs, ignore_errors=True)  # engines may create their log folder with logging off
            if not completed:  # MatchStopped / MatchInterrupted (with .result) or a refused engine: re-raised
                print(f"[TOURNAMENT] {actual_cat} (tc={tc}) was not completed: none of its results are stored"
                      + (f"; engine logs kept in {run_logs}" if log and run_logs.is_dir() else ""))

        print(f"\n[TOURNAMENT] {actual_cat} pairings this run (pentanomial Elo, 95% interval):")
        print(format_pairings(result["pairings"]))
        experiment_id = store_run(cnxn, result, engine_ids, engine_id, actual_cat, tc,
                                  dict(meta, run=result["run_id"], pairs=result["pairs"], games=result["games"]))
        # from here on the stored run's logs can always be loaded (--retry-ingest), even if the fit fails
        ids = keep_telemetry_ids(run_logs, experiment_id, engine_ids, opponents_in(pairings),
                                 result["run_dir"]) if log else None

        try:
            fit_id, _ = ratings.fit(cnxn, actual_cat)
        except BaseException:
            if ids is not None:
                print(f"[DATA] Engine logs kept; load them with: python -m tests.tournament --retry-ingest {run_logs}")
            raise
        if fit_id:
            print()
            print(ratings.format_history(cnxn, fit_id, highlight={engine_id}))

        if log:
            load_telemetry(cnxn, run_logs, ids)
        else:
            print("[DATA] Engine logging was disabled, skipping JSONL ingestion.")
        played.append(dict(category=actual_cat, time_control=tc, experiment_id=experiment_id,
                           run_dir=result["run_dir"], pairings=result["pairings"]))
    print(f"\n[TOURNAMENT] Results: {results_dir}")
    return played


def main(argv=None):
    parser = argparse.ArgumentParser(description="Progression test: rate a new engine against earlier versions")
    parser.add_argument("--engine", help="Candidate engine path (required unless --retry-ingest)")
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
    parser.add_argument("--book", default=str(DEFAULT_BOOK), help=f"Opening book (.pgn or .epd), {BOOK_PLIES} plies")
    parser.add_argument("--concurrency", default=None, type=int,
                        help="Concurrent games (default: every physical core but one); each game runs on its own core")
    parser.add_argument("--fastchess",
                        default=str(FASTCHESS), help="Path to the fastchess executable")
    parser.add_argument("--no-pin", action="store_true", help="Leave the placement of games to the OS")
    parser.add_argument("--tb", default=None, help="Syzygy tablebase dir(s) for adjudication (WDL files are enough)")
    parser.add_argument("--tbpieces", type=int, default=None)
    parser.add_argument("--pairs-per-job", type=int, default=4,
                        help="Opening pairs per fastchess process (amortizes engine start-up)")
    parser.add_argument("--quiet", action="store_true", help="No per-game lines")
    parser.add_argument("--log", action="store_true", help="Engine telemetry logging (uploaded to the database)")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="Where each tournament's folder goes (default logs/tournament_results)")
    parser.add_argument("--logroot", default=TOURNAMENT_LOG_DIR,
                        help="Root of the engine telemetry logs (each run logs into <logroot>/<run>/<engine name>/)")
    parser.add_argument("--retry-ingest", metavar="RUN_LOG_DIR",
                        help="Only load the preserved telemetry of an earlier run (<logroot>/<run>) into the database")

    # Accept a pre-built Namespace (programmatic callers like release.py)
    # as well as a raw argv list / sys.argv (CLI callers).
    if isinstance(argv, argparse.Namespace):
        args = argv
    else:
        args = parser.parse_args(argv)

    if getattr(args, "retry_ingest", None):
        return etl.retry_ingest(args.retry_ingest)
    if not args.engine:
        parser.error("--engine is required")

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
    try:
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
            pairs_per_job=getattr(args, "pairs_per_job", 4),
            quiet=getattr(args, "quiet", False),
            log=getattr(args, "log", False),
            results_dir=getattr(args, "results_dir", RESULTS_DIR),
            logroot=getattr(args, "logroot", TOURNAMENT_LOG_DIR),
        )

        return run_tournament(ns, cnxn, engine_id)
    finally:
        cnxn.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"[TOURNAMENT] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
