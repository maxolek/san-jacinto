#!/usr/bin/env python3
"""Round-robin tournament runner for Elo calculation.

Runs the candidate engine against all engines in the database at up to 4 time
controls (bullet, blitz, rapid, classical) and computes Elo ratings using the
standard pairwise comparison method.

Time control categories:
  - ultra_fast: < 25s/game       (default: 0:10+0.01)
  - bullet:     < 2 min/game     (default: 1:00+0.3)
  - blitz:      < 10 min/game    (default: 3+0.03)
  - rapid:      < 30 min/game    (default: 10+0.1)
  - classical:  >= 30 min/game   (default: 60+0.6)

Usage (standalone):
  python -m tests.tournament --engine engines/1.0.exe --tc blitz --games 50

Called from release.py as:
  tournament.run_tournament(args, cnxn, engine_id)
"""
import argparse
import math
import os
import re
import subprocess
import sqlite3
import platform
from pathlib import Path
import shutil
from data import etl

system = platform.system()

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
ENGINES_DIR = PROJECT_ROOT / "engines"
ENGINES_DEV_DIR = ENGINES_DIR / "dev"
ENGINES_PROD_DIR = ENGINES_DIR / "prod"
LOGS_DIR = PROJECT_ROOT / "logs"
TOURNAMENT_LOG_DIR = LOGS_DIR / "tournament_logs"

# Default time controls per category
TC_DEFAULTS = {
    "ultra_fast": "0:10+0.01",
    "bullet":     "1:00+0.3",
    "blitz":      "3:00+0.03",
    "rapid":      "10:00+0.1",
    "classical":  "30:00+0.6",
}

# default elo for oldest engine to build each version off of
# only needed for first tournament run (after that new elos are used)
ANCHOR_ELO = 1500.0
RATING_FLOOR = 100.0

def get_all_engines(cnxn, n=3, first=False):
    """Get last n=3 engine versions ++ first version and their exe paths from the DB."""
    first_engine_str = """,
        first_engine AS (                                              -- this is okay since n-1 + first_engine ==> n engines total as desired
            SELECT id, version FROM engines ORDER BY id ASC LIMIT 1
        )
    """

    rows = cnxn.execute(
        f"""
        WITH last_n AS (
            SELECT id, version FROM engines ORDER BY id DESC LIMIT {n} -- n --> n-1 due to candidate inclusion and later filtered out
        ){first_engine_str if first else ""}
        SELECT id, version FROM last_n
        UNION
        SELECT id, version FROM first_engine
        """
    ).fetchall()

    engines = []
    for row in rows:
        engine_id, version = row[0], row[1]
        # Try common exe naming patterns
        exe_path = ENGINES_PROD_DIR / f"{version}.exe"

        if not exe_path.exists():
            exe_path = ENGINES_DEV_DIR / f"{version}.exe"
            if not exe_path.exists():
                continue
            else:
                print(f" [TOURNAMENT] Using DEV version of {version} instead (Prod not found)")
            
        engines.append({"id": engine_id, "version": version, "path": str(exe_path)})
    return engines


def get_engines_by_names(cnxn, names):
    """Look up engine records (id, version, path) for explicitly named versions.

    `names` is a list of version strings (e.g. ["1.0", "0.9.3"]). Any name not
    found in the DB, or whose exe can't be located, is skipped with a warning.
    """
    if not names:
        return []

    placeholders = ",".join("?" for _ in names)
    rows = cnxn.execute(
        f"SELECT id, version FROM engines WHERE version IN ({placeholders})",
        list(names),
    ).fetchall()

    found_versions = {row[1] for row in rows}
    missing = [n for n in names if n not in found_versions]
    if missing:
        print(f"[TOURNAMENT] WARNING: requested engine(s) not found in DB: {missing}")
        for _ in missing:
            engine_filepath = f"engines/dev/{_}.exe"
            print(f"[TOURNAMENT] Engine {_} not registered, registering now...")
            
            _id = etl.register_engine(cnxn, {"engine_path": engine_filepath})
            _version = etl.probe_engine_metadata(engine_filepath)['version']
            rows.append([_id, _version])

    engines = []
    for row in rows:
        engine_id, version = row[0], row[1]
        exe_path = ENGINES_PROD_DIR / f"{version}.exe"
        
        if not exe_path.exists():
            exe_path = ENGINES_DEV_DIR / f"{version}.exe"
            if not exe_path.exists():
                print(f"[TOURNAMENT] WARNING: exe not found for engine version={version}")
                continue
            else:
                print(f" [TOURNAMENT] Using DEV version of {version} instead (Prod not found)")
            
        engines.append({"id": engine_id, "version": version, "path": str(exe_path)})
    return engines


def parse_tournament_output(output, engine_names):
    """Parse cutechess-cli round-robin output into pairwise results.
    
    Returns dict: {nameA: {nameB: {"wins": W, "losses": L, "draws": D}}}
    """
    #results = {name: {"wins": 0, "losses": 0, "draws": 0, "games": 0} for name in engine_names}
    pairwise = {name: {} for name in engine_names}

    # Match lines like: "Score of EngA vs EngB: 5 - 3 - 2  [0.600] 10"
    score_re = re.compile(
        r"Score of (.+?) vs (.+?):\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)"
    )
    # Match per-game finished lines like:
    # Finished game 1 (Candidate vs v0.2.2): 1-0 {White mates}
    finished_re = re.compile(r"^Finished game \d+ \((.+?) vs (.+?)\):\s*(1-0|0-1|1/2-1/2|1/2 - 1/2)")

    for line in output.splitlines():
        m = score_re.search(line)
        if m:
            name_a, name_b = m.group(1).strip(), m.group(2).strip()
            w, l, d = int(m.group(3)), int(m.group(4)), int(m.group(5))
            if name_a in pairwise:
                pairwise[name_a][name_b] = {"wins": w, "losses": l, "draws": d}
            if name_b in pairwise:
                pairwise[name_a][name_b] = {"wins": w, "losses": l, "draws": d}
            continue

        m2 = finished_re.search(line)
        if m2:
            name_a, name_b = m2.group(1).strip(), m2.group(2).strip()
            result = m2.group(3).strip()
            # normalize names
            if name_a not in pairwise or name_b not in pairwise:
                continue
            pairwise.setdefault(name_a, {})
            pairwise.setdefault(name_b, {})
            wa = pairwise[name_a].get(name_b, {"wins": 0, "losses": 0, "draws": 0})
            wb = pairwise[name_b].get(name_a, {"wins": 0, "losses": 0, "draws": 0})
            if result == '1-0':
                wa["wins"] += 1
                wb["losses"] += 1
            elif result == '0-1':
                wa["losses"] += 1
                wb["wins"] += 1
            else:
                wa["draws"] += 1
                wb["draws"] += 1
            pairwise[name_a][name_b] = wa
            pairwise[name_b][name_a] = wb
            continue

    return pairwise


def get_anchor_elo(cnxn, tc_category):
    """
    get the last registred engines elo as the anchor.
    falls back to ANCHOR_ELO if no prior ratings exist (first ever engine)
    """
    elo_col = f"elo_{tc_category}"
    row = cnxn.execute(f"""
        SELECT {elo_col} FROM engine_ratings
        WHERE {elo_col} IS NOT NULL
        ORDER BY id DESC LIMIT 1     
    """).fetchone()

    if row and row[0] is not None: return float(row[0])
    else: return ANCHOR_ELO

def compute_elo_diff(wins, losses, draws):
    """Compute Elo difference from score. Returns None if no games played."""
    games = wins + losses + draws
    if games == 0:
        return None
    score = (wins + draws * 0.5) / games
    # Avoid log(0) at extremes
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    return round(elo, 1)

def get_opponent_elo(cnxn, engine_id, tc_category):
    """get stored ratings for opponents. if none exist, assign anchor"""
    elo_col = f"elo_{tc_category}"
    row = cnxn.execute(
        f"SELECT {elo_col} FROM engine_ratings WHERE engine_id = ? ORDER BY id DESC LIMIT 1",
            (engine_id,),
    ).fetchone()
    if row and row[0] is not None: return float(row[0])
    else: return ANCHOR_ELO

def update_engine_rating(cnxn, engine_id, tc_category, new_elo, games_played):
    """ update or insert an engines'rating for a given TC category """
    elo_col = f"elo_{tc_category}"
    games_col = f"games_{tc_category}"

    existing = cnxn.execute(
        "SELECT id FROM engine_ratings WHERE engine_id = ?", (engine_id,)
    ).fetchone()

    if existing:
        # accumulate games, overwrite elo
        cnxn.execute(
            f"UPDATE engine_ratings SET {elo_col} = ?, {games_col} = COALESCE({games_col}, 0) + ? WHERE engine_id = ?",
            (new_elo, games_played, engine_id)
        )
    else:
        cnxn.execute(
            f"INSERT INTO engine_ratings (engine_id, {elo_col}, {games_col}) VALUES (?, ?, ?)",
            (engine_id, new_elo, games_played)
        )
    cnxn.commit()

def compute_performance_ratings(pairwise_results, opponent_elos):
    """
    FIDE-style performance rating from pairwise results
    
    for each opponent
        exp_score = N / (1 + 10^((Rc - Ro) / 400))
        act_score = wins + .5*draws

    find Rc (candidate rating) such that sum(expected) = sum(actual)
    uses iterative bisection

    returns: (performance_rating, total_games) or (None, 0)
    """

    # collect actual scores and game counts per opp
    matchups = []
    for opp_name, result in pairwise_results.items():
        if opp_name not in opponent_elos:continue

        w, l, d = result['wins'], result['losses'], result['draws']
        games = w+l+d
        if games == 0: continue

        actual = w + .5*d
        matchups.append((actual, games, opponent_elos[opp_name]))

    if not matchups: return None, 0

    total_games = sum(m[1] for m in matchups)
    total_actual = sum(m[0] for m in matchups)

    # edge cases: perfect/zero score
    if total_actual >= total_games: 
        # won everything: cap at avg_opp + 600
        avg_opp = sum(m[2]*m[1] for m in matchups) / total_games
        return round(avg_opp + 800.0, 1), total_games
    if total_actual <= 0:
        avg_opp = sum(m[2]*m[1] for m in matchups) / total_games 
        return round(avg_opp - 800.0, 1), total_games 
    
    # bisection: find Rc where sum(expect) == sum(actual)
    lo, hi = -1000.0, 5000.0
    for _ in range(100):
        mid = (lo+hi)/2
        expected = 0.0
        for actual_s, games, opp_elo in matchups:
            expected += games / (1.0 + 10**((opp_elo - mid) / 400.0))
        if expected < total_actual:
            lo = mid 
        else:
            hi = mid 

    return round((lo+hi)/2.0, 1), total_games


def compute_pool_elo(candidate_pairwise, opponents, opponent_ratings):
    """
    compute candidate elo from pairwise results against rated opponents.
    
    for each opponent with a known rating:
            candidate_elo_estimate = opponent_rating + elo_diff(candidate vs opponent)
            
    final rating = weighted avg of estimated (w=games_played)
    """
    estimates = []
    weights= []
    total_games = 0

    for opp in opponents:
        opp_name = f"v{opp['version']}"
        if opp_name not in candidate_pairwise: continue 

        pair = candidate_pairwise[opp_name]
        w, l, d = pair['wins'], pair['losses'], pair['draws']
        games = w+l+d
        if games == 0: continue 

        diff = compute_elo_diff(w, l, d)
        if diff is None: continue  

        opp_elo = opponent_ratings.get(opp['id'], ANCHOR_ELO)
        estimates.append(opp_elo+diff)
        weights.append(games)
        total_games += games 

    if not estimates: return None, 0

    # w_avg
    elo = sum(e*w for e,w in zip(estimates, weights)) / sum(weights)
    return round(elo, 1), total_games

def run_cutechess_tournament(candidate_path, opponents, tc, games_per_pair, cutechess_cli, concurrency = 2, book=None):
    """Run candidate vs all opponents using cutechess-cli.
    
    Returns raw stdout text for parsing.
    """
    TOURNAMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Build engine blocks
    engine_blocks = [
        "-engine", f"name=Candidate", f"cmd={os.path.abspath(candidate_path)}",
        f"dir={os.path.dirname(os.path.abspath(candidate_path))}",
    ]
    
    opponent_names = []
    for opp in opponents:
        name = f"v{opp['version']}"
        opponent_names.append(name)
        engine_blocks += [
            "-engine", f"name={name}", f"cmd={os.path.abspath(opp['path'])}",
            f"dir={os.path.dirname(os.path.abspath(opp['path']))}",
        ]

    # Book
    book_block = []
    if book:
        ext = os.path.splitext(book)[1][1:]
        book_block = [
            "-openings", f"file={os.path.abspath(book)}",
            f"format={ext}", "order=random", "plies=16",
        ]

    cmd = [
        cutechess_cli,
    ] + engine_blocks + [
        "-each", "proto=uci", f"tc={tc}",
        "-tournament", "round-robin",
        "-games", str(games_per_pair),
        "-repeat",
        "-maxmoves", "200",
        "-concurrency", f"{concurrency}",
        "-pgnout", str(TOURNAMENT_LOG_DIR / "tournament.pgn"),
    ] + book_block

    print(f"[TOURNAMENT] Running: tc={tc}, {len(opponents)} opponents, {games_per_pair} games each")
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
        print(f"[TOURNAMENT] WARNING: cutechess-cli exited with code {proc.returncode}")

    return "".join(output_lines), ["Candidate"] + opponent_names


def classify_tc(tc_str):
    """Classify a time control string into bullet/blitz/rapid/classical.
    
    Parses formats: "seconds+increment", "min:sec+increment", "seconds"
    Total game time estimate = base_time * 2 + 40 * increment (typical game ~80 plies)
    """
    tc_str = tc_str.strip()
    
    # Parse base time
    if ":" in tc_str.split("+")[0]:
        # "min:sec" format
        parts = tc_str.split("+")
        time_parts = parts[0].split(":")
        base_sec = int(time_parts[0]) * 60 + float(time_parts[1]) if len(time_parts) == 2 else float(time_parts[0]) * 60
        inc = float(parts[1]) if len(parts) > 1 else 0
    else:
        parts = tc_str.split("+")
        base_sec = float(parts[0])
        inc = float(parts[1]) if len(parts) > 1 else 0

    # Estimated game time in minutes (base + ~40 moves of increment per side)
    game_time_min = (base_sec + 40 * inc) / 60.0

    # ultra-fast:  < 15 seconds total game time (0.25 minutes)
    if game_time_min < 0.25:
        return "ultra_fast"
    elif game_time_min < 2:
        return "bullet"
    elif game_time_min < 10:
        return "blitz"
    elif game_time_min < 30:
        return "rapid"
    else:
        return "classical"


def run_tournament(args, cnxn, engine_id):
    """Main entry point from release.py.
    
    Runs tournaments at requested time controls and logs ratings.
    """
    candidate_path = args.engine
    cutechess_cli = args.cutechess_cli
    book = getattr(args, "opening_book", None)
    games_per_pair = getattr(args, "tournament_games", 20)
    tc_categories = getattr(args, "tournament_tc", ["blitz"])
    n_engines = getattr(args, "tournament_engines", 3)
    listed_opponents = getattr(args, "opps", [])
    concurrency = getattr(args, "concurrency")

    if len(listed_opponents) == 0:
        # Get all other engines from DB
        all_engines = get_all_engines(cnxn, n_engines, args.first_engine)
        # Exclude the candidate itself
        opponents = [e for e in all_engines if e["id"] != engine_id]
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

    # Start experiment
    experiment_id = etl.start_experiment(cnxn, "tournament", engine_id)

    ratings = {}
    for tc_cat in tc_categories:
        tc = TC_DEFAULTS.get(tc_cat, tc_cat)
        actual_cat = classify_tc(tc)

        print(f"\n[TOURNAMENT] === {actual_cat.upper()} (tc={tc}) ===")

        output, names = run_cutechess_tournament(
            candidate_path, opponents, tc, games_per_pair, cutechess_cli, concurrency, book
        )

        pairwise = parse_tournament_output(output, names)
        candidate_pairwise = pairwise.get("Candidate", {})

        # look up each opp's elo
        opponent_elos = {}
        for opp in opponents:
            opp_name = f"v{opp['version']}"
            opp_elo = get_opponent_elo(cnxn, opp['id'], actual_cat)
            opponent_elos[opp_name] = opp_elo 

        # print pairwise breakdown
        for opp in opponents:
            opp_name = f"v{opp['version']}"
            
            if opp_name in candidate_pairwise:
                p = candidate_pairwise[opp_name]
                diff = compute_elo_diff(p['wins'], p['losses'], p['draws'])
                opp_elo = opponent_elos[opp_name]
                diff_str = f"diff={diff:+.0f}" if diff is not None else "diff=N/A"
                print(f"  vs {opp_name:12s} ({opp_elo:.0f}): "
                      f"+{p['wins']} -{p['losses']} ={p['draws']}  {diff_str}")
                
        # compute performance rating
        perf_elo, _ = compute_performance_ratings(candidate_pairwise, opponent_elos)

        # compute pool elo (w_avg of opp_rating + pairwise_diff)
        candidate_elo, total_games = compute_pool_elo(candidate_pairwise, opponents,
                                                      {o['id']: opponent_elos[f"v{o['version']}"] for o in opponents})
        if candidate_elo is not None: 
            candidate_elo = max(candidate_elo, RATING_FLOOR)

        ratings[actual_cat] = {
            "elo": candidate_elo,
            "games": total_games,
        }

        avg_opp = (sum(opponent_elos[f"v{o['version']}"] for o in opponents) / len(opponents)
                   if opponents else 0)
        print(f"[TOURNAMENT] {actual_cat}: pool_elo={candidate_elo} perf_elo={candidate_elo} "
              f"avg_opp={avg_opp:.0f} games={total_games}")

    # update opponent ratings using standard elo formula
    # use full round-robin results (cand=current or anchor, opp=stored)
    # K-factor: 32 for engine with fewer games, 16 for established
    if candidate_elo is not None:
        elo_col = f"elo_{actual_cat}"
        candidate_old_elo = get_opponent_elo(cnxn, engine_id, actual_cat)
        candidate_has_rating = cnxn.execute(
            f"SELECT 1 FROM engine_ratings WHERE engine_id = ? AND {elo_col} IS NOT NULL",
            (engine_id,),
        ).fetchone() is not None

        candidate_start_elo = candidate_old_elo if candidate_has_rating else candidate_elo
        elo_map = {'Candidate': candidate_start_elo}
        elo_map.update(opponent_elos)

        # map to engine id
        id_map = {"Candidate": engine_id}
        for opp in opponents: id_map[f"v{opp['version']}"] = opp['id']

        K = 32
        # accumulate elo deltas for each engine across all pairings
        elo_deltas = {name: 0.0 for name in elo_map}
        games_count = {name: 0 for name in elo_map}

        for name_a, results_a in pairwise.items():
            if name_a not in elo_map: continue

            for name_b, p in results_a.items():
                if name_b not in elo_map: continue 
                # only process each pair once (alphabetically)
                if name_a >= name_b: continue 
                games = p['wins'] + p['losses'] + p['draws']
                if games == 0: continue 

                elo_a, elo_b = elo_map[name_a], elo_map[name_b]
                actual_a = p['wins'] + .5*p['draws']
                actual_b = p['losses'] + .5*p['draws']
                expected_a = games / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
                expected_b = games - expected_a

                elo_deltas[name_a] += K * (actual_a - expected_a)
                elo_deltas[name_b] += K * (actual_b - expected_b)
                games_count[name_a] += games
                games_count[name_b] += games
            
        print(f"\n[RATINGS] Updating ratings from all round-robin results:")
        for name, delta in elo_deltas.items():
            if games_count[name] == 0: continue 
            if name == 'Candidate' and not candidate_has_rating:
                continue

            old_elo = elo_map[name]
            new_elo = max(round(old_elo + delta, 1), RATING_FLOOR)
            print(f"  {name:12s}: {old_elo:.0f} -> {new_elo:.0f} "
                  f"(delta={delta:+.1f}, games={games_count[name]})")
            # update engine ratings in DB for this engine name
            engine_db_id = id_map.get(name)
            if engine_db_id is None:
                print(f"[WARN] No DB id found for {name}, skipping rating update")
                continue
            update_engine_rating(cnxn, engine_db_id, actual_cat, new_elo, games_count[name])

        # new candidates dont have an elo_delta
        if not candidate_has_rating and games_count.get('Candidate', 0) > 0:
            new_elo = round(candidate_elo, 1)
            candidate_games = games_count['Candidate']
            print(f"  Candidate    : (new) -> {new_elo:.0f} (games={candidate_games})")
            update_engine_rating(cnxn, engine_id, actual_cat, new_elo, candidate_games)

    # Consolidate per-instance logs and upload, like SPRT
    for basename in ("game", "search", "timing", "root_moves"):
        parts = sorted(TOURNAMENT_LOG_DIR.glob(f"{basename}_*.jsonl"))
        if not parts:
            continue
        out_path = TOURNAMENT_LOG_DIR / f"{basename}.jsonl"
        with open(out_path, "w", encoding="utf-8") as out_f:
            for part in parts:
                with open(part, "r", encoding="utf-8") as in_f:
                    shutil.copyfileobj(in_f, out_f)

    game_json = TOURNAMENT_LOG_DIR / "game.jsonl"
    search_json = TOURNAMENT_LOG_DIR / "search.jsonl"
    timing_json = TOURNAMENT_LOG_DIR / "timing.jsonl"
    root_moves_json = TOURNAMENT_LOG_DIR / "root_moves.jsonl"

    game_map = {}
    ingestion_ok = True
    if game_json.exists():
        try:
            game_map = etl.bulk_log_game(cnxn, game_json, experiment_id)
        except Exception as e:
            print(f"[DATA] bulk_log_game failed: {e}")
            ingestion_ok = False

    if search_json.exists():
        try:
            etl.bulk_log_search_and_timing(
                cnxn,
                search_json,
                game_map,
                timing_path=timing_json if timing_json.exists() else None,
                root_moves_path=root_moves_json if root_moves_json.exists() else None,
            )
        except Exception as e:
            print(f"[DATA] bulk_log_search_and_timing failed: {e}")
            ingestion_ok = False

    # finalize experiment
    from datetime import datetime, timezone
    etl.update_experiment(cnxn, experiment_id, {"end_time_utc": datetime.now(timezone.utc).isoformat()})

    # Only clear log directory if ingestion succeeded
    if ingestion_ok:
        etl.clear_log_dir(TOURNAMENT_LOG_DIR)
    else:
        print("[DATA] Log files preserved due to ingestion failure.")

    # !!! outdated ... printing perf-rating not new-rating !!!
    #print(f"\n[TOURNAMENT] Ratings logged for new engine (id={engine_id}"))
    #for cat, r in ratings.items():
    #    print(f"  {cat:12s}: Elo={r['elo']}, games={r['games']}")



def main(argv=None):
    parser = argparse.ArgumentParser(description="Round-robin tournament runner")
    parser.add_argument("--engine", required=True, help="Candidate engine path")
    parser.add_argument("--opps", nargs="+", default=[],
                        help="Explicit list of opponent engine version names "
                             "(e.g. --opps 1.0 0.9.3). Overrides --n_engines auto-selection.")
    parser.add_argument("--n_engines", default=3, help="Number of last versions in tournament")
    parser.add_argument("--first_engine", action="store_true", help="Whether to include the first ever PROD engine on top of n_engines")
    parser.add_argument("--tc", nargs="+", default=["blitz"],
                        help="Time controls to run. Accepts category names "
                             "(ultra_fast, bullet, blitz, rapid, classical), which "
                             "map to their default tc string, and/or explicit tc "
                             "strings (e.g. 15+0.15, 1:00+0.6) which are used as-is "
                             "and auto-classified into a category for rating purposes.")
    parser.add_argument("--games", type=int, default=10, help="Games per opponent per TC")
    parser.add_argument("--book", default=str(PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn"))
    parser.add_argument("--concurrency", default=2, type=int, help="Number of games to run simultaneously (uses 2*concurrency threads)")
    parser.add_argument("--cutechess-cli",
                        default=r"C:\Program Files (x86)\Cute Chess\cutechess-cli.exe")
 
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
 
    if system == "Windows":
        db_path = "F:/databases/chess.db"
    else:
        db_path = str(Path.home() / "Documents/databases/chess.db")
 
    cnxn = sqlite3.connect(db_path)
 
    # Get candidate engine ID
    meta = etl.probe_engine_metadata(args.engine)
    engine_id = etl.get_engine_id(cnxn, version=meta["version"])
 
    # auto-register if not found
    if engine_id is None:
        print(f"[TOURNAMENT] Engine {meta['version']} not registered, registering now...")
        engine_id = etl.register_engine(cnxn, {"engine_path": args.engine})
 
    # Build namespace for run_tournament
    ns = argparse.Namespace(
        engine=args.engine,
        cutechess_cli=getattr(args, "cutechess_cli"),
        opening_book=args.book,
        tournament_games=args.games,
        tournament_tc=args.tc,
        tournament_engines=args.n_engines,
        opps=args.opps,
        concurrency=args.concurrency,
    )
 
    run_tournament(ns, cnxn, engine_id)
    cnxn.close()
 
 
if __name__ == "__main__":
    main()
 