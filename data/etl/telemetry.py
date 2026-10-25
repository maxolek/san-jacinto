"""Engine telemetry of a match-runner run, linked to the games the runner recorded.

tests/sprt.py and tests/tournament.py give every engine of a run its own log folder
<run_logs>/<side>/ (UCI option log_dir), so each engine's logs are attributed to its own
build. The runner's games are authoritative (games rows, source 'match', stored by
ingest.bulk_log_match_games): an engine game is linked to the played game its searches
were made in (match_games.link_engine_games), and only an engine game that matches no
played game gets a row of its own (source 'engine'), so a played game is never stored twice.

Each side is one transaction, loaded completely or not at all (even if the process is
killed). The load is recorded in telemetry_loads (experiment_id, side), whose key refuses a
second load of the same side, and the side's logs are then removed, so a retry loads each
side exactly once. A side found in telemetry_loads is skipped before anything is read; the
run's games are read (its PGN replayed) once per load, and each side is linked to them
before its write transaction opens, which then holds only the inserts (a concurrent loader
that stored the side meanwhile is caught by the key). <run_logs>/engines.json holds the
run's ids for a retry (retry_ingest):

  an SPRT's: {"candidate": id, "baseline": id, "experiment": id, "loaded": [...], "run_dir": ...,
      "run_id": ...}; the sides are the folders candidate/ and baseline/, the runner's
      engines Candidate and Baseline.
  a run of any number of engines (telemetry_ids): {"engines": {runner name: engine id},
      "opponents": {runner name: engine id or null}, "experiment", "loaded", "run_dir",
      "run_id"}; each engine logs into the folder of its runner name, which is also its side.

An engine game that links to no played game is stored with the engine's opponent in the run
(`opponents`) as its other engine: NULL when the engine played several (a tournament's
candidate), as its logs do not say which.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from .ingest import bulk_log_game, bulk_log_search_and_timing
from .match_games import link_engine_games, pgn_moves, read_run, stored_games
from .paths import connect_raw, get_jsonl_paths
from .schema import ensure_schema
from .utils import consolidate_instance_logs


def telemetry_ids(experiment_id, engines, opponents=None, run_dir=None):
    """The ids (engines.json) of a run of any number of engines.

    engines: {runner name: engine id}, each engine logging into <run_logs>/<runner name>/;
    opponents: {runner name: the engine id of its only opponent in the run, or None};
    run_dir: the run folder whose games.jsonl the played games were stored from.
    """
    run_dir = Path(run_dir).resolve() if run_dir else None
    return dict(experiment=experiment_id, loaded=[], engines=dict(engines),
                opponents={name: (opponents or {}).get(name) for name in engines},
                run_dir=str(run_dir) if run_dir else None, run_id=run_dir.name if run_dir else None)


def run_sides(ids):
    """[(side, runner engine name, engine id, opponent engine id or None)] of a run's ids."""
    if "engines" in ids:
        opponents = ids.get("opponents") or {}
        return [(name, name, engine_id, opponents.get(name)) for name, engine_id in ids["engines"].items()]
    # an SPRT's (tests/sprt.py): folders candidate/ and baseline/
    return [("candidate", "Candidate", ids["candidate"], ids["baseline"]),
            ("baseline", "Baseline", ids["baseline"], ids["candidate"])]


def _recorded_games(cnxn, run_id, experiment_id, sides, by_engine):
    """(games, moves) of the run's stored played games, for linking when the run folder is gone.

    An SPRT's games are told apart by number (game 2k+1 of a pair has the Candidate as
    White); a run of several pairings needs each game's engines, named by their ids.
    """
    games, moves = stored_games(cnxn, run_id, experiment_id)
    if not by_engine:
        return games, moves
    names = {engine_id: name for _, name, engine_id, _ in sides}
    if len(names) < len(sides):
        raise ValueError("two engines of the run have the same engine id, so the stored games cannot tell them "
                         "apart; keep the run folder to link their telemetry")
    engines = {number: (white, black) for number, white, black in cnxn.execute(
        "SELECT run_game, white_engine_id, black_engine_id FROM games WHERE source = 'match' AND run_id = ? "
        "AND experiment_id = ?", (run_id, experiment_id)).fetchall()}
    for game in games:
        white, black = engines[game["game"]]
        game["white"], game["black"] = names.get(white), names.get(black)
    return games, moves


def ingest_telemetry(cnxn, run_logs, ids):
    """Load one run's engine logs (<run_logs>/<side>/), each side attributed to its own build.

    `ids`: the run's engine and experiment ids, as in its engines.json (see the module doc),
    and (when the runner's games were stored) its run_dir / run_id. Every run has its own
    log folder, so logs of other, aborted or concurrent runs are never mixed in. Updates
    ids["loaded"] and engines.json as sides are loaded. Returns True if everything was loaded.
    """
    run_logs = Path(run_logs)
    ensure_schema(cnxn)  # e.g. a migration added since the run (a retry): no transaction is open yet
    sides = run_sides(ids)
    run_dir = Path(ids["run_dir"]) if ids.get("run_dir") else None
    played, recorded = {}, {}  # run game number -> games.id of this run's played games; how to link to them
    if run_dir is not None:
        run_id = ids.get("run_id") or run_dir.name
        played = dict(cnxn.execute(
            "SELECT run_game, id FROM games WHERE source = 'match' AND run_id = ? AND experiment_id = ?",
            (run_id, ids["experiment"])).fetchall())
        if played and not (run_dir / "games.jsonl").exists():  # the run folder is gone: its stored games
            try:
                games, moves = _recorded_games(cnxn, run_id, ids["experiment"], sides, "engines" in ids)
            except ValueError as e:
                print(f"[DATA] JSONL ingestion failed: {e}")
                return False
            recorded = dict(games=games, moves=moves)
    ok = True
    source = None  # (games, moves) of the run's played games to link to: read once, for every side
    for side, name, engine_id, opponent_id in sides:
        log_dir = run_logs / side
        if side in ids.setdefault("loaded", []) or not log_dir.is_dir():
            continue
        print(f"[DATA] Logging {side} telemetry ...")
        if _loaded(cnxn, ids["experiment"], side):
            print(f"[DATA] {side} telemetry of experiment {ids['experiment']} was loaded already")
        else:
            try:
                # what only reads (the logs, the run's games; python-chess replays the PGN) comes before
                # the side's write transaction, so other writers are not kept waiting for it
                consolidate_instance_logs(log_dir)
                paths = get_jsonl_paths(log_dir)
                linked = {}
                if played:
                    if source is None:
                        source = recorded or dict(games=read_run(run_dir)[0], moves=pgn_moves(run_dir))
                    links = link_engine_games(log_dir, run_dir, engine_name=name, **source)
                    linked = {uuid: played[number] for uuid, number in links.items() if number in played}
            except Exception as e:
                print(f"[DATA] JSONL ingestion failed ({side}): {e}")
                ok = False
                continue
            try:
                # first, in the side's transaction: a side another loader stored meanwhile fails here,
                # before any of its rows are added again (the primary key)
                cnxn.execute("INSERT INTO telemetry_loads (experiment_id, side) VALUES (?, ?)",
                             (ids["experiment"], side))
            except sqlite3.IntegrityError as e:
                cnxn.rollback()
                if not _loaded(cnxn, ids["experiment"], side):  # another constraint, not the key
                    print(f"[DATA] JSONL ingestion failed ({side}): {e}")
                    ok = False
                    continue
                print(f"[DATA] {side} telemetry of experiment {ids['experiment']} was loaded already")
            else:
                try:
                    game_map = dict(linked)
                    if paths["game"].exists():
                        game_map.update(bulk_log_game(cnxn, paths["game"], ids["experiment"],
                                                      second_engine_id=opponent_id, engine_id=engine_id,
                                                      commit=False, skip_uuids=set(linked), source="engine"))
                    if paths["search"].exists():
                        bulk_log_search_and_timing(
                            cnxn,
                            paths["search"],
                            game_map,
                            engine_id=engine_id,
                            timing_path=paths["timing"] if paths["timing"].exists() else None,
                            root_moves_path=paths["root_moves"] if paths["root_moves"].exists() else None,
                            commit=False,
                        )
                    cnxn.commit()
                    if played:
                        print(f"[DATA] {side}: {len(linked)} engine game(s) linked to the played games, "
                              f"{len(game_map) - len(linked)} stored on their own")
                except Exception as e:
                    # any error, a record a constraint refuses (sqlite3.IntegrityError) included: the side
                    # and its telemetry_loads row are rolled back, its logs are kept for a retry
                    cnxn.rollback()
                    print(f"[DATA] JSONL ingestion failed ({side}): {e}")
                    ok = False
                    continue
        ids["loaded"].append(side)
        (run_logs / "engines.json").write_text(json.dumps(ids), encoding="utf-8")
        shutil.rmtree(log_dir, ignore_errors=True)
    return ok


def _loaded(cnxn, experiment_id, side):
    """True if the side's telemetry was stored (telemetry_loads)."""
    return cnxn.execute("SELECT 1 FROM telemetry_loads WHERE experiment_id = ? AND side = ?",
                        (experiment_id, side)).fetchone() is not None


def _raw_db(db=None):
    from .paths import RAW_DB  # read when called: CHESS_RAW_DB may be set after import
    return Path(db or RAW_DB)


def retry_ingest(run_logs, db=None):
    """Load a run's preserved telemetry with the engine ids recorded for it (engines.json).

    On success the run's log folder is removed. Returns True if everything was loaded.
    """
    run_logs = Path(run_logs)
    if not (run_logs / "engines.json").exists():
        raise FileNotFoundError(f"{run_logs / 'engines.json'} not found: pass the run's log folder "
                                "(<logroot>/<run>) that the run kept")
    db_path = _raw_db(db)
    if not db_path.exists():  # connecting would create an empty database the run's ids do not belong to
        raise FileNotFoundError(f"Database {db_path} not found: pass the --db (or set CHESS_RAW_DB) the run "
                                "was stored in")
    ids = json.loads((run_logs / "engines.json").read_text(encoding="utf-8"))
    cnxn = connect_raw(db_path)
    try:
        if ingest_telemetry(cnxn, run_logs, ids):
            shutil.rmtree(run_logs, ignore_errors=True)
            print(f"[DATA] Telemetry of experiment {ids['experiment']} loaded.")
            return True
        return False
    finally:
        cnxn.close()
