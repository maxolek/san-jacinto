"""Bulk ingestion functions: games, searches, timing, STS, SPRT."""
import json
import shutil
from pathlib import Path

from .paths import GAME_JSON, SEARCH_JSON, TIMING_JSON, ROOT_MOVES_JSON, GAMES_LOG_DIR, LOG_DIRS, get_jsonl_paths
from .utils import safe_val, safe, consolidate_instance_logs
from .db import get_engine_id, clear_log_dir, extract_engine_id_from_search, probe_engine_metadata, save_engine_config, register_engine
from .schema import ensure_schema
from .columns import MISMATCH, coerce

def _iter_json_objects_from_path(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                print(f"[WARN] skipping malformed JSON line {line_no} in {path}")
                continue

            if not isinstance(data, dict):
                print(f"[WARN] skipping non-object JSON line {line_no} in {path}: {type(data).__name__}")
                continue

            yield data

def _count_jsonl_rows(path):
    """Count non-empty lines in a JSONL file."""
    with open(path) as f:
        return sum(1 for line in f if line.strip())

# ─────────────────────────────────────────────────────────────────────────────
# Log fields -> columns
# ─────────────────────────────────────────────────────────────────────────────
# The loaders below read these log fields explicitly (some into a column of another
# name). Every other field goes into the column of the same name when the table has
# one (per-depth arrays: itdepth_x / treedepth_x -> column x), converted to its type,
# so a statistic a new engine version logs only needs a migration
# (data/databases/migrate_schema.py); earlier rows keep NULL. Nothing is lost on the
# way: a field with no column yet, or a value that doesn't fit its column's type, is
# kept in the row's extra_fields (JSON) and reported, and the migration that adds the
# column fills it from there. A field an engine doesn't log is NULL ("not measured"),
# never 0.

LINK_FIELDS = {"engine_id", "game_uuid", "search_uuid"}  # resolved to row ids, not stored as such
BOOKKEEPING_FIELDS = {"instance_id", "session", "type"}  # logging internals: stored only if a column exists
SEARCH_FIELDS = {
    "fen", "ply", "time_ms", "root_eval", "completed_depth", "max_depth", "max_qdepth", "best_move",
    "principal_variation", "nodes", "qnodes", "tt_stores", "tt_hits", "tt_fill", "fail_highs", "fail_lows",
    "aspiration_fail_high_researches", "aspiration_fail_low_researches", "fail_high_index",
    "see_prunes", "delta_prunes", "nmp", "nmp_failhigh", "tt_overwritten",
    "nmp_fail",  # nmp_failhigh before July 2026
}
ITERATION_FIELDS = {
    "itdepth_time_ms", "itdepth_eval", "itdepth_move", "itdepth_qdepth", "itdepth_nodes", "itdepth_qnodes",
    "itdepth_ttstores", "itdepth_tthits", "itdepth_ttfill", "itdepth_fail_highs", "itdepth_fail_lows",
    "itdepth_aspiration_failhigh_researches", "itdepth_aspiration_faillow_researches", "itdepth_fh_index",
    "itdepth_see_prunes", "itdepth_delta_prunes", "itdepth_pvs_researches", "itdepth_nmp", "itdepth_nmp_failhigh",
    "itdepth_nmp_fail",
}
TREE_FIELDS = {
    "treedepth_nodes", "treedepth_qnodes", "treedepth_tt_stores", "treedepth_tt_hits", "treedepth_fail_highs",
    "treedepth_fail_lows", "treedepth_fh_index", "treedepth_see_prunes", "treedepth_delta_prunes",
    "treedepth_pvs_researches", "treedepth_nmp", "treedepth_nmp_failhigh",
    "treedepth_nmp_fail",
}
GAME_FIELDS = {
    "side", "wtime", "btime", "winc", "binc", "movestogo", "depth", "nodes", "movetime", "result", "reason",
    "opening", "start_fen", "moves", "time_s",
}

SEARCH_COLUMNS = (
    "engine_id", "game_id", "sts_id", "fen", "ply", "time_ms", "eval", "completed_depth", "depth", "qdepth", "move",
    "principal_variation", "nodes", "qnodes", "tt_stores", "tt_hits", "tt_fill",
    "fail_highs", "fail_lows", "fail_high_researches", "fail_low_researches",
    "fh_index_0", "fh_index_1", "fh_index_2", "fh_index_3", "fh_index_4to7", "fh_index_8plus",
    "see_prunes", "delta_prunes", "nmp", "nmp_failhigh", "tt_overwritten",
)
ITERATION_COLUMNS = (
    "search_id", "depth", "time_ms", "eval", "move", "qdepth",
    "nodes", "qnodes", "tt_stores", "tt_hits", "tt_fill",
    "fail_highs", "fail_lows", "fail_high_researches", "fail_low_researches",
    "fh_index_0", "fh_index_1", "fh_index_2", "fh_index_3", "fh_index_4to7", "fh_index_8plus",
    "see_prunes", "delta_prunes", "pvs_researches", "nmp", "nmp_failhigh",
)
TREE_COLUMNS = (
    "search_id", "depth",
    "nodes", "qnodes", "tt_stores", "tt_hits", "fail_highs", "fail_lows",
    "fh_index_0", "fh_index_1", "fh_index_2", "fh_index_3", "fh_index_4to7", "fh_index_8plus",
    "see_prunes", "delta_prunes", "pvs_researches", "nmp", "nmp_failhigh",
)
GAME_COLUMNS = (
    "experiment_id", "white_engine_id", "black_engine_id",
    "wtime", "btime", "winc", "binc", "movestogo", "depth", "nodes", "movetime",
    "result", "termination", "opening", "start_fen", "moves", "run_time_s",
)


class _PassThrough:
    """The log fields of one table that no explicit mapping reads: into same-named columns
    (converted to the column's type) or, failing that, into extra_fields."""

    def __init__(self, cnxn, table, explicit_columns, prefix=""):
        info = cnxn.execute(f'PRAGMA table_info("{table}")').fetchall()
        self.order = [row[1] for row in info]
        self.types = {row[1]: row[2] for row in info
                      if row[1] not in explicit_columns and row[1] not in ("id", "extra_fields")}
        self.has_extra_fields = "extra_fields" in self.order
        self.prefix = prefix
        self.used = set()        # columns some row fills
        self.unstored = set()    # fields without a column
        self.mismatched = set()  # fields with a value that doesn't fit their column

    def split(self, fields):
        """({column: value}, {field: value}) for the fields (name -> value) of one row."""
        stored, extra = {}, {}
        for field, value in fields.items():
            if value is None:
                continue
            column = field[len(self.prefix):]
            if column in self.types:
                converted = coerce(value, self.types[column])
                if converted is MISMATCH:
                    self.mismatched.add(field)
                    extra[field] = value
                else:
                    stored[column] = converted
                    self.used.add(column)
            elif field not in BOOKKEEPING_FIELDS:
                self.unstored.add(field)
                extra[field] = value
        return stored, extra

    def columns(self, explicit, rows):
        """INSERT columns for rows of (explicit values, stored, extra)."""
        extra = ["extra_fields"] if self.has_extra_fields and any(row[2] for row in rows) else []
        return tuple(explicit) + tuple(c for c in self.order if c in self.used) + tuple(extra)

    def values(self, explicit, rows):
        used = [c for c in self.order if c in self.used]
        with_extra = self.has_extra_fields and any(row[2] for row in rows)
        return [tuple(values) + tuple(stored.get(c) for c in used)
                + ((json.dumps(extra) if extra else None,) if with_extra else ())
                for values, stored, extra in rows]


def _report_fields(kind, passes):
    """Report fields that have no column, or a value of another type, for the given tables."""
    unstored = sorted(set().union(*(p.unstored for p in passes)))
    mismatched = sorted(set().union(*(p.mismatched for p in passes)))
    kept = "Kept in extra_fields" if all(p.has_extra_fields for p in passes) else "Not stored"
    if unstored:
        print(f"[ETL] {kind} log fields with no column: {', '.join(unstored)}. {kept}; add a migration "
              "(data/databases/migrate_schema.py) to give them a column, which is filled from extra_fields.")
    if mismatched:
        print(f"[ETL] {kind} log values that don't fit their column's type: {', '.join(mismatched)}. {kept}.")
    return unstored, mismatched


def _first(data, *fields):
    """The first of `fields` the record has (current name first, then older names)."""
    return next((f for f in fields if f in data), fields[0])


def _insert_sql(table, columns):
    names = ", ".join(f'"{c}"' for c in columns)
    return f'INSERT INTO "{table}" ({names}) VALUES ({", ".join("?" * len(columns))})'


def _at_depth(data, field, d, past_end=0):
    """data[field][d] of a per-depth array: None if the engine does not log the field,
    `past_end` beyond the end of the array."""
    values = data.get(field)
    if not isinstance(values, list):
        return None
    return values[d] if d < len(values) else past_end


def _per_depth(data, field, d):
    """The value of an unmapped per-depth field for depth index d (None past its end;
    a field that isn't an array is kept once, with the first depth)."""
    values = data[field]
    if isinstance(values, list):
        return values[d] if d < len(values) else None
    return values if d == 0 else None


_ambiguous_versions_reported = set()


def _newest_build(cnxn, version):
    """The id of the newest engine row with this version string (None if none), reporting
    once when several builds share it: their logs can't tell them apart."""
    ids = [row[0] for row in cnxn.execute("SELECT id FROM engines WHERE version = ? ORDER BY id DESC", (version,))]
    if len(ids) > 1 and version not in _ambiguous_versions_reported:
        _ambiguous_versions_reported.add(version)
        print(f"[ETL] {len(ids)} builds are registered as version {version} (ids {', '.join(map(str, ids))}); "
              f"its log records go to the newest, id {ids[0]}. Pass engine ids to attribute them exactly.")
    return ids[0] if ids else None


def _engine_for_version(cnxn, version, by_version=None):
    """The engine id of a log record's version string: from `by_version` when given (exact),
    else the newest build of that version (registered first if unknown)."""
    if by_version and version in by_version:
        return by_version[version]
    return _newest_build(cnxn, version) or get_engine_id(cnxn, version)


def _prompt_register_engine(cnxn, version, sample_record):
    """Ask the user whether to register a new, unrecognized engine version.

    Uses fields from `sample_record` (the JSONL row that referenced this
    version) to suggest what to register, rather than guessing a fake path.
    """
    print(f"\n[ACTION NEEDED] Engine version '{version}' is not registered.")
    print(f"  Sample record: {sample_record}")

    suggested_path = sample_record.get("engine_path")
    if suggested_path:
        print(f"  Found engine_path in log: {suggested_path}")

    answer = input(f"Register engine '{version}'? [y/N]: ").strip().lower()
    if answer != "y":
        print(f"[INFO] Skipping engine '{version}' — not registered.")
        return None

    engine_path = suggested_path or input(f"  Enter engine_path for '{version}': ").strip()

    register_engine(cnxn, {"engine_path": engine_path, "version": version})

    row = cnxn.execute(
        "SELECT id FROM engines WHERE version=?", (version,)
    ).fetchone()

    if row is None:
        print(f"[ERROR] Registration failed for '{version}' — skipping.")
        return None

    print(f"[INFO] Registered '{version}' -> engine id {row[0]}")
    return row[0]


# ─────────────────────────────────────────────────────────────────────────────
# Top-level directory ingestion
# ─────────────────────────────────────────────────────────────────────────────

def log_games_directory(cnxn):
    """Ingest all game logs from the standard log directory."""
    ingest_log_dir(cnxn, GAMES_LOG_DIR)


def ingest_log_dir(cnxn, log_dir, clear=True):
    """Ingest all JSONL files from a given log directory."""
    ensure_schema(cnxn)
    paths = get_jsonl_paths(log_dir)
    game_map = {}
    search_ok = False

    consolidate_instance_logs(log_dir)

    if paths["game"].is_file():
        game_map = bulk_log_game(cnxn, paths["game"])
    else:
        print(f"[INFO] No game log found: {paths['game']}")

    if paths["search"].is_file():
        expected = _count_jsonl_rows(paths['search'])
        counts = bulk_log_search_and_timing(
            cnxn,
            paths["search"],
            game_map,
            timing_path=paths["timing"] if paths["timing"].is_file() else None,
            root_moves_path=paths["root_moves"] if paths["root_moves"].is_file() else None
        )
        accouted_for = counts['searches'] + counts['skipped']
        search_ok = accouted_for == expected
        if not search_ok:
            print(
                    f"[WARN] Search log row mismatch: {expected} rows in file, "
                    f"only {accouted_for} inserted+skipped ({counts})"
                )
    else:
        print(f"[INFO] No search log found: {paths['search']}")

    # clear only if searches+games
    if clear and search_ok and paths["search"].is_file():
        clear_log_dir(log_dir)


def ingest_all_log_dirs(cnxn):
    """Sweep all known log directories and ingest any JSONL data found."""
    for log_dir in LOG_DIRS:
        if not log_dir.exists():
            continue
        paths = get_jsonl_paths(log_dir)
        has_data = any(p.is_file() for p in paths.values())
        if has_data:
            print(f"[INGEST] Processing: {log_dir}")
            ingest_log_dir(cnxn, log_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment management
# ─────────────────────────────────────────────────────────────────────────────

def start_experiment(cnxn, experiment, engine_id, comparison_engine_id=None, info=None):
    """Start an experiment and return its ID."""
    cursor = cnxn.cursor()
    cursor.execute(
        """
        INSERT INTO experiments (engine_id, type, comparison_engine_id, metadata, start_time_utc)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (engine_id, experiment, comparison_engine_id, info)
    )
    cnxn.commit()
    cursor.execute("SELECT last_insert_rowid()")
    return cursor.fetchone()[0]


def update_experiment(cnxn, experiment_id, info=None):
    """Update experiment metadata columns."""
    if not info:
        return

    cols = ", ".join(f"{col} = ?" for col in info.keys())
    values = list(info.values()) + [experiment_id]

    sql = f"""
        UPDATE experiments
        SET {cols}
        WHERE id = ?
    """

    cur = cnxn.cursor()
    cur.execute(sql, values)
    cnxn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Single-row logging
# ─────────────────────────────────────────────────────────────────────────────


def log_perft(cnxn, perft_info) -> int:
    """Log a perft result and return its row ID."""
    cur = cnxn.execute(
        """
        INSERT INTO perft (
            experiment_id, fen, depth, nodes, expected_nodes, correct, time_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            perft_info["experiment_id"],
            perft_info["fen"],
            perft_info.get("depth"),
            perft_info.get("nodes"),
            perft_info.get("expected_nodes"),
            perft_info.get("correct"),
            perft_info.get("time_ms"),
        )
    )
    cnxn.commit()
    return cur.lastrowid

def log_engine_ratings(cnxn, engine_id, ratings) -> int:
    """Log engine elo ratings by time control category
    
    ratings: dict with keys like 'bullet', 'blitz', 'rapid', 'classical'
                each value is a dict with 'elo' and 'games'
    """
    # prepare values for each TC category (use None for missing elos, 0 for games)
    tc_keys = ["ultra_fast", "bullet", "blitz", "rapid", "classical"]
    elos = [None] * len(tc_keys)
    games = [0] * len(tc_keys)
    for i, k in enumerate(tc_keys):
        v = ratings.get(k)
        if isinstance(v, dict):
            elos[i] = v.get("elo")
            games[i] = int(v.get("games", 0) or 0)

    params = (
        elos[0], elos[1], elos[2], elos[3], elos[4],
        games[0], games[1], games[2], games[3], games[4],
        engine_id,
    )

    # If a row for this engine already exists, update it; otherwise insert.
    existing = cnxn.execute(
        "SELECT id FROM engine_ratings WHERE engine_id = ? LIMIT 1",
        (engine_id,),
    ).fetchone()

    if existing:
        cur = cnxn.execute(
            """
            UPDATE engine_ratings SET
                elo_ultra_fast = ?, elo_bullet = ?, elo_blitz = ?, elo_rapid = ?, elo_classical = ?,
                games_ultra_fast = ?, games_bullet = ?, games_blitz = ?, games_rapid = ?, games_classical = ?
            WHERE engine_id = ?
            """,
            params,
        )
    else:
        cur = cnxn.execute(
            """
            INSERT INTO engine_ratings (
                engine_id,
                elo_ultra_fast, elo_bullet, elo_blitz, elo_rapid, elo_classical,
                games_ultra_fast, games_bullet, games_blitz, games_rapid, games_classical
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (engine_id, elos[0], elos[1], elos[2], elos[3], elos[4],
             games[0], games[1], games[2], games[3], games[4]),
        )

    cnxn.commit()
    return cur.lastrowid

def log_sprt(cnxn, experiment_id, candidate_engine_id, baseline_engine_id, **kwargs) -> int:
    """Log an SPRT experiment result and return its row ID.

    kwargs are the run's arguments plus the final pentanomial snapshot of
    utils.pentanomial / tests.sprt. Columns added later (pentanomial counts,
    intervals, core-type summary, ...) are written only when the table has
    them, so databases that were not migrated yet keep working.
    """
    ensure_schema(cnxn)  # nodes_per_move, threads
    penta = kwargs.get("pentanomial") or [None] * 5
    detail = kwargs.get("pentanomial_detail") or {}

    values = dict(
        experiment_id=experiment_id,
        baseline_engine_id=baseline_engine_id,
        candidate_engine_id=candidate_engine_id,
        opening_book=kwargs.get("book"),
        book_depth=kwargs.get("book_depth"),
        time_control=kwargs.get("tc"),
        time_per_move=kwargs.get("time"),
        depth_per_move=kwargs.get("depth"),
        elo0=kwargs.get("elo0"),
        elo1=kwargs.get("elo1"),
        alpha=kwargs.get("alpha"),
        beta=kwargs.get("beta"),
        result=kwargs.get("result"),
        elo_diff=kwargs.get("elo_diff"),
        llr=kwargs.get("llr"),
        los=kwargs.get("los"),
        candidate_wins=kwargs.get("candidate_wins"),
        candidate_losses=kwargs.get("candidate_losses"),
        candidate_draws=kwargs.get("candidate_draws"),
        candidate_white_wins=kwargs.get("candidate_white_wins"),
        candidate_white_losses=kwargs.get("candidate_white_losses"),
        candidate_white_draws=kwargs.get("candidate_white_draws"),
        candidate_black_wins=kwargs.get("candidate_black_wins"),
        candidate_black_losses=kwargs.get("candidate_black_losses"),
        candidate_black_draws=kwargs.get("candidate_black_draws"),
        games_played=kwargs.get("games_played"),
        run_time_s=kwargs.get("runtime"),
        # pentanomial model (newer columns)
        pairs=kwargs.get("pairs"),
        penta_ll=penta[0], penta_ld=penta[1],
        penta_dd=detail.get("DD"), penta_wl=detail.get("WL"),
        penta_wd=penta[3], penta_ww=penta[4],
        score=kwargs.get("score"),
        elo_low=kwargs.get("elo_low"),
        elo_high=kwargs.get("elo_high"),
        nelo=kwargs.get("nelo"),
        nelo_low=kwargs.get("nelo_low"),
        nelo_high=kwargs.get("nelo_high"),
        draw_ratio=kwargs.get("draw_ratio"),
        variance_ratio=kwargs.get("variance_ratio"),
        concurrency=kwargs.get("slots") or kwargs.get("concurrency"),
        core_summary=json.dumps({core: {k: s.get(k) for k in ("pairs", "elo_diff", "elo_low", "elo_high",
                                                               "nelo", "pentanomial", "wins", "draws", "losses")}
                                 for core, s in (kwargs.get("by_core") or {}).items()}) if kwargs.get("by_core") else None,
        excluded_pairs=kwargs.get("voided_pairs"),
        run_id=kwargs.get("run_id"),
        book_seed=kwargs.get("book_seed"),
        env_host=kwargs.get("env_host"),
        env_cpu_model=kwargs.get("env_cpu_model"),
    )

    columns = [c for c in values if c in _table_columns(cnxn, "sprt")]
    cur = cnxn.execute(
        f"INSERT INTO sprt ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
        [values[c] for c in columns],
    )
    
    cnxn.commit()
    return cur.lastrowid

def _table_columns(cnxn, table):
    return {row[1] for row in cnxn.execute(f"PRAGMA table_info({table})").fetchall()}


def bulk_log_match_games(cnxn, run_dir, experiment_id, engine_ids) -> dict:
    """Insert one `games` row per game recorded by the match runner in `run_dir`.

    These rows carry the authoritative result and termination (also for
    adjudicated, resigned and time-forfeit games, which engines do not log),
    the opening, the moves from the run's PGN, and the environment: logical cpu,
    P/E core type, measured clock, host and CPU model. `engine_ids` maps the
    engine names used in the run (e.g. Candidate, Baseline) to engine ids.
    Returns {(run_id, run_game): games.id}.
    """
    from .match_games import game_rows

    rows = game_rows(run_dir, engine_ids, experiment_id)
    if not rows:
        return {}
    available = _table_columns(cnxn, "games")
    columns = [c for c in rows[0] if c in available]
    missing = sorted(set(rows[0]) - available)
    if missing:
        print(f"[DATA] games table lacks {len(missing)} newer column(s) ({', '.join(missing[:4])}...); "
              "run data/databases/migrate_schema.py to keep the environment labels")
    placeholders = ", ".join("?" for _ in columns)
    mapping = {}
    for row in rows:
        cur = cnxn.execute(f"INSERT INTO games ({', '.join(columns)}) VALUES ({placeholders})",
                           [row[c] for c in columns])
        mapping[(row["run_id"], row["run_game"])] = cur.lastrowid
    cnxn.commit()
    return mapping

# ─────────────────────────────────────────────────────────────────────────────
# Bulk logging
# ─────────────────────────────────────────────────────────────────────────────

def bulk_log_sts(cnxn, path, sts_id, **kwargs) -> None:
    """Bulk-load STS results from a JSONL file."""
    ensure_schema(cnxn)  # search_nodes
    rows = []

    for data in _iter_json_objects_from_path(path):
        move = data.get('engine_move')
        expected_moves = data.get('expected_moves') or []
        expected_move_1 = expected_moves[0] if len(expected_moves) > 0 else None
        expected_move_2 = expected_moves[1] if len(expected_moves) > 1 else None
        avoid_moves = data.get('avoid_moves') or []

        if "move_ok" in data:
            correct = bool(data["move_ok"])  # tests/sts.py's verdict, which checks every bm move
        elif expected_moves != []:
            correct = (move == expected_move_1 or move == expected_move_2)
        elif avoid_moves != []:
            correct = (move != avoid_moves[0])
        else:
            correct = False

        rows.append((
            sts_id,
            data.get("epd_file"),
            data.get("category"),
            data.get("fen"),
            kwargs.get("time"),
            kwargs.get("depth"),
            kwargs.get("nodes"),
            move,
            data.get("engine_score"),
            expected_move_1,
            data.get("expected_score"),
            expected_move_2,
            avoid_moves[0] if avoid_moves else None,
            correct
        ))

    cnxn.executemany(
        """
        INSERT INTO sts (
            experiment_id, suite, position_name,
            fen, search_time_ms, search_depth, search_nodes,
            engine_move, engine_score, expected_move, expected_score,
            alt_expected_move, avoid_move, move_is_correct
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows
    )
    cnxn.commit()


def bulk_log_game(cnxn, path, experiment_id=None, second_engine_id=None, engine_id=None, commit=True,
                  engine_ids_by_version=None) -> dict:
    """
    Bulk-load games from a JSONL file. Returns {game_uuid: game_id} mapping.
    
    Handles multi-engine JSONL files: if multiple records share the same game_uuid
    (from different engines), both engine IDs are resolved from the records themselves.
    Falls back to second_engine_id for the opponent when only one engine logged.
    Pass `engine_id` when the file holds one engine's logs (its own log directory):
    its records are attributed to it instead of being looked up by version string.
    engine_ids_by_version ({version: engine id}) attributes records exactly when several
    builds share a version (e.g. a release's DEV and Prod builds).
    commit=False leaves the rows in the caller's open transaction.
    """
    cursor = cnxn.cursor()
    passthrough = _PassThrough(cnxn, "games", GAME_COLUMNS)

    # first pass: collect all records grouped by game_uuid
    game_records = {} # game_uuid -> list of record dicts
    game_order = [] # preserve insertion order of first-seen game_uuids

    for data in _iter_json_objects_from_path(path):
        uuid = data['game_uuid']
        if uuid not in game_records:
            game_records[uuid] = []
            game_order.append(uuid)
        game_records[uuid].append(data)

    rows = []
    #game_uuids = []

    for uuid in game_order:
        records = game_records[uuid]
        white_engine_id = None
        black_engine_id = None

        # use the first record as the primary source for the game metadata
        primary = records[0]

        # resolve engine  IDs from all records 
        for rec in records:
            rec_engine_id = engine_id if engine_id is not None else \
                _engine_for_version(cnxn, rec.get('engine_id'), engine_ids_by_version)
            if rec.get('side') == 'white':
                white_engine_id = rec_engine_id
            else:
                black_engine_id = rec_engine_id

        # fall back to second_engine_id for the missing side
        if white_engine_id is None:
            white_engine_id = second_engine_id
        if black_engine_id is None:
            black_engine_id = second_engine_id

        stored, extra = passthrough.split(
            {f: v for f, v in primary.items() if f not in GAME_FIELDS and f not in LINK_FIELDS})
        rows.append(((
            experiment_id,
            white_engine_id,
            black_engine_id,
            primary.get("wtime"),
            primary.get("btime"),
            primary.get("winc"),
            primary.get("binc"),
            primary.get("movestogo"),
            primary.get("depth"),
            primary.get("nodes"),
            primary.get("movetime"),
            primary.get("result"),
            primary.get("reason"),
            primary.get("opening"),
            primary.get("start_fen"),
            json.dumps(primary.get("moves")),
            primary.get("time_s")
        ), stored, extra))

    cursor.executemany(_insert_sql("games", passthrough.columns(GAME_COLUMNS, rows)),
                       passthrough.values(GAME_COLUMNS, rows))
    _report_fields("game", [passthrough])

    # Recover inserted IDs deterministically
    cursor.execute(
        "SELECT id FROM games ORDER BY id DESC LIMIT ?",
        (len(rows),)
    )
    game_ids = [r[0] for r in cursor.fetchall()][::-1]

    if commit:
        cnxn.commit()
    return dict(zip(game_order, game_ids))


def bulk_log_search_and_timing(
    cnxn,
    search_path,
    game_map,
    engine_id=None,
    timing_path=None,
    root_moves_path=None,
    sts_id=None,
    commit=True,
    engine_ids_by_version=None
) -> dict:
    """Bulk-load searches, per-depth iterations, tree stats, and timing from JSONL.

    Supports search logs that mix multiple engines: each record's engine is
    resolved individually via its own `engine_version` field (cached so we
    don't re-query the DB per row). Pass `engine_id` explicitly to force a
    single engine for every record (old single-engine behavior), or
    engine_ids_by_version ({version: engine id}) when several builds share a version.
    commit=False leaves the rows in the caller's open transaction.
    """
    cursor = cnxn.cursor()

    searches_rows = []
    uuid_map = {}  # search_uuid -> search_id
    game_id = None
    skipped_uuids = set() # search_uuids dropped due to unresolved engine

    # version_string -> db_engine_id, avoids re-querying the DB for every row
    engine_id_cache = {}
    KNOWN_SKIP_VERSIONS = ['dev','DEV','Dev','debug','DEBUG','Debug','tomahawk','Tomahawk']

    def resolve_engine_id(data):
        """Return the DB engine_id (int) for a single record (or None if unresolved)."""
        if engine_id is not None:
            # Caller forced a single engine for the whole file.
            return engine_id

        version = data.get("engine_id")
        if version is None:
            # Fall back to file-level detection, for logs that don't
            # stamp engine_id on every record.
            version = extract_engine_id_from_search(search_path)
            if version is None:
                return None

        if version in engine_id_cache:
            return engine_id_cache[version]
        
        if engine_ids_by_version and version in engine_ids_by_version:
            engine_id_cache[version] = engine_ids_by_version[version]
            return engine_id_cache[version]

        if version in KNOWN_SKIP_VERSIONS:
            engine_id_cache[version] = None 
            return None
        
        # Check DB directly (the newest build when several share the version)
        row = _newest_build(cnxn, version)

        if row is not None:
            engine_id_cache[version] = row
            return row

        resolved = _prompt_register_engine(cnxn, version, data)
        engine_id_cache[version] = resolved 
        return resolved

    # log fields no explicit mapping reads: same-named columns or extra_fields (top of this file)
    search_pass = _PassThrough(cnxn, "searches", SEARCH_COLUMNS)
    iteration_pass = _PassThrough(cnxn, "searches_by_iteration", ITERATION_COLUMNS, prefix="itdepth_")
    tree_pass = _PassThrough(cnxn, "searches_by_tree_depth", TREE_COLUMNS, prefix="treedepth_")

    # ── SEARCHES ──
    for data in _iter_json_objects_from_path(search_path):

        row_engine_id = resolve_engine_id(data) 
        if row_engine_id in ['dev','test','tomahawk','DEV','TEST','DEBUG','debug','TOMAHAWK',None]: 
            skipped_uuids.add(data.get('search_uuid'))
            continue # unresolved / dev / test

        if not sts_id or game_map == {}:
            game_uuid = data["game_uuid"]
            game_id = game_map.get(game_uuid)

        stored, extra = search_pass.split({f: v for f, v in data.items()
                                           if f not in SEARCH_FIELDS and f not in LINK_FIELDS
                                           and not f.startswith(("itdepth_", "treedepth_"))})
        searches_rows.append(((
            row_engine_id,
            game_id,
            sts_id,
            data.get("fen"),
            data.get("ply"),
            data.get("time_ms"),
            data.get("root_eval"),
            data.get("completed_depth"),
            data.get("max_depth"),
            data.get("max_qdepth"),
            data.get("best_move"),
            safe_val(data.get("principal_variation")),
            data.get("nodes"),
            data.get("qnodes"),
            data.get("tt_stores"),
            data.get("tt_hits"),
            data.get("tt_fill"),
            data.get("fail_highs"),
            data.get("fail_lows"),
            data.get("aspiration_fail_high_researches"),
            data.get("aspiration_fail_low_researches"),
            # fail-high move index histogram buckets
            _at_depth(data, "fail_high_index", 0),
            _at_depth(data, "fail_high_index", 1),
            _at_depth(data, "fail_high_index", 2),
            _at_depth(data, "fail_high_index", 3),
            _at_depth(data, "fail_high_index", 4),
            _at_depth(data, "fail_high_index", 5),
            data.get("see_prunes"),
            data.get("delta_prunes"),
            data.get("nmp"),
            data.get(_first(data, "nmp_failhigh", "nmp_fail")),
            data.get("tt_overwritten"),
        ), stored, extra, data["search_uuid"]))

    # engines that never get logged — filtered by version string, not id
    #searches_rows = [
    #    row for row in searches_rows
    #    if row[-1] not in ('dev', 'test')
    #]

    search_rows = [row[:3] for row in searches_rows]
    cursor.executemany(_insert_sql("searches", search_pass.columns(SEARCH_COLUMNS, search_rows)),
                       search_pass.values(SEARCH_COLUMNS, search_rows))

    # Recover search IDs
    cursor.execute(
        "SELECT id FROM searches ORDER BY id DESC LIMIT ?",
        (len(searches_rows),)
    )
    search_ids = [r[0] for r in cursor.fetchall()][::-1]

    for row, sid in zip(searches_rows, search_ids):
        uuid_map[row[3]] = sid

    # ── PER-ITERATION DEPTH ──
    depth_rows = []
    for data in _iter_json_objects_from_path(search_path):
        search_id = uuid_map.get(data['search_uuid'])
        if search_id is None: continue

        n = len(data.get("itdepth_nodes", []))
        fh_index_data = data.get("itdepth_fh_index", [])
        unmapped = [f for f in data if f.startswith("itdepth_") and f not in ITERATION_FIELDS]
        for d in range(n):
            # Extract per-depth fh_index bucket (array of 6 values)
            fh_buckets = safe(fh_index_data, d) if d < len(fh_index_data) else None
            stored, extra = iteration_pass.split({f: _per_depth(data, f, d) for f in unmapped})
            depth_rows.append(((
                search_id,
                d + 1,
                _at_depth(data, "itdepth_time_ms", d),
                _at_depth(data, "itdepth_eval", d),
                _at_depth(data, "itdepth_move", d),
                _at_depth(data, "itdepth_qdepth", d),
                _at_depth(data, "itdepth_nodes", d),
                _at_depth(data, "itdepth_qnodes", d),
                _at_depth(data, "itdepth_ttstores", d),
                _at_depth(data, "itdepth_tthits", d),
                _at_depth(data, "itdepth_ttfill", d),
                _at_depth(data, "itdepth_fail_highs", d),
                _at_depth(data, "itdepth_fail_lows", d),
                _at_depth(data, "itdepth_aspiration_failhigh_researches", d),
                _at_depth(data, "itdepth_aspiration_faillow_researches", d),
                # fail-high index histogram per iteration depth
                fh_buckets[0] if isinstance(fh_buckets, list) and len(fh_buckets) > 0 else None,
                fh_buckets[1] if isinstance(fh_buckets, list) and len(fh_buckets) > 1 else None,
                fh_buckets[2] if isinstance(fh_buckets, list) and len(fh_buckets) > 2 else None,
                fh_buckets[3] if isinstance(fh_buckets, list) and len(fh_buckets) > 3 else None,
                fh_buckets[4] if isinstance(fh_buckets, list) and len(fh_buckets) > 4 else None,
                fh_buckets[5] if isinstance(fh_buckets, list) and len(fh_buckets) > 5 else None,
                _at_depth(data, "itdepth_see_prunes", d),
                _at_depth(data, "itdepth_delta_prunes", d),
                _at_depth(data, "itdepth_pvs_researches", d),
                _at_depth(data, "itdepth_nmp", d),
                _at_depth(data, _first(data, "itdepth_nmp_failhigh", "itdepth_nmp_fail"), d),
            ), stored, extra))

    cursor.executemany(_insert_sql("searches_by_iteration", iteration_pass.columns(ITERATION_COLUMNS, depth_rows)),
                       iteration_pass.values(ITERATION_COLUMNS, depth_rows))

    # ── PER-TREE DEPTH ──
    ply_rows = []
    for data in _iter_json_objects_from_path(search_path):
        search_id = uuid_map.get(data['search_uuid'])
        if search_id is None: continue

        n = len(data.get("treedepth_nodes", []))
        fh_index_tree = data.get("treedepth_fh_index", [])
        unmapped = [f for f in data if f.startswith("treedepth_") and f not in TREE_FIELDS]
        for d in range(n):
            fh_buckets = safe(fh_index_tree, d) if d < len(fh_index_tree) else None
            stored, extra = tree_pass.split({f: _per_depth(data, f, d) for f in unmapped})
            ply_rows.append(((
                search_id,
                d + 1,
                _at_depth(data, "treedepth_nodes", d),
                _at_depth(data, "treedepth_qnodes", d),
                _at_depth(data, "treedepth_tt_stores", d),
                _at_depth(data, "treedepth_tt_hits", d),
                _at_depth(data, "treedepth_fail_highs", d),
                _at_depth(data, "treedepth_fail_lows", d),
                # fail-high index histogram per tree depth
                fh_buckets[0] if isinstance(fh_buckets, list) and len(fh_buckets) > 0 else None,
                fh_buckets[1] if isinstance(fh_buckets, list) and len(fh_buckets) > 1 else None,
                fh_buckets[2] if isinstance(fh_buckets, list) and len(fh_buckets) > 2 else None,
                fh_buckets[3] if isinstance(fh_buckets, list) and len(fh_buckets) > 3 else None,
                fh_buckets[4] if isinstance(fh_buckets, list) and len(fh_buckets) > 4 else None,
                fh_buckets[5] if isinstance(fh_buckets, list) and len(fh_buckets) > 5 else None,
                _at_depth(data, "treedepth_see_prunes", d),
                _at_depth(data, "treedepth_delta_prunes", d),
                _at_depth(data, "treedepth_pvs_researches", d),
                _at_depth(data, "treedepth_nmp", d),
                _at_depth(data, _first(data, "treedepth_nmp_failhigh", "treedepth_nmp_fail"), d),
            ), stored, extra))

    cursor.executemany(_insert_sql("searches_by_tree_depth", tree_pass.columns(TREE_COLUMNS, ply_rows)),
                       tree_pass.values(TREE_COLUMNS, ply_rows))
    unstored, mismatched = _report_fields("search", [search_pass, iteration_pass, tree_pass])

    # ── TIMING ──
    timing_rows = []
    if timing_path:
        for data in _iter_json_objects_from_path(timing_path):
            if data.get("type") != "timing":
                continue

            search_id = uuid_map.get(data.get("search_uuid"))
            if search_id is None:
                continue

            for func, stats in data.items():
                if func in {
                    "engine_id", "instance_id", "type", "session",
                    "game_uuid", "search_uuid", "fen", "total_search_time_ms"
                }:
                    continue

                timing_rows.append((
                    search_id,
                    func,
                    stats.get("total_ms", 0),
                    stats.get("calls", 0),
                ))

        cursor.executemany(
            """
            INSERT INTO timing (search_id, function, total_time_ms, num_calls)
            VALUES (?, ?, ?, ?)
            """,
            timing_rows
        )

    # -- ROOT MOVES --
    root_moves_rows = []
    if root_moves_path and Path(root_moves_path).is_file():
        for data in _iter_json_objects_from_path(root_moves_path):
            search_id = uuid_map.get(data.get("search_uuid"))
            if search_id is None:
                continue
            depth = data.get("depth", 0)
            for idx, mv in enumerate(data.get("moves", [])):
                root_moves_rows.append((
                    search_id,
                    depth,
                    idx,
                    mv.get("move", ""),
                    mv.get("eval"),
                    mv.get("time_ms"),
                    mv.get("nodes"),
                ))

        cursor.executemany(
            """
            INSERT INTO root_moves (search_id, depth, move_index, move, eval, time_ms, nodes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            root_moves_rows
        )

    if commit:
        cnxn.commit()

    counts = {
        "searches": len(searches_rows),
        "depth_rows": len(depth_rows),
        "ply_rows": len(ply_rows),
        "timing_rows": len(timing_rows),
        "root_moves_rows": len(root_moves_rows),
        "skipped": len(skipped_uuids),
        "unstored_fields": unstored,
        "mismatched_fields": mismatched,
    }

    print(
        f"[INFO] Inserted {counts['searches']} searches, "
        f"{counts['depth_rows']} depth rows, {counts['ply_rows']} ply rows, "
        f"{counts['timing_rows']} timing rows, "
        f"{counts['root_moves_rows']} root move rows. "
        f"Skipped {counts['skipped']} records with unresolved/dev/test engines."
    )

    return counts
