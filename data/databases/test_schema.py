"""Schema check: does a database have the relations and columns the dashboard expects?

Profiles:
  analytics    the analytics DuckDB after the pipeline (tables, all views, aggregates)
  public-lite  a public snapshot (export default): small tables + aggregates only
  public-full  a public snapshot exported with --full (adds per-search facts and views)
Without --profile, a database with a snapshot_info table is checked as the
snapshot profile it records, anything else as `analytics`. Public snapshots
must match the publication allow-list exactly (no extra relations/columns).

Usage (from the repository root):
  python -m data.databases.test_schema
  python -m data.databases.test_schema --db path/to/snapshot.duckdb
"""
import argparse
import os
import sys
from pathlib import Path

import duckdb

from ..etl.paths import ANALYTICS_DB
from ..transforms.aggregates import AGG_COLUMNS, SAMPLE_SIZE
from ..transforms.transform_search import ANALYTICS_VIEWS
from .migrate_schema import GAMES_MATCH_COLUMNS, SPRT_PENTANOMIAL_COLUMNS
from .public_schema import (FORBIDDEN_COLUMNS, FORBIDDEN_RELATIONS, FULL_VIEWS, SNAPSHOT_INFO,
                            public_tables)

PROFILES = ("analytics", "public-lite", "public-full")

# Columns the analytics tables must have (they may have more).
ANALYTICS_TABLES = {
    "engines": ("id", "name", "version"),
    "experiments": ("id", "engine_id", "type", "start_time_utc"),
    "engine_ratings": ("engine_id", "engine_name", "engine_version"),
    "game_stats": ("id", "white_engine_id", "black_engine_id", "result", "termination", "opening",
                   "opening_eco", "opening_variation", "run_time_s", *(n for n, _ in GAMES_MATCH_COLUMNS)),
    "sprt_runs": ("id", "experiment_id", "baseline_engine_id", "candidate_engine_id", "result", "elo_diff",
                  "opening_book", *(n for n, _ in SPRT_PENTANOMIAL_COLUMNS)),
    "sts_runs": ("id", "experiment_id", "suite", "position_name", "position_type", "fen", "move_is_correct"),
    "search_stats": ("id", "engine_id", "game_id", "sts_id", "fen", "ply", "time_ms", "eval", "completed_depth",
                     "depth", "qdepth", "move", "nodes", "qnodes", "tt_stores", "tt_hits", "fail_highs",
                     "fail_lows", "nmp", "nmp_failhigh", "sf_eval", "sf_best_move", "sf_pv",
                     "sf_depth", "sf_multipv", "sf_move_rank", "sf_move_eval"),
    "iterative_deepening_stats": ("search_id", "depth", "qdepth", "time_ms", "eval", "move", "nodes", "qnodes",
                                  "tt_hits", "fail_highs"),
    "search_tree_stats": ("search_id", "depth", "nodes", "qnodes", "fail_highs", "fail_lows", "nmp"),
    "search_timings": ("search_id", "function", "total_time_ms", "num_calls"),
    "root_moves": ("id", "search_id", "move", "eval", "time_ms", "nodes"),
    "position_features": ("search_id", "game_phase", "position_type", "pos_tactical", "pos_positional"),
    "stockfish_moves": ("fen", "move", "rank", "eval", "pv", "depth", "multipv"),
}

# Columns the views must expose (they may expose more).
VIEW_COLUMNS = {
    "search_metrics": ("search_id", "engine_id", "game_id", "fen", "ply", "completed_depth", "total_nodes",
                       "total_time_ms", "nps", "qratio", "tt_hit_ratio", "fail_high_ratio", "eval", "eval_cp",
                       "mate", "mate_plies", "sf_eval", "sf_eval_cp", "eval_diff", "best_move_match",
                       "sf_move_rank", "sf_move_eval", "sf_cp_loss", "engine_move_rank",
                       "env_core_type", "env_mhz_mean", "env_host", "env_cpu_model"),
    "search_context": ("search_id", "engine_name", "engine_version", "game_score", "game_opening", "game_eco",
                       "game_opening_variation", "sts_suite", "sts_position_name", "sts_move_is_correct",
                       "env_core_type", "env_mhz_mean", "env_host", "env_cpu_model", "eval_cp", "mate"),
    "search_position_metrics": ("search_id", "game_phase", "position_type", "position_tactical_score"),
    "search_iteration_metrics": ("search_id", "depth", "eval", "eval_cp", "mate", "nps", "total_nodes"),
    "search_iteration_features": ("search_id", "depth", "ebf", "prior_eval_delta", "stddev_eval",
                                  "move_stability"),
    "search_tree_features": ("search_id", "depth", "total_nodes", "ebf", "qratio"),
    "search_iteration_summary": ("search_id", "avg_ebf", "geo_mean_ebf", "avg_eval", "eval_sign_flips"),
    "search_timing_summary": ("search_id", "total_search_time"),
    "search_features": ("search_id", "engine_id", "total_time_ms", "final_eval", "completed_depth",
                        "total_nodes", "nps", "qratio", "tt_hit_ratio", "fail_high_ratio", "avg_ebf"),
}


def _report(message):
    """Print without failing on consoles/pipes that can't encode a character."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    print(message.encode(encoding, "replace").decode(encoding, "replace"))


def _relations(cnxn):
    rows = cnxn.execute("""
        SELECT table_name, table_type FROM information_schema.tables
        WHERE table_catalog = current_database() AND table_schema = 'main'
    """).fetchall()
    return {name: ("VIEW" if kind == "VIEW" else "TABLE") for name, kind in rows}


def _columns(cnxn, relation):
    return [row[0] for row in cnxn.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
    """, [relation]).fetchall()]


def detect_profile(cnxn):
    if SNAPSHOT_INFO in _relations(cnxn):
        row = cnxn.execute(f"SELECT value FROM {SNAPSHOT_INFO} WHERE key = 'profile'").fetchone()
        if row and f"public-{row[0]}" in PROFILES:
            return f"public-{row[0]}"
    return "analytics"


def check_schema(cnxn, profile=None):
    """(errors, warnings, profile) for an open connection."""
    profile = profile or detect_profile(cnxn)
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; expected one of {', '.join(PROFILES)}")
    errors, warnings = [], []
    relations = _relations(cnxn)

    def require(relation, kind, columns, exact=False):
        if relation not in relations:
            errors.append(f"MISSING {kind}: {relation}")
            return
        if relations[relation] != kind:
            errors.append(f"WRONG KIND: {relation} is a {relations[relation].lower()}, expected a {kind.lower()}")
        actual = _columns(cnxn, relation)
        missing = [c for c in columns if c not in actual]
        if missing:
            errors.append(f"MISSING COLUMN(S) in {relation}: {', '.join(missing)}")
        if exact:
            extra = [c for c in actual if c not in columns]
            if extra:
                errors.append(f"UNEXPECTED COLUMN(S) in {relation}: {', '.join(extra)}")
            elif not missing and tuple(actual) != tuple(columns):
                errors.append(f"COLUMN ORDER of {relation} differs from the documented order")

    if profile == "analytics":
        for table, columns in ANALYTICS_TABLES.items():
            require(table, "TABLE", columns)
        for view in ANALYTICS_VIEWS:
            require(view, "VIEW", VIEW_COLUMNS[view])
        for table, columns in AGG_COLUMNS.items():
            require(table, "TABLE", columns, exact=True)
    else:
        full = profile == "public-full"
        expected = public_tables(full)
        for table, columns in expected.items():
            require(table, "TABLE", columns, exact=True)
        views = FULL_VIEWS if full else ()
        for view in views:
            require(view, "VIEW", VIEW_COLUMNS[view])
        unexpected = sorted(set(relations) - set(expected) - set(views))
        if unexpected:
            errors.append(f"UNEXPECTED RELATION(S) in a public snapshot: {', '.join(unexpected)}")
        for relation in FORBIDDEN_RELATIONS:
            if relation in relations:
                errors.append(f"FORBIDDEN RELATION published: {relation}")
        for table, columns in FORBIDDEN_COLUMNS.items():
            leaked = [c for c in columns if table in relations and c in _columns(cnxn, table)]
            if leaked:
                errors.append(f"FORBIDDEN COLUMN(S) published in {table}: {', '.join(leaked)}")

    # Row sanity
    if "search_stats" in relations and cnxn.execute("SELECT count(*) FROM search_stats").fetchone()[0] == 0:
        warnings.append("search_stats is empty (no data loaded)")
    if "agg_engine_summary" in relations and \
            cnxn.execute("SELECT count(*) FROM agg_engine_summary").fetchone()[0] == 0:
        warnings.append("agg_engine_summary is empty (no searches aggregated)")
    if "search_sample" in relations:
        rows = cnxn.execute("SELECT count(*) FROM search_sample").fetchone()[0]
        if rows > SAMPLE_SIZE:
            errors.append(f"search_sample has {rows:,} rows (limit {SAMPLE_SIZE:,})")
    return errors, warnings, profile


def validate_schema(path=None, profile=None, verbose=True) -> bool:
    """Check the database at `path` (default: CHESS_ANALYTICS_DB / ANALYTICS_DB); True when it passes."""
    path = Path(path or os.environ.get("CHESS_ANALYTICS_DB") or ANALYTICS_DB)
    if not path.is_file():
        _report(f"SCHEMA: database not found at {path}")
        _report("  (run the pipeline first: python -m data.databases.run_analytics_pipeline)")
        return False
    with duckdb.connect(str(path), read_only=True) as cnxn:
        errors, warnings, profile = check_schema(cnxn, profile)

    if verbose:
        _report("\n" + "=" * 60)
        _report(f"  SCHEMA VALIDATION REPORT ({profile})")
        _report("=" * 60)
        if not errors and not warnings:
            _report("  OK    All schema checks passed.")
        for w in warnings:
            _report(f"  WARN  {w}")
        for e in errors:
            _report(f"  ERROR {e}")
        _report("=" * 60)
        _report(f"  {len(errors)} errors, {len(warnings)} warnings")
    return not errors


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check a database against the dashboard's schema")
    parser.add_argument("--db", type=Path, help=f"Database to check (default: CHESS_ANALYTICS_DB or {ANALYTICS_DB})")
    parser.add_argument("--profile", choices=PROFILES, help="Expected layout (default: detected)")
    args = parser.parse_args(argv)
    return 0 if validate_schema(args.db, args.profile) else 1


if __name__ == "__main__":
    sys.exit(main())
