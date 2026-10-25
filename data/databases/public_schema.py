"""The publication boundary: which relations and columns a public snapshot holds.

export_public.py copies exactly these (allow-listed) columns; test_schema.py
checks snapshots against the same lists. Anything not listed -- engine
descriptions and compile flags (local paths), experiment metadata, game move
lists, dim_positions, compatibility views -- is never published.
"""
from ..transforms.aggregates import AGG_COLUMNS
from .migrate_schema import GAMES_MATCH_COLUMNS, SPRT_PENTANOMIAL_COLUMNS

SCHEMA_VERSION = 1
SNAPSHOT_INFO = "snapshot_info"   # (key, value): schema_version, profile, exported_at, searches, sample_rows
SNAPSHOT_INFO_COLUMNS = ("key", "value")

_FH_INDEX = ("fh_index_0", "fh_index_1", "fh_index_2", "fh_index_3", "fh_index_4to7", "fh_index_8plus")
# pvs_researches is derived for builds that log three kinds of re-search instead
# (transform_search.with_pvs_researches); the kinds are not published.
_PRUNING = ("see_prunes", "delta_prunes", "pvs_researches", "nmp", "nmp_failhigh")

# Small tables, published in every snapshot.
BASE_TABLE_COLUMNS = {
    "engines": (
        "id", "name", "version",
        "move_overhead_ms", "max_threads", "hash_size_mb", "pondering",
        "delta_prune_threshold", "see_prune_threshold", "aspiration_window", "aspiration_start_depth",
        "aspiration_depth_scale", "aspiration_research_scale", "draw_eval", "contempt", "r_nmp",
        "r_lmr_const", "r_lmr_denom", "lmr_move_order_threshold", "lmr_depth_threshold",
        "ingestion_timestamp_utc",
    ),
    "experiments": (
        "id", "engine_id", "comparison_engine_id", "type", "start_time_utc", "end_time_utc",
    ),
    "engine_ratings": (
        "id", "engine_id", "engine_name", "engine_version",
        "elo_ultra_fast", "elo_bullet", "elo_blitz", "elo_rapid", "elo_classical",
        "games_ultra_fast", "games_bullet", "games_blitz", "games_rapid", "games_classical",
        "ingestion_timestamp_utc",
    ),
    "game_stats": (
        "id", "experiment_id", "white_engine_id", "black_engine_id",
        "wtime", "winc", "btime", "binc", "movestogo", "depth", "nodes", "movetime",
        "result", "termination", "opening", "opening_eco", "opening_variation", "run_time_s",
        *(name for name, _ in GAMES_MATCH_COLUMNS),
        "ingestion_timestamp_utc",
    ),
    "sprt_runs": (
        "id", "experiment_id", "baseline_engine_id", "candidate_engine_id",
        "opening_book", "book_depth", "time_control", "time_per_move", "depth_per_move",
        "elo0", "elo1", "alpha", "beta", "result", "elo_diff", "llr", "los",
        "candidate_wins", "candidate_losses", "candidate_draws",
        "candidate_white_wins", "candidate_white_losses", "candidate_white_draws",
        "candidate_black_wins", "candidate_black_losses", "candidate_black_draws",
        "games_played", "run_time_s",
        *(name for name, _ in SPRT_PENTANOMIAL_COLUMNS),
        "ingestion_timestamp_utc",
    ),
    "sts_runs": (
        "id", "experiment_id", "suite", "position_name", "position_type", "fen",
        "search_time_ms", "search_depth", "engine_move", "engine_score", "expected_move", "expected_score",
        "alt_expected_move", "avoid_move", "move_is_correct", "ingestion_timestamp_utc",
    ),
}

# Per-search facts, published only by `export_public --full`.
FULL_TABLE_COLUMNS = {
    "search_stats": (
        "id", "engine_id", "game_id", "sts_id", "fen", "ply", "time_ms", "eval",
        "completed_depth", "depth", "qdepth", "move", "principal_variation",
        "nodes", "qnodes", "tt_stores", "tt_hits", "tt_fill", "fail_highs", "fail_lows",
        "fail_high_researches", "fail_low_researches", *_FH_INDEX, *_PRUNING, "tt_overwritten",
        "ingestion_timestamp_utc",
        "sf_eval", "sf_best_move", "sf_time_ms", "sf_computed_at", "sf_pv",
        "sf_depth", "sf_multipv", "sf_move_rank", "sf_move_eval",
    ),
    "iterative_deepening_stats": (
        "search_id", "depth", "time_ms", "eval", "move", "nodes", "qnodes", "qdepth",
        "tt_stores", "tt_hits", "tt_fill", "fail_highs", "fail_lows",
        "fail_high_researches", "fail_low_researches", *_FH_INDEX, *_PRUNING,
    ),
    "search_tree_stats": (
        "search_id", "depth", "nodes", "qnodes", "tt_stores", "tt_hits", "fail_highs", "fail_lows",
        *_FH_INDEX, *_PRUNING,
    ),
    "search_timings": ("search_id", "function", "total_time_ms", "num_calls"),
    "root_moves": ("id", "search_id", "depth", "move_index", "move", "eval", "time_ms", "nodes"),
    "position_features": (
        "search_id", "game_id", "game_phase", "position_type", "pos_tactical", "pos_positional", "pos_endgame",
        "balance", "white_backwards", "white_doubled", "white_passed",
        "black_backwards", "black_doubled", "black_passed",
        "white_shield_pawns", "white_open_files", "white_tropism",
        "black_shield_pawns", "black_open_files", "black_tropism",
        "white_num_moves", "white_capture_ratio", "white_check_ratio", "white_legal_enemy",
        "white_controlled_enemy", "black_num_moves", "black_capture_ratio", "black_check_ratio",
        "black_legal_enemy", "black_controlled_enemy",
    ),
    # Stockfish's lines of the published searches' positions (and the engine moves it scored alone).
    "stockfish_moves": (
        "fen", "move", "rank", "eval", "pv", "depth", "multipv", "time_ms", "stockfish", "computed_at",
    ),
}

# Views installed in --full snapshots, in dependency order. The compatibility
# views (search_iteration_summary, search_timing_summary, search_features) are not published.
FULL_VIEWS = (
    "search_metrics",
    "search_context",
    "search_position_metrics",
    "search_iteration_metrics",
    "search_iteration_features",
    "search_tree_features",
)

# Never published, whatever the allow-lists say.
FORBIDDEN_COLUMNS = {
    "engines": ("description", "compile_flags"),
    "experiments": ("metadata",),
    "game_stats": ("moves", "start_fen"),
}
FORBIDDEN_RELATIONS = ("dim_positions", "search_iteration_summary", "search_timing_summary", "search_features",
                       "schema_migrations")


def public_tables(full=False) -> dict:
    """{table: columns} of a snapshot (base tables, aggregates, snapshot_info, and per-search facts if full)."""
    tables = dict(BASE_TABLE_COLUMNS)
    if full:
        tables.update(FULL_TABLE_COLUMNS)
    tables.update(AGG_COLUMNS)
    tables[SNAPSHOT_INFO] = SNAPSHOT_INFO_COLUMNS
    return tables


def public_relations(full=False) -> set:
    return set(public_tables(full)) | (set(FULL_VIEWS) if full else set())
