"""
Data quality validation checks for the analytics database.

Runs after transforms to detect data integrity issues, range violations,
and schema inconsistencies. Reports warnings but does not block the pipeline
unless critical errors are found.

Usage (from the repository root):
  python -m data.transforms.validate [--db PATH]
"""
import argparse
import duckdb
import os
import sys
from pathlib import Path
from ..etl.paths import ANALYTICS_DB
from .aggregates import (AGG_TABLES, LOSS_BUCKETS, MIN_CORRELATION_N, ROOT_GAP_CAP, ROOT_INDEX_CAP,
                         SAMPLE_SIZE)

# Columns that hold file names: a path separator would publish a local path.
FILE_NAME_COLUMNS = (("sprt_runs", "opening_book"), ("sts_runs", "suite"))


def _print(message):
    """print() that survives consoles and pipes whose encoding lacks a character (Windows cp1252)."""
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    print(message.encode(encoding, "replace").decode(encoding, "replace"))


class ValidationResult:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warn(self, check, message, count=None):
        self.warnings.append((check, message, count))

    def error(self, check, message, count=None):
        self.errors.append((check, message, count))

    @property
    def passed(self):
        return len(self.errors) == 0

    def report(self):
        _print("\n" + "=" * 60)
        _print("  DATA QUALITY REPORT")
        _print("=" * 60)
        if not self.warnings and not self.errors:
            _print("  OK    All checks passed.")
        for check, msg, count in self.warnings:
            cnt = f" ({count:,} rows)" if count else ""
            _print(f"  WARN  [{check}] {msg}{cnt}")
        for check, msg, count in self.errors:
            cnt = f" ({count:,} rows)" if count else ""
            _print(f"  ERROR [{check}] {msg}{cnt}")
        _print("=" * 60)
        total_checks = len(self.warnings) + len(self.errors)
        _print(f"  {total_checks} issues: {len(self.errors)} errors, {len(self.warnings)} warnings")
        return self.passed


def _table_exists(cnxn, table):
    try:
        cnxn.execute(f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        return False


def _count(cnxn, query):
    return cnxn.execute(query).fetchone()[0]


def validate_searches(cnxn, result):
    """Validate search_stats table."""
    if not _table_exists(cnxn, 'search_stats'):
        result.warn("searches", "search_stats table missing - skipping search checks")
        return

    # FK: all searches reference valid engines
    orphan_engines = _count(cnxn, """
        SELECT COUNT(*) FROM search_stats s
        WHERE s.engine_id IS NOT NULL
          AND s.engine_id NOT IN (SELECT id FROM engines)
    """)
    if orphan_engines > 0:
        result.error("fk_engine", "Searches reference non-existent engine_id", orphan_engines)

    # Range: nodes >= 0
    bad_nodes = _count(cnxn, "SELECT COUNT(*) FROM search_stats WHERE nodes < 0 OR qnodes < 0")
    if bad_nodes > 0:
        result.error("range_nodes", "Searches with negative nodes/qnodes", bad_nodes)

    # Range: qnodes <= total nodes (nodes + qnodes)
    bad_q = _count(cnxn, "SELECT COUNT(*) FROM search_stats WHERE qnodes > nodes + qnodes")
    if bad_q > 0:
        result.warn("range_qnodes", "Searches where qnodes > total nodes (impossible)", bad_q)

    # Range: time >= 0
    bad_time = _count(cnxn, "SELECT COUNT(*) FROM search_stats WHERE time_ms < 0")
    if bad_time > 0:
        result.error("range_time", "Searches with negative time_ms", bad_time)

    # Range: depth > 0 and reasonable
    bad_depth = _count(cnxn, "SELECT COUNT(*) FROM search_stats WHERE completed_depth <= 0 OR completed_depth > 100")
    if bad_depth > 0:
        result.warn("range_depth", "Searches with depth <= 0 or > 100", bad_depth)

    # Range: NPS sanity (> 1B is suspicious for single-threaded)
    nps_outliers = _count(cnxn, """
        SELECT COUNT(*) FROM search_stats 
        WHERE time_ms > 0 AND (nodes + qnodes) / (time_ms / 1000.0) > 1e9
    """)
    if nps_outliers > 0:
        result.warn("range_nps", "Searches with NPS > 1 billion (suspicious)", nps_outliers)

    # Null eval where depth > 3 (should always have an eval)
    null_eval = _count(cnxn, "SELECT COUNT(*) FROM search_stats WHERE eval IS NULL AND completed_depth > 3")
    if null_eval > 0:
        result.warn("null_eval", "Searches with NULL eval but depth > 3", null_eval)

    # Duplicate UUIDs (if we can detect via fen+ply+engine combination)
    # Check for exact duplicates by fen + engine_id + completed_depth + time_ms
    dup_count = _count(cnxn, """
        SELECT COUNT(*) - COUNT(DISTINCT (fen || '|' || engine_id || '|' || completed_depth || '|' || time_ms))
        FROM search_stats WHERE fen IS NOT NULL
    """)
    if dup_count > 100:
        result.warn("duplicates", "Potential duplicate searches detected (same fen+engine+depth+time)", dup_count)


def validate_iterations(cnxn, result):
    """Validate iterative deepening stats."""
    if not _table_exists(cnxn, 'iterative_deepening_stats'):
        result.warn("iterations", "iterative_deepening_stats table missing")
        return

    # FK: all iterations reference valid searches
    orphan_iters = _count(cnxn, """
        SELECT COUNT(*) FROM iterative_deepening_stats i
        WHERE i.search_id NOT IN (SELECT id FROM search_stats)
    """)
    if orphan_iters > 0:
        result.error("fk_iter_search", "Iterations reference non-existent search_id", orphan_iters)

    # Consistency: depth values are positive and sequential
    bad_depth = _count(cnxn, "SELECT COUNT(*) FROM iterative_deepening_stats WHERE depth <= 0")
    if bad_depth > 0:
        result.warn("iter_depth", "Iterations with depth <= 0", bad_depth)

    # Range: nodes per iteration should be positive
    neg_nodes = _count(cnxn, "SELECT COUNT(*) FROM iterative_deepening_stats WHERE nodes < 0")
    if neg_nodes > 0:
        result.error("iter_neg_nodes", "Iterations with negative nodes", neg_nodes)


def validate_games(cnxn, result):
    """Validate game_stats table."""
    if not _table_exists(cnxn, 'game_stats'):
        result.warn("games", "game_stats table missing - skipping game checks")
        return

    # Result must be white/black/draw
    bad_result = _count(cnxn, """
        SELECT COUNT(*) FROM game_stats
        WHERE result NOT IN ('white', 'black', 'draw') AND result IS NOT NULL
    """)
    if bad_result > 0:
        result.warn("game_result", "Games with invalid result (not white/black/draw)", bad_result)

    # Duration sanity
    bad_time = _count(cnxn, "SELECT COUNT(*) FROM game_stats WHERE run_time_s < 0")
    if bad_time > 0:
        result.error("game_time", "Games with negative duration", bad_time)


def validate_tree(cnxn, result):
    """Validate search tree stats."""
    if not _table_exists(cnxn, 'search_tree_stats'):
        return

    orphans = _count(cnxn, """
        SELECT COUNT(*) FROM search_tree_stats t
        WHERE t.search_id NOT IN (SELECT id FROM search_stats)
    """)
    if orphans > 0:
        result.error("fk_tree_search", "Tree stats reference non-existent search_id", orphans)


def validate_features(cnxn, result):
    """Validate focused views without repeatedly executing the compatibility join."""
    if not _table_exists(cnxn, 'search_metrics'):
        result.error("features", "search_metrics missing - run data.transforms.transform_search")
        return

    # One-to-one joins require unique keys; CTAS imports do not retain constraints.
    for table, key in [('search_stats', 'id'), ('engines', 'id'), ('game_stats', 'id'),
                       ('sts_runs', 'id'), ('position_features', 'search_id')]:
        duplicates = _count(cnxn, f"""
            SELECT COUNT(*) FROM (
                SELECT {key} FROM {table} GROUP BY {key} HAVING COUNT(*) > 1
            )
        """)
        if duplicates:
            result.error('join_grain', f'{table}.{key} is not unique; metadata joins multiply searches', duplicates)

    # Ratios should be in [0, 1] (with some tolerance for floating point)
    ratio_cols = [
        'qratio', 'tt_hit_ratio', 'tt_store_ratio',
        'fail_high_ratio', 'fail_low_ratio', 'nmp_ratio',
    ]
    for col in ratio_cols:
        try:
            bad = _count(cnxn, f"SELECT COUNT(*) FROM search_metrics WHERE {col} < -0.01 OR {col} > 1.01")
            if bad > 0:
                result.warn(f"ratio_{col}", f"search_metrics.{col} outside [0,1]", bad)
        except duckdb.Error as exc:
            result.error('feature_query', str(exc))

    # The metadata joins of search_context must keep one row per search.
    if _table_exists(cnxn, 'search_context'):
        try:
            extra = _count(cnxn, "SELECT (SELECT COUNT(*) FROM search_context) - (SELECT COUNT(*) FROM search_metrics)")
            if extra:
                result.error('join_grain', 'search_context does not have one row per search', abs(extra))
        except duckdb.Error as exc:
            result.error('feature_query', str(exc))

    # EBF should be > 0 (where not null)
    if not _table_exists(cnxn, 'search_iteration_summary'):
        return
    try:
        bad_ebf = _count(cnxn, "SELECT COUNT(*) FROM search_iteration_summary WHERE avg_ebf < 0")
        if bad_ebf > 0:
            result.warn("ebf_negative", "search_iteration_summary.avg_ebf < 0", bad_ebf)
    except duckdb.Error as exc:
        result.error('feature_query', str(exc))


def validate_timing(cnxn, result):
    """Validate timing data."""
    if not _table_exists(cnxn, 'search_timings'):
        return

    neg_time = _count(cnxn, "SELECT COUNT(*) FROM search_timings WHERE total_time_ms < 0")
    if neg_time > 0:
        result.error("timing_neg", "Timing entries with negative time", neg_time)

    neg_calls = _count(cnxn, "SELECT COUNT(*) FROM search_timings WHERE num_calls < 0")
    if neg_calls > 0:
        result.error("timing_neg_calls", "Timing entries with negative call count", neg_calls)


def validate_aggregates(cnxn, result, required=True):
    """Pre-aggregated tables: presence, grain and internal consistency."""
    missing = [t for t in AGG_TABLES if not _table_exists(cnxn, t)]
    if missing:
        message = f"aggregate tables missing: {', '.join(missing)} - run data.transforms.aggregates"
        (result.error if required else result.warn)("aggregates", message)
        return

    checks = [
        ("agg_grain", "agg_engine_summary has duplicate (engine_id, core_type) rows",
         "SELECT COUNT(*) - COUNT(DISTINCT (engine_id, core_type)) FROM agg_engine_summary", "error"),
        ("agg_grain", "engines without a core_type = 'all' summary row",
         "SELECT COUNT(DISTINCT engine_id) - COUNT(DISTINCT engine_id) FILTER (WHERE core_type = 'all') "
         "FROM agg_engine_summary", "error"),
        ("agg_wdl", "agg_eval_wdl rows where wins + draws + losses <> positions",
         "SELECT COUNT(*) FROM agg_eval_wdl WHERE wins + draws + losses <> positions OR games > positions", "error"),
        ("agg_wdl", "agg_eval_wdl buckets outside [-1000, 1000] or not multiples of 25, or mate rows with a bucket",
         "SELECT COUNT(*) FROM agg_eval_wdl WHERE (mate = 0 AND (eval_bucket IS NULL OR abs(eval_bucket) > 1000 "
         "OR eval_bucket % 25 <> 0)) OR (mate <> 0 AND eval_bucket IS NOT NULL) OR mate NOT IN (-1, 0, 1)", "error"),
        ("agg_openings", "agg_openings rows where wins + draws + losses > games",
         "SELECT COUNT(*) FROM agg_openings WHERE wins + draws + losses > games OR level NOT IN ('family', 'variation')",
         "error"),
        ("agg_share", "shares outside [0, 1]",
         "SELECT (SELECT COUNT(*) FROM agg_hist WHERE share < 0 OR share > 1.000001) + "
         "(SELECT COUNT(*) FROM agg_moveorder WHERE share < 0 OR share > 1.000001) + "
         "(SELECT COUNT(*) FROM agg_tree_ply WHERE node_share < 0 OR node_share > 1.000001)", "error"),
        ("agg_hist", "agg_hist bins with bin_hi <= bin_lo",
         "SELECT COUNT(*) FROM agg_hist WHERE bin_hi <= bin_lo", "error"),
        ("agg_corr", f"agg_corr coefficients outside [-1, 1] or n < {MIN_CORRELATION_N}",
         f"SELECT COUNT(*) FROM agg_corr WHERE n < {MIN_CORRELATION_N} OR abs(pearson) > 1.000001 "
         "OR abs(spearman) > 1.000001", "error"),
        ("agg_rates", "negative NPS or counts in the aggregates",
         "SELECT (SELECT COUNT(*) FROM agg_engine_summary WHERE nps < 0 OR searches <= 0) + "
         "(SELECT COUNT(*) FROM agg_env WHERE nps < 0 OR knps_per_ghz < 0) + "
         "(SELECT COUNT(*) FROM agg_timing WHERE calls < 0 OR total_ms < 0)", "error"),
        ("agg_root", f"root move positions outside [-1, {ROOT_INDEX_CAP}], gaps outside [0, {ROOT_GAP_CAP}] "
                     "or non-positive counts",
         f"SELECT (SELECT COUNT(*) FROM agg_root_index WHERE move_index < 0 OR move_index > {ROOT_INDEX_CAP} "
         "OR moves <= 0) + "
         f"(SELECT COUNT(*) FROM agg_root_gap WHERE gap < 0 OR gap > {ROOT_GAP_CAP} OR searches <= 0) + "
         f"(SELECT COUNT(*) FROM agg_sf_root_order WHERE move_index < -1 OR move_index > {ROOT_INDEX_CAP} "
         "OR searches <= 0)", "error"),
        ("agg_sf", "agg_sf_moves ranks outside [0, lines], unknown loss buckets, or negative losses or counts",
         "SELECT COUNT(*) FROM agg_sf_moves WHERE sf_rank < 0 OR sf_rank > lines OR searches <= 0 "
         f"OR (loss_bucket IS NOT NULL AND loss_bucket NOT IN {tuple(LOSS_BUCKETS)}) OR loss_sum < 0 "
         "OR (loss_bucket IS NULL) <> (loss_sum IS NULL)", "error"),
        ("agg_sf", "agg_sf_root_effort shares that do not add up to one per search",
         "SELECT COUNT(*) FROM (SELECT engine_id, core_type, phase, SUM(node_share_sum) AS shares, "
         "MAX(searches) AS searches, MIN(searches) AS least FROM agg_sf_root_effort GROUP BY ALL) "
         "WHERE abs(shares - searches) > 1e-6 * searches OR searches <> least", "error"),
        ("search_sample", f"search_sample has more than {SAMPLE_SIZE:,} rows or duplicate searches",
         f"SELECT GREATEST(COUNT(*) - {SAMPLE_SIZE}, 0) + COUNT(*) - COUNT(DISTINCT search_id) FROM search_sample",
         "error"),
    ]
    if _table_exists(cnxn, 'game_stats'):
        # every game counts once per side (colour) in the family-level opening totals
        checks.append((
            "agg_openings", "agg_openings family totals differ from game_stats",
            """SELECT COUNT(*) FROM (
                   SELECT engine_id, SUM(games) AS games FROM agg_openings WHERE level = 'family' GROUP BY engine_id
               ) a FULL JOIN (
                   SELECT engine_id, COUNT(*) AS games FROM (
                       SELECT white_engine_id AS engine_id FROM game_stats WHERE white_engine_id IS NOT NULL
                       UNION ALL
                       SELECT black_engine_id FROM game_stats WHERE black_engine_id IS NOT NULL)
                   GROUP BY engine_id
               ) g ON a.engine_id = g.engine_id
               WHERE a.games IS DISTINCT FROM g.games""", "error"))
    for check, message, query, level in checks:
        try:
            bad = _count(cnxn, query)
        except duckdb.Error as exc:
            result.error('aggregate_query', f"{check}: {exc}")
            continue
        if bad:
            (result.error if level == "error" else result.warn)(check, message, bad)


def validate_publication(cnxn, result):
    """A public snapshot must not contain local paths."""
    for table, column in FILE_NAME_COLUMNS:
        if not _table_exists(cnxn, table):
            continue
        leaked = _count(cnxn, f"SELECT COUNT(*) FROM {table} "
                              f"WHERE strpos({column}, '/') > 0 OR strpos({column}, '\\') > 0")
        if leaked:
            result.error("privacy", f"{table}.{column} holds local paths - rerun the analytics pipeline", leaked)


def run_validation(cnxn, profile="analytics"):
    """Run all validation checks and return the result object.

    profile: 'analytics' (the analytics database), 'public-full' (a snapshot with
    per-search facts) or 'public-lite' (a snapshot with aggregates only).
    """
    result = ValidationResult()

    if profile != "public-lite":
        validate_searches(cnxn, result)
        validate_iterations(cnxn, result)
        validate_tree(cnxn, result)
        validate_timing(cnxn, result)
        validate_features(cnxn, result)
    validate_games(cnxn, result)
    validate_aggregates(cnxn, result, required=profile != "analytics")
    if profile.startswith("public"):
        validate_publication(cnxn, result)

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Data quality checks for the analytics database")
    parser.add_argument("--db", type=Path, help=f"Database (default: CHESS_ANALYTICS_DB or {ANALYTICS_DB})")
    parser.add_argument("--profile", default="analytics", choices=("analytics", "public-full", "public-lite"))
    args = parser.parse_args(argv)
    db = args.db or os.environ.get('CHESS_ANALYTICS_DB') or ANALYTICS_DB
    with duckdb.connect(str(db), read_only=True) as cnxn:
        result = run_validation(cnxn, args.profile)
        passed = result.report()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
