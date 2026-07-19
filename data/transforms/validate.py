"""
Data quality validation checks for the analytics database.

Runs after transforms to detect data integrity issues, range violations,
and schema inconsistencies. Reports warnings but does not block the pipeline
unless critical errors are found.
"""
import duckdb
import os
import sys
from ..etl.paths import ANALYTICS_DB


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
        print("\n" + "=" * 60)
        print("  DATA QUALITY REPORT")
        print("=" * 60)
        if not self.warnings and not self.errors:
            print("  ✓ All checks passed.")
        for check, msg, count in self.warnings:
            cnt = f" ({count:,} rows)" if count else ""
            print(f"  ⚠ [{check}] {msg}{cnt}")
        for check, msg, count in self.errors:
            cnt = f" ({count:,} rows)" if count else ""
            print(f"  ✗ [{check}] {msg}{cnt}")
        print("=" * 60)
        total_checks = len(self.warnings) + len(self.errors)
        print(f"  {total_checks} issues: {len(self.errors)} errors, {len(self.warnings)} warnings")
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
        result.warn("searches", "search_stats table missing — skipping search checks")
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
        result.warn("games", "game_stats table missing — skipping game checks")
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
    """Validate computed feature tables for ratio bounds."""
    if not _table_exists(cnxn, 'search_features'):
        result.warn("features", "search_features not yet built — skipping feature checks")
        return

    # Ratios should be in [0, 1] (with some tolerance for floating point)
    ratio_cols = [
        'qratio', 'tt_hit_ratio', 'tt_store_ratio',
        'fail_high_ratio', 'fail_low_ratio', 'nmp_ratio',
    ]
    for col in ratio_cols:
        try:
            bad = _count(cnxn, f"SELECT COUNT(*) FROM search_features WHERE {col} < -0.01 OR {col} > 1.01")
            if bad > 0:
                result.warn(f"ratio_{col}", f"search_features.{col} outside [0,1]", bad)
        except Exception:
            pass  # column may not exist

    # EBF should be > 0 (where not null)
    try:
        bad_ebf = _count(cnxn, "SELECT COUNT(*) FROM search_features WHERE avg_ebf < 0")
        if bad_ebf > 0:
            result.warn("ebf_negative", "search_features.avg_ebf < 0", bad_ebf)
    except Exception:
        pass


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


def run_validation(cnxn):
    """Run all validation checks and return the result object."""
    result = ValidationResult()

    validate_searches(cnxn, result)
    validate_iterations(cnxn, result)
    validate_games(cnxn, result)
    validate_tree(cnxn, result)
    validate_timing(cnxn, result)
    validate_features(cnxn, result)

    return result


if __name__ == "__main__":
    DB = os.environ.get('CHESS_ANALYTICS_DB') or str(ANALYTICS_DB)
    cnxn = duckdb.connect(DB, read_only=True)

    result = run_validation(cnxn)
    passed = result.report()
    cnxn.close()

    if not passed:
        sys.exit(1)
