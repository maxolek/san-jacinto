"""
Pipeline integration test — validates that the analytics DB schema
is consistent with what the dashboard expects.

Run after the pipeline to catch schema mismatches early:
  python -m data.databases.test_schema
"""
import duckdb
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.etl.paths import ANALYTICS_DB


# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED SCHEMAS (what the dashboard queries)
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_TABLES = {
    'search_stats': [
        'id', 'engine_id', 'fen', 'ply', 'time_ms', 'eval',
        'completed_depth', 'depth', 'qdepth', 'move',
        'nodes', 'qnodes', 'tt_stores', 'tt_hits',
        'fail_highs', 'fail_lows', 'nmp', 'nmp_failhigh',
    ],
    'engines': ['id', 'name', 'version'],
    'game_stats': ['id', 'result', 'opening', 'run_time_s'],
    'iterative_deepening_stats': [
        'search_id', 'depth', 'qdepth', 'time_ms', 'eval', 'move',
        'nodes', 'qnodes', 'tt_hits', 'fail_highs',
    ],
    'search_tree_stats': [
        'search_id', 'depth', 'nodes', 'qnodes',
        'fail_highs', 'fail_lows', 'nmp',
    ],
}

# Feature tables (built by transforms — may not exist on first run)
EXPECTED_FEATURE_TABLES = {
    'search_features': [
        'search_id', 'engine_id', 'fen', 'total_time_ms', 'final_eval',
        'completed_depth', 'total_nodes', 'nps', 'qratio',
        'tt_hit_ratio', 'fail_high_ratio', 'avg_ebf',
    ],
    'search_iteration_features': [
        'search_id', 'depth', 'qdepth', 'nodes', 'qnodes', 'nps',
        'ebf', 'qratio', 'tt_hit_ratio',
    ],
    'search_tree_features': [
        'search_id', 'depth', 'nodes', 'qnodes',
        'ebf', 'qratio', 'tt_hit_ratio',
    ],
}

OPTIONAL_TABLES = {
    'search_timings': ['search_id', 'function', 'total_time_ms', 'num_calls'],
    'root_moves': ['search_id', 'move', 'eval'],
    'sprt_runs': ['id', 'result'],
    'sts_runs': ['id', 'suite'],
    'engine_ratings': ['engine_id'],
}


def get_columns(cnxn, table):
    """Get column names for a table."""
    try:
        info = cnxn.execute(f"PRAGMA table_info('{table}')").fetchall()
        return {row[1] for row in info}
    except Exception:
        return set()


def validate_schema():
    DB = os.environ.get('CHESS_ANALYTICS_DB') or str(ANALYTICS_DB)

    if not Path(DB).exists():
        print(f"SKIP: Analytics DB not found at {DB}")
        print("  (Run the pipeline first: python -m data.databases.run_analytics_pipeline)")
        return True

    cnxn = duckdb.connect(DB, read_only=True)
    errors = []
    warnings = []

    # Check required tables
    for table, required_cols in EXPECTED_TABLES.items():
        cols = get_columns(cnxn, table)
        if not cols:
            errors.append(f"MISSING TABLE: {table}")
            continue
        for col in required_cols:
            if col not in cols:
                errors.append(f"MISSING COLUMN: {table}.{col}")

    # Check feature tables (warn only, they may not be built yet)
    for table, required_cols in EXPECTED_FEATURE_TABLES.items():
        cols = get_columns(cnxn, table)
        if not cols:
            warnings.append(f"Feature table not built yet: {table}")
            continue
        for col in required_cols:
            if col not in cols:
                errors.append(f"MISSING COLUMN in feature table: {table}.{col}")

    # Check optional tables (just warn)
    for table, required_cols in OPTIONAL_TABLES.items():
        cols = get_columns(cnxn, table)
        if not cols:
            warnings.append(f"Optional table missing: {table}")
            continue
        for col in required_cols:
            if col not in cols:
                warnings.append(f"Optional column missing: {table}.{col}")

    # Row count sanity
    try:
        search_count = cnxn.execute("SELECT COUNT(*) FROM search_stats").fetchone()[0]
        if search_count == 0:
            warnings.append("search_stats is empty (no data loaded)")
    except Exception:
        pass

    cnxn.close()

    # Report
    print("\n" + "=" * 60)
    print("  SCHEMA VALIDATION REPORT")
    print("=" * 60)

    if not errors and not warnings:
        print("  ✓ All schema checks passed.")
    for w in warnings:
        print(f"  ⚠ {w}")
    for e in errors:
        print(f"  ✗ {e}")

    print("=" * 60)
    print(f"  {len(errors)} errors, {len(warnings)} warnings")

    return len(errors) == 0


if __name__ == "__main__":
    passed = validate_schema()
    sys.exit(0 if passed else 1)
