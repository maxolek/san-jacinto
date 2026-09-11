"""Install query-time analytics views over stored facts and position analysis.

Run this module once to migrate existing feature tables, or after loading data.
No search/iteration/tree features are materialized. Views reflect changes to
their source tables immediately; ``search_features`` is a compatibility view.
"""
import argparse
from pathlib import Path

import duckdb

from ..etl.paths import ANALYTICS_DB


SQL_DIR = Path(__file__).with_name("sql")
# Dependencies precede their consumers. Keep these names stable for notebooks.
ANALYTICS_VIEWS = (
    "search_metrics",
    "search_context",
    "search_position_metrics",
    "search_iteration_metrics",
    "search_iteration_features",
    "search_tree_features",
    "search_iteration_summary",
    "search_timing_summary",
    "search_features",
)


def _replace_view(cnxn, name):
    """Replace a legacy materialized table or update an existing view."""
    query = (SQL_DIR / f"{name}.sql").read_text()
    kind = cnxn.execute("""
        SELECT table_type FROM information_schema.tables
        WHERE table_catalog = current_database()
          AND table_schema = 'main' AND table_name = ?
    """, [name]).fetchone()
    if kind and kind[0] == "BASE TABLE":
        cnxn.execute(f'DROP TABLE "{name}"')
    cnxn.execute(f'CREATE OR REPLACE VIEW "{name}" AS {query}')


def build_search_iterations_features(cnxn, full=False):
    """Keep the historical Python entry point; ``full`` is now unnecessary."""
    _replace_view(cnxn, "search_iteration_metrics")
    _replace_view(cnxn, "search_iteration_features")


def build_search_tree_features(cnxn, full=False):
    _replace_view(cnxn, "search_tree_features")


def build_search_features(cnxn, full=False):
    for name in ("search_metrics", "search_context", "search_position_metrics",
                 "search_iteration_summary", "search_timing_summary", "search_features"):
        _replace_view(cnxn, name)


def build_analytics_views(cnxn):
    """Migrate atomically, retaining legacy tables if any view fails to bind."""
    cnxn.execute("BEGIN TRANSACTION")
    try:
        for name in ANALYTICS_VIEWS:
            _replace_view(cnxn, name)
        cnxn.execute("COMMIT")
    except Exception:
        cnxn.execute("ROLLBACK")
        raise


def main():
    parser = argparse.ArgumentParser(description="Migrate/install analytics SQL views")
    parser.add_argument('--full', action='store_true',
                        help='Accepted for compatibility; views always use current facts')
    parser.parse_args()
    with duckdb.connect(str(ANALYTICS_DB)) as cnxn:
        build_analytics_views(cnxn)
    print(f"Installed {len(ANALYTICS_VIEWS)} analytics views (no feature tables built).")


if __name__ == "__main__":
    main()
