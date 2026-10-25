"""Export a consistent, standalone public snapshot without modifying the source.

  python -m data.databases.export_public --archive /tmp/analytics-public.tar.gz
  python -m data.databases.export_public --full --keep 5

A snapshot holds the small tables (engines, experiments, engine_ratings,
game_stats, sprt_runs, sts_runs -- allow-listed columns only, see
public_schema.py), every agg_* table, search_sample and snapshot_info. The
dashboard reads the aggregates, so by default the per-search facts are left
out; --full adds search_stats, the iteration/tree/timing/root-move facts,
position_features, Stockfish's lines of the searched positions
(stockfish_moves) and the views over them.

Only the newest --keep snapshots (default 3) are kept in the output folder
and bundled in the archive.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile

import duckdb

from ..etl.paths import ANALYTICS_DB, DATA_DIR
from ..transforms.aggregates import build_aggregates
from ..transforms.transform_positions import STOCKFISH_MOVES_DDL, ensure_stockfish_columns
from ..transforms.transform_search import PVS_RESEARCH_TABLES, build_analytics_views, with_pvs_researches
from ..transforms.validate import run_validation
from .public_schema import FULL_VIEWS, SCHEMA_VERSION, SNAPSHOT_INFO, public_tables
from .test_schema import validate_schema


# Everything the snapshot and its aggregates are computed from, copied into a
# private staging database. Explicit publication boundary: never copy
# arbitrary user tables or stored SQL.
STAGING_TABLES = (
    'engines', 'experiments', 'engine_ratings', 'game_stats', 'sprt_runs', 'sts_runs',
    'search_stats', 'iterative_deepening_stats', 'search_tree_stats',
    'search_timings', 'root_moves', 'position_features',
)
SNAPSHOT_NAME = re.compile(r'[0-9a-f]{64}\.duckdb')
DEFAULT_OUTPUT = DATA_DIR / 'dashboard/public/data'
DEFAULT_KEEP = 3
# Storage v1.2.0 is readable by the dashboard's DuckDB-WASM (DuckDB 1.4) and any DuckDB >= 1.2.
# Lite snapshots hold ~20 small tables: 16 KB blocks instead of 256 KB keep them ~10x smaller.
STORAGE_OPTIONS = {
    'lite': "STORAGE_VERSION 'v1.2.0', BLOCK_SIZE 16384",
    'full': "STORAGE_VERSION 'v1.2.0'",
}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _sql_string(value):
    return str(value).replace("'", "''")


def _columns(cnxn, table, catalog):
    return {row[0] for row in cnxn.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_catalog = ? AND table_schema = 'main' AND table_name = ?
    """, [catalog, table]).fetchall()}


def _stage(source, staging):
    """Copy the source tables, install the views and build the aggregates; returns the search count."""
    with duckdb.connect(str(staging)) as s:
        # ATTACH does not accept parameter placeholders in DuckDB.
        s.execute(f"ATTACH '{_sql_string(source)}' AS source (READ_ONLY)")
        present = {row[0] for row in s.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_catalog = 'source' AND table_schema = 'main' AND table_type = 'BASE TABLE'
        """).fetchall()}
        missing = [t for t in STAGING_TABLES if t not in present]
        if missing:
            raise ValueError(f"The analytics database lacks {', '.join(missing)}; run "
                             "python -m data.databases.run_analytics_pipeline first")
        # One transaction: a consistent snapshot of every table.
        s.execute('BEGIN TRANSACTION')
        try:
            for table in STAGING_TABLES:
                # pvs_researches as the views and aggs derive it: it is published, its kinds are not
                source_sql = (with_pvs_researches(f'source.main."{table}"') if table in PVS_RESEARCH_TABLES
                              else f'SELECT * FROM source.main."{table}"')
                s.execute(f'CREATE TABLE "{table}" AS {source_sql}')
                
            # Stockfish's analysis is optional (databases the pipeline built before it have none);
            # only the positions of the copied searches are kept.
            s.execute(STOCKFISH_MOVES_DDL)
            if 'stockfish_moves' in present:
                s.execute("""
                    INSERT INTO stockfish_moves
                    SELECT fen, move, rank, eval, pv, depth, multipv, time_ms, stockfish, computed_at
                    FROM source.main.stockfish_moves WHERE fen IN (SELECT fen FROM search_stats)
                """)
            s.execute('COMMIT')
        except Exception:
            s.execute('ROLLBACK')
            raise
        s.execute('DETACH source')
        ensure_stockfish_columns(s)   # searches loaded before the sf_* column set grew
        build_analytics_views(s)
        build_aggregates(s, verbose=False)
        searches = s.execute('SELECT COUNT(*) FROM search_stats').fetchone()[0]
        s.execute('CHECKPOINT')
    return searches


def _table_bytes(cnxn, table, block_size):
    blocks = cnxn.execute(f"SELECT COUNT(DISTINCT block_id) FROM pragma_storage_info('{table}') "
                          "WHERE block_id >= 0").fetchone()[0]
    return blocks * block_size


def _write_snapshot(staging, working, full, searches, exported_at):
    """Copy the allow-listed relations from staging into a new database; returns ({table: rows}, {table: bytes})."""
    profile = 'full' if full else 'lite'
    tables = public_tables(full)
    with duckdb.connect() as c:
        c.execute(f"ATTACH '{_sql_string(working)}' AS snapshot ({STORAGE_OPTIONS[profile]})")
        c.execute('USE snapshot')
        c.execute(f"ATTACH '{_sql_string(staging)}' AS staging (READ_ONLY)")
        c.execute('BEGIN TRANSACTION')
        try:
            for table, columns in tables.items():
                if table == SNAPSHOT_INFO:
                    continue
                missing = [col for col in columns if col not in _columns(c, table, 'staging')]
                if missing:
                    raise ValueError(f"{table} lacks the published column(s) {', '.join(missing)}; "
                                     "rerun the analytics pipeline (load_analytics --full) first")
                select = ', '.join(f'"{col}"' for col in columns)
                c.execute(f'CREATE TABLE "{table}" AS SELECT {select} FROM staging.main."{table}"')
            sample_rows = c.execute('SELECT COUNT(*) FROM search_sample').fetchone()[0]
            c.execute(f'CREATE TABLE "{SNAPSHOT_INFO}" (key VARCHAR, value VARCHAR)')
            c.executemany(f'INSERT INTO "{SNAPSHOT_INFO}" VALUES (?, ?)', [
                ('schema_version', str(SCHEMA_VERSION)), ('profile', profile), ('exported_at', exported_at),
                ('searches', str(searches)), ('sample_rows', str(sample_rows)),
            ])
            c.execute('COMMIT')
        except Exception:
            c.execute('ROLLBACK')
            raise
        c.execute('DETACH staging')
        if full:
            build_analytics_views(c, FULL_VIEWS)
        result = run_validation(c, profile=f'public-{profile}')
        if not result.passed:
            result.report()
            raise ValueError('Public snapshot failed data validation')
        counts = {name: c.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] for name in tables}
        c.execute('CHECKPOINT snapshot')
        block_size = c.execute("SELECT block_size FROM pragma_database_size() "
                               "WHERE database_name = 'snapshot'").fetchone()[0]
        sizes = {name: _table_bytes(c, name, block_size) for name in tables}
        c.execute('USE memory')
        c.execute('DETACH snapshot')
    return counts, sizes


def prune_snapshots(snapshots, keep, current):
    """Delete all but the `keep` newest snapshots (the current one always stays); returns (kept, removed)."""
    files = [p for p in snapshots.iterdir()
             if p.is_file() and not p.is_symlink() and SNAPSHOT_NAME.fullmatch(p.name)]
    files.sort(key=lambda p: (p.name != current.name, -p.stat().st_mtime, p.name))
    kept, removed = files[:keep], files[keep:]
    for path in removed:
        path.unlink()
    return kept, removed


def export_public(source=ANALYTICS_DB, output=DEFAULT_OUTPUT, archive=None, keep=DEFAULT_KEEP, full=False,
                  verbose=True):
    if keep < 1:
        raise ValueError('keep must be at least 1')
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Analytics database does not exist: {source}')
    if source.is_relative_to(output):
        raise ValueError('The source database must be outside the publication directory')
    if archive is not None:
        archive = Path(archive).resolve()
        if archive == source or archive.is_relative_to(output):
            raise ValueError('Archive must be outside both the source file and publication directory')
    output.mkdir(parents=True, exist_ok=True)
    snapshots = output / 'snapshots'
    snapshots.mkdir(exist_ok=True)
    profile = 'full' if full else 'lite'
    exported_at = datetime.now(timezone.utc).isoformat()

    # Build beside the final files so each rename is atomic on this filesystem.
    with tempfile.TemporaryDirectory(prefix='.export-', dir=output) as temporary:
        staging = Path(temporary) / 'staging.duckdb'
        working = Path(temporary) / 'snapshot.duckdb'
        searches = _stage(source, staging)
        if not searches:
            raise ValueError('Refusing to publish a snapshot with no searches')
        counts, sizes = _write_snapshot(staging, working, full, searches, exported_at)
        if not validate_schema(working, profile=f'public-{profile}', verbose=False):
            validate_schema(working, profile=f'public-{profile}', verbose=True)  # print the details
            raise ValueError('Public snapshot failed schema validation')

        digest = sha256(working)
        filename = f'{digest}.duckdb'
        destination = snapshots / filename
        if destination.exists():
            if sha256(destination) != digest:
                raise ValueError(f'Existing immutable snapshot is corrupt: {destination.name}')
            os.utime(destination)  # counts as the newest for retention
        else:
            working.replace(destination)
        row_counts = dict(counts)
        # The dashboard's manifest check expects the number of searches here; in a
        # lite snapshot it is the number of searches the aggregates summarize.
        row_counts['search_stats'] = counts.get('search_stats', searches)
        manifest = {
            'schema_version': SCHEMA_VERSION,
            'exported_at': exported_at,
            'profile': profile,
            'snapshot': f'snapshots/{filename}',
            'sha256': digest,
            'size_bytes': destination.stat().st_size,
            'searches': searches,
            'row_counts': row_counts,
        }
        pending_manifest = Path(temporary) / 'manifest.json'
        pending_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
        pending_manifest.replace(output / 'manifest.json')

    kept, removed = prune_snapshots(snapshots, keep, destination)
    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.bundle-', dir=archive.parent) as temporary:
            pending = Path(temporary) / 'analytics-public.tar.gz'
            # DuckDB files are compressed already: level 6 is as small as 9, and much faster.
            with tarfile.open(pending, 'w:gz', compresslevel=6) as bundle:
                bundle.add(output / 'manifest.json', arcname='data/manifest.json')
                # Retain the last snapshots for browsers still using a prior manifest.
                for snapshot in sorted(kept):
                    bundle.add(snapshot, arcname=f'data/snapshots/{snapshot.name}')
            pending.replace(archive)

    if verbose:
        print(f"Snapshot ({profile}): {manifest['size_bytes'] / 2 ** 20:.2f} MiB, {searches:,} searches summarized")
        for name, rows in counts.items():
            print(f"  {name:.<30s} {rows:>10,} rows  ~{sizes[name] / 2 ** 20:7.2f} MiB")
        print(f"Kept {len(kept)} snapshot(s); removed {len(removed)} older one(s)")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=ANALYTICS_DB)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--archive', type=Path, help='Optional release bundle, outside the public data directory')
    parser.add_argument('--keep', type=int, default=DEFAULT_KEEP,
                        help=f'Snapshots kept in the output folder and archive (default: {DEFAULT_KEEP})')
    parser.add_argument('--full', action='store_true',
                        help='Also publish the per-search facts and views (much larger)')
    args = parser.parse_args()
    manifest = export_public(args.source, args.output, args.archive, keep=args.keep, full=args.full)
    print(f"Exported {manifest['searches']:,} searches to {args.output}")
    if args.archive:
        print(f'Release asset: {args.archive}')


if __name__ == '__main__':
    main()
