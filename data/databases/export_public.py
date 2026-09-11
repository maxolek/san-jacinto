"""Export a consistent, standalone public snapshot without modifying the source.

python -m data.databases.export_public --archive /tmp/analytics-public.tar.gz
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tempfile

import duckdb

from ..etl.paths import ANALYTICS_DB, DATA_DIR
from ..transforms.transform_search import build_analytics_views
from ..transforms.validate import run_validation
from .test_schema import validate_schema


# Explicit publication boundary: never copy arbitrary user tables or stored SQL.
PUBLIC_TABLES = (
    'engines', 'experiments', 'engine_ratings', 'game_stats', 'sprt_runs', 'sts_runs',
    'search_stats', 'iterative_deepening_stats', 'search_tree_stats',
    'search_timings', 'root_moves', 'dim_positions', 'position_features',
)
SNAPSHOT_NAME = re.compile(r'[0-9a-f]{64}\.duckdb')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_public(source=ANALYTICS_DB, output=DATA_DIR / 'dashboard/public/data', archive=None):
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

    # Build beside the final files so each rename is atomic on this filesystem.
    with tempfile.TemporaryDirectory(prefix='.export-', dir=output) as temporary:
        working = Path(temporary) / 'snapshot.duckdb'
        with duckdb.connect(str(working)) as c:
            # ATTACH does not accept parameter placeholders in DuckDB.
            source_sql = str(source).replace("'", "''")
            c.execute(f"ATTACH '{source_sql}' AS source (READ_ONLY)")
            c.execute('BEGIN TRANSACTION')
            try:
                for table in PUBLIC_TABLES:
                    c.execute(f'CREATE TABLE "{table}" AS SELECT * FROM source.main."{table}"')
                c.execute('COMMIT')
            except Exception:
                c.execute('ROLLBACK')
                raise
            c.execute('DETACH source')
            build_analytics_views(c)
            result = run_validation(c)
            if not result.passed:
                result.report()
                raise ValueError('Public snapshot failed data validation')
            counts = {name: c.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                      for name in PUBLIC_TABLES}
            if not counts['search_stats']:
                raise ValueError('Refusing to publish a snapshot with no searches')
            c.execute('CHECKPOINT')
        if not validate_schema(working):
            raise ValueError('Public snapshot failed schema validation')

        digest = sha256(working)
        filename = f'{digest}.duckdb'
        destination = snapshots / filename
        if destination.exists():
            if sha256(destination) != digest:
                raise ValueError(f'Existing immutable snapshot is corrupt: {destination.name}')
        else:
            working.replace(destination)
        manifest = {
            'schema_version': 1,
            'exported_at': datetime.now(timezone.utc).isoformat(),
            'snapshot': f'snapshots/{filename}',
            'sha256': digest,
            'size_bytes': destination.stat().st_size,
            'row_counts': counts,
        }
        pending_manifest = Path(temporary) / 'manifest.json'
        pending_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
        pending_manifest.replace(output / 'manifest.json')

    if archive is not None:
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.bundle-', dir=archive.parent) as temporary:
            pending = Path(temporary) / 'analytics-public.tar.gz'
            with tarfile.open(pending, 'w:gz') as bundle:
                bundle.add(output / 'manifest.json', arcname='data/manifest.json')
                # Retain old snapshots for browsers still using a prior manifest.
                for snapshot in sorted(snapshots.iterdir()):
                    if snapshot.is_file() and not snapshot.is_symlink() and SNAPSHOT_NAME.fullmatch(snapshot.name):
                        bundle.add(snapshot, arcname=f'data/snapshots/{snapshot.name}')
            pending.replace(archive)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ANALYTICS_DB)
    parser.add_argument('--output', type=Path, default=DATA_DIR / 'dashboard/public/data')
    parser.add_argument('--archive', type=Path, help='Optional release bundle, outside the public data directory')
    args = parser.parse_args()
    manifest = export_public(args.source, args.output, args.archive)
    print(f"Exported {manifest['row_counts']['search_stats']:,} searches to {args.output}")
    if args.archive:
        print(f'Release asset: {args.archive}')


if __name__ == '__main__':
    main()
