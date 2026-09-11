"""Extract a verified publication bundle for a static build (standard library only)."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tempfile


FILE = re.compile(r'data/snapshots/([0-9a-f]{64})\.duckdb')


def prepare_public_data(archive, public_directory):
    public = Path(public_directory)
    public.mkdir(parents=True, exist_ok=True)
    destination = public / 'data'
    if destination.exists():
        raise FileExistsError('Choose a public directory without an existing data folder')
    with tempfile.TemporaryDirectory(prefix='.data-', dir=public.parent) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, 'r:gz') as bundle:
            members = bundle.getmembers()
            names = set()
            for member in members:
                if (not member.isfile() or member.name in names or
                        (member.name != 'data/manifest.json' and not FILE.fullmatch(member.name))):
                    raise ValueError(f'Unexpected archive entry: {member.name}')
                names.add(member.name)
            # Leave room for the bundled DuckDB-WASM engine under Pages' site limit.
            if sum(member.size for member in members) > 900 * 1024 ** 2:
                raise ValueError('Publication exceeds the 900 MiB static site data budget')
            if 'data/manifest.json' not in names:
                raise ValueError('Publication manifest is missing')
            bundle.extractall(staging, filter='data')

        data = staging / 'data'
        manifest = json.loads((data / 'manifest.json').read_text())
        if not isinstance(manifest, dict):
            raise ValueError('Invalid publication manifest')
        digest = manifest.get('sha256')
        path = manifest.get('snapshot')
        counts = manifest.get('row_counts')
        if (manifest.get('schema_version') != 1 or not isinstance(digest, str) or
                not re.fullmatch('[0-9a-f]{64}', digest) or
                path != f'snapshots/{digest}.duckdb' or
                type(manifest.get('size_bytes')) is not int or manifest['size_bytes'] <= 0 or
                not isinstance(counts, dict) or type(counts.get('search_stats')) is not int or counts['search_stats'] <= 0):
            raise ValueError('Invalid publication manifest')
        datetime.fromisoformat(manifest['exported_at'])
        current = data / path
        if not current.is_file() or current.stat().st_size != manifest['size_bytes']:
            raise ValueError('Current snapshot is missing or has an incorrect size')
        for name in names:
            match = FILE.fullmatch(name)
            if match:
                with (staging / name).open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != match.group(1):
                    raise ValueError(f'Snapshot checksum failed: {name}')
        data.replace(destination)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('public_directory', type=Path)
    args = parser.parse_args()
    manifest = prepare_public_data(args.archive, args.public_directory)
    print(f"Verified publication: {manifest['row_counts']['search_stats']:,} searches")
