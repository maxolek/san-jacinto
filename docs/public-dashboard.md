# Public analytics dashboard

The public version is a static website. Visitors open a URL and the browser
loads a versioned DuckDB snapshot automatically. Queries and cross-filtering
run in DuckDB-WASM; there is no continuously running Python service. The
published data is downloadable by anyone who can access the site.

## Prepare a publication

From the repository root, with the usual analytics dependencies installed:

```sh
python3 -m data.databases.export_public --archive /tmp/analytics-public.tar.gz
```

`--source /path/to/chess_analytics.duckdb` overrides `CHESS_ANALYTICS_DB` and the
platform default. `--output /path/to/public/data` overrides the default
`data/dashboard/public/data` directory. The source can still contain the old
materialized feature tables: the export recreates the current SQL views in a
fresh database without changing the source.

The exporter:

1. Reads all selected facts/metadata in one transaction through a read-only
   attachment, preserving a consistent snapshot.
2. Copies only the 13 tables listed in `PUBLIC_TABLES`, including stored
   position features and Stockfish evaluations. All columns/rows in those
   tables are published. Unrelated tables, attached raw databases, migration
   history, and arbitrary source views are excluded.
3. Installs the current analytics views and validates the result. Missing
   inputs, invalid joins, or an empty search table stop publication.
4. Writes `snapshots/<sha256>.duckdb`, then atomically replaces `manifest.json`.
   The manifest records the export timestamp, row counts, size, and checksum.
5. Creates `analytics-public.tar.gz` as a separate release asset. It contains
   the manifest and retained immutable snapshots; it contains no source code.

Run one export at a time against each output directory. Failed database
exports leave the previous manifest/bundle intact. Old snapshots are retained
so a browser that opened the previous version can finish its queries. Continue
using the same output directory between exports to retain that history.
Generated data is ignored by Git; release assets keep binary history out of
the source repository.

## Publish with GitHub Pages

The checked-in workflow is `.github/workflows/dashboard-pages.yml`. It runs
only when manually dispatched; ordinary pushes do not publish a site.

1. Review and commit/push the dashboard, workflow, and export changes when
   ready. Neither the export command nor this workflow commits or pushes code.
2. In the repository's **Settings → Pages**, choose **GitHub Actions** as the
   publishing source. This may require repository administrator access.
3. Create a repository release with a unique data tag (for example,
   `analytics-2026-09-10`). Upload the generated `analytics-public.tar.gz`
   as a release asset. Treat each published data release as immutable.
4. In **Actions → Publish analytics dashboard → Run workflow**, enter that
   release tag as `data_release` and run the workflow from the reviewed branch.
5. Use the URL reported by the deployment job. A successful workflow must be
   verified before treating the site as live.

The workflow downloads only the selected repository release, validates the
archive's paths and snapshot checksums, installs locked npm dependencies,
runs the data-loading tests, builds the dashboard, and deploys the static
artifact. Relative asset paths support the repository subpath as well as a
custom domain. No database credentials are needed in the frontend.

The bundle verifier reserves 900 MiB for data to leave room for the bundled
DuckDB engine within GitHub Pages' 1 GB published-site limit. Browser memory
and query latency may become limiting earlier. If the dataset grows beyond
comfortable browser use, reconsider separate Parquet files/object storage or
a hosted query service rather than accumulating unlimited snapshots.

For another static host, deploy `data/dashboard/dist/` after exporting data
and running `npm run build` in `data/dashboard`. The host should serve `.wasm`
with its correct MIME type and support HTTP byte-range requests for the DuckDB
file. Hosting data and frontend together avoids cross-origin configuration.

## Refresh or roll back

Run the analytics pipeline, export again, upload a new data release, and
manually dispatch the workflow with that release tag. Each deployment carries
one complete manifest plus its data files. Existing browsers keep using the
snapshot they already opened; reloading the page fetches the current manifest
with cache bypass and shows its publication time. The timestamp means
**export time**, not the time of the last engine experiment.

To roll back the data, dispatch the same reviewed code with a previous data
release tag. If the schema or frontend contract has changed, use the matching
code revision as well. Retain released bundles for reproducible rollbacks.

## Local usage and checks

With no manifest in development, the existing file picker still appears.
`?local=1` explicitly selects a local file even when public data exists.
`?db=https://example.org/data.duckdb` overrides the manifest with another HTTP
URL (the remote host must permit the request). Production displays an error
if its default publication is unavailable; it does not silently show demo data.

```sh
python3 -m unittest data.databases.test_analytics_views data.databases.test_public_export
cd data/dashboard
npm ci
npm test
npm run build
```

`prepare_public_data.py ARCHIVE PUBLIC_DIRECTORY` can also verify/extract a
release bundle locally using Python 3.12+. Use an output without an existing
`data/` folder; it will not overwrite an existing publication.

## Current setup status

The publication code and workflow are prepared. No release, commit, push, or
live deployment was performed. The database under `~/Documents/databases`
remains inaccessible to the agent because of macOS file permissions. Run the
export yourself in a terminal with access, or copy the database to an accessible
location and pass it with `--source`, before publishing real data.

References: [Vite static deployment](https://vite.dev/guide/static-deploy),
[GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
[GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).
