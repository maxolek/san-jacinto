# Public analytics dashboard

The public version is a static website. Visitors open a URL and the browser
loads a versioned DuckDB snapshot automatically. Queries and cross-filtering
run in DuckDB-WASM; there is no continuously running Python service. The
published data is downloadable by anyone who can access the site.

## Using the dashboard

- **Engines and baseline.** The chips under the navigation choose which engine
  versions appear on every tab (one shared Mosaic selection filters all charts),
  and the baseline menu picks the version the others are measured against on
  Compare, Hardware and Relationships (it also orders the Timing chart). Each
  version keeps one colour everywhere, taken from the colour-blind-safe
  Okabe–Ito palette in registration order. Every version is selected at first,
  up to eight (the newest eight when there are more).
- **Core type.** The *Cores* control next to the chips (All, P-cores, E-cores,
  Unpinned) restricts every tab to games that ran on that kind of core, as
  recorded by the match runner. Charts whose data is not split by core type say
  so while a core type is chosen, and the Hardware tab always compares all of
  them side by side. When the snapshot records only one kind of core, or none,
  a short note replaces the control.
- **Tabs** are grouped into *Results* (Overview, Games, Openings, Eval vs WDL,
  SPRT), *Search* (Searches, Iterations, Tree depth, Pruning, Move order, Time
  management, Root moves) and *Engine* (Compare, Trends, Relationships,
  Hardware, Timing, Positions, Eval quality). Every chart has a one-line caption
  saying what it shows, and a note saying whether its data came pre-aggregated
  in the snapshot or was computed in the browser.
- **Links are shareable.** The address keeps the tab, the selected engines, the
  baseline, the core type and the opening grouping, for example
  `…/#tab=compare&engines=3,4,5&base=4&cores=P`.
- **Loading.** The snapshot is read with HTTP range requests, so a visitor
  downloads only the pages the open tab needs (a few MB for a lite snapshot);
  the first tab's code and the query engine load in parallel and the other
  tabs load on demand. Every tab works from a lite snapshot's pre-aggregated
  tables except Root moves' "Largest misses", which lists individual positions
  and says so; panels that need match-runner data (clock speeds, game pairs,
  per-core SPRT results) say so when an older snapshot lacks it.

## Prepare a publication

From the repository root, with the usual analytics dependencies installed:

```sh
python3 -m data.databases.export_public --archive /tmp/analytics-public.tar.gz
```

`--source /path/to/chess_analytics.duckdb` overrides `CHESS_ANALYTICS_DB` and the
platform default. `--output /path/to/public/data` overrides the default
`data/dashboard/public/data` directory. `--keep N` sets how many snapshots are
retained (default 3). `--full` also publishes the per-search tables (see below).
The source can still contain the old materialized feature tables: the export
recreates the current SQL views in a separate staging database without changing
the source.

The exporter:

1. Reads the facts and metadata in one transaction through a read-only
   attachment, which gives a consistent snapshot. It copies them into a private
   staging database next to the output.
2. Installs the analytics views there, and rebuilds every `agg_*` table and
   `search_sample` from that data (`data/transforms/aggregates.py`).
3. Writes a new database that holds only the publication allow-list from
   `data/databases/public_schema.py`. It contains:
   - The small tables with the allowed columns only. Engine descriptions and
     compile flags, experiment metadata (machine info, local paths) and game
     move lists are never published.
   - The aggregates and `search_sample`.
   - A `snapshot_info` table.

   With `--full` it also copies `search_stats`, the iteration, tree, timing and
   root-move facts and `position_features`, and installs six of the views.
   `dim_positions`, the compatibility views, migration history, attached raw
   databases and arbitrary source tables or views are always excluded.
4. Validates the data (`data.transforms.validate`) and checks the schema
   (`data.databases.test_schema`), which must match the allow-list exactly.
   Missing inputs, invalid joins, leaked columns or an empty search table stop
   publication.
5. Writes `snapshots/<sha256>.duckdb`, then atomically replaces `manifest.json`.
   - The manifest records the export timestamp, the profile, the number of
     searches, row counts, size and checksum.
   - Snapshots use DuckDB storage v1.2.0, which the dashboard's DuckDB-WASM
     reads. Lite snapshots use 16 KB blocks, which keeps them small.
6. Keeps only the newest `--keep` snapshots and creates
   `analytics-public.tar.gz` as a separate release asset. The asset contains the
   manifest and exactly those snapshots, and no source code.

The default ("lite") snapshot is what the dashboard needs. The per-search
tables make a snapshot many times larger. On the synthetic 104,640-search
database:

| Export | Snapshot size |
|---|---:|
| Lite (default) | 3.3 MiB (0.7 MiB as `.tar.gz`) |
| `--full` | 114.8 MiB |
| Previous export (all 13 tables) | 113.3 MiB |

The exact contents are listed in [dashboard-data.md](dashboard-data.md).

Run one export at a time against each output directory. Failed database
exports leave the previous manifest/bundle intact. The previous snapshots
(within `--keep`) are retained so a browser that opened an earlier version can
finish its queries. Continue using the same output directory between exports.
Generated data is ignored by Git; release assets keep binary history out of
the source repository.

## Publish with GitHub Pages

One command does the export, the release and the deployment (it needs the
GitHub CLI, `gh`, logged in with access to the repository):

```sh
python -m data.databases.publish --refresh --watch   # rebuild analytics, export, release, deploy, wait
python -m data.databases.publish                     # export, release, deploy
python -m data.databases.publish --dry-run           # export only; print the gh commands
```

With `--refresh`, Stockfish (when found) analyses only positions it has not
analysed before: its results are kept in the analytics database's
`stockfish_moves` table across full reloads. The first run analyses every
searched position, about 90 positions a second with 8 processes on a laptop
(depth 10, top 5 lines); `--skip-stockfish` skips new analysis and still
publishes what was analysed earlier.

It exports the snapshot, creates a release `analytics-<UTC time>` holding the
bundle (not marked as the latest release), and starts the Pages workflow for
it. The one-time setup is step 2 below, and the workflow file must be on the
branch the workflow runs from (`--ref`, default: the default branch). The
manual steps are:

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
   release tag as `data_release` and run the workflow from `main`. The
   `github-pages` environment normally only allows the default branch.
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
with its correct MIME type. It should also support HTTP byte-range requests
for the DuckDB file and serve it without `Content-Encoding`: the dashboard
always opens snapshots with range reads (it does not trust HEAD responses,
which GitHub Pages answers with 200 even for ranged requests) and only falls
back to one full download when a host ignores ranges. Hosting data and
frontend together avoids cross-origin configuration.

## Refresh or roll back

`python -m data.databases.publish --refresh` does all of this. By hand: run
the analytics pipeline, export again, upload a new data release, and manually
dispatch the workflow with that release tag:

```sh
python -m data.databases.run_analytics_pipeline
python -m data.databases.export_public --archive /tmp/analytics-public.tar.gz
```

Each deployment carries one complete manifest plus its data files. Existing
browsers keep using the snapshot they already opened; reloading the page
fetches the current manifest with cache bypass and shows its publication time.
The timestamp means **export time**, not the time of the last engine experiment.

To roll back the data, dispatch the same reviewed code with a previous data
release tag. If the schema or frontend contract has changed, use the matching
code revision as well. Retain released bundles for reproducible rollbacks.

## Local usage and checks

With no manifest in development, the existing file picker still appears.
`?local=1` explicitly selects a local file even when public data exists; the
file is read in slices as queries need it, not copied into memory, so a full
analytics database also works (tabs without `agg_*` tables compute their
aggregates once, in the browser, as temporary tables).
`?db=https://example.org/data.duckdb` overrides the manifest with another HTTP
URL (the remote host must permit the request). Production displays an error
if its default publication is unavailable; it does not silently show demo data.

```sh
python -m unittest discover -s data -p "test_*.py"
cd data/dashboard
npm ci
npm test
npm run build
```

The same Python tests and `npm test` run in CI on every push and pull request
(`.github/workflows/checks.yml`).

`prepare_public_data.py ARCHIVE PUBLIC_DIRECTORY` can also verify/extract a
release bundle locally using Python 3.12+. Use an output without an existing
`data/` folder; it will not overwrite an existing publication.

## Current setup status

The publication code and workflow are prepared. Nothing has been released or
deployed yet. The analytics database paths default to `F:/databases` on
Windows and `~/Documents/databases` elsewhere (`data/etl/paths.py`). Set
`CHESS_RAW_DB` and `CHESS_ANALYTICS_DB`, or pass `--source`, to export from
another location.

References: [Vite static deployment](https://vite.dev/guide/static-deploy),
[GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
[GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).
