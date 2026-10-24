# San Jacinto dashboard

Interactive analytics for the Tomahawk chess engine, built with
[Mosaic](https://uwdata.github.io/mosaic/) and DuckDB-WASM. All queries run in
the visitor's browser; the site is static and needs no server.

## Setup

```bash
# Node.js 18+ (CI uses 22)
cd data/dashboard
npm ci
npm run dev      # http://localhost:3000
npm test         # unit tests of the pure modules (node:test)
npm run build    # static site in dist/
```

Published snapshots load automatically when `public/data/manifest.json` exists.
`?local=1` opens a file picker (the file is read in slices, never uploaded or
copied into memory), and `?db=https://example.org/file.duckdb` loads another
HTTP URL.

## Architecture

**Loading.** `main.js` starts three things at once: the manifest request, the
DuckDB-WASM engine (download and compile) and the code of the first tab.
`db.open()` is called with `duckdbConfig()` from `data-source.js`:
`allowFullHTTPReads: true, forceFullHTTPReads: false, reliableHeadRequests: false`.
The snapshot is then `ATTACH`ed read-only as `db` and read with HTTP range
requests, so a visitor downloads only the pages the open tab touches. GitHub
Pages answers a HEAD carrying a Range header with 200, which is why HEAD
responses are not trusted; a host without range support still works through one
full download. The whole schema is read with one `duckdb_columns()` query.

**Datasets.** Charts never read the snapshot directly. `datasets.js` defines
named datasets (`defs-search.js`, `defs-results.js`, `defs-engine.js`); each is
built once per session into a small temporary table `ds_<name>` in the
in-memory database:

1. from the snapshot's pre-aggregated table (`agg_*`, `search_sample`) when it
   exists and has the needed columns, mapped to the names in
   [docs/dashboard-data.md](../../docs/dashboard-data.md);
2. otherwise computed in the browser from the analytics views and facts (local
   analytics files and `--full` snapshots), with the same column names.

Each chart's note says which of the two it used. After the first tab is drawn,
the remaining pre-aggregated tables are copied in the background, so later tab
switches do not wait for the network. Nothing heavy stays on the main thread:
the temporary tables live in the DuckDB worker, and charts draw aggregates, the
search sample or binned marks, never one mark per search.

**Engines.** `engines.js` loads the engines (labelled `name version`, ordered by
id), assigns each a fixed Okabe–Ito colour in that order (`palette.js`), and
owns one Mosaic selection, `$engines`. Charts that query a table filter by it
(`vg.from(table, { filterBy })`); charts drawn from query results use its SQL
predicate (`engineWhere()`), so every tab follows the same selection. The
baseline is the version the others are compared with. By default every engine
is selected, up to the eight the palette can tell apart (the newest eight when
there are more).

**Core type.** Games from the match runner record the core type they ran on
(`P`, `E`, or `-` when not pinned). The engine bar's *Cores* control
(All, P-cores, E-cores, Unpinned) sets one shared filter in `settings.js` that
every tab applies: `coreRows()` picks the matching rows of tables that carry
`'all'` rows (`agg_*`), `coreFilter()` filters per-game and per-search rows,
and `engineScoped({ core: true })` adds the same clause to Mosaic selections.
Games whose core moved (`mixed`) or is not recorded (`unknown`) count only
under All. Aggregates that are not split by core type (tree ply, timings,
correlations, root moves computed from the raw log) say so in a short note while a core type is chosen,
and Hardware always shows every core type side by side. When a snapshot has no
core types to choose from, the control is replaced by a note.

**Older snapshots.** Match-runner fields (core type and clock, game results
linked to searches, pentanomial SPRT counts, opening variations) are the
primary path. When a snapshot lacks them, the tabs stay and the affected panels
say what is missing instead of hiding.

**Tabs** are ES modules in `src/views/`, imported on first use and kept after
that, so returning to a tab is instant. Tab, engines, baseline, core type and
the opening grouping are kept in the URL hash (`state.js`, for example
`#tab=compare&engines=4,5&base=4&cores=P&open=variation`), so links can be
shared.

**Style.** `style.css` defines an editorial look: warm paper background, dark
ink, one oxblood accent for the current tab and links, serif headings and
numbers, sentence-case labels and tabular numerals, with a warm dark variant
under `prefers-color-scheme: dark`. `ui.js` holds the panel, stat line, control,
table and chart helpers; every chart gets a one-sentence caption.

| File | Role |
|---|---|
| `src/main.js` | Start-up, navigation, engine bar, lazy tabs, background prefetch |
| `src/data-source.js` | Manifest and URL resolution, DuckDB open options |
| `src/db.js` | Queries, schema of the attached snapshot |
| `src/datasets.js`, `src/defs-*.js` | Dataset registry: pre-aggregated table or live fallback |
| `src/engines.js`, `src/palette.js`, `src/state.js`, `src/settings.js` | Selection, colours, URL state, shared settings (core type, opening grouping) |
| `src/stats.js` | Pure statistics: Wilson intervals, ratio errors, W/D/L buckets, logistic fit, histogram quantiles, Elo, NPS split |
| `src/stockfish.js` | Pure helpers of the Stockfish comparison: evaluation and mate labels, Lichess links, rank and loss shares, mean loss interval |
| `src/ui.js`, `src/dom.js`, `src/style.css` | Layout, controls, chart theme |

## Tabs

| Group | Tab | What it shows | Data |
|---|---|---|---|
| Results | Overview | Stat line, a table of every selected version, distributions of a chosen metric, searches per engine | `agg_engine_summary`, `agg_hist`, `game_stats` |
| | Games | Results from each engine's side, how games ended, game length, time losses, illegal moves, crashes and adjudications per engine, most played openings | `game_stats` |
| | Openings | Score by opening with 95% intervals, frequency, ECO volumes, search depth by opening; family or family + variation | `agg_openings`, `search_sample` |
| | Eval vs WDL | Win/draw/loss share and expected score per eval bucket of the side to move (the width is read from the data), games under the axis, intervals with games as the sample size, logistic fit, mate scores; one engine or overlaid, by phase | `agg_eval_wdl` |
| | SPRT | Verdicts (pass, fail, inconclusive), a timeline from `experiments.start_time_utc`, game-pair outcomes of recent runs, every run with pairs, pentanomial counts, Elo and nElo intervals and LOS, and the per-core-type split | `sprt_runs`, `experiments` |
| Search | Searches | Depth against a chosen effort measure (median and middle half), linked histograms that brush each other | `search_sample` |
| | Iterations | A measure per iteration depth with an "up to depth" interval slider, how many searches reach each depth, cumulative time | `agg_depth` |
| | Tree depth | Where the nodes are per ply (brushable) and a chosen measure per ply | `agg_tree_ply` |
| | Pruning | SEE, delta and null-move rates along iterations or plies, and per engine | `agg_depth`, `agg_tree_ply`, `agg_engine_summary` |
| | Move order | Which move index caused each cutoff, per engine; first-move and late cutoffs by depth or ply | `agg_moveorder` |
| | Time management | Time per iteration, search-time distribution, cumulative time by depth, time against nodes | `agg_depth`, `agg_hist`, `search_sample` |
| | Root moves | Effort by root move order, how clear the choice was, most chosen moves; then, by phase, the chosen move against Stockfish's top N lines: first-choice, top-3 and top-N shares, mean centipawn loss with 95% intervals, rank and loss distributions, root effort on Stockfish's lines, where its best move stood in the root ordering, and the largest misses with Lichess links | `agg_root_index`, `agg_root_gap`, `agg_root_best`, `agg_sf_moves`, `agg_sf_root_effort`, `agg_sf_root_order`; the largest misses need `search_metrics` and `stockfish_moves` (full snapshots and local files) |
| Engine | Compare | One column per engine with differences against the baseline and 95% margins (NPS as Σnodes/Σtime), quantile boxes and CDFs, SPRT result against the baseline | `agg_engine_summary`, `agg_hist`, `search_sample` |
| | Trends | Key metrics across versions in registration order, with intervals | `agg_engine_summary` |
| | Relationships | Per-engine correlation matrix (Spearman or Pearson, or the change against the baseline) and a linked scatter whose brush filters the charts below | `agg_corr`, `search_sample` |
| | Hardware | NPS against the measured clock, knps per GHz by core type, and the NPS change split into code and hardware | `agg_env` |
| | Timing | Profiled functions: share of search time, µs per call, calls and ms per search | `agg_timing` |
| | Positions | Measures by game phase and position type, tactical against positional scores | `agg_positions`, `search_sample` |
| | Eval quality | Agreement with Stockfish: error distribution, best-move agreement by depth, scatter | `search_sample`, `agg_hist` |

## Adding a tab

1. Create `src/views/mytab.js` exporting `async function render()` that returns a
   DOM node or fragment. Use `panel({ title, caption, body })` from `ui.js`, and
   give every chart a one-sentence caption.
2. Read data through `dataset('name')` (add a definition to a `defs-*.js` file
   if needed) and filter with `$engines` or `engineWhere()`, and by core type
   with `coreRows()` / `coreFilter()` from `settings.js` or
   `engineScoped({ core: true })`; if the data is not split by core type, add
   `coreScopeNote()` from `ui.js` instead. When a snapshot lacks the data, show
   a short `emptyNote()` saying what is missing rather than hiding the panel.
3. Add the tab to `TABS` in `src/main.js` with a lazy `load: () => import(...)`.

## Data requirements

The dashboard reads what `python -m data.databases.export_public` publishes
(see [docs/dashboard-data.md](../../docs/dashboard-data.md)): `engines`,
`experiments`, `game_stats`, `sprt_runs`, the `agg_*` tables and
`search_sample`. A `--full` snapshot or a local analytics database adds the
per-search tables and views (`search_metrics`, `iterative_deepening_stats`,
`search_tree_stats`, `search_timings`, `root_moves`, `position_features`,
`stockfish_moves`); without `agg_*` tables the tabs compute their aggregates
from those instead, except the Stockfish comparison on Root moves, which needs
its `agg_sf_*` tables.

To refresh a local analytics database without reloading facts:

```bash
python -m data.transforms.transform_search   # views
python -m data.transforms.aggregates         # agg_* tables and search_sample
python -m data.databases.test_schema
```

Run these from the repository root; `CHESS_ANALYTICS_DB` selects the database.
`python -m data.databases.run_analytics_pipeline` runs the same steps as part of
a full rebuild from the raw database (load, positions, views, aggregates,
validation and the schema check).
See [public deployment and updates](../../docs/public-dashboard.md).
