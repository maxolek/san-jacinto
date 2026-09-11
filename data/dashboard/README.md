# Chess Engine Analytics Dashboard

Interactive analytics dashboard powered by [Mosaic](https://uwdata.github.io/mosaic/) + DuckDB-WASM. All computation happens in-browser — no server required.

## Setup

```bash
# Install Node.js (v18+) if not already installed:
# https://nodejs.org/

cd data/dashboard
npm install
npm run dev
```

Then open `http://localhost:3000`. Published snapshots load automatically when present.

Use `?local=1` for the file picker, or `?db=https://example.org/file.duckdb`
to load another HTTP URL. A path on your computer is loaded through the file picker.

## Architecture

- **DuckDB-WASM**: Runs the full DuckDB engine in the browser via WebAssembly
- **Mosaic**: Cross-filtering framework — brush one chart, all linked charts update instantly
- **Vite**: Dev server and bundler (fast HMR in development, optimized static build)

All queries run in-browser. Local files are not uploaded; public snapshots are
downloaded from the website and can be accessed by its visitors.

## Building for deployment

First export the data from the repository root:

```bash
python3 -m data.databases.export_public --archive /tmp/analytics-public.tar.gz
```

Then build from `data/dashboard`:

```bash
npm run build
```

Produces a `dist/` folder containing the frontend and exported snapshots.
The manually triggered GitHub Pages workflow consumes the export as a repository
release asset. See [public deployment and updates](../../docs/public-dashboard.md).
Generated snapshots are excluded from Git. Keep the data output directory between
exports so previous snapshot URLs remain available across updates.

## Tabs

| Tab | Description |
|-----|-------------|
| Overview | Summary metrics + key distributions |
| Searches | Cross-filtered scatter plots (depth, eval, nodes, TT) |
| Games | Results, openings, termination types |
| Trends | How metrics evolve across engine versions |
| Compare | Side-by-side engine box plots |
| Iterations | Per-iteration depth analysis |
| Tree Depth | Tree depth statistics |
| Timing | Function-level profiling breakdown |
| SPRT | Experiment results and timeline |

## Adding new tabs

1. Create `src/views/mytab.js` exporting an `async function renderMyTab()`
2. Import and add to the `TABS` array in `src/main.js`
3. The function should return a DOM element (use helpers from `src/util.js`)

## Data requirements

Expects a DuckDB file with these relations (from analytics pipeline):
- `engines`, `experiments`, `game_stats`
- `search_metrics` for search-level counters and ratios
- `search_context` for engine/game/STS metadata
- `search_position_metrics` for position analysis
- `search_iteration_metrics`, `search_tree_features` for per-depth charts
- `search_iteration_features` for optional branching/stability window queries
- `search_timings`, `root_moves`, `sprt_runs`

The analytics relations are SQL views over facts. Charts aggregate at query
time; no wide fact table is required. The old `search_features` name remains
a compatibility view for analysis scripts. Older analytics files with physical
feature tables are still detected, but should be migrated for consistent columns.

To migrate an existing database without reloading facts or rerunning Stockfish:

```bash
python -m data.transforms.transform_search
python -m data.databases.test_schema
```

Run these commands from the repository root. `CHESS_ANALYTICS_DB` selects a
database path. Reload the database in the dashboard after migration.
See [migration details](../../docs/olap-migration.md).
