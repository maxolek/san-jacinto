# Query-driven OLAP migration

The analytics layer stores measurements at their original grain and computes
SQL metrics when queried. Position analysis and Stockfish evaluations remain
stored because they are expensive to reproduce. The dashboard keeps using
DuckDB-WASM and Mosaic; hosting is unchanged.

## Apply to an existing database

Back up the analytics database while no writer is using it, then run from the
repository root:

```sh
python -m data.transforms.transform_search
python -m data.databases.test_schema
python -m data.transforms.validate
```

Set `CHESS_ANALYTICS_DB=/path/to/chess_analytics.duckdb` to override the default.
The first command atomically replaces `search_features`,
`search_iteration_features`, and `search_tree_features` tables with views and
installs their supporting views. If a definition fails, the transaction rolls
back, including any dropped feature tables. Running it again is safe. No fact
reload, position recomputation, or Stockfish run is needed for this migration.
`--full` is accepted by this command for compatibility but does not materialize
anything. Reopen the migrated file in the dashboard.

The normal analytics pipeline installs these views after loading facts and
computing position features. A pipeline `--full` still reloads source facts and
resets stored Stockfish columns as before; it is unnecessary for this migration.

## Query surfaces

SQL definitions live in `data/transforms/sql/`:

| View | Use |
|---|---|
| `search_metrics` | Overview, searches, comparison, trends, pruning, evaluation quality, time management |
| `search_context` | Opening analysis and move ordering by engine |
| `search_position_metrics` | Position classifications and performance |
| `search_iteration_metrics` | Iteration charts: counters, ratios and evaluation values without windows |
| `search_iteration_features` | Iteration metrics plus branching and stability windows |
| `search_tree_features` | Tree-depth counters and ratios |
| `search_iteration_summary` | One row per search with aggregated iteration metrics |
| `search_timing_summary` | One row per search with a timing pivot |
| `search_features` | Compatibility join for notebooks/scripts using the previous wide table |

For example, an engine comparison needs only search metrics:

```sql
SELECT engine_id, COUNT(*) AS searches,
       AVG(completed_depth) AS avg_depth,
       SUM(total_nodes) / NULLIF(SUM(total_time_ms) / 1000.0, 0) AS aggregate_nps
FROM search_metrics
GROUP BY engine_id;
```

`AVG(nps)` weights each search equally; the ratio of summed nodes and time
weights by elapsed time. Both are valid, distinct statistics. Existing chart
averages retain their meaning. Join child facts only after aggregating to the
desired grain. Metadata IDs and position search IDs must be unique; validation
now checks that assumption.

## Corrected metric behavior

- Zero denominators return NULL, including iteration growth and timing shares.
  A fail-high count of one now produces a valid ratio.
- Geometric EBF uses natural logs and positive observations; unavailable/zero
  EBF values no longer contribute an artificial value of one.
- Search-level evaluation sign flips sum all iteration transitions. Each
  iteration's `eval_sign_flips` remains a zero/one transition indicator.
- `stddev_last5_eval` uses the current iteration and four preceding iterations.
- `eval_diff` is computed from current evaluations. An obsolete stored column
  in an older `search_stats` table is ignored and no longer maintained.
- Best-move agreement uses `sf_best_move`. `sf_pv` contains a continuation, not
  ranked candidate moves: `engine_move_rank` is 1 for a known best-move match
  and NULL otherwise, since other ranks cannot be recovered from this data.
- Searches without a known game result have NULL `game_score`, not a loss.
- Search views supply both historical aliases and the dashboard's raw counter
  names, including `move`, `tt_hits`, `fail_highs`, and `total_nmp_failhigh`.

## Refresh and performance limits

Views immediately reflect changes already made to the DuckDB source tables,
including late position/evaluation enrichment and edits to existing iteration
rows. This does not change the SQLite loader's append-by-ID policy: corrections
and late child records in SQLite still need to be synchronized into DuckDB.
The position-analysis cache also retains its existing incremental behavior.

Simple search queries avoid metadata joins and iteration windows entirely.
Ordinary iteration charts use the separate `search_iteration_metrics` view,
so even iteration NPS/ratio charts avoid the branching/stability windows.
Window metrics and the full compatibility view can cost more at query time
than the previous materialized tables. Benchmark actual browser workloads
before adding any targeted cache. Dropping tables frees internal storage for
reuse; it does not guarantee that an existing database file shrinks on disk.

Regression tests use a fixture built from the repository's raw SQLite schema:

```sh
python -m unittest data.databases.test_analytics_views
cd data/dashboard
npm run build
```

The tests cover migration, reruns, transaction rollback, source updates,
one-row-per-search joins, metric edge cases, and full/incremental loader
compatibility. The schema check requires the new views, so a missing query
layer can no longer silently pass as an unfinished optional feature build.

The browser attaches the analytics file read-only and exposes its relations
through temporary aliases in a writable in-memory database. Mosaic can build
small cross-filter caches there without writing to the source file.

## Migration verification

Native DuckDB 1.4.4, synthetic data (20,002 searches / 80,008 iterations), five
warm-query runs, median milliseconds:

| Operation | Views | Previous materialized tables |
|---|---:|---:|
| Install views / build feature tables (single run) | 26.46 | 223.40 |
| Overview aggregates | 1.06 | 0.20 |
| Engine comparison | 1.20 | 0.26 |
| Iteration NPS by depth | 1.36 | 0.35 |
| Iteration EBF by depth, including windows | 29.14 | 0.65 |

Unchanged search totals, evaluations, NPS, average iteration NPS, and timing
averages matched the previous transform across the synthetic dataset. These
numbers demonstrate the build/query tradeoff, not production latency or file
size savings. The existing database in Documents was inaccessible to the test
process because of macOS file permissions, so it was not migrated or benchmarked.
Browser verification used a small fixture: all 16 tabs and 83 selector options
completed without console errors after fixing the cache workspace.
