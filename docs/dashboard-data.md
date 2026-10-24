# Dashboard data contract

Shared by the pipeline and the dashboard.

- **Python side** (pipeline): `data/databases/*`, `data/transforms/*`.
- **Browser side** (dashboard): `data/dashboard/*`.
- **Ingestion**: `data/etl/ingest.py`, `data/etl/environment.py`, `tests/sprt.py`.

Sections 1–3 are the agreed contract. Section 3 also lists the exact columns of
every pre-aggregated table. Sections 4 and 5 describe the public snapshot and
the checks that enforce all of this. The column lists live in code, in
`data/transforms/aggregates.py` (`AGG_COLUMNS`) and
`data/databases/public_schema.py`, and the tests compare those against the
databases.

## 1. Raw SQLite additions

These columns are in the `CREATE TABLE` statements of `init_raw_db.py`.
Existing databases get them from migrations in `migrate_schema.py`:
`2026_09_28_games_match_environment` and `2026_09_28_sprt_pentanomial`. Run
`python -m data.databases.migrate_schema`, or `python -m data.databases.init_raw_db`,
which also migrates.

`games` (all nullable):

| Column | Meaning |
|---|---|
| `source` TEXT | `'engine'` (engine log) or `'match'` (match-runner record, which has the authoritative result) |
| `run_id` TEXT | Match-runner run directory name, e.g. `20260928_101129_354256` |
| `run_game` INTEGER | Game number within the run (SPRT numbering; pairs are (1,2), (3,4), …) |
| `pair_index` INTEGER | 1-based opening pair within the run |
| `env_host`, `env_cpu_model`, `env_os` TEXT | Machine |
| `env_cpu` INTEGER | Logical CPU the game ran on |
| `env_core_type` TEXT | `'P'`, `'E'` or `'-'` (not pinned) |
| `env_cluster` INTEGER | L2 cluster id of the core |
| `env_concurrency` INTEGER | Games running at the same time on the machine |
| `env_mhz_mean`, `env_mhz_min`, `env_mhz_max` REAL | Measured clock of that CPU during the game |
| `env_util_mean` REAL | CPU utilisation of that CPU during the game (%) |
| `env_power_source` TEXT | `'ac'` or `'battery'` |
| `env_core_mixed` INTEGER | 1 if an engine was found off its core. These games are excluded from SPRT. |

Integer result and termination codes:

- **result**: 1 white, 2 black, 3 draw. 0 (ongoing) and 4 (aborted) become NULL.
- **termination**: 1 checkmate, 2 stalemate, 3 threefold, 4 fiftymove, 5 time,
  6 resign, 7 adjudication (including the move limit), 8 insufficient_material,
  9 illegal_move, 10 crash (disconnect or stall).

`sprt` (all nullable). `elo_diff` stays the **measured** Elo. New databases
declare `elo0`, `elo1` and `elo_diff` as REAL. Older ones declare them INTEGER,
and the loader reads them as DOUBLE anyway.

- Counts: `pairs`, `penta_ll`, `penta_ld`, `penta_dd`, `penta_wl`, `penta_wd`, `penta_ww` INTEGER
- Statistics: `score`, `elo_low`, `elo_high`, `nelo`, `nelo_low`, `nelo_high`, `draw_ratio`, `variance_ratio` REAL
- `concurrency`, `excluded_pairs`, `book_seed` INTEGER
- `core_summary` TEXT: JSON, `{"P": {pairs, elo, elo_low, elo_high, pentanomial}, "E": {...}}`
- `run_id`, `env_host`, `env_cpu_model` TEXT

`experiments.metadata` (existing TEXT) holds JSON with machine information for
match-runner experiments. It is not published.

## 2. Analytics DuckDB

`load_analytics` creates these tables:

- **`game_stats`**
  - Every `games` column above except `opening`.
  - `result` decoded to `'white'`, `'black'`, `'draw'` or NULL.
  - `termination` decoded to `'checkmate'`, `'stalemate'`, `'threefold'`,
    `'fiftymove'`, `'time'`, `'resign'`, `'adjudication'`,
    `'insufficient_material'`, `'illegal_move'` or `'crash'`.
  - `opening`, `opening_eco` and `opening_variation` come from
    `data.etl.openings.get_opening_details(moves)`. `''` means not found. The
    recorded `"Name, Variation"` label is used instead (split at the first
    `", "`) in two cases:
    - the moves can't all be replayed from the start position, as with games
      from an EPD book;
    - the moves name no opening.
- **`sprt_runs`**
  - Every `sprt` column.
  - `elo_diff` is the measured Elo; it is never overwritten.
  - `opening_book` is reduced to the file name. This works for Windows and
    POSIX paths.
- **`sts_runs`**: `suite` is reduced to the file name, and `position_type` is added.
- **`search_stats`**
  - The raw searches, plus `sf_eval`, `sf_best_move`, `sf_time_ms`,
    `sf_computed_at`, `sf_pv`, `sf_eval_pov`, `sf_depth`, `sf_multipv`,
    `sf_move_rank` and `sf_move_eval`, filled by `transform_positions` from
    `stockfish_moves` (below).
  - `sf_eval` is from the side to move, like the engine's `eval`.
  - `sf_best_move`, `sf_eval` and `sf_pv` come from Stockfish's first line.
  - `sf_multipv` is the number of lines analysed (the N of "Stockfish's top
    N") and `sf_depth` their depth.
  - `sf_move_rank` is the engine's move among the lines: 1..`sf_multipv`, 0
    when it is not among them, NULL when unknown (no move, or not a legal one).
    `sf_move_eval` is Stockfish's evaluation of that move, side to move. A move
    outside the lines is searched alone (UCI `searchmoves`) at the depth of the
    position's lines, so it has an evaluation too. Separate searches don't
    always agree: such a move can come out above Stockfish's lower lines, and
    now and then above the first (its loss is then 0). NULL when the engine
    searched a different move than the one asked for.
  - `sf_eval_pov = 'stm'` marks the convention. Values stored from White's view
    by older versions are converted once by `transform_positions`.
- **`stockfish_moves`**: Stockfish's analysis per position, created by
  `transform_positions` (not by `load_analytics`). One row per (position, move):
  - `fen`, `move` (UCI), `rank` (Stockfish's line, 1 = best; NULL for an engine
    move outside the lines, scored alone), `eval` (side to move, engine scale),
    `pv` (JSON list of UCI moves starting with `move`), `depth`, `multipv`
    (lines requested; NULL on single-move rows), `time_ms`, `stockfish` (the
    engine's UCI name) and `computed_at`.
  - Positions are analysed once; a position is analysed again only when a run
    asks for more depth or more lines, and then at the larger of the stored
    and the requested value of each, so an analysis never gets shallower or
    narrower. The table is never dropped by `load_analytics --full`, which
    empties the `sf_*` columns: they are filled again from here, with or
    without Stockfish.
  - Analyses start from a cleared hash, so a result depends only on the
    position, the depth, the number of lines and the Stockfish version.
- **Other tables**: `iterative_deepening_stats`, `search_tree_stats`,
  `search_timings`, `root_moves`, `dim_positions` and `position_features`.

Incremental loads add any new raw column to the analytics table and insert by
column name. `engines`, `experiments`, `engine_ratings`, `sprt_runs` and
`sts_runs` are small, so they are reloaded in full every time. `game_stats` is
rebuilt once when an older loader built it: the loader version is recorded in
`analytics_meta`. Otherwise only new ids are appended, so other rows changed in
SQLite need `--full`.

Views, installed by `transform_search`:

| View | Additions in this version |
|---|---|
| `search_metrics` | `eval_cp` is the eval, or NULL for a mate score. `mate` is +1 (side to move mates), −1 (gets mated), 0 otherwise, or NULL. `mate_plies` is ±(100000 − \|eval\|). `sf_eval_cp`. `eval_diff = eval − sf_eval` when both are centipawns. `sf_depth`, `sf_multipv`, `sf_move_rank` and `sf_move_eval` as in `search_stats`. `sf_cp_loss` = max(0, clamp(`sf_eval`) − clamp(`sf_move_eval`)), each clamped to ±1000 cp so a mate counts as 1000, NULL when unknown. `engine_move_rank` is `sf_move_rank`, or for older analyses without the lines 1 when the move is `sf_best_move` and NULL otherwise. `env_core_type`, `env_core_mixed`, `env_mhz_mean`, `env_host`, `env_cpu_model` come from the game (`game_id`). Also `tt_fill`, `total_pvs_researches`, `see_prune_ratio` and `delta_prune_ratio`. |
| `search_context` | `search_metrics` plus engine, game (`game_score`, `game_opening`, `game_eco`, `game_opening_variation`) and STS columns. The STS columns now join on `sts_runs.experiment_id = searches.sts_id` plus the position's EPD fields; `searches.sts_id` holds the STS *experiment* id. One row per search. |
| `search_iteration_metrics` | `eval_cp`, `mate` |
| `search_iteration_features` | Eval deltas and deviations use `eval_cp`. Sign flips still use the raw eval. |
| `search_iteration_summary` | `max_eval`, `avg_eval` and `stddev_eval` ignore mate scores. |

Engine mate scores are ±(100000 − ply), and \|eval\| ≥ 99000 is treated as mate.
Averages of evaluations must use `eval_cp`, or `abs(eval) < 99000` for tables
such as `root_moves`.

## 3. Pre-aggregated tables

`data/transforms/aggregates.py: build_aggregates(cnxn)` builds these. It runs
after the views in the analytics pipeline and again for every public snapshot.
The dashboard reads these tables first. It falls back to live aggregation on
the views only when a table is missing, which happens with local files.

Conventions:

- **Nodes**: `nodes` means all nodes searched, main search plus quiescence
  (`nodes + qnodes`).
- **NPS**: `nps` is Σnodes / Σtime, over searches that have both. It is not the
  mean of per-search NPS; that is `nps_mean` in `agg_engine_summary`.
- **Speed per clock**: `knps_per_ghz` is Σnodes / Σ(time_s × MHz / 1000) / 1000,
  over searches whose game has `env_mhz_mean`.
- **Distributions**: per-search distributions have `_mean`, `_sd`, `_p10`,
  `_p25`, `_p50`, `_p75` and `_p90` columns (`quantile_cont`).
- **Rates**: `_rate` columns are ratios of sums. All of them divide by nodes,
  except these:
  - `fh_first_rate` = fail-highs on the first move / fail-highs
  - `nmp_success_rate` = null-move cutoffs / null-move searches
  - `see_prune_rate` and `delta_prune_rate` divide by qnodes
- **`core_type`**
  - `'P'`, `'E'` or `'-'`: the game's `env_core_type`.
  - `'mixed'`: the game has `env_core_mixed = 1`.
  - `'unknown'`: no match-runner data (engine-log games, STS runs).
  - `'all'`: every search of the engine.
- **`phase`**: `'opening'`, `'midgame'`, `'endgame'` from `position_features`,
  or `'unknown'`, or `'all'`.
- **Evaluations** are from the side to move. Mate scores never enter a
  centipawn statistic.
- **Types**: ids and counts are INTEGER. Sums, ratios and large counts
  (`agg_moveorder.count`, `agg_timing.calls`) are DOUBLE, so JavaScript gets
  plain numbers.
- **Missing engines**: searches without an engine are ignored.

### `agg_engine_summary`: per engine × core_type

| Column | Meaning |
|---|---|
| `engine_id`, `core_type` | Key |
| `searches`, `games` | Searches, and the distinct games they come from |
| `nodes`, `time_ms`, `nps` | Totals and aggregate NPS |
| `knps_per_ghz`, `mhz_mean` | Clock-normalised speed. `mhz_mean` is time-weighted. |
| `depth_mean`, `depth_sd`, `depth_p10`, `depth_p25`, `depth_p50`, `depth_p75`, `depth_p90` | Completed depth: mean, standard deviation and quantiles |
| `seldepth_mean`, `seldepth_p50` | Selective depth (max quiescence depth) |
| `time_ms_mean`, `time_ms_sd`, `time_ms_p10`, `time_ms_p25`, `time_ms_p50`, `time_ms_p75`, `time_ms_p90` | Search time per search |
| `nodes_mean`, `nodes_sd`, `nodes_p10`, `nodes_p25`, `nodes_p50`, `nodes_p75`, `nodes_p90` | Nodes per search |
| `nps_mean`, `nps_sd`, `nps_p10`, `nps_p25`, `nps_p50`, `nps_p75`, `nps_p90` | Per-search NPS. Use `nps` for the engine's speed. |
| `ebf` | Geometric mean of the per-search effective branching factor. That factor is the geometric mean of iteration-to-iteration node growth. |
| `qratio`, `tt_hit_rate`, `tt_store_rate`, `fail_high_rate`, `fail_low_rate`, `fh_first_rate`, `nmp_rate`, `nmp_success_rate`, `see_prune_rate`, `delta_prune_rate` | Ratios of sums |
| `tt_fill_mean`, `fail_high_researches_mean`, `fail_low_researches_mean` | Per-search means. The researches are aspiration re-searches. |
| `eval_abs_mean` | Mean \|eval_cp\|, without mates |
| `mate_searches` | Searches that returned a mate score |
| `sf_searches`, `sf_best_move_match`, `sf_mae_cp` | Stockfish coverage, best-move agreement rate, and mean \|eval_diff\|. They are NULL or 0 without Stockfish. |

### `agg_depth`: per engine × core_type × iterative-deepening depth

Each row only covers searches that reached that depth. `searches` gives that
coverage.

| Column | Meaning |
|---|---|
| `engine_id`, `core_type`, `depth` | Key |
| `searches` | Searches with an iteration at this depth |
| `nodes`, `qnodes`, `time_ms`, `nps` | Iteration totals and aggregate NPS |
| `nodes_mean`, `time_ms_mean` | Per-iteration means |
| `cum_time_ms_mean` | Mean time from the start of the search to the end of this iteration |
| `ebf`, `time_growth` | Geometric mean of nodes(d)/nodes(d−1) and time(d)/time(d−1) |
| `qratio`, `tt_hit_rate`, `fail_high_rate`, `fail_low_rate`, `fh_first_rate`, `see_prune_rate`, `delta_prune_rate`, `nmp_rate`, `nmp_success_rate`, `pvs_research_rate` | Ratios of sums |
| `fail_high_researches_mean`, `fail_low_researches_mean` | Aspiration re-searches per iteration |
| `eval_change_mean` | Mean \|eval_cp(d) − eval_cp(d−1)\|, without mates |
| `move_change_rate` | Share of iterations whose best move differs from the previous iteration |
| `eval_sign_flip_rate` | Share of iterations whose eval sign flips |

### `agg_tree_ply`: per engine × tree ply (`search_tree_stats.depth`)

| Column | Meaning |
|---|---|
| `engine_id`, `ply`, `searches` | Key, and the searches with nodes at this ply |
| `nodes`, `qnodes` | Totals |
| `node_share` | Share of the engine's nodes at this ply (sums to 1) |
| `node_growth` | nodes(ply) / nodes(previous ply) |
| `qratio`, `tt_hit_rate`, `tt_store_rate`, `fail_high_rate`, `fail_low_rate`, `fh_first_rate`, `see_prune_rate`, `delta_prune_rate`, `nmp_rate`, `nmp_success_rate`, `pvs_research_rate` | Ratios of sums |

### `agg_timing`: per engine × timed function

| Column | Meaning |
|---|---|
| `engine_id`, `function` | Key (`ROOT`, `MAKEMOVE`, `MOVEGEN`, `NNUE`, `EVAL`, `SEE`, `TT_PROBE`, …) |
| `calls`, `total_ms` | Totals |
| `avg_us_per_call` | total_ms × 1000 / calls |
| `share` | total_ms / the engine's total search time (its `ROOT` timer). Timers nest, so shares don't add up to 1. |
| `searches`, `ms_per_search`, `calls_per_search` | Coverage and per-search averages |

### `agg_moveorder`: fail-high move-index histogram

| Column | Meaning |
|---|---|
| `engine_id`, `dimension`, `dim_value` | `dimension` is one of: `'all'` (with `dim_value` `'all'`), `'core_type'`, `'phase'`, `'depth'` (iteration) or `'tree_ply'`. `dim_value` is text. |
| `fh_index_bucket` | `'0'`, `'1'`, `'2'`, `'3'`, `'4-7'`, `'8+'` |
| `count` | Fail-highs in the bucket |
| `share` | count / the fail-highs of that engine × dimension × dim_value |
| `dim_order`, `bucket_order` | Numeric sort keys. `dim_order` is the depth or ply, or phase 1–3, and is NULL otherwise. `bucket_order` is 0–5. |

### `agg_eval_wdl`: game outcome by evaluation

| Column | Meaning |
|---|---|
| `engine_id`, `core_type`, `phase` | Key. `core_type` and `phase` include `'all'` rows. |
| `eval_bucket` | round(eval_cp / 25) × 25, clamped to [−1000, 1000]. It is NULL on mate rows. |
| `positions` | Searches (positions) in the bucket whose game has a result |
| `games` | Distinct games among them. Don't add these up across buckets. |
| `wins`, `draws`, `losses` | Positions whose game the side to move won, drew or lost (`game_stats.result`). wins + draws + losses = positions. |
| `mate` | +1 the side to move has a mate score, −1 it gets mated, 0 for centipawn rows |

### `agg_openings`: results by opening, from each engine's perspective

| Column | Meaning |
|---|---|
| `engine_id` | The engine. A game counts once for each side the engine played, so a self-play game counts twice. |
| `opening` | Family name. `'(unknown)'` when no opening was found. |
| `opening_variation` | NULL on `level = 'family'` rows. On `'variation'` rows it is the variation name, or `''` for the family's main line. |
| `opening_eco` | ECO code, or a range such as `'B20-B99'` when a family spans several codes |
| `games` | Games, including those without a result |
| `wins`, `draws`, `losses` | From the engine's side. They are ≤ games. |
| `level` | `'family'` (group by `opening`) or `'variation'` (group by `opening`, `opening_variation`). The dashboard's toggle picks one. |

### `agg_env`: speed by measured clock

Only searches whose game has `env_mhz_mean` are included.

| Column | Meaning |
|---|---|
| `engine_id`, `core_type`, `mhz_bucket` | Key. `mhz_bucket` is the lower bound of a 100 MHz bucket. |
| `searches`, `nodes`, `time_ms`, `nps`, `knps_per_ghz` | As above |
| `mhz_mean` | Time-weighted mean clock in the bucket |

### `agg_hist`: per-engine histograms with bins shared across engines

| Column | Meaning |
|---|---|
| `engine_id`, `core_type`, `metric` | Key |
| `bin_lo`, `bin_hi` | Bin edges in the metric's units. Values outside the range are counted in the first or last bin. |
| `count`, `share` | Searches in the bin, and count / the searches of that engine × core_type × metric |
| `scale` | `'log'` means the bins are equal in log10 (draw with a log x axis). `'linear'` otherwise. |

Metrics and their bins:

- `depth`, `seldepth`: unit bins [k, k+1).
- `time_ms`, `nodes`, `nps`: 40 log bins between the 0.5th and 99.5th percentile of all engines.
- `ebf`, `tt_hit_rate`, `fail_high_rate`: 40 linear bins over the same percentile range.
- `eval_cp`: 50 cp bins over [−1000, 1000].
- `eval_diff`: 25 cp bins over [−500, 500]. Only present with Stockfish data.
- `qratio`, `fh_first_rate`, `nmp_success_rate`: 0.025 bins over [0, 1].

### `agg_corr`: metric correlations per engine

| Column | Meaning |
|---|---|
| `engine_id`, `metric_x`, `metric_y` | Both orders of every pair (x ≠ y). The diagonal is omitted. |
| `n` | Searches with both metrics. Pairs with n < 30 are dropped. |
| `pearson`, `spearman` | Pearson and Spearman correlation over those searches. NULL for a constant metric. |

Metrics: `depth`, `seldepth`, `nodes`, `time_ms`, `nps`, `ebf`, `tt_hit_rate`,
`qratio`, `fail_high_rate`, `fail_low_rate`, `fh_first_rate`, `nmp_rate`,
`fail_high_researches`, `fail_low_researches`, `abs_eval_cp` (no mates) and
`ply`. An engine with more than 200,000 searches uses a deterministic
subsample of that size.

### `agg_positions`: per engine × core_type × phase × position type

This one is not in the original contract. It replaces the per-search
`position_features` in public snapshots.

| Column | Meaning |
|---|---|
| `engine_id`, `core_type`, `phase`, `position_type` | Key. Every dimension also has `'all'` rows. `position_type` is `Tactical`, `Positional`, `Endgame` or `Unknown`. |
| `searches`, `depth_mean`, `time_ms_mean`, `nodes_mean`, `nps` | Volume and speed |
| `tt_hit_rate`, `fail_high_rate`, `fh_first_rate`, `ebf`, `eval_abs_mean` | As in `agg_engine_summary` |
| `sf_searches`, `sf_best_move_match`, `sf_mae_cp` | Stockfish agreement |

### Root moves: `agg_root_index`, `agg_root_gap`, `agg_root_best`

These are the Root moves tab's first charts. They use the root moves of each
search's deepest complete iteration: the deepest logged `depth` of `root_moves`
with as many moves as the search ever logged at one depth, so every root move
was searched. The engine logs every iteration, and the last one is usually cut
off by the clock; it would show where the time ran out rather than how the
engine ordered its moves. They have `core_type` `'all'` rows but no phase.

| Table and column | Meaning |
|---|---|
| `agg_root_index`: `engine_id`, `core_type`, `move_index` | Key. `move_index` is the move's place in the engine's root order, 0 = searched first; 10 means index 10 or later (the eleventh move on). |
| `agg_root_index`: `moves`, `nodes`, `time_ms` | Root moves at that place, and the nodes and milliseconds the engine spent on them |
| `agg_root_gap`: `engine_id`, `core_type`, `gap`, `searches` | Best minus second-best root score, `floor(/10) × 10` cp, 500 meaning 500 or more. Mate scores and searches with fewer than two scored moves are left out. Only the first move's score is exact (the others come from null-window searches), so the gap is a lower bound. |
| `agg_root_best`: `engine_id`, `core_type`, `move`, `searches` | The engine's best moves (UCI) and how often it returned them: the 50 most frequent per engine and core type |

### Against Stockfish: `agg_sf_moves`, `agg_sf_root_effort`, `agg_sf_root_order`

These compare each search with Stockfish's top lines of its position
(`stockfish_moves`). They are empty until `transform_positions` ran with
Stockfish. Every table has `core_type` and `phase` `'all'` rows. When
positions were analysed with different numbers of lines, `lines` is the
smallest, and ranks above it count as outside the lines, so every position is
judged alike.

`agg_sf_moves`: the engine's move against Stockfish's lines, per engine ×
core_type × phase × rank × loss bucket.

| Column | Meaning |
|---|---|
| `engine_id`, `core_type`, `phase` | Key, with `'all'` rows |
| `sf_rank` | Stockfish's line of the engine's move, 1..`lines`; 0 = not among the lines |
| `loss_bucket` | Lower edge (cp) of the move's loss (`sf_cp_loss`): 0 (exactly 0), 1 (1–9), 10 (10–24), 25 (25–49), 50 (50–99), 100 (100–199), 200 (200–399), 400 (400 or more); NULL when the loss is unknown |
| `searches` | Searches in the cell |
| `loss_sum`, `loss_sq_sum` | Σ loss and Σ loss² (NULL with `loss_bucket` NULL): the mean loss and its standard error |
| `lines` | The number of Stockfish lines the ranks refer to |
| `depth_min`, `depth_max` | Stockfish's depth over the cell's searches |

`agg_sf_root_effort`: how the engine spent its root effort (deepest complete
iteration) on Stockfish's lines, per engine × core_type × phase × rank.

| Column | Meaning |
|---|---|
| `engine_id`, `core_type`, `phase` | Key, with `'all'` rows |
| `sf_rank` | 1..`lines`: the root move that is Stockfish's line of that rank; 0: all the other root moves |
| `searches` | Searches of the group with root-move logs and Stockfish lines: the denominator, the same on every `sf_rank` row of the group |
| `moves`, `nodes` | Root moves counted and the nodes spent on them |
| `node_share_sum` | Σ over searches of (nodes on these moves / all root nodes of the search). `node_share_sum / searches` is the mean share per search; over the ranks of a group it adds up to `searches`. |
| `lines` | As in `agg_sf_moves` |

`agg_sf_root_order`: where Stockfish's first-line move was in the engine's
root order (deepest complete iteration), per engine × core_type × phase.

| Column | Meaning |
|---|---|
| `engine_id`, `core_type`, `phase` | Key, with `'all'` rows |
| `move_index` | 0 = searched first, …, 10 = index 10 or later (the eleventh move on), −1 = not searched in that iteration |
| `searches` | Searches with root-move logs and Stockfish lines |

### `search_sample`: up to 20,000 searches for scatter plots

The sample is stratified by engine. Each engine gets an equal share, and
engines with fewer searches keep all of theirs, with the rest shared out. It
is deterministic: the order is a fixed hash of the search id.

| Column | Meaning |
|---|---|
| `search_id`, `engine_id`, `game_id`, `core_type`, `mhz` | Identity and environment |
| `ply`, `stm` (`'w'` or `'b'`), `phase`, `position_type`, `pos_tactical`, `pos_positional` | Position |
| `depth`, `seldepth`, `nodes`, `qnodes`, `time_ms`, `nps`, `ebf` | Search effort |
| `eval`, `eval_cp`, `mate` | Evaluation, side to move |
| `tt_hit_rate`, `qratio`, `fail_high_rate`, `fail_low_rate`, `fh_first_rate`, `nmp_rate` | Per-search ratios |
| `sf_eval`, `eval_diff`, `best_move_match` | Stockfish comparison (NULL without Stockfish) |
| `game_score` | Result for the side to move: 1 win, 0 draw, −1 loss, or NULL |
| `sample_weight` | The engine's searches / its sampled searches. Use it as a weight for estimates from the sample. |

## 4. Public snapshot

`python -m data.databases.export_public [--full] [--keep N] [--archive FILE]`.

A snapshot always contains:

- **The small tables, allow-listed columns only.**
  - `engines`: no `description` or `compile_flags`, which can contain local paths.
  - `experiments`: no `metadata`.
  - `game_stats`: no `moves` or `start_fen`.
  - `engine_ratings`, `sprt_runs`, `sts_runs`: all columns.
- Every `agg_*` table and `search_sample`, rebuilt from the snapshot's own data.
- `snapshot_info(key, value)`, with the keys `schema_version`, `profile`
  (`'lite'` or `'full'`), `exported_at`, `searches` and `sample_rows`.

By default (the `lite` profile) the snapshot has no per-search tables and no
views. `--full` adds:

- The per-search facts: `search_stats`, `iterative_deepening_stats`,
  `search_tree_stats`, `search_timings`, `root_moves` and `position_features`.
- `stockfish_moves`, restricted to the positions of those searches.
- The views `search_metrics`, `search_context`, `search_position_metrics`,
  `search_iteration_metrics`, `search_iteration_features` and
  `search_tree_features`.

These are never published: `dim_positions`, the compatibility views
`search_iteration_summary`, `search_timing_summary` and `search_features`, and
`schema_migrations`.

`manifest.json`:

- **Unchanged from `schema_version: 1`**: `schema_version`, `exported_at`,
  `snapshot`, `sha256`, `size_bytes` and `row_counts`.
- **New fields**: `profile` and `searches`.
- **`row_counts`** covers every table in the snapshot. `row_counts.search_stats`
  is the number of searches summarised, even in a lite snapshot that has no
  `search_stats` table.

Only the newest `--keep` snapshots (default 3) stay in `snapshots/`, and the
archive bundles exactly those.

Storage format:

- DuckDB storage version v1.2.0, readable by DuckDB ≥ 1.2 and by
  duckdb-wasm ≥ 1.29 (the dashboard uses 1.32.0, which is DuckDB 1.4.3).
- Lite snapshots use 16 KB blocks, which makes them about 10× smaller than with
  the default 256 KB blocks.

## 5. Checks

- **Data quality**: `python -m data.transforms.validate [--profile analytics|public-lite|public-full]`.
  It checks grain, ranges and aggregate consistency (W/D/L sums, opening
  totals against `game_stats`, shares, correlation bounds, sample size). For
  public profiles it also checks that no file-name column holds a local path.
- **Schema**: `python -m data.databases.test_schema [--db FILE]`.
  - It checks the relations and columns in this document.
  - Public snapshots must match the allow-list exactly: no extra relations or
    columns, and no forbidden columns.
- **Tests**: `python -m unittest discover -s data -p "test_*.py"`. These build
  a synthetic raw database from generated logs, then run the pipeline without
  Stockfish (and once more with a fake UCI engine standing in for it), the
  exports and the migrations.
