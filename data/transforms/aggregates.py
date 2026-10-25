"""Pre-aggregated tables for the dashboard (see docs/dashboard-data.md).

build_aggregates(cnxn) creates every agg_* table and search_sample from the
analytics tables and views. The analytics pipeline runs it after installing
the views, and data/databases/export_public.py runs it again for every public
snapshot, so the dashboard never has to aggregate raw search rows.

Conventions:
  * `nodes` always means all nodes searched (main search + quiescence).
  * NPS is sum(nodes) / sum(time); per-search distributions have _mean, _sd
    and _p10.._p90 columns.
  * Evaluations are from the side to move. Mate scores (|eval| >= 99000) are
    left out of every centipawn statistic and counted separately.
  * core_type is the game's env_core_type ('P', 'E', '-'), 'mixed' when an
    engine ran off its core, 'unknown' without match-runner data; 'all' rows
    cover every search of the engine.
  * Counts are INTEGER, sums and ratios DOUBLE (friendly to JavaScript).

Usage (from the repository root):
  python -m data.transforms.aggregates [--db PATH] [--sample-size N]
"""
import argparse
import math
import os
from pathlib import Path

import duckdb

from ..etl.paths import ANALYTICS_DB
from .transform_positions import STOCKFISH_MOVES_DDL

SAMPLE_SIZE = 20_000          # search_sample rows (stratified by engine)
MIN_CORRELATION_N = 30        # agg_corr drops metric pairs with fewer observations
CORRELATION_MAX_ROWS = 200_000  # per engine; larger engines use a deterministic subsample
MATE_THRESHOLD = 99_000       # |eval| at or above this is a mate score
EVAL_BUCKET_CP = 25
EVAL_CLAMP_CP = 1000
MHZ_BUCKET = 100
HIST_BINS = 40

# Deterministic pseudo-random order (Knuth multiplicative hash of the search id).
STABLE_ORDER = "(CAST(search_id AS HUGEINT) * 2654435761) % 4294967296, search_id"

# Output schema of every table, in column order (contract columns first).
AGG_COLUMNS = {
    "agg_engine_summary": (
        "engine_id", "core_type", "searches", "games", "nodes", "time_ms", "nps", "knps_per_ghz", "mhz_mean",
        "depth_mean", "depth_sd", "depth_p10", "depth_p25", "depth_p50", "depth_p75", "depth_p90",
        "seldepth_mean", "seldepth_p50",
        "time_ms_mean", "time_ms_sd", "time_ms_p10", "time_ms_p25", "time_ms_p50", "time_ms_p75", "time_ms_p90",
        "nodes_mean", "nodes_sd", "nodes_p10", "nodes_p25", "nodes_p50", "nodes_p75", "nodes_p90",
        "nps_mean", "nps_sd", "nps_p10", "nps_p25", "nps_p50", "nps_p75", "nps_p90",
        "ebf", "qratio", "tt_hit_rate", "tt_store_rate", "tt_fill_mean", "fail_high_rate", "fail_low_rate",
        "fh_first_rate", "nmp_rate", "nmp_success_rate", "see_prune_rate", "delta_prune_rate",
        "fail_high_researches_mean", "fail_low_researches_mean",
        "eval_abs_mean", "mate_searches", "sf_searches", "sf_best_move_match", "sf_mae_cp",
    ),
    "agg_depth": (
        "engine_id", "core_type", "depth", "searches", "nodes", "qnodes", "time_ms", "nps",
        "nodes_mean", "time_ms_mean", "cum_time_ms_mean", "ebf", "time_growth",
        "qratio", "tt_hit_rate", "fail_high_rate", "fail_low_rate", "fh_first_rate",
        "see_prune_rate", "delta_prune_rate", "nmp_rate", "nmp_success_rate", "pvs_research_rate",
        "fail_high_researches_mean", "fail_low_researches_mean",
        "eval_change_mean", "move_change_rate", "eval_sign_flip_rate",
    ),
    "agg_tree_ply": (
        "engine_id", "ply", "searches", "nodes", "qnodes", "node_share", "node_growth",
        "qratio", "tt_hit_rate", "tt_store_rate", "fail_high_rate", "fail_low_rate", "fh_first_rate",
        "see_prune_rate", "delta_prune_rate", "nmp_rate", "nmp_success_rate", "pvs_research_rate",
    ),
    "agg_timing": (
        "engine_id", "function", "calls", "total_ms", "avg_us_per_call", "share",
        "searches", "ms_per_search", "calls_per_search",
    ),
    "agg_moveorder": (
        "engine_id", "dimension", "dim_value", "fh_index_bucket", "count", "share", "dim_order", "bucket_order",
    ),
    "agg_eval_wdl": (
        "engine_id", "core_type", "phase", "eval_bucket", "positions", "games", "wins", "draws", "losses", "mate",
    ),
    "agg_openings": (
        "engine_id", "opening", "opening_variation", "opening_eco", "games", "wins", "draws", "losses", "level",
    ),
    "agg_env": (
        "engine_id", "core_type", "mhz_bucket", "searches", "nodes", "time_ms", "nps", "knps_per_ghz", "mhz_mean",
    ),
    "agg_hist": (
        "engine_id", "core_type", "metric", "bin_lo", "bin_hi", "count", "share", "scale",
    ),
    "agg_corr": (
        "engine_id", "metric_x", "metric_y", "n", "pearson", "spearman",
    ),
    "agg_positions": (
        "engine_id", "core_type", "phase", "position_type", "searches", "depth_mean", "time_ms_mean",
        "nodes_mean", "nps", "tt_hit_rate", "fail_high_rate", "fh_first_rate", "ebf", "eval_abs_mean",
        "sf_searches", "sf_best_move_match", "sf_mae_cp",
    ),
    "agg_root_index": ("engine_id", "core_type", "move_index", "moves", "nodes", "time_ms"),
    "agg_root_gap": ("engine_id", "core_type", "gap", "searches"),
    "agg_root_best": ("engine_id", "core_type", "move", "searches"),
    "agg_sf_moves": (
        "engine_id", "core_type", "phase", "sf_rank", "loss_bucket", "searches", "loss_sum", "loss_sq_sum",
        "lines", "depth_min", "depth_max",
    ),
    "agg_sf_root_effort": (
        "engine_id", "core_type", "phase", "sf_rank", "searches", "moves", "nodes", "node_share_sum", "lines",
    ),
    "agg_sf_root_order": ("engine_id", "core_type", "phase", "move_index", "searches"),
    "search_sample": (
        "search_id", "engine_id", "game_id", "core_type", "mhz", "ply", "stm", "phase", "position_type",
        "pos_tactical", "pos_positional", "depth", "seldepth", "nodes", "qnodes", "time_ms", "nps", "ebf",
        "eval", "eval_cp", "mate", "tt_hit_rate", "qratio", "fail_high_rate", "fail_low_rate", "fh_first_rate",
        "nmp_rate", "sf_eval", "eval_diff", "best_move_match", "game_score", "sample_weight",
    ),
}
AGG_TABLES = tuple(AGG_COLUMNS)
# Empty until the positions have Stockfish's lines (transform_positions with Stockfish).
STOCKFISH_TABLES = ("agg_sf_moves", "agg_sf_root_effort", "agg_sf_root_order")
# Lower edges (cp) of agg_sf_moves.loss_bucket: 0 means exactly 0, 1 means 1-9, and so on.
LOSS_BUCKETS = (0, 1, 10, 25, 50, 100, 200, 400)
ROOT_INDEX_CAP = 10       # root move positions from here on are counted together
ROOT_GAP_CAP = 500        # cp
ROOT_BEST_MOVES = 50      # agg_root_best keeps the most frequent moves per engine and core type

# Relations build_aggregates reads.
REQUIRED_RELATIONS = (
    "search_metrics", "search_iteration_metrics", "search_tree_features", "game_stats", "position_features",
    "iterative_deepening_stats", "search_tree_stats", "search_timings", "root_moves",
)

# Per-search metrics of agg_corr: (name, expression over the base table).
CORRELATION_METRICS = (
    ("depth", "depth"),
    ("seldepth", "seldepth"),
    ("nodes", "nodes"),
    ("time_ms", "time_ms"),
    ("nps", "nps"),
    ("ebf", "ebf"),
    ("tt_hit_rate", "tt_hit_rate"),
    ("qratio", "qratio"),
    ("fail_high_rate", "fail_high_rate"),
    ("fail_low_rate", "fail_low_rate"),
    ("fh_first_rate", "fh_first_rate"),
    ("nmp_rate", "nmp_rate"),
    ("fail_high_researches", "fail_high_researches"),
    ("fail_low_researches", "fail_low_researches"),
    ("abs_eval_cp", "abs(eval_cp)"),
    ("ply", "ply"),
)

# agg_hist metrics: (name, expression, binning). Binning is 'int' (unit bins),
# 'log' / 'linear' (HIST_BINS bins between the 0.5th and 99.5th percentile of
# all engines) or a fixed (low, high, width). Values outside the range are
# counted in the first/last bin.
HIST_METRICS = (
    ("depth", "depth", "int"),
    ("seldepth", "seldepth", "int"),
    ("time_ms", "time_ms", "log"),
    ("nodes", "nodes", "log"),
    ("nps", "nps", "log"),
    ("ebf", "ebf", "linear"),
    ("eval_cp", "eval_cp", (-EVAL_CLAMP_CP, EVAL_CLAMP_CP, 50)),
    ("eval_diff", "eval_diff", (-500, 500, 25)),
    ("tt_hit_rate", "tt_hit_rate", "linear"),
    ("qratio", "qratio", (0.0, 1.0, 0.025)),
    ("fail_high_rate", "fail_high_rate", "linear"),
    ("fh_first_rate", "fh_first_rate", (0.0, 1.0, 0.025)),
    ("nmp_success_rate", "nmp_success_rate", (0.0, 1.0, 0.025)),
)

# One row per search with the columns every aggregate needs (a temporary table).
BASE_SQL = """
CREATE OR REPLACE TEMP TABLE _agg_base AS
WITH iteration_growth AS (
    -- geometric mean growth of the nodes per iteration (effective branching factor)
    SELECT search_id, exp(avg(ln(growth))) AS ebf
    FROM (
        SELECT search_id,
               (nodes + qnodes) / NULLIF(LAG(nodes + qnodes) OVER (PARTITION BY search_id ORDER BY depth), 0) AS growth
        FROM iterative_deepening_stats
    )
    WHERE growth > 0
    GROUP BY search_id
)
SELECT
    CAST(m.search_id AS INTEGER) AS search_id,
    CAST(m.engine_id AS INTEGER) AS engine_id,
    CAST(m.game_id AS INTEGER) AS game_id,
    CAST(m.ply AS INTEGER) AS ply,
    CASE WHEN g.env_core_mixed = 1 THEN 'mixed'
         ELSE coalesce(nullif(g.env_core_type, ''), 'unknown') END AS core_type,
    CAST(g.env_mhz_mean AS DOUBLE) AS mhz,
    split_part(m.fen, ' ', 2) AS stm,
    g.result AS game_result,
    coalesce(pf.game_phase, 'unknown') AS phase,
    coalesce(pf.position_type, 'Unknown') AS position_type,
    CAST(pf.pos_tactical AS DOUBLE) AS pos_tactical,
    CAST(pf.pos_positional AS DOUBLE) AS pos_positional,
    CAST(m.completed_depth AS INTEGER) AS depth,
    CAST(m.max_qdepth AS INTEGER) AS seldepth,
    CAST(m.total_nodes AS DOUBLE) AS nodes,
    CAST(m.total_qnodes AS DOUBLE) AS qnodes,
    CAST(m.total_time_ms AS DOUBLE) AS time_ms,
    CAST(m.nps AS DOUBLE) AS nps,
    CAST(m.eval AS INTEGER) AS eval,
    CAST(m.eval_cp AS INTEGER) AS eval_cp,
    CAST(m.mate AS INTEGER) AS mate,
    CAST(m.tt_hits AS DOUBLE) AS tt_hits,
    CAST(m.tt_stores AS DOUBLE) AS tt_stores,
    CAST(m.tt_fill AS DOUBLE) AS tt_fill,
    CAST(m.fail_highs AS DOUBLE) AS fail_highs,
    CAST(m.fail_lows AS DOUBLE) AS fail_lows,
    CAST(m.total_fh_index_0 AS DOUBLE) AS fh_0,
    CAST(m.total_fh_index_1 AS DOUBLE) AS fh_1,
    CAST(m.total_fh_index_2 AS DOUBLE) AS fh_2,
    CAST(m.total_fh_index_3 AS DOUBLE) AS fh_3,
    CAST(m.total_fh_index_4to7 AS DOUBLE) AS fh_4to7,
    CAST(m.total_fh_index_8plus AS DOUBLE) AS fh_8plus,
    CAST(m.total_see_prunes AS DOUBLE) AS see_prunes,
    CAST(m.total_delta_prunes AS DOUBLE) AS delta_prunes,
    CAST(m.nmp AS DOUBLE) AS nmp,
    CAST(m.nmp_failhigh AS DOUBLE) AS nmp_failhigh,
    CAST(m.total_fail_high_researches AS DOUBLE) AS fail_high_researches,
    CAST(m.total_fail_low_researches AS DOUBLE) AS fail_low_researches,
    CAST(m.qratio AS DOUBLE) AS qratio,
    CAST(m.tt_hit_ratio AS DOUBLE) AS tt_hit_rate,
    CAST(m.fail_high_ratio AS DOUBLE) AS fail_high_rate,
    CAST(m.fail_low_ratio AS DOUBLE) AS fail_low_rate,
    CAST(m.fh_index_0_ratio AS DOUBLE) AS fh_first_rate,
    CAST(m.nmp_ratio AS DOUBLE) AS nmp_rate,
    CAST(m.nmp_failhigh_ratio AS DOUBLE) AS nmp_success_rate,
    ig.ebf,
    CAST(m.sf_eval AS INTEGER) AS sf_eval,
    CAST(m.eval_diff AS DOUBLE) AS eval_diff,
    CAST(m.best_move_match AS INTEGER) AS best_move_match,
    m.fen,
    m.move,
    CAST(m.sf_depth AS INTEGER) AS sf_depth,
    CAST(m.sf_multipv AS INTEGER) AS sf_multipv,
    CAST(m.sf_move_rank AS INTEGER) AS sf_move_rank,
    CAST(m.sf_cp_loss AS DOUBLE) AS sf_cp_loss
FROM search_metrics m
LEFT JOIN game_stats g ON g.id = m.game_id
LEFT JOIN position_features pf ON pf.search_id = m.search_id
LEFT JOIN iteration_growth ig ON ig.search_id = m.search_id
WHERE m.engine_id IS NOT NULL
"""


def _core_type(column="core_type"):
    return f"CASE WHEN GROUPING({column}) = 1 THEN 'all' ELSE {column} END"


def _quantiles(expr, name, levels=(10, 25, 50, 75, 90)):
    return ",\n    ".join(f"quantile_cont({expr}, {level / 100}) AS {name}_p{level}" for level in levels)


def _distribution(expr, name):
    return (f"avg({expr}) AS {name}_mean,\n    stddev_samp({expr}) AS {name}_sd,\n    "
            + _quantiles(expr, name))


# Totals over searches that have both nodes and time, and the clock-normalised speed.
_NPS_SQL = """
    sum(nodes) FILTER (WHERE time_ms IS NOT NULL) / NULLIF(sum(time_ms) FILTER (WHERE nodes IS NOT NULL) / 1000.0, 0)
"""
_KNPS_PER_GHZ_SQL = """
    sum(nodes) FILTER (WHERE time_ms IS NOT NULL AND mhz > 0)
        / NULLIF(sum(time_ms / 1000.0 * mhz / 1000.0) FILTER (WHERE nodes IS NOT NULL AND mhz > 0), 0) / 1000.0
"""
_MHZ_MEAN_SQL = """
    coalesce(sum(time_ms * mhz) FILTER (WHERE mhz > 0) / NULLIF(sum(time_ms) FILTER (WHERE mhz > 0), 0),
             avg(mhz) FILTER (WHERE mhz > 0))
"""

ENGINE_SUMMARY_SQL = f"""
CREATE OR REPLACE TABLE agg_engine_summary AS
SELECT
    engine_id,
    {_core_type()} AS core_type,
    CAST(count(*) AS INTEGER) AS searches,
    CAST(count(DISTINCT game_id) AS INTEGER) AS games,
    sum(nodes) FILTER (WHERE time_ms IS NOT NULL) AS nodes,
    sum(time_ms) FILTER (WHERE nodes IS NOT NULL) AS time_ms,
    {_NPS_SQL} AS nps,
    {_KNPS_PER_GHZ_SQL} AS knps_per_ghz,
    {_MHZ_MEAN_SQL} AS mhz_mean,
    {_distribution('depth', 'depth')},
    avg(seldepth) AS seldepth_mean,
    quantile_cont(seldepth, 0.5) AS seldepth_p50,
    {_distribution('time_ms', 'time_ms')},
    {_distribution('nodes', 'nodes')},
    {_distribution('nps', 'nps')},
    exp(avg(ln(ebf)) FILTER (WHERE ebf > 0)) AS ebf,
    sum(qnodes) / NULLIF(sum(nodes), 0) AS qratio,
    sum(tt_hits) / NULLIF(sum(nodes), 0) AS tt_hit_rate,
    sum(tt_stores) / NULLIF(sum(nodes), 0) AS tt_store_rate,
    avg(tt_fill) AS tt_fill_mean,
    sum(fail_highs) / NULLIF(sum(nodes), 0) AS fail_high_rate,
    sum(fail_lows) / NULLIF(sum(nodes), 0) AS fail_low_rate,
    sum(fh_0) / NULLIF(sum(fail_highs), 0) AS fh_first_rate,
    sum(nmp) / NULLIF(sum(nodes), 0) AS nmp_rate,
    sum(nmp_failhigh) / NULLIF(sum(nmp), 0) AS nmp_success_rate,
    sum(see_prunes) / NULLIF(sum(qnodes), 0) AS see_prune_rate,
    sum(delta_prunes) / NULLIF(sum(qnodes), 0) AS delta_prune_rate,
    avg(fail_high_researches) AS fail_high_researches_mean,
    avg(fail_low_researches) AS fail_low_researches_mean,
    avg(abs(eval_cp)) AS eval_abs_mean,
    CAST(count(*) FILTER (WHERE mate <> 0) AS INTEGER) AS mate_searches,
    CAST(count(sf_eval) AS INTEGER) AS sf_searches,
    avg(best_move_match) AS sf_best_move_match,
    avg(abs(eval_diff)) AS sf_mae_cp
FROM _agg_base
GROUP BY GROUPING SETS ((engine_id), (engine_id, core_type))
ORDER BY engine_id, core_type
"""

DEPTH_SQL = f"""
CREATE OR REPLACE TABLE agg_depth AS
WITH it AS (
    SELECT i.search_id, i.depth,
           CAST(i.nodes + i.qnodes AS DOUBLE) AS nodes,
           CAST(i.qnodes AS DOUBLE) AS qnodes,
           CAST(i.time_ms AS DOUBLE) AS time_ms,
           i.tt_hits, i.fail_highs, i.fail_lows, i.fail_high_researches, i.fail_low_researches,
           i.fh_index_0, i.see_prunes, i.delta_prunes, i.pvs_researches, i.nmp, i.nmp_failhigh,
           i.eval, i.move,
           CASE WHEN abs(i.eval) < {MATE_THRESHOLD} THEN i.eval END AS eval_cp,
           LAG(i.nodes + i.qnodes) OVER w AS prev_nodes,
           LAG(i.time_ms) OVER w AS prev_time_ms,
           LAG(CASE WHEN abs(i.eval) < {MATE_THRESHOLD} THEN i.eval END) OVER w AS prev_eval_cp,
           LAG(i.eval) OVER w AS prev_eval,
           LAG(i.move) OVER w AS prev_move,
           SUM(i.time_ms) OVER (w ROWS UNBOUNDED PRECEDING) AS cum_time_ms
    FROM search_iteration_metrics i   -- iterative_deepening_stats with pvs_researches derived
    WINDOW w AS (PARTITION BY i.search_id ORDER BY i.depth)
)
SELECT
    b.engine_id,
    {_core_type('b.core_type')} AS core_type,
    CAST(it.depth AS INTEGER) AS depth,
    CAST(count(*) AS INTEGER) AS searches,
    sum(it.nodes) AS nodes,
    sum(it.qnodes) AS qnodes,
    sum(it.time_ms) AS time_ms,
    sum(it.nodes) FILTER (WHERE it.time_ms IS NOT NULL)
        / NULLIF(sum(it.time_ms) FILTER (WHERE it.nodes IS NOT NULL) / 1000.0, 0) AS nps,
    avg(it.nodes) AS nodes_mean,
    avg(it.time_ms) AS time_ms_mean,
    avg(it.cum_time_ms) AS cum_time_ms_mean,
    exp(avg(ln(it.nodes / it.prev_nodes)) FILTER (WHERE it.nodes > 0 AND it.prev_nodes > 0)) AS ebf,
    exp(avg(ln(it.time_ms / it.prev_time_ms)) FILTER (WHERE it.time_ms > 0 AND it.prev_time_ms > 0)) AS time_growth,
    sum(it.qnodes) / NULLIF(sum(it.nodes), 0) AS qratio,
    sum(it.tt_hits) / NULLIF(sum(it.nodes), 0) AS tt_hit_rate,
    sum(it.fail_highs) / NULLIF(sum(it.nodes), 0) AS fail_high_rate,
    sum(it.fail_lows) / NULLIF(sum(it.nodes), 0) AS fail_low_rate,
    sum(it.fh_index_0) / NULLIF(sum(it.fail_highs), 0) AS fh_first_rate,
    sum(it.see_prunes) / NULLIF(sum(it.qnodes), 0) AS see_prune_rate,
    sum(it.delta_prunes) / NULLIF(sum(it.qnodes), 0) AS delta_prune_rate,
    sum(it.nmp) / NULLIF(sum(it.nodes), 0) AS nmp_rate,
    sum(it.nmp_failhigh) / NULLIF(sum(it.nmp), 0) AS nmp_success_rate,
    sum(it.pvs_researches) / NULLIF(sum(it.nodes), 0) AS pvs_research_rate,
    avg(it.fail_high_researches) AS fail_high_researches_mean,
    avg(it.fail_low_researches) AS fail_low_researches_mean,
    avg(abs(it.eval_cp - it.prev_eval_cp)) AS eval_change_mean,
    avg(CASE WHEN it.prev_move IS NULL OR it.move IS NULL THEN NULL
             WHEN it.move <> it.prev_move THEN 1.0 ELSE 0.0 END) AS move_change_rate,
    avg(CASE WHEN it.prev_eval IS NULL OR it.eval IS NULL THEN NULL
             WHEN (it.eval > 0 AND it.prev_eval < 0) OR (it.eval < 0 AND it.prev_eval > 0) THEN 1.0
             ELSE 0.0 END) AS eval_sign_flip_rate
FROM it
JOIN _agg_base b ON b.search_id = it.search_id
GROUP BY GROUPING SETS ((b.engine_id, it.depth), (b.engine_id, b.core_type, it.depth))
ORDER BY 1, 2, 3
"""

TREE_PLY_SQL = """
CREATE OR REPLACE TABLE agg_tree_ply AS
WITH t AS (
    SELECT b.engine_id, t.depth AS ply,
           count(*) AS searches,
           CAST(sum(t.nodes + t.qnodes) AS DOUBLE) AS nodes,
           CAST(sum(t.qnodes) AS DOUBLE) AS qnodes,
           CAST(sum(t.tt_hits) AS DOUBLE) AS tt_hits,
           CAST(sum(t.tt_stores) AS DOUBLE) AS tt_stores,
           CAST(sum(t.fail_highs) AS DOUBLE) AS fail_highs,
           CAST(sum(t.fail_lows) AS DOUBLE) AS fail_lows,
           CAST(sum(t.fh_index_0) AS DOUBLE) AS fh_0,
           CAST(sum(t.see_prunes) AS DOUBLE) AS see_prunes,
           CAST(sum(t.delta_prunes) AS DOUBLE) AS delta_prunes,
           CAST(sum(t.nmp) AS DOUBLE) AS nmp,
           CAST(sum(t.nmp_failhigh) AS DOUBLE) AS nmp_failhigh,
           CAST(sum(t.pvs_researches) AS DOUBLE) AS pvs_researches
    FROM search_tree_features t   -- search_tree_stats with pvs_researches derived
    JOIN _agg_base b ON b.search_id = t.search_id
    GROUP BY b.engine_id, t.depth
)
SELECT
    engine_id,
    CAST(ply AS INTEGER) AS ply,
    CAST(searches AS INTEGER) AS searches,
    nodes,
    qnodes,
    nodes / NULLIF(sum(nodes) OVER (PARTITION BY engine_id), 0) AS node_share,
    nodes / NULLIF(LAG(nodes) OVER (PARTITION BY engine_id ORDER BY ply), 0) AS node_growth,
    qnodes / NULLIF(nodes, 0) AS qratio,
    tt_hits / NULLIF(nodes, 0) AS tt_hit_rate,
    tt_stores / NULLIF(nodes, 0) AS tt_store_rate,
    fail_highs / NULLIF(nodes, 0) AS fail_high_rate,
    fail_lows / NULLIF(nodes, 0) AS fail_low_rate,
    fh_0 / NULLIF(fail_highs, 0) AS fh_first_rate,
    see_prunes / NULLIF(qnodes, 0) AS see_prune_rate,
    delta_prunes / NULLIF(qnodes, 0) AS delta_prune_rate,
    nmp / NULLIF(nodes, 0) AS nmp_rate,
    nmp_failhigh / NULLIF(nmp, 0) AS nmp_success_rate,
    pvs_researches / NULLIF(nodes, 0) AS pvs_research_rate
FROM t
ORDER BY engine_id, ply
"""

TIMING_SQL = """
CREATE OR REPLACE TABLE agg_timing AS
WITH t AS (
    SELECT b.engine_id, st.search_id, st.function,
           CAST(st.total_time_ms AS DOUBLE) AS total_ms, CAST(st.num_calls AS DOUBLE) AS calls
    FROM search_timings st
    JOIN _agg_base b ON b.search_id = st.search_id
),
engine_time AS (
    -- the engine's total search time: its ROOT timer, else the time of the searches with timings
    SELECT s.engine_id, coalesce(r.root_ms, s.search_ms) AS search_ms
    FROM (
        SELECT b.engine_id, sum(b.time_ms) AS search_ms
        FROM _agg_base b
        WHERE b.search_id IN (SELECT search_id FROM t)
        GROUP BY b.engine_id
    ) s
    LEFT JOIN (
        SELECT engine_id, sum(total_ms) AS root_ms FROM t WHERE function = 'ROOT' GROUP BY engine_id
    ) r ON r.engine_id = s.engine_id
)
SELECT
    t.engine_id,
    t.function,
    sum(t.calls) AS calls,
    sum(t.total_ms) AS total_ms,
    sum(t.total_ms) * 1000.0 / NULLIF(sum(t.calls), 0) AS avg_us_per_call,
    sum(t.total_ms) / NULLIF(any_value(e.search_ms), 0) AS share,
    CAST(count(DISTINCT t.search_id) AS INTEGER) AS searches,
    sum(t.total_ms) / count(DISTINCT t.search_id) AS ms_per_search,
    sum(t.calls) / count(DISTINCT t.search_id) AS calls_per_search
FROM t
LEFT JOIN engine_time e ON e.engine_id = t.engine_id
GROUP BY t.engine_id, t.function
ORDER BY t.engine_id, total_ms DESC
"""

_FH_SUMS_BASE = "sum(fh_0), sum(fh_1), sum(fh_2), sum(fh_3), sum(fh_4to7), sum(fh_8plus)"
_FH_SUMS_FACT = """CAST(sum({t}.fh_index_0) AS DOUBLE), CAST(sum({t}.fh_index_1) AS DOUBLE),
           CAST(sum({t}.fh_index_2) AS DOUBLE), CAST(sum({t}.fh_index_3) AS DOUBLE),
           CAST(sum({t}.fh_index_4to7) AS DOUBLE), CAST(sum({t}.fh_index_8plus) AS DOUBLE)"""

MOVEORDER_SQL = f"""
CREATE OR REPLACE TABLE agg_moveorder AS
WITH totals (engine_id, dimension, dim_value, dim_order, f0, f1, f2, f3, f4to7, f8plus) AS (
    SELECT engine_id, 'all', 'all', CAST(NULL AS INTEGER), {_FH_SUMS_BASE}
    FROM _agg_base GROUP BY engine_id
    UNION ALL
    SELECT engine_id, 'core_type', core_type, CAST(NULL AS INTEGER), {_FH_SUMS_BASE}
    FROM _agg_base GROUP BY engine_id, core_type
    UNION ALL
    SELECT engine_id, 'phase', phase,
           CASE phase WHEN 'opening' THEN 1 WHEN 'midgame' THEN 2 WHEN 'endgame' THEN 3 END, {_FH_SUMS_BASE}
    FROM _agg_base GROUP BY engine_id, phase
    UNION ALL
    SELECT b.engine_id, 'depth', CAST(i.depth AS VARCHAR), CAST(i.depth AS INTEGER), {_FH_SUMS_FACT.format(t='i')}
    FROM iterative_deepening_stats i JOIN _agg_base b ON b.search_id = i.search_id
    GROUP BY b.engine_id, i.depth
    UNION ALL
    SELECT b.engine_id, 'tree_ply', CAST(t.depth AS VARCHAR), CAST(t.depth AS INTEGER), {_FH_SUMS_FACT.format(t='t')}
    FROM search_tree_stats t JOIN _agg_base b ON b.search_id = t.search_id
    GROUP BY b.engine_id, t.depth
),
long AS (
    SELECT engine_id, dimension, dim_value, dim_order,
           unnest(['0', '1', '2', '3', '4-7', '8+']) AS fh_index_bucket,
           unnest([0, 1, 2, 3, 4, 5]) AS bucket_order,
           unnest([coalesce(f0, 0), coalesce(f1, 0), coalesce(f2, 0), coalesce(f3, 0),
                   coalesce(f4to7, 0), coalesce(f8plus, 0)]) AS n
    FROM totals
)
SELECT
    engine_id,
    dimension,
    dim_value,
    fh_index_bucket,
    CAST(n AS DOUBLE) AS "count",
    n / NULLIF(sum(n) OVER (PARTITION BY engine_id, dimension, dim_value), 0) AS share,
    dim_order,
    CAST(bucket_order AS INTEGER) AS bucket_order
FROM long
QUALIFY sum(n) OVER (PARTITION BY engine_id, dimension, dim_value) > 0
ORDER BY engine_id, dimension, dim_order NULLS FIRST, dim_value, bucket_order
"""

EVAL_WDL_SQL = f"""
CREATE OR REPLACE TABLE agg_eval_wdl AS
WITH s AS (
    SELECT engine_id, core_type, phase, game_id, mate,
           CASE WHEN mate = 0 THEN CAST(greatest(-{EVAL_CLAMP_CP}, least({EVAL_CLAMP_CP},
                round(eval_cp / {EVAL_BUCKET_CP}.0) * {EVAL_BUCKET_CP})) AS INTEGER) END AS eval_bucket,
           CASE WHEN game_result = 'draw' THEN 'draw'
                WHEN game_result = CASE stm WHEN 'w' THEN 'white' ELSE 'black' END THEN 'win'
                ELSE 'loss' END AS outcome
    FROM _agg_base
    WHERE eval IS NOT NULL AND game_result IN ('white', 'black', 'draw') AND stm IN ('w', 'b')
)
SELECT
    engine_id,
    {_core_type()} AS core_type,
    CASE WHEN GROUPING(phase) = 1 THEN 'all' ELSE phase END AS phase,
    eval_bucket,
    CAST(count(*) AS INTEGER) AS positions,
    CAST(count(DISTINCT game_id) AS INTEGER) AS games,
    CAST(count(*) FILTER (WHERE outcome = 'win') AS INTEGER) AS wins,
    CAST(count(*) FILTER (WHERE outcome = 'draw') AS INTEGER) AS draws,
    CAST(count(*) FILTER (WHERE outcome = 'loss') AS INTEGER) AS losses,
    mate
FROM s
GROUP BY engine_id, mate, eval_bucket, CUBE (core_type, phase)
ORDER BY engine_id, core_type, phase, mate, eval_bucket
"""

OPENINGS_SQL = """
CREATE OR REPLACE TABLE agg_openings AS
WITH sides AS (
    SELECT white_engine_id AS engine_id, 'white' AS colour, result, opening, opening_variation, opening_eco
    FROM game_stats WHERE white_engine_id IS NOT NULL
    UNION ALL
    SELECT black_engine_id AS engine_id, 'black' AS colour, result, opening, opening_variation, opening_eco
    FROM game_stats WHERE black_engine_id IS NOT NULL
),
g AS (
    SELECT CAST(engine_id AS INTEGER) AS engine_id,
           coalesce(nullif(opening, ''), '(unknown)') AS opening,
           coalesce(opening_variation, '') AS opening_variation,
           nullif(opening_eco, '') AS opening_eco,
           CASE WHEN result = colour THEN 'win'
                WHEN result = 'draw' THEN 'draw'
                WHEN result IN ('white', 'black') THEN 'loss' END AS outcome
    FROM sides
)
SELECT
    engine_id,
    opening,
    CASE WHEN GROUPING(opening_variation) = 1 THEN NULL ELSE opening_variation END AS opening_variation,
    CASE WHEN min(opening_eco) = max(opening_eco) THEN min(opening_eco)
         ELSE min(opening_eco) || '-' || max(opening_eco) END AS opening_eco,
    CAST(count(*) AS INTEGER) AS games,
    CAST(count(*) FILTER (WHERE outcome = 'win') AS INTEGER) AS wins,
    CAST(count(*) FILTER (WHERE outcome = 'draw') AS INTEGER) AS draws,
    CAST(count(*) FILTER (WHERE outcome = 'loss') AS INTEGER) AS losses,
    CASE WHEN GROUPING(opening_variation) = 1 THEN 'family' ELSE 'variation' END AS level
FROM g
GROUP BY GROUPING SETS ((engine_id, opening), (engine_id, opening, opening_variation))
ORDER BY engine_id, level, games DESC, opening, opening_variation
"""

ENV_SQL = f"""
CREATE OR REPLACE TABLE agg_env AS
WITH b AS (
    SELECT engine_id, core_type, mhz, nodes, time_ms,
           CAST(floor(mhz / {MHZ_BUCKET}) * {MHZ_BUCKET} AS INTEGER) AS mhz_bucket
    FROM _agg_base
    WHERE mhz > 0 AND nodes IS NOT NULL AND time_ms IS NOT NULL
)
SELECT
    engine_id,
    {_core_type()} AS core_type,
    mhz_bucket,
    CAST(count(*) AS INTEGER) AS searches,
    sum(nodes) AS nodes,
    sum(time_ms) AS time_ms,
    sum(nodes) / NULLIF(sum(time_ms) / 1000.0, 0) AS nps,
    sum(nodes) / NULLIF(sum(time_ms / 1000.0 * mhz / 1000.0), 0) / 1000.0 AS knps_per_ghz,
    coalesce(sum(time_ms * mhz) / NULLIF(sum(time_ms), 0), avg(mhz)) AS mhz_mean
FROM b
GROUP BY GROUPING SETS ((engine_id, mhz_bucket), (engine_id, core_type, mhz_bucket))
ORDER BY engine_id, core_type, mhz_bucket
"""

POSITIONS_SQL = f"""
CREATE OR REPLACE TABLE agg_positions AS
SELECT
    engine_id,
    {_core_type()} AS core_type,
    CASE WHEN GROUPING(phase) = 1 THEN 'all' ELSE phase END AS phase,
    CASE WHEN GROUPING(position_type) = 1 THEN 'all' ELSE position_type END AS position_type,
    CAST(count(*) AS INTEGER) AS searches,
    avg(depth) AS depth_mean,
    avg(time_ms) AS time_ms_mean,
    avg(nodes) AS nodes_mean,
    {_NPS_SQL} AS nps,
    sum(tt_hits) / NULLIF(sum(nodes), 0) AS tt_hit_rate,
    sum(fail_highs) / NULLIF(sum(nodes), 0) AS fail_high_rate,
    sum(fh_0) / NULLIF(sum(fail_highs), 0) AS fh_first_rate,
    exp(avg(ln(ebf)) FILTER (WHERE ebf > 0)) AS ebf,
    avg(abs(eval_cp)) AS eval_abs_mean,
    CAST(count(sf_eval) AS INTEGER) AS sf_searches,
    avg(best_move_match) AS sf_best_move_match,
    avg(abs(eval_diff)) AS sf_mae_cp
FROM _agg_base
GROUP BY engine_id, CUBE (core_type, phase, position_type)
ORDER BY engine_id, core_type, phase, position_type
"""

# Tables built by one SQL statement, in build order.
TABLE_SQL = {
    "agg_engine_summary": ENGINE_SUMMARY_SQL,
    "agg_depth": DEPTH_SQL,
    "agg_tree_ply": TREE_PLY_SQL,
    "agg_timing": TIMING_SQL,
    "agg_moveorder": MOVEORDER_SQL,
    "agg_eval_wdl": EVAL_WDL_SQL,
    "agg_openings": OPENINGS_SQL,
    "agg_env": ENV_SQL,
    "agg_positions": POSITIONS_SQL,
}


def _phase(column="phase"):
    return f"CASE WHEN GROUPING({column}) = 1 THEN 'all' ELSE {column} END"


# Root moves (root_moves) of each search's deepest complete iteration: the deepest logged
# depth with as many root moves as the search ever logged at one depth, so every move was
# searched. The engine logs every iteration, and the last one is usually cut off by the
# clock: it shows where the time ran out, not how the engine ordered and spent its effort.
# Only the first move's eval is exact (the others come from null-window searches), so only
# best-minus-second uses evals.
ROOT_SQL = {
    "agg_root_index": f"""
        CREATE OR REPLACE TABLE agg_root_index AS
        WITH x AS (
            SELECT b.engine_id, b.core_type, least(r.move_index, {ROOT_INDEX_CAP}) AS move_index, r.nodes, r.time_ms
            FROM _agg_root r JOIN _agg_base b ON b.search_id = r.search_id
        )
        SELECT engine_id, {_core_type()} AS core_type, CAST(move_index AS INTEGER) AS move_index,
               CAST(count(*) AS INTEGER) AS moves, sum(nodes) AS nodes, sum(time_ms) AS time_ms
        FROM x
        GROUP BY GROUPING SETS ((engine_id, move_index), (engine_id, core_type, move_index))
        ORDER BY engine_id, core_type, move_index
    """,
    "agg_root_gap": f"""
        CREATE OR REPLACE TABLE agg_root_gap AS
        WITH g AS (
            SELECT search_id, max(eval) AS best, list(eval ORDER BY eval DESC)[2] AS second
            FROM _agg_root WHERE abs(eval) < {MATE_THRESHOLD} GROUP BY search_id
        ), x AS (
            SELECT b.engine_id, b.core_type, least({ROOT_GAP_CAP}, floor((g.best - g.second) / 10.0) * 10) AS gap
            FROM g JOIN _agg_base b ON b.search_id = g.search_id
            WHERE g.second IS NOT NULL
        )
        SELECT engine_id, {_core_type()} AS core_type, CAST(gap AS INTEGER) AS gap, CAST(count(*) AS INTEGER) AS searches
        FROM x
        GROUP BY GROUPING SETS ((engine_id, gap), (engine_id, core_type, gap))
        ORDER BY engine_id, core_type, gap
    """,
    "agg_root_best": f"""
        CREATE OR REPLACE TABLE agg_root_best AS
        WITH counts AS (
            SELECT engine_id, {_core_type()} AS core_type, CAST(move AS VARCHAR) AS move,
                   CAST(count(*) AS INTEGER) AS searches
            FROM _agg_base WHERE move IS NOT NULL AND move <> ''
            GROUP BY GROUPING SETS ((engine_id, move), (engine_id, core_type, move))
        )
        SELECT engine_id, core_type, move, searches FROM counts
        QUALIFY row_number() OVER (PARTITION BY engine_id, core_type ORDER BY searches DESC, move) <= {ROOT_BEST_MOVES}
        ORDER BY engine_id, core_type, searches DESC, move
    """,
}


def _build_root(cnxn):
    complete = ""
    if "depth" in _table_columns(cnxn, "root_moves"):
        complete = """
            JOIN (
                SELECT search_id, max(depth) AS depth FROM (
                    SELECT search_id, depth, count(*) AS moves,
                           max(count(*)) OVER (PARTITION BY search_id) AS most
                    FROM root_moves GROUP BY search_id, depth
                ) WHERE moves = most GROUP BY search_id
            ) d ON d.search_id = r.search_id AND d.depth = r.depth
        """
    cnxn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _agg_root AS
        SELECT CAST(r.search_id AS INTEGER) AS search_id, CAST(r.move_index AS INTEGER) AS move_index,
               CAST(r.move AS VARCHAR) AS move, CAST(r.eval AS INTEGER) AS eval,
               CAST(r.nodes AS DOUBLE) AS nodes, CAST(r.time_ms AS DOUBLE) AS time_ms
        FROM root_moves r {complete}
    """)
    for sql in ROOT_SQL.values():
        cnxn.execute(sql)


def _stockfish_sql(lines):
    """agg_sf_* statements. Ranks above `lines` (positions analysed with more lines than
    the smallest analysis) count as outside the lines, so every position is judged alike."""
    bucket = " ".join(f"WHEN sf_cp_loss >= {edge} THEN {edge}" for edge in reversed(LOSS_BUCKETS[1:]))
    # Stockfish's lines of each position, as ranks 1..lines.
    ranked = f"""
        SELECT fen, move, CAST(rank AS INTEGER) AS rank FROM stockfish_moves WHERE rank BETWEEN 1 AND {lines}
        QUALIFY row_number() OVER (PARTITION BY fen, move ORDER BY computed_at DESC) = 1
    """
    return {
        "agg_sf_moves": f"""
            CREATE OR REPLACE TABLE agg_sf_moves AS
            WITH b AS (
                SELECT engine_id, core_type, phase, sf_depth, sf_cp_loss AS loss,
                       CASE WHEN sf_move_rank > {lines} THEN 0 ELSE sf_move_rank END AS sf_rank,
                       CASE WHEN sf_cp_loss IS NULL THEN NULL {bucket} ELSE 0 END AS loss_bucket
                FROM _agg_base WHERE sf_move_rank IS NOT NULL
            )
            SELECT engine_id, {_core_type()} AS core_type, {_phase()} AS phase,
                   CAST(sf_rank AS INTEGER) AS sf_rank, CAST(loss_bucket AS INTEGER) AS loss_bucket,
                   CAST(count(*) AS INTEGER) AS searches, sum(loss) AS loss_sum, sum(loss * loss) AS loss_sq_sum,
                   CAST({lines} AS INTEGER) AS lines,
                   CAST(min(sf_depth) AS INTEGER) AS depth_min, CAST(max(sf_depth) AS INTEGER) AS depth_max
            FROM b
            GROUP BY engine_id, sf_rank, loss_bucket, CUBE (core_type, phase)
            ORDER BY engine_id, core_type, phase, sf_rank, loss_bucket
        """,
        "agg_sf_root_effort": f"""
            CREATE OR REPLACE TABLE agg_sf_root_effort AS
            WITH lines AS ({ranked}),
            per_move AS (
                SELECT b.search_id, b.engine_id, b.core_type, b.phase, coalesce(l.rank, 0) AS sf_rank, r.nodes
                FROM _agg_root r
                JOIN _agg_base b ON b.search_id = r.search_id
                LEFT JOIN lines l ON l.fen = b.fen AND l.move = r.move
                WHERE b.sf_depth IS NOT NULL
            ), totals AS (
                SELECT search_id, sum(nodes) AS total FROM per_move GROUP BY search_id HAVING sum(nodes) > 0
            ), per_search AS (
                SELECT p.search_id, p.engine_id, p.core_type, p.phase, p.sf_rank,
                       count(*) AS moves, sum(p.nodes) AS nodes, sum(p.nodes) / any_value(t.total) AS share
                FROM per_move p JOIN totals t ON t.search_id = p.search_id
                GROUP BY p.search_id, p.engine_id, p.core_type, p.phase, p.sf_rank
            ), grouped AS (
                SELECT engine_id, {_core_type()} AS core_type, {_phase()} AS phase, sf_rank,
                       CAST(sum(moves) AS INTEGER) AS moves, sum(nodes) AS nodes, sum(share) AS node_share_sum
                FROM per_search
                GROUP BY engine_id, sf_rank, CUBE (core_type, phase)
            ), searches AS (
                SELECT engine_id, {_core_type()} AS core_type, {_phase()} AS phase,
                       CAST(count(*) AS INTEGER) AS searches
                FROM (SELECT DISTINCT search_id, engine_id, core_type, phase FROM per_search)
                GROUP BY engine_id, CUBE (core_type, phase)
            )
            SELECT g.engine_id, g.core_type, g.phase, CAST(g.sf_rank AS INTEGER) AS sf_rank, s.searches,
                   g.moves, g.nodes, g.node_share_sum, CAST({lines} AS INTEGER) AS lines
            FROM grouped g JOIN searches s USING (engine_id, core_type, phase)
            ORDER BY g.engine_id, g.core_type, g.phase, sf_rank
        """,
        "agg_sf_root_order": f"""
            CREATE OR REPLACE TABLE agg_sf_root_order AS
            WITH best AS (
                SELECT fen, move FROM stockfish_moves WHERE rank = 1
                QUALIFY row_number() OVER (PARTITION BY fen ORDER BY computed_at DESC) = 1
            ), per_search AS (
                SELECT b.search_id, b.engine_id, b.core_type, b.phase,
                       -- least() skips NULLs: map "not searched" to -1 before capping
                       least(coalesce(min(r.move_index), -1), {ROOT_INDEX_CAP}) AS move_index
                FROM _agg_base b
                JOIN (SELECT DISTINCT search_id FROM _agg_root) logged ON logged.search_id = b.search_id
                JOIN best s ON s.fen = b.fen
                LEFT JOIN _agg_root r ON r.search_id = b.search_id AND r.move = s.move
                GROUP BY b.search_id, b.engine_id, b.core_type, b.phase
            )
            SELECT engine_id, {_core_type()} AS core_type, {_phase()} AS phase,
                   CAST(move_index AS INTEGER) AS move_index, CAST(count(*) AS INTEGER) AS searches
            FROM per_search
            GROUP BY engine_id, move_index, CUBE (core_type, phase)
            ORDER BY engine_id, core_type, phase, move_index
        """,
    }


def _build_stockfish(cnxn):
    lines = cnxn.execute("SELECT min(sf_multipv) FROM _agg_base WHERE sf_depth IS NOT NULL").fetchone()[0]
    for sql in _stockfish_sql(int(lines or 0)).values():
        cnxn.execute(sql)

def _relation_exists(cnxn, name):
    row = cnxn.execute("""
        SELECT 1 FROM information_schema.tables
        WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = ?
    """, [name]).fetchone()
    return row is not None


def _table_columns(cnxn, name):
    return tuple(row[0] for row in cnxn.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
    """, [name]).fetchall())


def _hist_bins(cnxn, expr, binning):
    """(low, width, bins, scale) for a metric, or None without data. `low`/`width` are log10 for 'log'."""
    if isinstance(binning, tuple):
        low, high, width = binning
        return float(low), float(width), max(1, round((high - low) / width)), "linear"
    positive = " AND x > 0" if binning == "log" else ""
    row = cnxn.execute(f"""
        SELECT min(x), max(x), quantile_cont(x, 0.005), quantile_cont(x, 0.995)
        FROM (SELECT CAST({expr} AS DOUBLE) AS x FROM _agg_base) WHERE x IS NOT NULL {positive}
    """).fetchone()
    lowest, highest, p_low, p_high = row
    if lowest is None:
        return None
    if binning == "int":
        low = math.floor(lowest)
        width = max(1, math.ceil((math.floor(highest) + 1 - low) / 120))
        return float(low), float(width), max(1, math.ceil((math.floor(highest) + 1 - low) / width)), "linear"
    if binning == "log":
        p_low, p_high = math.log10(p_low), math.log10(p_high)
    if p_high <= p_low:
        return p_low, 1.0, 1, "log" if binning == "log" else "linear"
    return p_low, (p_high - p_low) / HIST_BINS, HIST_BINS, "log" if binning == "log" else "linear"


def _build_hist(cnxn):
    cnxn.execute("""
        CREATE OR REPLACE TABLE agg_hist (
            engine_id INTEGER, core_type VARCHAR, metric VARCHAR, bin_lo DOUBLE, bin_hi DOUBLE,
            "count" INTEGER, share DOUBLE, scale VARCHAR
        )
    """)
    for metric, expr, binning in HIST_METRICS:
        spec = _hist_bins(cnxn, expr, binning)
        if spec is None:
            continue
        low, width, bins, scale = spec
        if scale == "log":
            value = f"log10(greatest(CAST({expr} AS DOUBLE), {10 ** low!r}))"
            edge = "power(10, {})"
        else:
            value = f"CAST({expr} AS DOUBLE)"
            edge = "{}"
        index = f"CAST(least({bins - 1}, greatest(0, floor(({value} - {low!r}) / {width!r}))) AS INTEGER)"
        cnxn.execute(f"""
            INSERT INTO agg_hist
            SELECT engine_id, core_type, '{metric}', {edge.format(f'{low!r} + idx * {width!r}')},
                   {edge.format(f'{low!r} + (idx + 1) * {width!r}')}, n,
                   n / sum(n) OVER (PARTITION BY engine_id, core_type), '{scale}'
            FROM (
                SELECT engine_id, {_core_type()} AS core_type, idx, CAST(count(*) AS INTEGER) AS n
                FROM (SELECT engine_id, core_type, {index} AS idx FROM _agg_base WHERE {expr} IS NOT NULL)
                GROUP BY GROUPING SETS ((engine_id, idx), (engine_id, core_type, idx))
            )
            ORDER BY engine_id, core_type, idx
        """)


def _finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def _build_corr(cnxn):
    """Pearson and Spearman correlations per engine over pairwise-complete searches."""
    cnxn.execute("""
        CREATE OR REPLACE TABLE agg_corr (
            engine_id INTEGER, metric_x VARCHAR, metric_y VARCHAR, n INTEGER, pearson DOUBLE, spearman DOUBLE
        )
    """)
    names = [name for name, _ in CORRELATION_METRICS]
    columns = ", ".join(f"CAST({expr} AS DOUBLE) AS {name}" for name, expr in CORRELATION_METRICS)
    rows = []
    engines = [r[0] for r in cnxn.execute("SELECT DISTINCT engine_id FROM _agg_base ORDER BY 1").fetchall()]
    for engine_id in engines:
        frame = cnxn.execute(f"""
            SELECT {columns} FROM _agg_base WHERE engine_id = ?
            ORDER BY {STABLE_ORDER} LIMIT {CORRELATION_MAX_ROWS}
        """, [engine_id]).df()
        valid = frame.notna().astype("int64")
        counts = valid.T.dot(valid)
        pearson = frame.corr(method="pearson", min_periods=MIN_CORRELATION_N)
        spearman = frame.corr(method="spearman", min_periods=MIN_CORRELATION_N)
        for x in names:
            for y in names:
                n = int(counts.loc[x, y])
                if x == y or n < MIN_CORRELATION_N:
                    continue
                rows.append((engine_id, x, y, n, _finite(pearson.loc[x, y]), _finite(spearman.loc[x, y])))
    if rows:
        # plain parameters (a DataFrame with string columns fails on some duckdb/pandas pairs), 500 rows a statement
        for start in range(0, len(rows), 500):
            chunk = rows[start:start + 500]
            placeholders = ", ".join(["(?, ?, ?, ?, ?, ?)"] * len(chunk))
            cnxn.execute(f"INSERT INTO agg_corr VALUES {placeholders}", [v for row in chunk for v in row])


def allocate_sample(counts, size):
    """Equal share of `size` per engine; engines with fewer searches keep them all and the rest is shared."""
    quotas, remaining = {}, size
    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    for i, (engine_id, count) in enumerate(ordered):
        quotas[engine_id] = min(count, remaining // (len(ordered) - i))
        remaining -= quotas[engine_id]
    return quotas


def _build_sample(cnxn, size):
    counts = dict(cnxn.execute("SELECT engine_id, count(*) FROM _agg_base GROUP BY engine_id").fetchall())
    quotas = allocate_sample(counts, size)
    cnxn.execute("CREATE OR REPLACE TEMP TABLE _agg_quota (engine_id INTEGER, quota BIGINT, searches BIGINT)")
    if quotas:
        cnxn.executemany("INSERT INTO _agg_quota VALUES (?, ?, ?)",
                         [(engine_id, quota, counts[engine_id]) for engine_id, quota in quotas.items()])
    cnxn.execute(f"""
        CREATE OR REPLACE TABLE search_sample AS
        SELECT
            b.search_id, b.engine_id, b.game_id, b.core_type, b.mhz, b.ply, b.stm, b.phase, b.position_type,
            b.pos_tactical, b.pos_positional, b.depth, b.seldepth, b.nodes, b.qnodes, b.time_ms, b.nps, b.ebf,
            b.eval, b.eval_cp, b.mate, b.tt_hit_rate, b.qratio, b.fail_high_rate, b.fail_low_rate,
            b.fh_first_rate, b.nmp_rate, b.sf_eval, b.eval_diff, b.best_move_match,
            CASE WHEN b.game_result = 'draw' THEN 0
                 WHEN b.game_result = CASE b.stm WHEN 'w' THEN 'white' WHEN 'b' THEN 'black' END THEN 1
                 WHEN b.game_result IN ('white', 'black') AND b.stm IN ('w', 'b') THEN -1 END AS game_score,
            CAST(q.searches AS DOUBLE) / q.quota AS sample_weight
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY engine_id ORDER BY {STABLE_ORDER}) AS sample_rank
            FROM _agg_base
        ) b
        JOIN _agg_quota q ON q.engine_id = b.engine_id
        WHERE b.sample_rank <= q.quota
        ORDER BY b.engine_id, b.search_id
    """)
    cnxn.execute("DROP TABLE _agg_quota")


def build_aggregates(cnxn, sample_size=SAMPLE_SIZE, verbose=True):
    """Create (or replace) every agg_* table and search_sample; returns {table: rows}.

    Needs the analytics views (search_metrics) and the fact tables; the tables
    are replaced in one transaction, so a failure keeps the previous ones.
    """
    missing = [name for name in REQUIRED_RELATIONS if not _relation_exists(cnxn, name)]
    if missing:
        raise RuntimeError(f"build_aggregates needs {', '.join(missing)}: run the analytics pipeline "
                           "(load_analytics, transform_positions, transform_search) first")
    if not _relation_exists(cnxn, "stockfish_moves"):
        cnxn.execute(STOCKFISH_MOVES_DDL)   # no Stockfish analysis yet: the agg_sf_* tables stay empty
    cnxn.execute("BEGIN TRANSACTION")
    try:
        cnxn.execute(BASE_SQL)
        for sql in TABLE_SQL.values():
            cnxn.execute(sql)
        _build_hist(cnxn)
        _build_corr(cnxn)
        _build_root(cnxn)
        _build_stockfish(cnxn)
        _build_sample(cnxn, sample_size)
        cnxn.execute("DROP TABLE _agg_root")
        cnxn.execute("DROP TABLE _agg_base")
        for table, expected in AGG_COLUMNS.items():
            actual = _table_columns(cnxn, table)
            if actual != expected:
                raise RuntimeError(f"{table} columns {actual} differ from AGG_COLUMNS {expected}")
        cnxn.execute("COMMIT")
    except Exception:
        cnxn.execute("ROLLBACK")
        raise
    sizes = {table: cnxn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] for table in AGG_TABLES}
    if verbose:
        print("  Aggregates (rows):")
        for table, rows in sizes.items():
            print(f"    {table:.<28s} {rows:>8,}")
    return sizes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the dashboard's pre-aggregated tables")
    parser.add_argument("--db", type=Path, help=f"Analytics database (default: CHESS_ANALYTICS_DB or {ANALYTICS_DB})")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="search_sample rows (default: 20000)")
    args = parser.parse_args(argv)
    db = args.db or os.environ.get("CHESS_ANALYTICS_DB") or ANALYTICS_DB
    with duckdb.connect(str(db)) as cnxn:
        build_aggregates(cnxn, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
