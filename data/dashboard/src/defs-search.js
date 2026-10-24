/**
 * Search-level datasets: per-engine summary, histograms, iterations, tree plies, timings, move ordering.
 * Column names follow the pre-aggregated tables (docs/dashboard-data.md); the live fallbacks produce the same names.
 */
import { has, pick, schema, src } from './db.js';
import { define, col, selectList, searchTable, envOf, gameCore, quantileCols, QUANTILES, HIST_BINS, binSelect,
  withAllCores } from './datasets.js';

const aggCore = T => (pick(T, 'core_type') ? `coalesce(CAST(core_type AS VARCHAR), 'all')` : `'all'`);

function gameJoin(alias = 's') {
  return has('game_stats', 'env_core_type') ? `LEFT JOIN ${src('game_stats')} g ON g.id = ${alias}.game_id` : '';
}

/** One row per search with pipeline-style names, from search_metrics (full snapshots and local analytics files). */
export function searchBase() {
  const T = searchTable();
  if (!T) return null;
  const env = envOf('m', T);
  const c = (name, expr = `m.${name}`) => (has(T, name) ? `${expr}::DOUBLE` : 'NULL::DOUBLE');
  return `SELECT m.engine_id, ${env.core} AS core_type, m.game_id, m.search_id, ${c('ply')} AS ply,
      ${c('completed_depth')} AS depth, ${c('max_qdepth')} AS seldepth, ${c('total_nodes')} AS nodes,
      ${c('total_qnodes')} AS qnodes, ${c('total_time_ms')} AS time_ms, ${c('nps')} AS nps,
      CASE WHEN abs(m.eval) < 99000 THEN m.eval::DOUBLE END AS eval_cp,
      ${c('total_tt_hits')} AS tt_hits, ${c('total_fail_highs')} AS fail_highs, ${c('total_fh_index_0')} AS fh0,
      ${c('nmp')} AS nmp, ${c('nmp_failhigh')} AS nmp_fh, ${c('total_see_prunes')} AS see, ${c('total_delta_prunes')} AS delta,
      ${c('tt_hit_ratio')} AS tt_hit_rate, ${c('qratio')} AS qratio, ${c('fail_high_ratio')} AS fail_high_rate,
      ${c('fail_low_ratio')} AS fail_low_rate, ${c('fh_index_0_ratio')} AS fh_first_rate, ${c('nmp_ratio')} AS nmp_rate,
      ${c('nmp_failhigh_ratio')} AS nmp_success_rate, ${env.mhz} AS mhz,
      ${c('sf_eval')} AS sf_eval, ${c('eval_diff')} AS eval_diff, ${c('best_move_match')} AS best_move_match
    FROM ${src(T)} m ${env.join} WHERE m.engine_id IS NOT NULL`;
}

/* ---------------------------------------------------------------- per-engine summary */
const SUMMARY_COLS = ['searches', 'games', 'nodes', 'time_ms', 'nps', 'knps_per_ghz', 'mhz_mean',
  'depth_mean', 'depth_sd', ...QUANTILES.map(([k]) => `depth_p${k}`),
  'time_ms_mean', 'time_ms_sd', ...QUANTILES.map(([k]) => `time_ms_p${k}`),
  'nodes_mean', 'nodes_sd', ...QUANTILES.map(([k]) => `nodes_p${k}`),
  'nps_mean', ...QUANTILES.map(([k]) => `nps_p${k}`), 'ebf', 'seldepth_mean',
  'qratio', 'tt_hit_rate', 'fail_high_rate', 'fail_low_rate', 'fh_first_rate', 'nmp_rate', 'nmp_success_rate',
  'see_prune_rate', 'delta_prune_rate', 'eval_abs_mean', 'sf_searches', 'sf_best_move_match', 'sf_mae_cp'];

define('summary', {
  agg: 'agg_engine_summary',
  fromAgg() {
    const T = 'agg_engine_summary';
    const list = selectList(Object.fromEntries(SUMMARY_COLS.map(name => [name, col(T, name)])), ['searches', 'nodes', 'time_ms']);
    if (!list) return null;
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${aggCore(T)} AS core_type, ${list} FROM ${src(T)}`;
  },
  live() {
    const base = searchBase();
    if (!base) return null;
    const { core, groupBy } = withAllCores([]);
    return `WITH b AS (${base})
      SELECT engine_id, ${core}, count(*)::DOUBLE AS searches, count(DISTINCT game_id)::DOUBLE AS games,
        sum(nodes) AS nodes, sum(time_ms) AS time_ms, sum(nodes) / nullif(sum(time_ms) / 1000, 0) AS nps,
        sum(nodes) FILTER (WHERE mhz > 0) / nullif(sum(time_ms / 1000 * mhz / 1000) FILTER (WHERE mhz > 0), 0) / 1000 AS knps_per_ghz,
        sum(mhz * time_ms) / nullif(sum(CASE WHEN mhz IS NOT NULL THEN time_ms END), 0) AS mhz_mean,
        sum(nodes * nodes) AS nodes_sq, sum(time_ms * time_ms) AS time_sq, sum(nodes * time_ms) AS nodes_time,
        sum(tt_hits) AS tt_hits, sum(tt_hits * tt_hits) AS tt_hits_sq, sum(tt_hits * nodes) AS tt_hits_nodes,
        sum(qnodes) AS qnodes, sum(qnodes * qnodes) AS qnodes_sq, sum(qnodes * nodes) AS qnodes_nodes,
        sum(fail_highs) AS fh, sum(fail_highs * fail_highs) AS fh_sq, sum(fh0) AS fh0, sum(fh0 * fh0) AS fh0_sq,
        sum(fh0 * fail_highs) AS fh0_fh, sum(nmp) AS nmp, sum(nmp * nmp) AS nmp_sq, sum(nmp_fh) AS nmp_fh,
        sum(nmp_fh * nmp_fh) AS nmp_fh_sq, sum(nmp_fh * nmp) AS nmp_fh_nmp,
        avg(depth) AS depth_mean, stddev_samp(depth) AS depth_sd, ${quantileCols('depth', 'depth')},
        avg(time_ms) AS time_ms_mean, stddev_samp(time_ms) AS time_ms_sd, ${quantileCols('time_ms', 'time_ms')},
        avg(nodes) AS nodes_mean, stddev_samp(nodes) AS nodes_sd, ${quantileCols('nodes', 'nodes')},
        avg(nps) AS nps_mean, ${quantileCols('nps', 'nps')}, NULL::DOUBLE AS ebf, avg(seldepth) AS seldepth_mean,
        sum(qnodes) / nullif(sum(nodes), 0) AS qratio, sum(tt_hits) / nullif(sum(nodes), 0) AS tt_hit_rate,
        sum(fail_highs) / nullif(sum(nodes), 0) AS fail_high_rate, NULL::DOUBLE AS fail_low_rate,
        sum(fh0) / nullif(sum(fail_highs), 0) AS fh_first_rate, sum(nmp) / nullif(sum(nodes), 0) AS nmp_rate,
        sum(nmp_fh) / nullif(sum(nmp), 0) AS nmp_success_rate, sum(see) / nullif(sum(qnodes), 0) AS see_prune_rate,
        sum(delta) / nullif(sum(qnodes), 0) AS delta_prune_rate, avg(abs(eval_cp)) AS eval_abs_mean,
        NULL::DOUBLE AS sf_searches, NULL::DOUBLE AS sf_best_move_match, NULL::DOUBLE AS sf_mae_cp
      FROM b ${groupBy}`;
  },
});

/* ---------------------------------------------------------------- histograms per engine and metric */
define('hist', {
  agg: 'agg_hist',
  fromAgg() {
    const T = 'agg_hist';
    if (!has(T, 'engine_id', 'metric', 'bin_lo', 'bin_hi', 'count')) return null;
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${aggCore(T)} AS core_type, CAST(metric AS VARCHAR) AS metric,
      bin_lo::DOUBLE AS bin_lo, bin_hi::DOUBLE AS bin_hi, count::DOUBLE AS count,
      ${pick(T, 'scale') ? 'CAST(scale AS VARCHAR)' : 'NULL::VARCHAR'} AS scale FROM ${src(T)}`;
  },
  rollup: { keys: ['engine_id', 'metric', 'bin_lo', 'bin_hi', 'scale'], sums: ['count'] },
  live() {
    const base = searchBase();
    if (!base) return null;
    return `WITH b AS (${base}), u AS (SELECT * FROM b UNION ALL SELECT * REPLACE ('all' AS core_type) FROM b)
      ${Object.keys(HIST_BINS).map(m => binSelect(m, 'u')).join(' UNION ALL ')}`;
  },
  derive: body => `SELECT *, count / sum(count) OVER (PARTITION BY engine_id, core_type, metric) AS share,
      (bin_lo + bin_hi) / 2 AS mid_lin, CASE WHEN bin_lo > 0 THEN sqrt(bin_lo * bin_hi) END AS mid_log FROM ${body}`,
});

/* ---------------------------------------------------------------- iterative deepening, per depth */
const RATE_COLS = ['qratio', 'tt_hit_rate', 'fail_high_rate', 'fail_low_rate', 'fh_first_rate', 'see_prune_rate',
  'delta_prune_rate', 'nmp_rate', 'nmp_success_rate'];

define('depth', {
  agg: 'agg_depth',
  fromAgg() {
    const T = 'agg_depth';
    const d = pick(T, 'depth', 'iter_depth');
    if (!d) return null;
    const fhr = col(T, 'fail_high_researches_mean'), flr = col(T, 'fail_low_researches_mean');
    const map = {
      searches: col(T, 'searches'), nodes: col(T, 'nodes'), qnodes: col(T, 'qnodes'), time_ms: col(T, 'time_ms'),
      nodes_mean: col(T, 'nodes_mean'), time_ms_mean: col(T, 'time_ms_mean'), nps: col(T, 'nps'), ebf: col(T, 'ebf'),
      ...Object.fromEntries(RATE_COLS.map(r => [r, col(T, r)])),
      researches_mean: fhr || flr ? `coalesce(${fhr ?? 0}, 0) + coalesce(${flr ?? 0}, 0)` : null,
      eval_change_mean: col(T, 'eval_change_mean'), move_change_rate: col(T, 'move_change_rate'),
    };
    const list = selectList(map, ['searches']);
    if (!list) return null;
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${aggCore(T)} AS core_type, CAST("${d}" AS INTEGER) AS depth, ${list}
      FROM ${src(T)}`;
  },
  rollup: { keys: ['engine_id', 'core_type', 'depth'], sums: ['searches', 'nodes', 'qnodes', 'time_ms'] },
  live() {
    if (!has('iterative_deepening_stats', 'search_id', 'depth', 'nodes') || !schema.has('search_stats')) return null;
    const { core, groupBy } = withAllCores(['depth']);
    return `WITH it AS (
        SELECT s.engine_id, ${gameCore()} AS core_type, i.depth, i.time_ms::DOUBLE AS time_ms,
          (i.nodes + i.qnodes)::DOUBLE AS nodes, i.qnodes::DOUBLE AS qnodes, i.tt_hits::DOUBLE AS tt_hits,
          i.fail_highs::DOUBLE AS fail_highs, i.fail_lows::DOUBLE AS fail_lows, i.fh_index_0::DOUBLE AS fh0,
          i.see_prunes::DOUBLE AS see, i.delta_prunes::DOUBLE AS delta, i.nmp::DOUBLE AS nmp,
          i.nmp_failhigh::DOUBLE AS nmp_fh, (i.fail_high_researches + i.fail_low_researches)::DOUBLE AS researches,
          CASE WHEN abs(i.eval) < 99000 AND abs(lag(i.eval) OVER w) < 99000 THEN abs(i.eval - lag(i.eval) OVER w) END AS eval_delta,
          CASE WHEN lag(i.move) OVER w IS NULL THEN NULL WHEN i.move IS DISTINCT FROM lag(i.move) OVER w THEN 1.0 ELSE 0.0 END AS move_changed,
          (i.nodes + i.qnodes)::DOUBLE / nullif(lag(i.nodes + i.qnodes) OVER w, 0) AS growth
        FROM ${src('iterative_deepening_stats')} i JOIN ${src('search_stats')} s ON s.id = i.search_id ${gameJoin('s')}
        WHERE s.engine_id IS NOT NULL
        WINDOW w AS (PARTITION BY i.search_id ORDER BY i.depth))
      SELECT engine_id, ${core}, depth, count(*)::DOUBLE AS searches, sum(nodes) AS nodes, sum(qnodes) AS qnodes,
        sum(time_ms) AS time_ms, avg(nodes) AS nodes_mean, avg(time_ms) AS time_ms_mean,
        sum(nodes) / nullif(sum(time_ms) / 1000, 0) AS nps, exp(avg(ln(CASE WHEN growth > 0 THEN growth END))) AS ebf,
        sum(qnodes) / nullif(sum(nodes), 0) AS qratio, sum(tt_hits) / nullif(sum(nodes), 0) AS tt_hit_rate,
        sum(fail_highs) / nullif(sum(nodes), 0) AS fail_high_rate, sum(fail_lows) / nullif(sum(nodes), 0) AS fail_low_rate,
        sum(fh0) / nullif(sum(fail_highs), 0) AS fh_first_rate, sum(see) / nullif(sum(qnodes), 0) AS see_prune_rate,
        sum(delta) / nullif(sum(qnodes), 0) AS delta_prune_rate, sum(nmp) / nullif(sum(nodes), 0) AS nmp_rate,
        sum(nmp_fh) / nullif(sum(nmp), 0) AS nmp_success_rate, avg(researches) AS researches_mean,
        avg(eval_delta) AS eval_change_mean, avg(move_changed) AS move_change_rate
      FROM it ${groupBy}`;
  },
  derive: body => `SELECT *,
      searches / nullif(max(searches) OVER (PARTITION BY engine_id, core_type), 0) AS reached_share,
      sum(time_ms) OVER (PARTITION BY engine_id, core_type ORDER BY depth)
        / nullif(sum(time_ms) OVER (PARTITION BY engine_id, core_type), 0) AS time_share_cum
    FROM ${body}`,
});

/* ---------------------------------------------------------------- search tree, per ply */
define('tree', {
  agg: 'agg_tree_ply',
  fromAgg() {
    const T = 'agg_tree_ply';
    const p = pick(T, 'ply', 'depth', 'tree_ply');
    if (!p) return null;
    const map = { searches: col(T, 'searches'), nodes: col(T, 'nodes'), qnodes: col(T, 'qnodes'),
      ...Object.fromEntries(RATE_COLS.map(r => [r, col(T, r)])) };
    const list = selectList(map, ['nodes']);
    if (!list) return null;
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, CAST("${p}" AS INTEGER) AS ply, ${list} FROM ${src(T)}`;
  },
  live() {
    if (!has('search_tree_stats', 'search_id', 'depth', 'nodes') || !schema.has('search_stats')) return null;
    return `SELECT s.engine_id, t.depth AS ply, count(*)::DOUBLE AS searches,
        sum(t.nodes + t.qnodes)::DOUBLE AS nodes, sum(t.qnodes)::DOUBLE AS qnodes,
        sum(t.qnodes)::DOUBLE / nullif(sum(t.nodes + t.qnodes), 0) AS qratio,
        sum(t.tt_hits)::DOUBLE / nullif(sum(t.nodes + t.qnodes), 0) AS tt_hit_rate,
        sum(t.fail_highs)::DOUBLE / nullif(sum(t.nodes + t.qnodes), 0) AS fail_high_rate,
        sum(t.fail_lows)::DOUBLE / nullif(sum(t.nodes + t.qnodes), 0) AS fail_low_rate,
        sum(t.fh_index_0)::DOUBLE / nullif(sum(t.fail_highs), 0) AS fh_first_rate,
        sum(t.see_prunes)::DOUBLE / nullif(sum(t.qnodes), 0) AS see_prune_rate,
        sum(t.delta_prunes)::DOUBLE / nullif(sum(t.qnodes), 0) AS delta_prune_rate,
        sum(t.nmp)::DOUBLE / nullif(sum(t.nodes + t.qnodes), 0) AS nmp_rate,
        sum(t.nmp_failhigh)::DOUBLE / nullif(sum(t.nmp), 0) AS nmp_success_rate
      FROM ${src('search_tree_stats')} t JOIN ${src('search_stats')} s ON s.id = t.search_id
      WHERE s.engine_id IS NOT NULL GROUP BY ALL`;
  },
  derive: body => `SELECT *, nodes / nullif(searches, 0) AS nodes_mean,
      nodes / nullif(sum(nodes) OVER (PARTITION BY engine_id), 0) AS node_share FROM ${body}`,
});

/* ---------------------------------------------------------------- function timings */
define('timing', {
  agg: 'agg_timing',
  deps: ['summary'],
  fromAgg({ summary }) {
    const T = 'agg_timing';
    const f = pick(T, 'function', 'func', 'name');
    const calls = col(T, 'calls', 'num_calls'), total = col(T, 'total_ms', 'total_time_ms');
    if (!f || !calls || !total) return null;
    const searches = col(T, 'searches');
    const fallback = summary ? `(SELECT searches FROM ds_summary x WHERE x.engine_id = t.engine_id AND x.core_type = 'all')` : 'NULL::DOUBLE';
    return `SELECT CAST(t.engine_id AS INTEGER) AS engine_id, CAST(t."${f}" AS VARCHAR) AS function,
        ${calls.replace('TRY_CAST("', 'TRY_CAST(t."')} AS calls, ${total.replace('TRY_CAST("', 'TRY_CAST(t."')} AS total_ms,
        coalesce(${searches ? searches.replace('TRY_CAST("', 'TRY_CAST(t."') : 'NULL::DOUBLE'}, ${fallback}) AS searches
      FROM ${src(T)} t`;
  },
  live() {
    if (!has('search_timings', 'search_id', 'function', 'num_calls', 'total_time_ms')) return null;
    return `WITH t AS (SELECT s.engine_id, t.search_id, t."function" AS function, t.num_calls, t.total_time_ms
          FROM ${src('search_timings')} t JOIN ${src('search_stats')} s ON s.id = t.search_id WHERE s.engine_id IS NOT NULL),
        f AS (SELECT engine_id, function, sum(num_calls)::DOUBLE AS calls, sum(total_time_ms)::DOUBLE AS total_ms
          FROM t GROUP BY ALL),
        n AS (SELECT engine_id, count(DISTINCT search_id)::DOUBLE AS searches FROM t GROUP BY 1)
      SELECT f.*, n.searches FROM f JOIN n USING (engine_id)`;
  },
  derive: body => `SELECT *, total_ms * 1000 / nullif(calls, 0) AS us_per_call, total_ms / nullif(searches, 0) AS ms_per_search,
      calls / nullif(searches, 0) AS calls_per_search,
      total_ms / nullif(coalesce(max(CASE WHEN upper(function) = 'ROOT' THEN total_ms END) OVER (PARTITION BY engine_id),
                                 max(total_ms) OVER (PARTITION BY engine_id)), 0) AS share
    FROM ${body}`,
});

/* ---------------------------------------------------------------- move ordering: which move index caused the cutoff */
const BUCKET_CASE = b => `CASE
    WHEN ${b} IN ('0', 'fh_index_0', 'fh0', 'idx0') THEN '0' WHEN ${b} IN ('1', 'fh_index_1', 'fh1', 'idx1') THEN '1'
    WHEN ${b} IN ('2', 'fh_index_2', 'fh2', 'idx2') THEN '2' WHEN ${b} IN ('3', 'fh_index_3', 'fh3', 'idx3') THEN '3'
    WHEN ${b} LIKE '%4%' THEN '4-7' WHEN ${b} LIKE '%8%' THEN '8+' ELSE ${b} END`;
const DIM_CASE = d => `CASE WHEN ${d} IN ('all', 'overall', 'total') THEN 'all'
    WHEN ${d} LIKE '%iter%' OR ${d} = 'depth' THEN 'iter_depth'
    WHEN ${d} LIKE '%tree%' OR ${d} = 'ply' THEN 'tree_ply'
    WHEN ${d} LIKE '%complete%' THEN 'completed_depth' ELSE ${d} END`;

define('moveorder', {
  agg: 'agg_moveorder',
  fromAgg() {
    const T = 'agg_moveorder';
    const b = pick(T, 'fh_index_bucket', 'bucket', 'fh_bucket');
    const c = col(T, 'count', 'n', 'cutoffs');
    if (!b || !c || !has(T, 'dimension')) return null;
    const v = pick(T, 'dim_value', 'value');
    const order = pick(T, 'dim_order') ? 'TRY_CAST(dim_order AS DOUBLE)' : 'NULL::DOUBLE';
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${DIM_CASE('lower(CAST(dimension AS VARCHAR))')} AS dimension,
      coalesce(${order}, ${v ? `TRY_CAST("${v}" AS DOUBLE)` : 'NULL::DOUBLE'}) AS dim_value,
      ${v ? `CAST("${v}" AS VARCHAR)` : 'NULL::VARCHAR'} AS dim_label,
      ${BUCKET_CASE(`lower(replace(CAST("${b}" AS VARCHAR), ' ', ''))`)} AS bucket, ${c} AS count FROM ${src(T)}`;
  },
  live() {
    if (!has('search_stats', 'fh_index_0')) return null;
    const fh = p => `sum(${p}fh_index_0)::DOUBLE AS b0, sum(${p}fh_index_1)::DOUBLE AS b1, sum(${p}fh_index_2)::DOUBLE AS b2,
      sum(${p}fh_index_3)::DOUBLE AS b3, sum(${p}fh_index_4to7)::DOUBLE AS b4, sum(${p}fh_index_8plus)::DOUBLE AS b8`;
    const parts = [
      `SELECT engine_id, 'all' AS dimension, NULL::DOUBLE AS dim_value, ${fh('')} FROM ${src('search_stats')} WHERE engine_id IS NOT NULL GROUP BY ALL`,
      `SELECT engine_id, 'completed_depth', completed_depth::DOUBLE, ${fh('')} FROM ${src('search_stats')} WHERE engine_id IS NOT NULL GROUP BY ALL`,
    ];
    if (has('iterative_deepening_stats', 'fh_index_0')) {
      parts.push(`SELECT s.engine_id, 'iter_depth', i.depth::DOUBLE, ${fh('i.')} FROM ${src('iterative_deepening_stats')} i
        JOIN ${src('search_stats')} s ON s.id = i.search_id GROUP BY ALL`);
    }
    if (has('search_tree_stats', 'fh_index_0')) {
      parts.push(`SELECT s.engine_id, 'tree_ply', t.depth::DOUBLE, ${fh('t.')} FROM ${src('search_tree_stats')} t
        JOIN ${src('search_stats')} s ON s.id = t.search_id GROUP BY ALL`);
    }
    return `SELECT engine_id, dimension, dim_value, CAST(dim_value AS VARCHAR) AS dim_label,
        CASE k WHEN 'b0' THEN '0' WHEN 'b1' THEN '1' WHEN 'b2' THEN '2' WHEN 'b3' THEN '3' WHEN 'b4' THEN '4-7' ELSE '8+' END AS bucket,
        cnt AS count
      FROM (${parts.join(' UNION ALL ')}) x UNPIVOT (cnt FOR k IN (b0, b1, b2, b3, b4, b8))`;
  },
  derive: body => `SELECT *, CASE bucket WHEN '0' THEN 0 WHEN '1' THEN 1 WHEN '2' THEN 2 WHEN '3' THEN 3 WHEN '4-7' THEN 4 ELSE 5 END AS bucket_ord,
      count / nullif(sum(count) OVER (PARTITION BY engine_id, dimension, dim_label), 0) AS share
    FROM ${body}`,
});
