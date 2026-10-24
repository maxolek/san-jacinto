/** Engine-level datasets: hardware, the search sample, correlations, positions, eval quality, root moves. */
import { has, pick, query, schema, src } from './db.js';
import { dataset, define, col, gameCore, searchTable, envOf } from './datasets.js';
import { searchBase } from './defs-search.js';

const aggCore = T => (pick(T, 'core_type') ? `coalesce(CAST(core_type AS VARCHAR), 'all')` : `'all'`);

/* ---------------------------------------------------------------- speed against measured clock */
define('env', {
  agg: 'agg_env',
  fromAgg() {
    const T = 'agg_env';
    const nodes = col(T, 'nodes'), time = col(T, 'time_ms');
    if (!has(T, 'engine_id', 'mhz_bucket') || !nodes || !time) return null;
    const mean = col(T, 'mhz_mean');
    const nps = col(T, 'nps'), k = col(T, 'knps_per_ghz');
    // clock of the bucket: published mean, else implied by nps / knps_per_ghz, else the bucket centre
    const mhz = mean ?? (nps && k ? `${nps} / nullif(${k}, 0)` : 'mhz_bucket + 50');
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${aggCore(T)} AS core_type, mhz_bucket::DOUBLE AS mhz_bucket,
        ${col(T, 'searches') ?? 'NULL::DOUBLE'} AS searches, ${nodes} AS nodes, ${time} AS time_ms, ${time} * ${mhz} AS mhz_time
      FROM ${src(T)} WHERE ${aggCore(T)} <> 'all'`;
  },
  live() {
    const base = searchBase();
    if (!base || envOf('m', searchTable()).mhz === 'NULL::DOUBLE') return null;
    return `WITH b AS (${base})
      SELECT engine_id, core_type, floor(mhz / 100) * 100 AS mhz_bucket, count(*)::DOUBLE AS searches,
        sum(nodes) AS nodes, sum(time_ms) AS time_ms, sum(mhz * time_ms) AS mhz_time
      FROM b WHERE mhz > 0 AND time_ms > 0 GROUP BY ALL`;
  },
  derive: body => `SELECT *, nodes / nullif(time_ms / 1000, 0) AS nps, mhz_time / nullif(time_ms, 0) AS mhz_mean,
      nodes / nullif(time_ms / 1000, 0) / 1000 / nullif(mhz_time / nullif(time_ms, 0) / 1000, 0) AS knps_per_ghz
    FROM ${body}`,
});

/* ---------------------------------------------------------------- stratified sample of searches (for scatter plots) */
// Columns of search_sample the dashboard uses (docs/dashboard-data.md); present ones are copied.
const SAMPLE_TEXT = new Set(['core_type', 'stm', 'phase', 'position_type']);
const SAMPLE_COLS = ['search_id', 'game_id', 'core_type', 'mhz', 'ply', 'stm', 'phase', 'position_type', 'pos_tactical',
  'pos_positional', 'depth', 'seldepth', 'nodes', 'qnodes', 'time_ms', 'nps', 'ebf', 'eval_cp', 'mate', 'tt_hit_rate', 'qratio',
  'fail_high_rate', 'fail_low_rate', 'fh_first_rate', 'nmp_rate', 'sf_eval', 'eval_diff', 'best_move_match', 'game_score',
  'sample_weight'];

define('sample', {
  agg: 'search_sample',
  fromAgg() {
    const T = 'search_sample';
    if (!has(T, 'engine_id')) return null;
    const cols = SAMPLE_COLS.filter(c => has(T, c)).map(c => (SAMPLE_TEXT.has(c) ? `CAST(s."${c}" AS VARCHAR) AS ${c}` : `TRY_CAST(s."${c}" AS DOUBLE) AS ${c}`));
    if (!has(T, 'eval_cp') && has(T, 'eval')) cols.push('CASE WHEN abs(s.eval) < 99000 THEN s.eval::DOUBLE END AS eval_cp');
    if (!has(T, 'search_id')) cols.push('row_number() OVER ()::DOUBLE AS search_id');
    return `SELECT CAST(s.engine_id AS INTEGER) AS engine_id, ${cols.join(', ')} FROM ${src(T)} s`;
  },
  async live() {
    const base = searchBase();
    if (!base) return null;
    const [{ n }] = await query('SELECT count(*)::DOUBLE AS n FROM ds_engines');
    const per = Math.ceil(20000 / Math.max(1, n));
    const pf = has('position_features', 'search_id', 'game_phase');
    return `WITH b AS (${base})
      SELECT b.* EXCLUDE (tt_hits, fail_highs, fh0, nmp, nmp_fh, see, delta, nmp_success_rate)
        ${pf ? `, pf.game_phase AS phase, pf.position_type, pf.pos_tactical::DOUBLE AS pos_tactical,
          pf.pos_positional::DOUBLE AS pos_positional` : ''}
      FROM b ${pf ? `LEFT JOIN ${src('position_features')} pf ON pf.search_id = b.search_id` : ''}
      QUALIFY row_number() OVER (PARTITION BY b.engine_id ORDER BY hash(b.search_id)) <= ${per}`;
  },
  derive: body => `SELECT *, abs(eval_cp) AS abs_eval_cp FROM ${body}`,
});

/* a thinner, deterministic subset for drawing points (the charts beside it use the full sample) */
define('sample_points', {
  deps: ['sample'],
  engine: false,
  live: ({ sample }) => (sample ? `SELECT * FROM ds_sample
      QUALIFY row_number() OVER (PARTITION BY engine_id ORDER BY hash(search_id)) <= 700` : null),
});

/* ---------------------------------------------------------------- correlations between search metrics, per engine */
export const CORR_METRICS = ['depth', 'seldepth', 'nodes', 'time_ms', 'nps', 'ebf', 'tt_hit_rate', 'qratio', 'fail_high_rate',
  'fail_low_rate', 'fh_first_rate', 'nmp_rate', 'fail_high_researches', 'fail_low_researches', 'abs_eval_cp', 'ply', 'mhz'];

define('corr', {
  agg: 'agg_corr',
  deps: ['sample'],
  fromAgg() {
    const T = 'agg_corr';
    if (!has(T, 'engine_id', 'metric_x', 'metric_y')) return null;
    const p = col(T, 'pearson') ?? 'NULL::DOUBLE';
    const s = col(T, 'spearman') ?? 'NULL::DOUBLE';
    return `WITH c AS (SELECT CAST(engine_id AS INTEGER) AS engine_id, CAST(metric_x AS VARCHAR) AS metric_x,
          CAST(metric_y AS VARCHAR) AS metric_y, ${col(T, 'n') ?? 'NULL::DOUBLE'} AS n, ${p} AS pearson, ${s} AS spearman
        FROM ${src(T)} WHERE metric_x <> metric_y),
      u AS (SELECT * FROM c UNION ALL SELECT engine_id, metric_y, metric_x, n, pearson, spearman FROM c)
      SELECT DISTINCT ON (engine_id, metric_x, metric_y) * FROM u`;
  },
  async live({ sample }) {
    if (!sample) return null;
    const metrics = CORR_METRICS.filter(m => sample.columns.has(m));
    if (metrics.length < 2) return null;
    const ranks = metrics.map(m => `CASE WHEN ${m} IS NOT NULL THEN rank() OVER (PARTITION BY engine_id ORDER BY ${m}) END AS r_${m}`);
    const parts = [];
    metrics.forEach((a, i) => metrics.slice(i + 1).forEach(b => parts.push(
      `SELECT engine_id, '${a}' AS metric_x, '${b}' AS metric_y, count(${a} + ${b})::DOUBLE AS n,
        corr(${a}, ${b}) AS pearson, corr(r_${a}, r_${b}) AS spearman FROM r GROUP BY engine_id`)));
    return `WITH r AS (SELECT engine_id, ${metrics.join(', ')}, ${ranks.join(', ')} FROM ds_sample),
      c AS (${parts.join(' UNION ALL ')})
      SELECT * FROM c WHERE n >= 30 UNION ALL SELECT engine_id, metric_y, metric_x, n, pearson, spearman FROM c WHERE n >= 30`;
  },
  liveOrigin: 'computed in your browser from the search sample',
});

/* ---------------------------------------------------------------- Stockfish comparison */
let perspective;
/**
 * Engine evals are from the side to move. Current snapshots store Stockfish's eval the same way
 * (sf_eval_pov = 'stm'); older ones used White's view, which the correlation with the engine eval reveals.
 */
export function sfPerspective() {
  const T = searchTable();
  if (!perspective) {
    perspective = (async () => {
      if (!T || !has(T, 'sf_eval', 'eval', 'fen')) return null;
      const [r] = await query(`SELECT count(*)::DOUBLE AS n, corr(eval, sf_eval) AS raw,
          corr(eval, CASE WHEN split_part(fen, ' ', 2) = 'b' THEN -sf_eval ELSE sf_eval END) AS flipped
        FROM ${src(T)} WHERE sf_eval IS NOT NULL AND abs(eval) < 99000 AND abs(sf_eval) < 99000`);
      if (!(r?.n > 0)) return null;
      return (r.flipped ?? -1) > (r.raw ?? -1) + 0.05 ? 'white' : 'stm';
    })();
  }
  return perspective;
}

const sfStm = (p, alias = 'm') => (p === 'white'
  ? `CASE WHEN split_part(${alias}.fen, ' ', 2) = 'b' THEN -${alias}.sf_eval ELSE ${alias}.sf_eval END` : `${alias}.sf_eval`);

/** Whether the sample carries Stockfish columns with data (lite snapshots have no per-search tables). */
async function sampleWithSf() {
  const sample = await dataset('sample');
  if (!sample?.columns.has('eval_diff')) return null;
  const [{ n }] = await query('SELECT count(eval_diff)::DOUBLE AS n FROM ds_sample');
  return n > 0 ? sample : null;
}

const QUALITY_AGGS = `count(diff)::DOUBLE AS n, avg(abs(diff)) AS mae, avg(diff) AS bias, median(abs(diff)) AS median_ae,
  avg(hit) AS match_rate, count(hit)::DOUBLE AS match_n, count(*) FILTER (WHERE abs(diff) > 200)::DOUBLE AS blunders`;
const byCore = keys => `GROUP BY GROUPING SETS ((engine_id, core_type, ${keys}), (engine_id, ${keys}))`;
const coreOrAll = `CASE WHEN grouping(core_type) = 1 THEN 'all' ELSE core_type END AS core_type`;

define('quality', {
  async live() {
    const p = await sfPerspective();
    if (p) {
      const T = searchTable();
      const env = envOf('m', T);
      const match = has(T, 'best_move_match') ? 'm.best_move_match' : has(T, 'sf_best_move', 'move')
        ? `CASE WHEN m.sf_best_move IS NULL THEN NULL WHEN m.move = m.sf_best_move THEN 1 ELSE 0 END` : 'NULL';
      return `WITH b AS (SELECT m.engine_id, ${env.core} AS core_type, m.completed_depth::DOUBLE AS depth,
            m.eval::DOUBLE - (${sfStm(p)})::DOUBLE AS diff, ${match} AS hit
          FROM ${src(T)} m ${env.join} WHERE m.sf_eval IS NOT NULL AND abs(m.eval) < 99000 AND abs(m.sf_eval) < 99000)
        SELECT engine_id, ${coreOrAll}, depth, ${QUALITY_AGGS} FROM b ${byCore('depth')}`;
    }
    const sample = await sampleWithSf();
    if (!sample) return null;
    const core = sample.columns.has('core_type') ? 'core_type' : `'unknown'`;
    return `WITH b AS (SELECT engine_id, ${core} AS core_type, depth, eval_diff AS diff, best_move_match AS hit
        FROM ds_sample WHERE eval_diff IS NOT NULL)
      SELECT engine_id, ${coreOrAll}, depth, ${QUALITY_AGGS} FROM b ${byCore('depth')}`;
  },
  liveOrigin: 'computed in your browser (from the search sample when the snapshot has no per-search tables)',
});

define('quality_hist', {
  deps: ['hist'],
  async live({ hist }) {
    const p = await sfPerspective();
    if (p) {
      const T = searchTable();
      const env = envOf('m', T);
      return `WITH b AS (SELECT m.engine_id, ${env.core} AS core_type, greatest(-500, least(499, m.eval - (${sfStm(p)}))) AS diff
          FROM ${src(T)} m ${env.join} WHERE m.sf_eval IS NOT NULL AND abs(m.eval) < 99000 AND abs(m.sf_eval) < 99000)
        SELECT engine_id, ${coreOrAll}, floor(diff / 25) * 25 AS bin_lo, floor(diff / 25) * 25 + 25 AS bin_hi,
          count(*)::DOUBLE AS count FROM b ${byCore('bin_lo, bin_hi')}`;
    }
    if (!hist) return null;
    const [{ n }] = await query(`SELECT count(*)::DOUBLE AS n FROM ds_hist WHERE metric = 'eval_diff'`);
    return n > 0 ? `SELECT engine_id, core_type, bin_lo, bin_hi, count FROM ds_hist WHERE metric = 'eval_diff'` : null;
  },
  derive: body => `SELECT *, count / sum(count) OVER (PARTITION BY engine_id, core_type) AS share FROM ${body}`,
});

define('quality_points', {
  async live() {
    const p = await sfPerspective();
    if (p) {
      const T = searchTable();
      const env = envOf('m', T);
      return `SELECT m.engine_id, ${env.core} AS core_type, m.eval::DOUBLE AS eval, (${sfStm(p)})::DOUBLE AS sf_eval
        FROM ${src(T)} m ${env.join}
        WHERE m.sf_eval IS NOT NULL AND abs(m.eval) < 1500 AND abs(m.sf_eval) < 1500
        QUALIFY row_number() OVER (PARTITION BY m.engine_id ORDER BY hash(m.search_id)) <= 1500`;
    }
    const sample = await sampleWithSf();
    if (!sample) return null;
    return `SELECT engine_id, ${sample.columns.has('core_type') ? 'core_type' : `'unknown'`} AS core_type, eval_cp AS eval, sf_eval
      FROM ds_sample WHERE sf_eval IS NOT NULL AND abs(eval_cp) < 1500 AND abs(sf_eval) < 1500
      QUALIFY row_number() OVER (PARTITION BY engine_id ORDER BY hash(search_id)) <= 1500`;
  },
});

/* ---------------------------------------------------------------- positions: phase and type */
define('positions', {
  agg: 'agg_positions',
  fromAgg() {
    const T = 'agg_positions';
    if (!has(T, 'engine_id', 'phase', 'position_type', 'searches')) return null;
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${aggCore(T)} AS core_type, CAST(phase AS VARCHAR) AS game_phase,
        CAST(position_type AS VARCHAR) AS position_type, ${col(T, 'searches')} AS searches, ${col(T, 'depth_mean') ?? 'NULL'} AS depth_mean,
        ${col(T, 'time_ms_mean') ?? 'NULL'} AS time_mean, ${col(T, 'nps') ?? 'NULL'} AS nps,
        ${col(T, 'sf_searches') ?? 'NULL'} AS sf_n, ${col(T, 'sf_mae_cp') ?? 'NULL'} AS sf_mae,
        ${col(T, 'sf_best_move_match') ?? 'NULL'} AS sf_match
      FROM ${src(T)} WHERE phase <> 'all' AND position_type <> 'all'`;
  },
  rollup: { keys: ['engine_id', 'core_type', 'game_phase', 'position_type'], sums: ['searches'] },
  async live() {
    const T = searchTable();
    if (!T || !has('position_features', 'search_id', 'game_phase')) return null;
    const p = await sfPerspective();
    const env = envOf('m', T);
    const sf = p ? `, count(sf)::DOUBLE AS sf_n, avg(abs(eval - sf)) FILTER (WHERE abs(eval) < 99000 AND abs(sf) < 99000) AS sf_mae,
        avg(hit) AS sf_match` : ', NULL::DOUBLE AS sf_n, NULL::DOUBLE AS sf_mae, NULL::DOUBLE AS sf_match';
    const type = has('position_features', 'position_type') ? `coalesce(pf.position_type, 'Unknown')` : `'Unknown'`;
    return `WITH b AS (SELECT m.engine_id, ${env.core} AS core_type, coalesce(pf.game_phase, 'unknown') AS game_phase,
          ${type} AS position_type, m.completed_depth::DOUBLE AS depth, m.total_time_ms::DOUBLE AS time_ms,
          m.total_nodes::DOUBLE AS nodes, m.eval::DOUBLE AS eval, ${p ? `(${sfStm(p)})::DOUBLE` : 'NULL::DOUBLE'} AS sf,
          ${has(T, 'best_move_match') ? 'm.best_move_match::DOUBLE' : 'NULL::DOUBLE'} AS hit
        FROM ${src(T)} m ${env.join} JOIN ${src('position_features')} pf ON pf.search_id = m.search_id
        WHERE m.engine_id IS NOT NULL)
      SELECT engine_id, CASE WHEN grouping(core_type) = 1 THEN 'all' ELSE core_type END AS core_type, game_phase, position_type,
        count(*)::DOUBLE AS searches, avg(depth) AS depth_mean, avg(time_ms) AS time_mean,
        sum(nodes) / nullif(sum(time_ms) / 1000, 0) AS nps ${sf}
      FROM b GROUP BY GROUPING SETS ((engine_id, core_type, game_phase, position_type), (engine_id, game_phase, position_type))`;
  },
});

define('positions_points', {
  async live() {
    const sample = await dataset('sample');
    if (sample?.columns.has('pos_tactical') && sample.columns.has('pos_positional')) {
      return `SELECT engine_id, ${sample.columns.has('core_type') ? 'core_type' : `'unknown'`} AS core_type, pos_tactical AS tactical,
          pos_positional AS positional, depth AS completed_depth FROM ds_sample
        WHERE pos_tactical IS NOT NULL AND pos_positional IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY engine_id ORDER BY hash(search_id)) <= 1500`;
    }
    if (!has('position_features', 'search_id', 'pos_tactical', 'pos_positional') || !schema.has('search_stats')) return null;
    const join = has('game_stats', 'env_core_type') ? `LEFT JOIN ${src('game_stats')} g ON g.id = s.game_id` : '';
    return `SELECT s.engine_id, ${gameCore()} AS core_type, pf.pos_tactical::DOUBLE AS tactical, pf.pos_positional::DOUBLE AS positional,
        s.completed_depth::DOUBLE AS completed_depth
      FROM ${src('search_stats')} s JOIN ${src('position_features')} pf ON pf.search_id = s.id ${join}
      WHERE pf.pos_tactical IS NOT NULL AND pf.pos_positional IS NOT NULL
      QUALIFY row_number() OVER (PARTITION BY s.engine_id ORDER BY hash(s.id)) <= 1500`;
  },
});

/* ---------------------------------------------------------------- root moves and Stockfish's lines */
// Snapshots carry these pre-aggregated (agg_root_*, agg_sf_*: split by core type, with 'all' rows). Without agg_root_*
// the root_* datasets come from the root-move log (full snapshots and local files), as core_type 'all' rows only.
// Root moves of each search's deepest complete iteration: the deepest logged depth with as many root moves as the search
// ever logged at one depth (as data/transforms/aggregates.py); the last iteration is usually cut off by the clock.
const rootBase = () => `SELECT s.engine_id, r.search_id, r.move_index, r.nodes::DOUBLE AS nodes, r.time_ms::DOUBLE AS time_ms,
    r.eval::DOUBLE AS eval
  FROM ${src('root_moves')} r JOIN ${src('search_stats')} s ON s.id = r.search_id
  ${has('root_moves', 'depth') ? `JOIN (SELECT search_id, max(depth) AS d FROM (
      SELECT search_id, depth, count(*) AS moves, max(count(*)) OVER (PARTITION BY search_id) AS most
      FROM ${src('root_moves')} GROUP BY search_id, depth) WHERE moves = most GROUP BY 1) m
    ON m.search_id = r.search_id AND m.d = r.depth` : ''}
  WHERE s.engine_id IS NOT NULL`;
const aggPhase = T => (pick(T, 'phase') ? `coalesce(CAST(phase AS VARCHAR), 'unknown')` : `'all'`);

define('root_index', {
  agg: 'agg_root_index',
  fromAgg() {
    const T = 'agg_root_index';
    if (!has(T, 'engine_id', 'move_index', 'nodes')) return null;
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${aggCore(T)} AS core_type, least(move_index, 10)::DOUBLE AS move_index,
        sum(${col(T, 'moves') ?? 'NULL::DOUBLE'}) AS moves, sum(${col(T, 'nodes')}) AS nodes,
        sum(${col(T, 'time_ms') ?? 'NULL::DOUBLE'}) AS time_ms
      FROM ${src(T)} WHERE engine_id IS NOT NULL AND move_index IS NOT NULL GROUP BY ALL`;
  },
  live() {
    if (!has('root_moves', 'search_id', 'move_index', 'nodes') || !schema.has('search_stats')) return null;
    return `SELECT engine_id, 'all' AS core_type, least(move_index, 10)::DOUBLE AS move_index, count(*)::DOUBLE AS moves,
        sum(nodes) AS nodes, sum(time_ms) AS time_ms FROM (${rootBase()}) b GROUP BY ALL`;
  },
  derive: body => `SELECT *, nodes / nullif(sum(nodes) OVER (PARTITION BY engine_id, core_type), 0) AS node_share,
      time_ms / nullif(sum(time_ms) OVER (PARTITION BY engine_id, core_type), 0) AS time_share FROM ${body}`,
});

define('root_gap', {
  agg: 'agg_root_gap',
  fromAgg() {
    const T = 'agg_root_gap';
    if (!has(T, 'engine_id', 'gap', 'searches')) return null;
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${aggCore(T)} AS core_type, gap::DOUBLE AS gap,
        ${col(T, 'searches')} AS searches
      FROM ${src(T)} WHERE engine_id IS NOT NULL AND gap IS NOT NULL`;
  },
  live() {
    if (!has('root_moves', 'search_id', 'eval') || !schema.has('search_stats')) return null;
    return `WITH g AS (SELECT engine_id, search_id, max(eval) AS best, list(eval ORDER BY eval DESC)[2] AS second
          FROM (${rootBase()}) b WHERE abs(eval) < 99000 GROUP BY ALL)
      SELECT engine_id, 'all' AS core_type, least(500, floor((best - second) / 10) * 10) AS gap, count(*)::DOUBLE AS searches
      FROM g WHERE second IS NOT NULL GROUP BY ALL`;
  },
  derive: body => `SELECT *, searches / nullif(sum(searches) OVER (PARTITION BY engine_id, core_type), 0) AS share FROM ${body}`,
});

define('root_best', {
  agg: 'agg_root_best',
  fromAgg() {
    const T = 'agg_root_best';
    if (!has(T, 'engine_id', 'move', 'searches')) return null;
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${aggCore(T)} AS core_type, CAST(move AS VARCHAR) AS move,
        ${col(T, 'searches')} AS searches
      FROM ${src(T)} WHERE engine_id IS NOT NULL AND move IS NOT NULL`;
  },
  live() {
    if (!has('search_stats', 'move', 'engine_id')) return null;
    return `SELECT engine_id, 'all' AS core_type, move, count(*)::DOUBLE AS searches FROM ${src('search_stats')}
      WHERE move IS NOT NULL AND engine_id IS NOT NULL GROUP BY ALL`;
  },
});

/* the engine's chosen move against Stockfish's lines (sf_rank 0: outside them; loss_bucket NULL: loss unknown) */
define('sf_moves', {
  agg: 'agg_sf_moves',
  fromAgg() {
    const T = 'agg_sf_moves';
    if (!has(T, 'engine_id', 'sf_rank', 'searches')) return null;
    const c = name => col(T, name) ?? 'NULL::DOUBLE';
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${aggCore(T)} AS core_type, ${aggPhase(T)} AS phase,
        CAST(sf_rank AS INTEGER) AS sf_rank, ${pick(T, 'loss_bucket') ? 'TRY_CAST(loss_bucket AS INTEGER)' : 'NULL::INTEGER'} AS loss_bucket,
        ${c('searches')} AS searches, ${c('loss_sum')} AS loss_sum, ${c('loss_sq_sum')} AS loss_sq_sum, ${c('lines')} AS lines,
        ${c('depth_min')} AS depth_min, ${c('depth_max')} AS depth_max
      FROM ${src(T)} WHERE engine_id IS NOT NULL AND sf_rank IS NOT NULL`;
  },
});

/* root nodes spent on Stockfish's line 1..N (sf_rank 0: every other root move); `searches` is each group's denominator */
define('sf_root_effort', {
  agg: 'agg_sf_root_effort',
  fromAgg() {
    const T = 'agg_sf_root_effort';
    if (!has(T, 'engine_id', 'sf_rank', 'searches', 'node_share_sum')) return null;
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${aggCore(T)} AS core_type, ${aggPhase(T)} AS phase,
        CAST(sf_rank AS INTEGER) AS sf_rank, ${col(T, 'searches')} AS searches, ${col(T, 'moves') ?? 'NULL::DOUBLE'} AS moves,
        ${col(T, 'nodes') ?? 'NULL::DOUBLE'} AS nodes, ${col(T, 'node_share_sum')} AS node_share_sum
      FROM ${src(T)} WHERE engine_id IS NOT NULL AND sf_rank IS NOT NULL`;
  },
  derive: body => `SELECT *, node_share_sum / nullif(searches, 0) AS node_share FROM ${body}`,
});

/* where Stockfish's first-line move stood in the engine's root ordering (10: 10 or later, -1: not searched) */
define('sf_root_order', {
  agg: 'agg_sf_root_order',
  fromAgg() {
    const T = 'agg_sf_root_order';
    if (!has(T, 'engine_id', 'move_index', 'searches')) return null;
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${aggCore(T)} AS core_type, ${aggPhase(T)} AS phase,
        CAST(move_index AS INTEGER) AS move_index, ${col(T, 'searches')} AS searches
      FROM ${src(T)} WHERE engine_id IS NOT NULL AND move_index IS NOT NULL`;
  },
  derive: body => `SELECT *, searches / nullif(sum(searches) OVER (PARTITION BY engine_id, core_type, phase), 0) AS share
    FROM ${body}`,
});

/*
 * The positions where the chosen move lost the most (per-search data: full snapshots and local files only).
 * Keeps the worst 25 (engine, position, move) per engine, core type and phase, which contains the worst 25 of any
 * selection of them; Stockfish's lines come along as "rank:move:eval" items.
 */
define('sf_misses', {
  live() {
    const T = searchTable();
    const move = T ? pick(T, 'move', 'best_move') : null;
    if (!move || !has(T, 'fen', 'sf_cp_loss') || !has('stockfish_moves', 'fen', 'move', 'rank', 'eval')) return null;
    const env = envOf('m', T);
    const pf = has(T, 'search_id') && has('position_features', 'search_id', 'game_phase');
    const c = (name, type) => (has(T, name) ? `TRY_CAST(m.${name} AS ${type})` : `NULL::${type}`);
    return `WITH m AS (
        SELECT m.engine_id, ${env.core} AS core_type,
          ${pf ? `CASE WHEN pf.game_phase = 'middlegame' THEN 'midgame' ELSE coalesce(pf.game_phase, 'unknown') END` : `'unknown'`} AS phase,
          m.fen, CAST(m.${move} AS VARCHAR) AS move, m.sf_cp_loss::DOUBLE AS loss, ${c('sf_move_rank', 'INTEGER')} AS sf_rank,
          ${c('sf_move_eval', 'DOUBLE')} AS move_eval, ${c('sf_multipv', 'INTEGER')} AS lines
        FROM ${src(T)} m ${env.join} ${pf ? `LEFT JOIN ${src('position_features')} pf ON pf.search_id = m.search_id` : ''}
        WHERE m.engine_id IS NOT NULL AND m.fen IS NOT NULL AND m.sf_cp_loss > 0),
      g AS (SELECT engine_id, core_type, phase, fen, move, max(loss) AS loss, max(sf_rank) AS sf_rank,
          max(move_eval) AS move_eval, max(lines) AS lines, count(*)::DOUBLE AS searches
        FROM m GROUP BY engine_id, core_type, phase, fen, move),
      t AS (SELECT * FROM g QUALIFY row_number() OVER (PARTITION BY engine_id, core_type, phase ORDER BY loss DESC, fen, move) <= 25),
      l AS (SELECT s.fen, string_agg(CAST(s.rank AS VARCHAR) || ':' || s.move || ':' || coalesce(CAST(s.eval AS VARCHAR), ''), ','
            ORDER BY s.rank) AS sf_lines
        FROM ${src('stockfish_moves')} s WHERE s.rank IS NOT NULL AND s.fen IN (SELECT fen FROM t) GROUP BY s.fen)
      SELECT t.*, l.sf_lines FROM t LEFT JOIN l ON l.fen = t.fen`;
  },
});
