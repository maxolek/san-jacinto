/** Game-level datasets: eval vs result, openings, game sides, SPRT runs. */
import { has, pick, schema, src } from './db.js';
import { dataset, define, col, gameCore } from './datasets.js';

const variationExpr = () => (has('game_stats', 'opening_variation') ? 'g.opening_variation' : 'NULL::VARCHAR');

/* ---------------------------------------------------------------- eval (side to move) vs the game result */
define('wdl', {
  agg: 'agg_eval_wdl',
  fromAgg() {
    const T = 'agg_eval_wdl';
    if (!has(T, 'engine_id', 'eval_bucket', 'wins', 'draws', 'losses')) return null;
    const core = pick(T, 'core_type') ? `coalesce(CAST(core_type AS VARCHAR), 'all')` : `'all'`;
    const ph = pick(T, 'phase', 'game_phase');
    const phase = ph ? `coalesce(CAST("${ph}" AS VARCHAR), 'unknown')` : `'all'`;
    const mate = pick(T, 'mate') ? 'coalesce(CAST(mate AS INTEGER), 0)' : '0';
    const positions = col(T, 'positions', 'n', 'count') ?? '(wins + draws + losses)::DOUBLE';
    const games = col(T, 'games', 'n_games') ?? 'NULL::DOUBLE';
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, ${core} AS core_type, ${phase} AS phase,
      TRY_CAST(eval_bucket AS DOUBLE) AS eval_bucket, ${mate} AS mate, ${positions} AS positions, ${games} AS games,
      wins::DOUBLE AS wins, draws::DOUBLE AS draws, losses::DOUBLE AS losses FROM ${src(T)}`;
  },
  rollup: { keys: ['engine_id', 'core_type', 'phase', 'eval_bucket', 'mate'], sums: ['positions', 'games', 'wins', 'draws', 'losses'],
    dims: ['core_type', 'phase'] },
  live() {
    if (!has('search_stats', 'eval', 'fen', 'game_id') || !has('game_stats', 'result')) return null;
    const pf = has('position_features', 'search_id', 'game_phase');
    return `WITH s AS (
        SELECT m.engine_id, ${gameCore()} AS core_type,
          ${pf ? `coalesce(pf.game_phase, 'unknown')` : `'unknown'`} AS phase, m.game_id, m.eval,
          split_part(m.fen, ' ', 2) = 'w' AS wtm, g.result
        FROM ${src('search_stats')} m JOIN ${src('game_stats')} g ON g.id = m.game_id
        ${pf ? `LEFT JOIN ${src('position_features')} pf ON pf.search_id = m.id` : ''}
        WHERE g.result IN ('white', 'black', 'draw') AND m.eval IS NOT NULL),
      c AS (SELECT *, CASE WHEN eval >= 99000 THEN 1 WHEN eval <= -99000 THEN -1 ELSE 0 END AS mate,
          CASE WHEN abs(eval) < 99000 THEN CAST(greatest(-1000, least(1000, round(eval / 25.0) * 25)) AS DOUBLE) END AS eval_bucket,
          result = CASE WHEN wtm THEN 'white' ELSE 'black' END AS win, result = 'draw' AS draw FROM s)
      SELECT engine_id, CASE WHEN grouping(core_type) = 1 THEN 'all' ELSE core_type END AS core_type,
        CASE WHEN grouping(phase) = 1 THEN 'all' ELSE phase END AS phase, eval_bucket, mate,
        count(*)::DOUBLE AS positions, count(DISTINCT game_id)::DOUBLE AS games,
        count(*) FILTER (WHERE win)::DOUBLE AS wins, count(*) FILTER (WHERE draw)::DOUBLE AS draws,
        count(*) FILTER (WHERE NOT win AND NOT draw)::DOUBLE AS losses
      FROM c GROUP BY GROUPING SETS ((engine_id, core_type, phase, eval_bucket, mate), (engine_id, phase, eval_bucket, mate),
        (engine_id, core_type, eval_bucket, mate), (engine_id, eval_bucket, mate))`;
  },
});

/* ---------------------------------------------------------------- openings from each engine's point of view */
define('openings', {
  agg: 'agg_openings',
  fromAgg() {
    const T = 'agg_openings';
    if (!has(T, 'engine_id', 'opening', 'games', 'wins', 'draws', 'losses')) return null;
    const v = pick(T, 'opening_variation', 'variation') ? `CAST("${pick(T, 'opening_variation', 'variation')}" AS VARCHAR)` : 'NULL::VARCHAR';
    const eco = pick(T, 'opening_eco', 'eco') ? `CAST("${pick(T, 'opening_eco', 'eco')}" AS VARCHAR)` : 'NULL::VARCHAR';
    return `SELECT CAST(engine_id AS INTEGER) AS engine_id, CAST(opening AS VARCHAR) AS opening, ${v} AS opening_variation,
      ${eco} AS opening_eco, games::DOUBLE AS games, wins::DOUBLE AS wins, draws::DOUBLE AS draws, losses::DOUBLE AS losses
      FROM ${src(T)} ${has(T, 'level') ? `WHERE level = 'variation' OR NOT EXISTS (SELECT 1 FROM ${src(T)} WHERE level = 'variation')` : ''}`;
  },
  live() {
    if (!has('game_stats', 'white_engine_id', 'black_engine_id', 'result', 'opening')) return null;
    const eco = has('game_stats', 'opening_eco') ? 'g.opening_eco' : 'NULL::VARCHAR';
    const side = (engine, winner, loser) => `SELECT g.${engine} AS engine_id, g.opening, ${variationExpr()} AS opening_variation,
        ${eco} AS opening_eco, CASE g.result WHEN '${winner}' THEN 1 WHEN 'draw' THEN 0.5 WHEN '${loser}' THEN 0 END AS pts
      FROM ${src('game_stats')} g`;
    return `WITH sides AS (${side('white_engine_id', 'white', 'black')} UNION ALL ${side('black_engine_id', 'black', 'white')})
      SELECT engine_id, opening, opening_variation, opening_eco, count(*)::DOUBLE AS games,
        count(*) FILTER (WHERE pts = 1)::DOUBLE AS wins, count(*) FILTER (WHERE pts = 0.5)::DOUBLE AS draws,
        count(*) FILTER (WHERE pts = 0)::DOUBLE AS losses
      FROM sides WHERE engine_id IS NOT NULL AND pts IS NOT NULL GROUP BY ALL`;
  },
});

/* ---------------------------------------------------------------- one row per (game, engine side) */
define('game_sides', {
  live() {
    if (!has('game_stats', 'id', 'white_engine_id', 'black_engine_id', 'result')) return null;
    const c = name => (has('game_stats', name) ? `g.${name}` : 'NULL');
    const side = (engine, other, color, winner, loser) => `SELECT g.id AS game_id, g.${engine} AS engine_id,
        g.${other} AS opponent_id, '${color}' AS color, g.result, ${c('termination')} AS termination,
        ${c('run_time_s')}::DOUBLE AS run_time_s, ${c('opening')} AS opening, ${variationExpr()} AS opening_variation,
        ${c('opening_eco')} AS opening_eco, ${gameCore()} AS core_type,
        CASE g.result WHEN '${winner}' THEN 1.0 WHEN 'draw' THEN 0.5 WHEN '${loser}' THEN 0.0 END AS score
      FROM ${src('game_stats')} g WHERE g.${engine} IS NOT NULL`;
    return `${side('white_engine_id', 'black_engine_id', 'white', 'white', 'black')} UNION ALL
      ${side('black_engine_id', 'white_engine_id', 'black', 'black', 'white')}`;
  },
});

/* ---------------------------------------------------------------- SPRT runs with their start time */
define('sprt', {
  engine: false,
  live() {
    if (!schema.has('sprt_runs')) return null;
    const exp = has('sprt_runs', 'experiment_id') && has('experiments', 'id', 'start_time_utc');
    const start = exp ? 'x.start_time_utc' : has('sprt_runs', 'ingestion_timestamp_utc') ? 'r.ingestion_timestamp_utc' : 'NULL::TIMESTAMP';
    return `SELECT r.*, epoch_ms(CAST(${start} AS TIMESTAMP))::DOUBLE AS start_ms
      FROM ${src('sprt_runs')} r ${exp ? `LEFT JOIN ${src('experiments')} x ON x.id = r.experiment_id` : ''}`;
  },
});

/* ---------------------------------------------------------------- mean search depth per opening */
define('opening_depth', {
  async live() {
    if (!has('game_stats', 'id', 'opening')) return null;
    if (has('search_stats', 'game_id', 'completed_depth')) {
      return `SELECT s.engine_id, ${gameCore()} AS core_type, g.opening, ${variationExpr()} AS opening_variation,
          count(*)::DOUBLE AS searches, avg(s.completed_depth) AS depth_mean
        FROM ${src('search_stats')} s JOIN ${src('game_stats')} g ON g.id = s.game_id
        WHERE g.opening IS NOT NULL AND g.opening <> '' GROUP BY ALL`;
    }
    // lite snapshots: estimate from the search sample (a random sample within each engine)
    const sample = await dataset('sample');
    if (!sample?.columns.has('game_id') || !sample.columns.has('depth')) return null;
    return `SELECT s.engine_id, ${sample.columns.has('core_type') ? 's.core_type' : `'unknown'`} AS core_type, g.opening,
        ${variationExpr()} AS opening_variation, count(*)::DOUBLE AS searches, avg(s.depth) AS depth_mean
      FROM ds_sample s JOIN ${src('game_stats')} g ON g.id = s.game_id
      WHERE g.opening IS NOT NULL AND g.opening <> '' GROUP BY ALL`;
  },
  liveOrigin: 'computed in your browser (from the search sample when the snapshot has no per-search tables)',
});
