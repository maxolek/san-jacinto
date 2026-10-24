/**
 * Datasets: every chart reads a small temporary table built once per session.
 * A dataset comes from the snapshot's pre-aggregated table (agg_*) when it exists and has the
 * needed columns, otherwise it is computed in the browser from the analytics views/facts.
 */
import { query, exec, has, pick, schema, src, quoteIdent } from './db.js';

const registry = new Map();
/** name -> { table, origin, rows, ms, columns, notes } for built datasets. */
export const datasetInfo = new Map();
const DEFS = {};

export function define(name, def) {
  DEFS[name] = def;
}

/** Build (once) and return info about a dataset, or null when the snapshot cannot provide it. */
export function dataset(name) {
  if (!registry.has(name)) {
    registry.set(name, build(name).catch(error => {
      registry.delete(name);
      throw error;
    }));
  }
  return registry.get(name);
}

async function build(name) {
  const def = DEFS[name];
  if (!def) throw new Error(`Unknown dataset ${name}`);
  const t0 = performance.now();
  const deps = {};
  for (const dep of def.deps ?? []) deps[dep] = await dataset(dep);
  let plan = null;
  if (def.agg && schema.has(def.agg) && def.fromAgg) {
    const sql = def.fromAgg(deps);
    if (sql) plan = { sql, origin: `pre-aggregated in the snapshot (${def.agg})`, fromAgg: true };
  }
  if (!plan && def.live) {
    const sql = await def.live(deps);
    if (sql) plan = { sql, origin: def.liveOrigin ?? 'computed in your browser from the snapshot', fromAgg: false };
  }
  if (!plan) return null;
  const table = `ds_${name}`, raw = `${table}_raw`;
  const notes = [];
  await exec(`CREATE OR REPLACE TEMP TABLE ${raw} AS ${plan.sql}`);
  if (plan.fromAgg && def.rollup) {
    for (const dim of def.rollup.dims ?? ['core_type']) {
      const keys = def.rollup.keys.filter(k => k !== dim);
      if (await ensureAllRows(raw, keys, def.rollup.sums, dim)) notes.push(`'${dim} = all' rows summed in the browser`);
    }
  }
  const body = def.engine === false ? `SELECT * FROM ${raw}`
    : `SELECT t.*, e.label AS engine, e.ord AS engine_ord FROM ${raw} t LEFT JOIN ds_engines e ON e.id = t.engine_id`;
  await exec(`CREATE OR REPLACE TEMP TABLE ${table} AS ${def.derive ? def.derive(`(${body})`) : body}`);
  await exec(`DROP TABLE IF EXISTS ${raw}`);
  if (def.post) notes.push(...((await def.post(table, plan, deps)) ?? []));
  const [{ n }] = await query(`SELECT count(*)::DOUBLE AS n FROM ${table}`);
  const cols = await query(`SELECT column_name FROM duckdb_columns() WHERE table_name = '${table}' AND database_name = 'temp'`);
  const info = { name, table, origin: plan.origin, fromAgg: plan.fromAgg, rows: n, ms: performance.now() - t0,
    columns: new Set(cols.map(c => c.column_name)), notes };
  datasetInfo.set(name, info);
  return info;
}

/* ------------------------------------------------------------------ SQL helpers */

const num = expr => `TRY_CAST(${expr} AS DOUBLE)`;

/** Column expression for the first matching candidate, as DOUBLE, or null. */
export function col(table, ...candidates) {
  const c = pick(table, ...candidates);
  return c ? num(quoteIdent(c)) : null;
}

/** SELECT list from { out: expression-or-null }; required outputs must resolve. */
export function selectList(map, required = []) {
  for (const r of required) if (!map[r]) return null;
  return Object.entries(map).map(([out, expr]) => `${expr ?? 'NULL::DOUBLE'} AS ${quoteIdent(out)}`).join(', ');
}

/** The table holding search-level metrics (current views first, legacy compatibility view last). */
export function searchTable() {
  return schema.has('search_metrics') ? 'search_metrics' : schema.has('search_features') ? 'search_features' : null;
}

/** Core type the way the pipeline labels it: P, E, - (unpinned), mixed (engine left its core) or unknown. */
function coreExpr(coreCol, mixedCol) {
  const base = `coalesce(nullif(${coreCol}, ''), 'unknown')`;
  return mixedCol ? `CASE WHEN ${mixedCol} = 1 THEN 'mixed' ELSE ${base} END` : base;
}

/** Core type and clock of the game a search belongs to, from the view or through game_stats. */
export function envOf(searchAlias, table) {
  const inView = has(table, 'env_core_type');
  const inGames = has('game_stats', 'env_core_type');
  const join = !inView && inGames ? `LEFT JOIN ${src('game_stats')} g_env ON g_env.id = ${searchAlias}.game_id` : '';
  const a = inView ? searchAlias : 'g_env';
  const mixed = inView ? (has(table, 'env_core_mixed') ? `${a}.env_core_mixed` : null) : has('game_stats', 'env_core_mixed') ? `${a}.env_core_mixed` : null;
  const core = inView || inGames ? coreExpr(`${a}.env_core_type`, mixed) : `'unknown'`;
  const mhz = has(table, 'env_mhz_mean') ? `${searchAlias}.env_mhz_mean`
    : has('game_stats', 'env_mhz_mean') && join ? 'g_env.env_mhz_mean' : 'NULL::DOUBLE';
  return { join, core, mhz };
}

/** Core type of a game in game_stats (alias g). */
export const gameCore = () => (has('game_stats', 'env_core_type')
  ? coreExpr('g.env_core_type', has('game_stats', 'env_core_mixed') ? 'g.env_core_mixed' : null) : `'unknown'`);

/** Adds rows with core_type 'all' (sums over core types) when a pre-aggregated table has none. */
export async function ensureAllRows(table, keys, sums, dim = 'core_type') {
  const [{ n }] = await query(`SELECT count(*)::DOUBLE AS n FROM ${table} WHERE ${dim} = 'all'`);
  if (n > 0) return false;
  const k = keys.map(quoteIdent).join(', ');
  const s = sums.map(c => `sum(${quoteIdent(c)}) AS ${quoteIdent(c)}`).join(', ');
  await exec(`INSERT INTO ${table} BY NAME SELECT ${k}, 'all' AS ${dim}, ${s} FROM ${table} GROUP BY ALL`);
  return true;
}

export const QUANTILES = [[10, 0.1], [25, 0.25], [50, 0.5], [75, 0.75], [90, 0.9]];
export const quantileCols = (expr, name) => QUANTILES.map(([k, p]) => `quantile_cont(${expr}, ${p}) AS ${name}_p${k}`).join(', ');

/**
 * Histogram binning of the live fallback, using agg_hist's metric names:
 * metric -> [expression over the base rows, kind, parameter]. Log bins use 20 per decade.
 */
export const HIST_BINS = {
  depth: ['depth', 'lin', 1],
  nodes: ['nodes', 'log', 20],
  time_ms: ['time_ms', 'log', 20],
  nps: ['nps', 'log', 40],
  eval_cp: ['greatest(-1000, least(999, eval_cp))', 'lin', 50],
  tt_hit_rate: ['tt_hit_rate', 'lin', 0.01],
  qratio: ['least(qratio, 0.999)', 'lin', 0.025],
  fh_first_rate: ['least(fh_first_rate, 0.999)', 'lin', 0.025],
  fail_high_rate: ['fail_high_rate', 'lin', 0.002],
};

export function binSelect(metric, from) {
  const [expr, kind, p] = HIST_BINS[metric];
  const lo = kind === 'lin' ? `floor((${expr}) / ${p}) * ${p}` : `pow(10, floor(log10(${expr}) * ${p}) / ${p})`;
  const hi = kind === 'lin' ? `(floor((${expr}) / ${p}) + 1) * ${p}` : `pow(10, (floor(log10(${expr}) * ${p}) + 1) / ${p})`;
  const cond = kind === 'lin' ? `(${expr}) IS NOT NULL` : `(${expr}) > 0`;
  return `SELECT engine_id, core_type, '${metric}' AS metric, ${lo} AS bin_lo, ${hi} AS bin_hi, count(*)::DOUBLE AS count,
      '${kind === 'log' ? 'log' : 'linear'}' AS scale
    FROM ${from} WHERE ${cond} GROUP BY ALL`;
}

/** Rolls rows up to core_type 'all' with GROUPING SETS; `keys` excludes engine_id and core_type. */
export function withAllCores(keys) {
  const k = keys.length ? ', ' + keys.join(', ') : '';
  return {
    core: `CASE WHEN grouping(core_type) = 1 THEN 'all' ELSE core_type END AS core_type`,
    groupBy: `GROUP BY GROUPING SETS ((engine_id, core_type${k}), (engine_id${k}))`,
  };
}
