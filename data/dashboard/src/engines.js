/**
 * Engines and the global engine selection. One Mosaic selection ($engines) filters every chart
 * by engine_id; a baseline engine is used for differences. Colours are fixed per engine version.
 */
import { Selection, clausePoints } from '@uwdata/mosaic-core';
import { and } from '@uwdata/mosaic-sql';
import { assignColors } from './palette.js';
import { coreType, onCoreType } from './settings.js';
import { defaultBaseline, sanitizeBaseline, sanitizeSelection } from './state.js';
import { exec, query, src, quoteString, has } from './db.js';

export const $engines = Selection.intersect();
const SOURCE = { reset() {} };
const listeners = new Set();

export const engines = {
  all: [],
  byId: new Map(),
  selected: [],
  baseline: null,
};

export async function loadEngines({ dark = false } = {}) {
  const hasName = has('engines', 'name'), hasVersion = has('engines', 'version');
  const rows = has('engines', 'id') ? await query(`SELECT id::DOUBLE AS id, ${hasName ? 'name' : 'NULL AS name'},
      ${hasVersion ? 'CAST(version AS VARCHAR)' : 'NULL'} AS version FROM ${src('engines')} ORDER BY id`) : [];
  if (!rows.length && has('search_stats', 'engine_id')) {
    rows.push(...await query(`SELECT DISTINCT engine_id::DOUBLE AS id, NULL AS name, NULL AS version
      FROM ${src('search_stats')} WHERE engine_id IS NOT NULL ORDER BY 1`));
  }
  engines.all = assignColors(rows, { dark });
  engines.byId = new Map(engines.all.map(e => [e.id, e]));
  const values = engines.all.map(e => `(${e.id}, ${quoteString(e.label)}, ${quoteString(e.color)}, ${e.order},
    ${e.name == null ? 'NULL' : quoteString(e.name)}, ${e.version == null ? 'NULL' : quoteString(e.version)})`);
  await exec(`CREATE OR REPLACE TEMP TABLE ds_engines (id INTEGER, label VARCHAR, color VARCHAR, ord INTEGER,
    name VARCHAR, version VARCHAR)`);
  if (values.length) await exec(`INSERT INTO ds_engines VALUES ${values.join(', ')}`);
  return engines.all;
}

export function setSelection(ids, baseline) {
  const selected = sanitizeSelection(ids, engines.all.map(e => e.id));
  engines.selected = selected;
  engines.baseline = sanitizeBaseline(baseline ?? engines.baseline, selected);
  $engines.update(clausePoints(['engine_id'], selected.map(id => [id]), { source: SOURCE }));
  for (const fn of listeners) fn(engines);
}

export function setBaseline(id) {
  engines.baseline = sanitizeBaseline(id, engines.selected);
  for (const fn of listeners) fn(engines);
}

export function toggleEngine(id) {
  const has = engines.selected.includes(id);
  const next = has ? engines.selected.filter(x => x !== id) : [...engines.selected, id];
  if (!next.length) return;
  setSelection(next, engines.baseline === id && has ? defaultBaseline(next) : engines.baseline);
}

/** Subscribe to selection or baseline changes; returns an unsubscribe function. */
export function onEngines(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export const selectedEngines = () => engines.selected.map(id => engines.byId.get(id)).filter(Boolean);
export const baselineEngine = () => engines.byId.get(engines.baseline) ?? null;
export const engineById = id => engines.byId.get(Number(id));

/** The version alone when every engine shares one name (tick labels), otherwise the full label. */
export function shortLabel(engine) {
  if (!engine) return '';
  const names = new Set(engines.all.map(e => e.name));
  return names.size === 1 && engine.version ? String(engine.version) : engine.label;
}
export const sqlIdList = () => (engines.selected.length ? engines.selected.join(', ') : 'NULL');

/** SQL predicate of the global engine selection, for queries that feed array-based charts. */
export function engineWhere(column = 'engine_id') {
  const predicates = $engines.predicate(null) ?? [];
  const sql = predicates.length ? String(and(predicates)) : 'TRUE';
  return column === 'engine_id' ? sql : sql.replaceAll('"engine_id"', column);
}

/**
 * A selection for one tab that also carries the global engine clause (and keeps receiving it).
 * With cross = true, brushes in a chart do not filter that chart itself. With core = true the
 * selection also follows the global core type (for tables with per-game or per-search core_type,
 * or with 'all' rows when allRows = true).
 */
export function engineScoped({ cross = false, core = false, allRows = false } = {}) {
  const sel = cross ? Selection.crossfilter({ include: [$engines] }) : Selection.intersect({ include: [$engines] });
  for (const clause of $engines.clauses) sel.update(clause);
  if (core) {
    const source = { reset() {} };
    const apply = () => {
      const c = coreType();
      const value = allRows || c !== 'all' ? [[c]] : null;
      sel.update(clausePoints(value ? ['core_type'] : [], value, { source }));
    };
    apply();
    onCoreType(apply);
  }
  return sel;
}

/** A clause source for per-tab filters (metric, core type, phase ...). */
export function filterSource(name) {
  return { name, reset() {} };
}

export { clausePoints };
