/** Shareable view state in the URL hash: #tab=compare&engines=1,3,5&base=3&cores=P&open=variation. */

const OPENING_MODES = ['family', 'variation'];
const CORE_TYPES = ['P', 'E', '-'];
/** Engines selected by default: one distinct colour each (the Okabe-Ito palette has eight). */
export const DEFAULT_ENGINE_LIMIT = 8;

export function parseHash(hash) {
  const params = new URLSearchParams(String(hash || '').replace(/^#/, ''));
  const ids = value => (value ?? '').split(',').map(Number).filter(n => Number.isSafeInteger(n) && n >= 0);
  const base = Number(params.get('base'));
  const open = params.get('open');
  const cores = params.get('cores');
  return {
    tab: params.get('tab') || null,
    engines: params.has('engines') ? ids(params.get('engines')) : null,
    base: params.has('base') && Number.isSafeInteger(base) ? base : null,
    cores: CORE_TYPES.includes(cores) ? cores : null,
    open: OPENING_MODES.includes(open) ? open : null,
  };
}

export function formatHash({ tab, engines, base, cores, open } = {}) {
  const params = new URLSearchParams();
  if (tab) params.set('tab', tab);
  if (engines?.length) params.set('engines', engines.join(','));
  if (base != null) params.set('base', String(base));
  if (cores && cores !== 'all') params.set('cores', cores);
  if (open && open !== 'family') params.set('open', open);
  const text = params.toString().replaceAll('%2C', ',');
  return text ? `#${text}` : '';
}

/** Every engine up to the limit, otherwise the newest ones (ids are registration order). */
export function defaultSelection(ids, max = DEFAULT_ENGINE_LIMIT) {
  const sorted = [...ids].sort((a, b) => a - b);
  return sorted.length <= max ? sorted : sorted.slice(-max);
}

/** Keep requested ids that exist, in id order; fall back to the default selection. */
export function sanitizeSelection(requested, available) {
  const have = new Set(available);
  const kept = [...new Set((requested ?? []).filter(id => have.has(id)))].sort((a, b) => a - b);
  return kept.length ? kept : defaultSelection(available);
}

/** The previous version of the newest selected engine, or the only one. */
export function defaultBaseline(selected) {
  const sorted = [...selected].sort((a, b) => a - b);
  return sorted.length > 1 ? sorted[sorted.length - 2] : sorted[0] ?? null;
}

export function sanitizeBaseline(requested, selected) {
  return selected.includes(requested) ? requested : defaultBaseline(selected);
}

export const CORE_LABELS = { all: 'All', P: 'P-cores', E: 'E-cores', '-': 'Unpinned' };

/**
 * Core types a view can be filtered by: the ones a game can be pinned to (P, E, unpinned) that occur.
 * 'mixed' (an engine left its core) and 'unknown' (no match-runner data) only count in All.
 */
export function pinnedCores(cores) {
  const present = new Set(cores);
  return CORE_TYPES.filter(c => present.has(c));
}

/** Filter options: All, then each pinned core type, or just All when there is nothing to choose between. */
export function coreOptions(cores) {
  const kinds = [...new Set(cores)].filter(c => c != null && c !== 'all');
  const options = [{ value: 'all', label: CORE_LABELS.all }];
  if (kinds.length < 2) return options;
  return options.concat(pinnedCores(cores).map(c => ({ value: c, label: CORE_LABELS[c] })));
}

/** Short explanation when a snapshot offers no core-type choice, otherwise null. */
export function coreNote(cores) {
  if (coreOptions(cores).length > 1) return null;
  const pinned = pinnedCores(cores);
  if (pinned.length === 1) return pinned[0] === '-' ? 'every game ran unpinned' : `every game ran on ${CORE_LABELS[pinned[0]]}`;
  return 'not recorded in this snapshot';
}
