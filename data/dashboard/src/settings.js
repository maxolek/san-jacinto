/** UI settings shared by several tabs: how openings are grouped and which core type is shown. */
const listeners = new Set();
const settings = { openingMode: 'family', coreType: 'all', coreTypes: [] };

export const openingMode = () => settings.openingMode;
/** Core type every tab filters by ('all', 'P', 'E' or '-'). */
export const coreType = () => settings.coreType;
/** Core types the snapshot can be filtered by (empty when the snapshot has no core-type data). */
export const coreTypes = () => settings.coreTypes;

function emit(key) {
  for (const [k, fn] of listeners) if (k === key || k === '*') fn(settings);
}

export function setOpeningMode(mode) {
  if (mode !== 'family' && mode !== 'variation') return;
  settings.openingMode = mode;
  emit('openingMode');
}

export function setCoreTypes(types) {
  settings.coreTypes = types;
  if (settings.coreType !== 'all' && !types.includes(settings.coreType)) settings.coreType = 'all';
}

export function setCoreType(type) {
  const next = type === 'all' || settings.coreTypes.includes(type) ? type : 'all';
  if (next === settings.coreType) return;
  settings.coreType = next;
  emit('coreType');
}

/** Subscribe to one setting ('openingMode', 'coreType') or to all of them ('*'). */
export function onSetting(key, fn) {
  const entry = [key, fn];
  listeners.add(entry);
  return () => listeners.delete(entry);
}
export const onSettings = fn => onSetting('*', fn);
export const onCoreType = fn => onSetting('coreType', fn);
export const onOpeningMode = fn => onSetting('openingMode', fn);

/** SQL predicate for the current core type on a table with a core_type column that has 'all' rows. */
export const coreRows = (column = 'core_type') => `${column} = '${settings.coreType}'`;
/** SQL predicate for the current core type on per-game or per-search rows (no 'all' rows). */
export const coreFilter = (column = 'core_type') => (settings.coreType === 'all' ? 'TRUE' : `${column} = '${settings.coreType}'`);

/** SQL expression for the opening label in the current mode (table alias optional). */
export function openingLabelSQL(alias = '') {
  const a = alias ? `${alias}.` : '';
  return settings.openingMode === 'variation'
    ? `CASE WHEN ${a}opening_variation IS NULL OR ${a}opening_variation = '' THEN ${a}opening || ' (main line)'
        ELSE ${a}opening || ': ' || ${a}opening_variation END`
    : `${a}opening`;
}
