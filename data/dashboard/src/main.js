/**
 * San Jacinto dashboard: starts DuckDB-WASM, opens the published snapshot with HTTP range reads,
 * and renders tabs lazily. Each tab is built once and kept, so returning to it is instant.
 */
import { coordinator, wasmConnector } from '@uwdata/mosaic-core';
import * as duckdb from '@duckdb/duckdb-wasm';
import duckdb_wasm from '@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url';
import mvp_worker from '@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url';
import duckdb_wasm_eh from '@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url';
import eh_worker from '@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url';
import { duckdbConfig, resolveDataSource } from './data-source.js';
import { attach, has, query, schema, src } from './db.js';
import { dataset, datasetInfo, gameCore } from './datasets.js';
import './defs-search.js';
import './defs-results.js';
import './defs-engine.js';
import { engines, loadEngines, onEngines, setBaseline, setSelection, toggleEngine } from './engines.js';
import { coreType, onSettings, openingMode, setCoreType, setCoreTypes, setOpeningMode } from './settings.js';
import { CORE_LABELS, coreNote, coreOptions, formatHash, parseHash, pinnedCores } from './state.js';
import { el, fmt } from './dom.js';

const TABS = [
  { group: 'Results', id: 'overview', label: 'Overview', load: () => import('./views/overview.js') },
  { group: 'Results', id: 'games', label: 'Games', load: () => import('./views/games.js') },
  { group: 'Results', id: 'openings', label: 'Openings', load: () => import('./views/openings.js') },
  { group: 'Results', id: 'wdl', label: 'Eval vs WDL', load: () => import('./views/wdl.js') },
  { group: 'Results', id: 'sprt', label: 'SPRT', load: () => import('./views/sprt.js') },
  { group: 'Search', id: 'searches', label: 'Searches', load: () => import('./views/searches.js') },
  { group: 'Search', id: 'iterations', label: 'Iterations', load: () => import('./views/iterations.js') },
  { group: 'Search', id: 'tree', label: 'Tree depth', load: () => import('./views/tree.js') },
  { group: 'Search', id: 'pruning', label: 'Pruning', load: () => import('./views/pruning.js') },
  { group: 'Search', id: 'moveorder', label: 'Move order', load: () => import('./views/moveorder.js') },
  { group: 'Search', id: 'timemgmt', label: 'Time management', load: () => import('./views/timemgmt.js') },
  { group: 'Search', id: 'rootmoves', label: 'Root moves', load: () => import('./views/rootmoves.js') },
  { group: 'Engine', id: 'compare', label: 'Compare', load: () => import('./views/compare.js') },
  { group: 'Engine', id: 'trends', label: 'Trends', load: () => import('./views/trends.js') },
  { group: 'Engine', id: 'relationships', label: 'Relationships', load: () => import('./views/relationships.js') },
  { group: 'Engine', id: 'hardware', label: 'Hardware', load: () => import('./views/hardware.js') },
  { group: 'Engine', id: 'timing', label: 'Timing', load: () => import('./views/timing.js') },
  { group: 'Engine', id: 'positions', label: 'Positions', load: () => import('./views/positions.js') },
  { group: 'Engine', id: 'quality', label: 'Eval quality', load: () => import('./views/quality.js') },
];
// Small pre-aggregated tables copied into memory while the visitor reads the first tab.
const PREFETCH = [['summary', 'agg_engine_summary'], ['hist', 'agg_hist'], ['depth', 'agg_depth'], ['wdl', 'agg_eval_wdl'],
  ['openings', 'agg_openings'], ['moveorder', 'agg_moveorder'], ['tree', 'agg_tree_ply'], ['timing', 'agg_timing'],
  ['env', 'agg_env'], ['sample', 'search_sample'], ['corr', 'agg_corr']];

const state = { tab: 'overview', manifest: null, source: null, cores: [] };
const panels = new Map();
const dark = window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false;
// load milestones, readable from the console as __timings (ms since navigation start)
const timings = window.__timings = [];
const mark = name => timings.push([name, Math.round(performance.now())]);
window.__datasets = datasetInfo;

function setStatus(text, kind = '') {
  const s = document.getElementById('status');
  if (!s) return;
  s.textContent = text;
  s.className = kind;
}

async function startDuckDB() {
  const bundle = await duckdb.selectBundle({
    mvp: { mainModule: duckdb_wasm, mainWorker: mvp_worker },
    eh: { mainModule: duckdb_wasm_eh, mainWorker: eh_worker },
  });
  const worker = new Worker(bundle.mainWorker);
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  mark('engine instantiated');
  await db.open(duckdbConfig());
  mark('engine opened');
  coordinator().databaseConnector(wasmConnector({ duckdb: db }));
  coordinator().logger({ debug() {}, info() {}, log() {}, warn: console.warn, error: console.error });
  return db;
}

async function init() {
  const base = new URL(import.meta.env.BASE_URL, window.location.href);
  const hash = parseHash(window.location.hash);
  if (hash.tab && TABS.some(t => t.id === hash.tab)) state.tab = hash.tab;
  if (hash.open) setOpeningMode(hash.open);
  // start fetching the first tab's code (and the chart library) while the engine downloads
  TABS.find(t => t.id === state.tab)?.load().catch(() => {});
  try {
    setStatus('Starting the query engine\u2026');
    // the manifest request overlaps with downloading and compiling the engine
    const [source, db] = await Promise.all([
      resolveDataSource(window.location.href, base, { development: import.meta.env.DEV }),
      startDuckDB(),
    ]);
    window.__duckdb = db;
    if (!source) {
      setStatus('Choose a local database');
      showFilePicker(db);
      return;
    }
    state.manifest = source.manifest;
    state.source = source.url;
    setStatus('Opening the snapshot\u2026');
    await attach(source.url);
    mark('snapshot attached');
    await startDashboard(hash);
  } catch (error) {
    console.error('Init failed:', error);
    setStatus('The data could not be loaded', 'error');
    showLoadError(error.message);
  }
}

async function startDashboard(hash) {
  await loadEngines({ dark });
  mark('engines loaded');
  setSelection(hash.engines, hash.base);
  state.cores = await detectCoreTypes();
  setCoreTypes(pinnedCores(state.cores));
  if (hash.cores) setCoreType(hash.cores);
  showFreshness();
  renderEngineBar();
  renderNav();
  onEngines(() => { renderEngineBar(); writeHash(false); });
  onSettings(() => { renderEngineBar(); writeHash(false); });
  setStatus('');
  await switchTab(state.tab, { push: false });
  mark('first tab rendered');
  setTimeout(prefetch, 1500);
  setTimeout(() => TABS.forEach(t => t.load().catch(() => {})), 2500);
}

/** Core types in the snapshot, labelled the way the aggregates label them. */
async function detectCoreTypes() {
  try {
    if (has('agg_engine_summary', 'core_type')) {
      return (await query(`SELECT DISTINCT CAST(core_type AS VARCHAR) AS c FROM ${src('agg_engine_summary')}`)).map(r => r.c);
    }
    if (has('game_stats', 'env_core_type')) {
      return (await query(`SELECT DISTINCT ${gameCore()} AS c FROM ${src('game_stats')} g`)).map(r => r.c);
    }
  } catch (error) {
    console.warn('Core types could not be read:', error);
  }
  return [];
}

async function prefetch() {
  for (const [name, table] of PREFETCH) {
    if (!schema.has(table)) continue;
    try { await dataset(name); } catch { /* the tab reports the problem when it is opened */ }
  }
}

function showFreshness() {
  const node = document.getElementById('freshness');
  if (!node) return;
  const m = state.manifest;
  const parts = [];
  if (m) {
    parts.push(`Snapshot of ${fmt.date(Date.parse(m.exported_at))}`);
    if (m.row_counts?.search_stats) parts.push(`${fmt.int(m.row_counts.search_stats)} searches`);
    if (m.row_counts?.game_stats) parts.push(`${fmt.int(m.row_counts.game_stats)} games`);
  } else {
    parts.push(/^https?:/.test(state.source ?? '') ? 'Snapshot from a custom address' : 'Local file');
  }
  parts.push(`${engines.all.length} engine version${engines.all.length === 1 ? '' : 's'}`);
  node.textContent = parts.join(' \u00b7 ');
}

/* ------------------------------------------------------------------ engine selector */
function renderEngineBar() {
  const bar = document.getElementById('engine-bar');
  if (!bar) return;
  const chips = engines.all.map(e => {
    const on = engines.selected.includes(e.id);
    return el('button', { type: 'button', class: 'chip', 'aria-pressed': String(on),
      title: `${on ? 'Hide' : 'Show'} ${e.label}`, onclick: () => toggleEngine(e.id) },
    el('i', { style: { background: e.color } }), e.label);
  });
  const baseline = el('select', { id: 'baseline', 'aria-label': 'Baseline engine' }, engines.selected.map(id => {
    const option = el('option', { value: String(id) }, engines.byId.get(id).label);
    option.selected = id === engines.baseline;
    return option;
  }));
  baseline.addEventListener('change', () => setBaseline(Number(baseline.value)));
  const quick = [
    ['all', () => setSelection(engines.all.map(e => e.id), engines.baseline)],
    ['latest two', () => {
      const ids = engines.all.slice(-2).map(e => e.id);
      setSelection(ids, ids[0]);
    }],
  ].map(([label, fn]) => el('button', { type: 'button', class: 'link', onclick: fn }, label));
  const coreChoices = coreOptions(state.cores);
  const cores = coreChoices.length > 1
    ? el('div', { class: 'segmented', role: 'radiogroup', 'aria-label': 'Core type' }, coreChoices.map(o => el('button', {
      type: 'button', role: 'radio', 'aria-checked': String(o.value === coreType()), onclick: () => setCoreType(o.value),
      title: o.value === 'all' ? 'Every game, whatever core it ran on' : `Only games that ran on ${CORE_LABELS[o.value]}`,
    }, o.label)))
    : el('span', { class: 'bar-note' }, coreNote(state.cores));
  bar.replaceChildren(
    el('span', { class: 'bar-label' }, 'Engines'),
    el('div', { class: 'chips' }, chips),
    el('span', { class: 'quick' }, 'Show ', quick[0], ' or ', quick[1]),
    el('span', { class: 'cores' }, el('span', { class: 'bar-label' }, 'Cores'), cores),
    el('label', { class: 'baseline' }, el('span', {}, 'Baseline'), baseline));
}

/* ------------------------------------------------------------------ navigation */
function renderNav() {
  const nav = document.getElementById('tabs');
  const groups = [...new Set(TABS.map(t => t.group))];
  nav.replaceChildren(...groups.map(g => el('div', { class: 'nav-group' },
    el('span', { class: 'nav-label' }, g),
    TABS.filter(t => t.group === g).map(t => el('button', { type: 'button', 'data-tab': t.id,
      'aria-current': t.id === state.tab ? 'page' : null, onclick: () => switchTab(t.id) }, t.label)))));
}

function writeHash(push) {
  const hash = formatHash({ tab: state.tab, engines: engines.selected, base: engines.baseline, cores: coreType(),
    open: openingMode() });
  if (hash === window.location.hash) return;
  const url = window.location.pathname + window.location.search + hash;
  if (push) history.pushState(null, '', url);
  else history.replaceState(null, '', url);
}

async function switchTab(id, { push = true } = {}) {
  const tab = TABS.find(t => t.id === id) ?? TABS[0];
  state.tab = tab.id;
  document.querySelectorAll('#tabs button').forEach(b => {
    if (b.dataset.tab === tab.id) b.setAttribute('aria-current', 'page');
    else b.removeAttribute('aria-current');
  });
  writeHash(push);
  document.title = `${tab.label} \u00b7 San Jacinto`;
  for (const [key, node] of panels) node.hidden = key !== tab.id;
  if (panels.has(tab.id)) return;
  const node = el('div', { class: 'tab-panel', 'data-tab': tab.id }, el('div', { class: 'loading' }, 'Loading\u2026'));
  panels.set(tab.id, node);
  document.getElementById('content').appendChild(node);
  try {
    const mod = await tab.load();
    node.replaceChildren(await mod.render());
  } catch (error) {
    console.error(`Tab ${tab.id} failed:`, error);
    node.replaceChildren(el('section', { class: 'panel' },
      el('p', { class: 'error' }, `This tab could not be drawn: ${error.message}`)));
  }
}

window.addEventListener('popstate', () => {
  if (!panels.size) return;
  // a pasted or edited link: follow its engines, baseline, opening grouping and tab
  const hash = parseHash(window.location.hash);
  const same = (a, b) => a.length === b.length && [...a].sort((x, y) => x - y).every((x, i) => x === b[i]);
  if (hash.engines && (!same(hash.engines, engines.selected) || (hash.base != null && hash.base !== engines.baseline))) {
    setSelection(hash.engines, hash.base);
  } else if (hash.base != null && hash.base !== engines.baseline) {
    setBaseline(hash.base);
  }
  if (hash.open && hash.open !== openingMode()) setOpeningMode(hash.open);
  if ((hash.cores ?? 'all') !== coreType()) setCoreType(hash.cores ?? 'all');
  if (hash.tab && hash.tab !== state.tab) switchTab(hash.tab, { push: false });
});

/* ------------------------------------------------------------------ errors and local files */
function showLoadError(message) {
  const retry = el('button', { type: 'button', onclick: () => window.location.reload() }, 'Try again');
  const local = el('a', { href: `${import.meta.env.BASE_URL}?local=1` }, 'open a local database');
  document.getElementById('content').replaceChildren(el('section', { class: 'panel narrow' },
    el('h3', {}, 'The data could not be loaded'), el('p', {}, message), el('p', {}, retry, ' or ', local, '.')));
}

function showFilePicker(db) {
  const input = el('input', { type: 'file', accept: '.duckdb,.db' });
  const note = el('p', { class: 'caption' }, 'Pick an analytics DuckDB file. It is read in your browser and never uploaded. ',
    'You can also pass ', el('code', {}, '?db=https://\u2026/file.duckdb'), ' in the address.');
  document.getElementById('content').replaceChildren(el('section', { class: 'panel narrow' },
    el('h3', {}, 'Open a local database'), note, input));
  input.addEventListener('change', async () => {
    const file = input.files?.[0];
    if (!file) return;
    setStatus(`Opening ${file.name}\u2026`);
    try {
      // read slices on demand instead of copying the whole file into memory
      await db.registerFileHandle(file.name, file, duckdb.DuckDBDataProtocol.BROWSER_FILEREADER, true);
      state.source = file.name;
      await attach(file.name);
      document.getElementById('content').replaceChildren();
      await startDashboard(parseHash(window.location.hash));
    } catch (error) {
      console.error('File load failed:', error);
      setStatus(`Could not open ${file.name}`, 'error');
      note.textContent = error.message;
    }
  });
}

init();
