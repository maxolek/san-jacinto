/** DOM helpers, number formatting, panels, controls and the shared chart theme. */
import { coordinator } from '@uwdata/mosaic-core';
import { Plot } from '@uwdata/mosaic-plot';
import * as vg from '@uwdata/vgplot';
import { el } from './dom.js';
import { engines, onEngines, selectedEngines } from './engines.js';
import { coreType, onCoreType } from './settings.js';
import { CORE_LABELS } from './state.js';

export { el, fmt } from './dom.js';

/* ------------------------------------------------------------------ layout pieces */

/** A titled panel with a one-sentence caption saying what it shows and how to read it. */
export function panel({ title, caption, controls, body, note, span = 'full', id } = {}) {
  return el('section', { class: `panel span-${span}`, id },
    el('header', { class: 'panel-head' },
      title ? el('h3', {}, title) : null,
      caption ? el('p', { class: 'caption' }, caption) : null),
    controls ? el('div', { class: 'controls' }, controls) : null,
    el('div', { class: 'panel-body' }, body),
    note ? el('p', { class: 'note' }, note) : null);
}

/** Big serif numbers with small captions separated by thin rules. */
export function statLine(items) {
  return el('dl', { class: 'stat-line' }, items.filter(Boolean).map(({ value, label, detail }) =>
    el('div', { class: 'stat' },
      el('dt', {}, label, detail ? el('span', { class: 'detail' }, detail) : null),
      el('dd', {}, value ?? '\u2014'))));
}

export const loading = (text = 'Loading\u2026') => el('div', { class: 'loading' }, text);
export const emptyNote = text => el('p', { class: 'empty' }, text);

export function sourceNote(info) {
  if (!info) return null;
  return `Data ${info.origin}${info.notes?.length ? '; ' + info.notes.join('; ') : ''}.`;
}

/* ------------------------------------------------------------------ controls */

export function selectInput({ label, options, value, onChange, id }) {
  const select = el('select', { id });
  for (const o of options) {
    const opt = el('option', { value: String(o.value) }, o.label ?? String(o.value));
    if (String(o.value) === String(value)) opt.selected = true;
    select.appendChild(opt);
  }
  select.addEventListener('change', () => onChange(select.value));
  return label ? el('label', { class: 'control' }, el('span', {}, label), select) : select;
}

export function segmented({ label, options, value, onChange }) {
  const group = el('div', { class: 'segmented', role: 'radiogroup', 'aria-label': label });
  const buttons = options.map(o => el('button', { type: 'button', role: 'radio', 'aria-checked': String(o.value === value),
    onclick: () => {
      buttons.forEach(b => b.setAttribute('aria-checked', String(b === btnFor(o))));
      onChange(o.value);
    } }, o.label));
  const btnFor = o => buttons[options.indexOf(o)];
  group.append(...buttons);
  return label ? el('div', { class: 'control' }, el('span', {}, label), group) : group;
}

export function checkbox({ label, checked, onChange }) {
  const input = el('input', { type: 'checkbox' });
  input.checked = Boolean(checked);
  input.addEventListener('change', () => onChange(input.checked));
  return el('label', { class: 'control check' }, input, el('span', {}, label));
}

/* ------------------------------------------------------------------ charts */

/** Width available to a chart in a panel spanning `span` of the content column. */
export function chartWidth(span = 'full') {
  const content = document.getElementById('content');
  const w = Math.min(1180, content?.clientWidth || 1100);
  const inner = span === 'half' ? (w - 24) / 2 : w;
  return Math.max(300, Math.floor(inner - 50));
}

/** Plot with the house style: light grid, thin marks, centred x label, labels without arrows. */
export function plot(...directives) {
  const p = new Plot();
  [vg.xLabelArrow('none'), vg.yLabelArrow('none'), vg.xLabelAnchor('center'),
    vg.marginTop(24), vg.marginLeft(58), vg.marginBottom(40), vg.marginRight(16), ...directives].flat().filter(Boolean).forEach(d => d(p));
  // marks drawing arrays need no database client; only table-backed marks join the coordinator
  for (const mark of p.marks) if (!mark.hasOwnData()) coordinator().connect(mark);
  p.update();
  return el('div', { class: 'chart' }, p.element);
}

/** Stop the database clients of charts inside `node` (used before a chart is replaced). */
export function releaseCharts(node) {
  for (const element of node.querySelectorAll('.plot')) {
    for (const mark of element.value?.marks ?? []) {
      if (mark.coordinator) coordinator().disconnect(mark);
    }
  }
}

/** Fixed engine colours for every chart (the domain covers all engines so colours never shift). */
export function engineColors() {
  return [vg.colorDomain(engines.all.map(e => e.label)), vg.colorRange(engines.all.map(e => e.color))];
}

/** Label only whole-number ticks (depths, plies, move indices). */
export const intTicks = (axis = 'x') => (axis === 'x' ? vg.xTickFormat : vg.yTickFormat)(d => (Number.isInteger(d) ? String(d) : ''));

/** Legend of the currently selected engines, kept in sync with the global selection. */
export function engineLegend() {
  const node = el('div', { class: 'legend' });
  const draw = () => node.replaceChildren(...selectedEngines().map(e =>
    el('span', { class: 'legend-item' }, el('i', { style: { background: e.color } }), e.label)));
  draw();
  onEngines(draw);
  return node;
}

/** Static legend for non-engine series. */
export function seriesLegend(items) {
  return el('div', { class: 'legend' }, items.map(({ label, color, dash }) =>
    el('span', { class: 'legend-item' },
      el('i', { class: dash ? 'dash' : '', style: dash ? { borderColor: color } : { background: color } }), label)));
}

/**
 * Redraw `container` whenever the engine selection (or another subscribed source) changes.
 * `draw` returns nodes; it should only build charts from arrays so old charts leave no live queries.
 */
export function reactive(container, draw, subscribe = []) {
  let token = 0;
  const run = async () => {
    const mine = ++token;
    try {
      const out = await draw();
      if (mine !== token) return;
      releaseCharts(container);
      container.replaceChildren(...[out].flat().filter(Boolean));
    } catch (error) {
      console.error(error);
      if (mine === token) container.replaceChildren(el('p', { class: 'error' }, `This chart could not be drawn: ${error.message}`));
    }
  };
  container.replaceChildren(loading());
  run();
  onEngines(run);
  onCoreType(run);
  for (const sub of subscribe) sub(run);
  return run;
}

/**
 * Simple table. columns: [{ label, value: row => node|string, num: bool, title }].
 */
export function dataTable(columns, rows, { rowClass } = {}) {
  return el('div', { class: 'table-wrap' }, el('table', { class: 'data-table' },
    el('thead', {}, el('tr', {}, columns.map(c => el('th', { class: c.num ? 'num' : null, title: c.title }, c.label)))),
    el('tbody', {}, rows.map(r => el('tr', { class: rowClass?.(r) }, columns.map(c => el('td', { class: c.num ? 'num' : null }, c.value(r))))))));
}

/** Coloured dot and label for an engine. */
export function engineName(engine) {
  return engine ? el('span', { class: 'engine-name' }, el('i', { class: 'swatch', style: { background: engine.color } }), engine.label)
    : '\u2014';
}

export { coreOptions } from './state.js';

/**
 * A short note for data that is not split by core type; it appears only while a core type is selected,
 * so nobody mistakes an all-games chart for a P-core one.
 */
export function coreScopeNote(what, applies = () => true) {
  const node = el('p', { class: 'tab-note span-full' });
  const update = () => {
    const c = coreType();
    node.hidden = c === 'all' || !applies();
    node.textContent = `${what} not split by core type, so ${what.startsWith('This') ? 'it covers' : 'they cover'} all core types, not only ${c === '-' ? 'unpinned games' : CORE_LABELS[c] ?? c}.`;
  };
  update();
  onCoreType(update);
  node.update = update;
  return node;
}
