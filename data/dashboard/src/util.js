/**
 * Shared utilities for building dashboard views.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';

// Chart color palette
export const COLORS = ['#818cf8', '#34d399', '#f472b6', '#fbbf24', '#22d3ee', '#fb923c'];

/**
 * Run a SQL query and return results as an array of plain objects.
 */
export async function sql(query) {
  const result = await coordinator().query(query);
  return Array.from(result);
}

/**
 * Create a DOM element with optional class and children.
 */
export function el(tag, attrs = {}, ...children) {
  const elem = document.createElement(tag);
  for (const [key, val] of Object.entries(attrs)) {
    if (key === 'class') elem.className = val;
    else if (key === 'style' && typeof val === 'object') {
      Object.assign(elem.style, val);
    } else if (key.startsWith('on')) {
      elem.addEventListener(key.slice(2).toLowerCase(), val);
    } else {
      elem.setAttribute(key, val);
    }
  }
  for (const child of children) {
    if (typeof child === 'string') elem.appendChild(document.createTextNode(child));
    else if (child) elem.appendChild(child);
  }
  return elem;
}

/**
 * Create a panel wrapper.
 */
export function panel(title, ...children) {
  return el('div', { class: 'panel' },
    el('div', { class: 'panel-title' }, title),
    ...children
  );
}

/**
 * Create a metric card.
 */
export function metricCard(value, label) {
  return el('div', { class: 'metric-card' },
    el('div', { class: 'value' }, String(value)),
    el('div', { class: 'label' }, label),
  );
}

/**
 * Create a summary stat (smaller than metric card).
 */
export function summaryStat(value, label) {
  return el('div', { class: 'summary-stat' },
    el('div', { class: 'stat-value' }, String(value)),
    el('div', { class: 'stat-label' }, label),
  );
}

/**
 * Create a summary row from an object of { label: value }.
 */
export function summaryRow(stats) {
  const row = el('div', { class: 'summary-row' });
  for (const [label, value] of Object.entries(stats)) {
    row.appendChild(summaryStat(fmt(value), label));
  }
  return row;
}

/**
 * Create a controls bar with Mosaic inputs and/or native selects.
 */
export function controlsBar(...children) {
  const bar = el('div', { class: 'controls-bar' });
  for (const child of children) {
    if (child) bar.appendChild(child);
  }
  return bar;
}

/**
 * Create a labeled control group (label + element).
 */
export function controlGroup(label, input) {
  return el('div', { class: 'control-group' },
    el('label', {}, label),
    input,
  );
}

/**
 * Create a native <select> that triggers a callback on change.
 * options: [{value, label}]
 */
export function nativeSelect(options, onChange, defaultValue) {
  const select = el('select', {});
  for (const opt of options) {
    const option = el('option', { value: opt.value }, opt.label || opt.value);
    if (opt.value === defaultValue) option.selected = true;
    select.appendChild(option);
  }
  select.addEventListener('change', () => onChange(select.value));
  return select;
}

/**
 * Format a number with commas.
 */
export function fmt(n) {
  if (n == null) return '—';
  const num = Number(n);
  if (isNaN(num)) return '—';
  if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(1) + 'M';
  if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(1) + 'K';
  if (Number.isInteger(num)) return num.toLocaleString();
  return num.toFixed(2);
}

/**
 * Create a container for a Mosaic vgplot element.
 */
export function plotPanel(title, plot) {
  const container = el('div', { class: 'panel' },
    el('div', { class: 'panel-title' }, title),
    el('div', { class: 'plot-container' }),
  );
  container.querySelector('.plot-container').appendChild(plot);
  return container;
}

/**
 * Check if a table exists in the current database.
 */
export async function hasTable(name) {
  return (window.__tables || []).includes(name);
}

/**
 * Get the search table name (search_features or search_stats).
 */
export function getSearchTable() {
  const tables = window.__tables || [];
  if (tables.includes('search_features')) return 'search_features';
  if (tables.includes('search_stats')) return 'search_stats';
  return null;
}

/**
 * Get the iteration table name.
 */
export function getIterTable() {
  const tables = window.__tables || [];
  if (tables.includes('search_iteration_features')) return 'search_iteration_features';
  if (tables.includes('iterative_deepening_stats')) return 'iterative_deepening_stats';
  return null;
}

/**
 * Get the tree table name.
 */
export function getTreeTable() {
  const tables = window.__tables || [];
  if (tables.includes('search_tree_features')) return 'search_tree_features';
  if (tables.includes('search_tree_stats')) return 'search_tree_stats';
  return null;
}
