/** Relationships: how search metrics move together, per engine, with a linked scatter and brushing. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { CORR_METRICS } from '../defs-engine.js';
import { engineById, engines, engineScoped, onEngines, selectedEngines } from '../engines.js';
import { divergingColor, DIVERGING, textOn } from '../palette.js';
import { chartWidth, checkbox, coreScopeNote, el, emptyNote, engineColors, engineLegend, intTicks, panel, plot, reactive, releaseCharts,
  segmented, selectInput, sourceNote } from '../ui.js';

export const METRIC_LABELS = {
  depth: 'Depth', seldepth: 'Selective depth', nodes: 'Nodes', time_ms: 'Time (ms)', nps: 'Nodes per second', ebf: 'Branching factor', abs_eval_cp: 'Absolute eval',
  eval_cp: 'Eval', tt_hit_rate: 'TT hit rate', qratio: 'Quiescence share', fh_first_rate: 'First-move cutoffs', fail_low_rate: 'Fail-low rate',
  fail_high_rate: 'Fail-high rate', nmp_rate: 'Null-move rate', nmp_success_rate: 'Null-move success', ply: 'Game ply',
  mhz: 'Clock (MHz)', fail_high_researches: 'Fail-high re-searches', fail_low_researches: 'Fail-low re-searches',
};
const LOG = new Set(['nodes', 'time_ms']);
const label = m => METRIC_LABELS[m] ?? m.replaceAll('_', ' ');

export async function render() {
  const frag = document.createDocumentFragment();
  const [corr, sample, points] = await Promise.all([dataset('corr'), dataset('sample'), dataset('sample_points')]);
  if (!sample) {
    frag.append(panel({ body: emptyNote('This snapshot has no search-level data.') }));
    return frag;
  }
  frag.append(el('p', { class: 'tab-intro' },
    'Which search measures move together? The matrix shows rank correlations for one engine (or how they differ from the baseline). ',
    'Click a cell to plot that pair below, then drag across the scatter to filter the charts beside it.'));

  const opts = { coef: 'spearman', engine: null, diff: false, x: null, y: null };
  const sampleMetrics = CORR_METRICS.filter(m => sample.columns.has(m));
  opts.x = sampleMetrics.includes('time_ms') ? 'time_ms' : sampleMetrics[0];
  opts.y = sampleMetrics.includes('nodes') ? 'nodes' : sampleMetrics[1];
  const subs = new Set();
  const changed = () => subs.forEach(fn => fn());

  if (corr) {
    const engineSelect = el('span');
    const drawSelect = () => {
      const sel = selectedEngines();
      if (!sel.some(e => e.id === opts.engine)) opts.engine = sel[sel.length - 1]?.id ?? null;
      engineSelect.replaceChildren(selectInput({ label: 'Engine', value: opts.engine,
        options: sel.map(e => ({ value: e.id, label: e.label })), onChange: v => { opts.engine = Number(v); changed(); } }));
    };
    drawSelect();
    onEngines(drawSelect);
    const matrix = el('div');
    reactive(matrix, async () => {
      const rows = await query(`SELECT engine_id, metric_x, metric_y, n, ${opts.coef} AS r FROM ds_corr
        WHERE engine_id IN (${opts.engine ?? 'NULL'}${opts.diff && engines.baseline != null ? `, ${engines.baseline}` : ''})`);
      if (!rows.length) return emptyNote('No correlations for this engine.');
      const present = new Set(rows.flatMap(r => [r.metric_x, r.metric_y]));
      const ms = [...CORR_METRICS.filter(m => present.has(m)), ...[...present].filter(m => !CORR_METRICS.includes(m)).sort()];
      const get = (id, a, b) => (a === b ? 1 : rows.find(r => r.engine_id === id && r.metric_x === a && r.metric_y === b)?.r);
      const diff = opts.diff && engines.baseline !== opts.engine;
      const value = (a, b) => {
        const r = get(opts.engine, a, b);
        if (!diff) return r;
        const rb = get(engines.baseline, a, b);
        return r != null && rb != null ? r - rb : null;
      };
      const table = el('table', { class: 'matrix' },
        el('thead', {}, el('tr', {}, el('th'), ms.map(m => el('th', { scope: 'col' }, label(m))))),
        el('tbody', {}, ms.map(a => el('tr', {}, el('th', { scope: 'row' }, label(a)), ms.map(b => {
          const v = value(a, b);
          const diag = a === b;
          const bg = diag || v == null ? 'transparent' : divergingColor(v, diff ? 0.5 : 1);
          const td = el('td', {
            title: diag ? label(a) : `${label(a)} and ${label(b)}: ${v == null ? 'no data' : v.toFixed(2)}`,
            class: diag ? 'diag' : (a === opts.x && b === opts.y) || (a === opts.y && b === opts.x) ? 'current' : null,
            style: { background: bg, color: v == null ? 'var(--ink-3)' : textOn(bg === 'transparent' ? '#F6F3EC' : bg) },
            onclick: () => {
              if (diag || !sample.columns.has(a) || !sample.columns.has(b)) return;
              opts.x = b; opts.y = a; changed();
            },
          }, diag ? '' : v == null ? '\u00b7' : v.toFixed(2).replace('-', '\u2212'));
          return td;
        })))));
      const key = el('div', { class: 'scale-key' }, diff ? '\u22120.5' : '\u22121',
        el('span', { class: 'bar', style: { background: `linear-gradient(90deg, ${DIVERGING.join(', ')})` } }), diff ? '+0.5' : '+1',
        el('span', {}, diff ? `difference from ${engineById(engines.baseline)?.label}` : `${opts.coef === 'spearman' ? 'Spearman rank' : 'Pearson'} correlation`));
      return [table, key];
    }, [fn => subs.add(fn)]);
    frag.append(panel({
      title: 'Correlation matrix',
      caption: 'Each cell is the correlation between two measures across one engine\u2019s searches (oxblood: they rise together, blue: one falls as the other rises, pale: little relation), or, with \u201cdifference\u201d ticked, how that correlation changed against the baseline.',
      controls: [
        engineSelect,
        segmented({ label: 'Coefficient', value: opts.coef, options: [{ value: 'spearman', label: 'Spearman' }, { value: 'pearson', label: 'Pearson' }],
          onChange: v => { opts.coef = v; changed(); } }),
        checkbox({ label: 'Difference from baseline', checked: false, onChange: v => { opts.diff = v; changed(); } }),
      ],
      body: [coreScopeNote('This correlation matrix is'), matrix], note: sourceNote(corr),
    }));
  }

  const $rel = engineScoped({ cross: true, core: sample.columns.has('core_type') });
  const scatter = el('div');
  const axisX = el('span'), axisY = el('span');
  const drawAxes = () => {
    const options = sampleMetrics.map(m => ({ value: m, label: label(m) }));
    axisX.replaceChildren(selectInput({ label: 'Across', options, value: opts.x, onChange: v => { opts.x = v; changed(); } }));
    axisY.replaceChildren(selectInput({ label: 'Up', options, value: opts.y, onChange: v => { opts.y = v; changed(); } }));
  };
  const drawScatter = () => {
    drawAxes();
    releaseCharts(scatter);
    const table = points ? 'ds_sample_points' : 'ds_sample';
    scatter.replaceChildren(engineLegend(), plot(
      vg.dot(vg.from(table, { filterBy: $rel }), { x: opts.x, y: opts.y, fill: 'engine', r: 1.6, fillOpacity: 0.45 }),
      vg.intervalXY({ as: $rel, brush: { fill: 'none', stroke: 'currentColor', strokeDasharray: '3,3' } }),
      ...engineColors(), vg.xScale(LOG.has(opts.x) ? 'log' : 'linear'), vg.yScale(LOG.has(opts.y) ? 'log' : 'linear'),
      LOG.has(opts.x) ? vg.xTicks(6) : null, LOG.has(opts.y) ? vg.yTicks(6) : null,
      vg.xGrid(true), vg.yGrid(true), vg.xLabel(label(opts.x)), vg.yLabel(label(opts.y)),
      vg.width(chartWidth()), vg.height(400)));
  };
  subs.add(drawScatter);
  frag.append(panel({
    title: 'Two measures against each other',
    caption: 'Each dot is one sampled search coloured by engine (a thinned subset is drawn; the charts below use the whole sample), and dragging a rectangle makes the charts below describe only the searches inside it.',
    controls: [axisX, axisY], body: scatter,
  }));
  drawScatter();

  const linked = (title, caption, directives) => panel({ span: 'half', title, caption, body: plot(...directives) });
  if (sample.columns.has('depth')) {
    frag.append(linked('Depth of the selected searches', 'Completed depth of the searches inside the rectangle, stacked by engine.', [
      vg.rectY(vg.from('ds_sample', { filterBy: $rel }), { x: vg.bin('depth', { step: 1 }), y: vg.count(), fill: 'engine', inset: 0.5 }),
      ...engineColors(), vg.yGrid(true), intTicks(), vg.xLabel('Completed depth (plies)'), vg.yLabel('Searches'), vg.width(chartWidth('half')), vg.height(220)]));
  }
  if (sample.columns.has('phase')) {
    frag.append(linked('Game phase of the selected searches', 'How many of the selected searches came from each phase of the game, stacked by engine.', [
      vg.barX(vg.from('ds_sample', { filterBy: $rel }), { x: vg.count(), y: 'phase', fill: 'engine', inset: 0.5 }),
      ...engineColors(), vg.xGrid(true), vg.xLabel('Selected searches'), vg.yLabel(null), vg.marginLeft(90),
      vg.width(chartWidth('half')), vg.height(220)]));
  } else if (sample.columns.has('nps')) {
    frag.append(linked('Speed of the selected searches', 'Nodes per second of the searches inside the rectangle.', [
      vg.rectY(vg.from('ds_sample', { filterBy: $rel }), { x: vg.bin('nps'), y: vg.count(), fill: 'engine', inset: 0.5 }),
      ...engineColors(), vg.yGrid(true), vg.xLabel('Nodes per second'), vg.yLabel('Searches'), vg.width(chartWidth('half')), vg.height(220)]));
  }
  return frag;
}
