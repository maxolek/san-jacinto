/** Trends: how key metrics move from version to version, in registration order. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineById, engineWhere, shortLabel } from '../engines.js';
import { coreRows, coreType } from '../settings.js';
import { chartWidth, el, emptyNote, engineColors, panel, plot, reactive, selectInput, sourceNote } from '../ui.js';
import { fillSampleErrors, METRICS } from './compare.js';

const Z = 1.96;
const SMALL = ['Nodes per second', 'Mean depth', 'Median time', 'TT hit rate', 'First-move cutoffs', 'Quiescence share'];
// y axis title and scale factor per metric
const AXES = {
  'Nodes per second': ['Million nodes per second', 1e-6],
  'Mean depth': ['Mean completed depth (plies)', 1],
  'Median time': ['Median search time (ms)', 1],
  'Mean nodes': ['Mean nodes per search (millions)', 1e-6],
  'TT hit rate': ['TT hits per node (%)', 100],
  'First-move cutoffs': ['Cutoffs by the first move (%)', 100],
  'Quiescence share': ['Quiescence share of nodes (%)', 100],
  'Null-move success': ['Null moves that failed high (%)', 100],
};
const CAPTIONS = {
  'Nodes per second': 'Total nodes over total search time; the Hardware tab separates code speed from clock speed.',
  'Mean depth': 'Average completed iterative-deepening depth per search.',
  'Median time': 'Half of the searches took less than this.',
  'TT hit rate': 'Transposition-table hits per node searched.',
  'First-move cutoffs': 'Share of beta cutoffs caused by the first move tried; higher means better move ordering.',
  'Quiescence share': 'Share of all nodes spent in quiescence search.',
};

export async function render() {
  const frag = document.createDocumentFragment();
  const summary = await dataset('summary');
  if (!summary) {
    frag.append(panel({ body: emptyNote('This snapshot has no search data.') }));
    return frag;
  }
  frag.append(el('p', { class: 'tab-intro' },
    'Versions run left to right in the order they were registered, not alphabetically, so 1.10 follows 1.9. ',
    'Whiskers are 95% intervals where one can be computed.'));

  const load = async () => {
    const rows = await query(`SELECT * FROM ds_summary WHERE ${coreRows()} AND ${engineWhere()} ORDER BY engine_ord`);
    await fillSampleErrors(rows, coreType());
    return rows.map(r => ({ ...r, short: shortLabel(engineById(r.engine_id)) }));
  };
  const chart = (metric, rows, { width, height }) => {
    const [axis, scale] = AXES[metric.label] ?? [metric.label, 1];
    const data = rows.map(r => {
      const { v, se } = metric.get(r);
      const ok = Number.isFinite(se);
      return { engine: r.engine, short: r.short, v: v * scale, lo: ok ? (v - Z * se) * scale : null, hi: ok ? (v + Z * se) * scale : null,
        title: `${r.engine}: ${metric.format(v)}` };
    }).filter(d => Number.isFinite(d.v));
    if (!data.length) return emptyNote('Not available in this snapshot.');
    return plot(
      vg.lineY(data, { x: 'short', y: 'v', stroke: 'currentColor', strokeOpacity: 0.35, strokeWidth: 1 }),
      vg.ruleX(data.filter(d => d.lo != null), { x: 'short', y1: 'lo', y2: 'hi', stroke: 'engine', strokeWidth: 1.5 }),
      vg.dot(data, { x: 'short', y: 'v', fill: 'engine', r: 4, tip: true, title: 'title' }),
      ...engineColors(), vg.xDomain(data.map(d => d.short)), vg.yGrid(true), vg.xLabel(null), vg.yLabel(axis),
      vg.marginBottom(34), vg.width(width), vg.height(height));
  };

  let chosen = METRICS[0].label;
  const main = el('div');
  const redraw = reactive(main, async () => chart(METRICS.find(m => m.label === chosen), await load(), { width: chartWidth(), height: 300 }));
  frag.append(panel({
    title: 'One metric across versions',
    caption: 'Each dot is one version; the thin line only connects neighbours to guide the eye, it does not imply anything between versions.',
    controls: selectInput({ label: 'Metric', value: chosen, options: METRICS.map(m => ({ value: m.label, label: m.label })),
      onChange: v => { chosen = v; redraw(); } }),
    body: main, note: sourceNote(summary),
  }));

  for (const label of SMALL) {
    const metric = METRICS.find(m => m.label === label);
    const body = el('div');
    reactive(body, async () => chart(metric, await load(), { width: chartWidth('half'), height: 220 }));
    frag.append(panel({ span: 'half', title: label, caption: CAPTIONS[label], body }));
  }

  const volume = el('div');
  reactive(volume, async () => {
    const rows = await load();
    return plot(
      vg.barY(rows, { x: 'short', y: 'searches', fill: 'engine', tip: true }),
      ...engineColors(), vg.xDomain(rows.map(r => r.short)), vg.xPadding(0.55), vg.yGrid(true), vg.xLabel(null),
      vg.yLabel('Searches logged'), vg.marginBottom(34), vg.width(chartWidth()), vg.height(200));
  });
  frag.append(panel({ title: 'Data behind each version',
    caption: 'Searches logged per version; the fewer there are, the wider the whiskers above.', body: volume }));
  return frag;
}
