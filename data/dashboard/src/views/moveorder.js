/** Move order: which move in the ordering produced each beta cutoff. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineWhere } from '../engines.js';
import { coreType } from '../settings.js';
import { chartWidth, coreScopeNote, el, emptyNote, engineColors, engineLegend, fmt, intTicks, panel, plot, reactive, segmented,
  seriesLegend, sourceNote } from '../ui.js';

const BUCKETS = ['0', '1', '2', '3', '4-7', '8+'];
const BUCKET_LABELS = { 0: 'First move', 1: 'Second', 2: 'Third', 3: 'Fourth', '4-7': 'Fifth to eighth', '8+': 'Ninth or later' };
const BUCKET_COLORS = ['#7E2E1E', '#A8553A', '#C98652', '#DDB07A', '#EBD2A6', '#F4E9D2'];
const DIMENSIONS = [
  { value: 'iter_depth', label: 'Iteration depth', axis: 'Iteration depth (plies)' },
  { value: 'tree_ply', label: 'Tree ply', axis: 'Ply from the root' },
  { value: 'completed_depth', label: 'Completed depth', axis: 'Completed depth of the search (plies)' },
];

export async function render() {
  const frag = document.createDocumentFragment();
  const mo = await dataset('moveorder');
  if (!mo) {
    frag.append(panel({ body: emptyNote('This snapshot has no fail-high index counters.') }));
    return frag;
  }
  const [{ n: coreRowCount }] = await query(`SELECT count(*) AS n FROM ds_moveorder WHERE dimension = 'core_type'`);
  const scope = () => (coreType() !== 'all' && coreRowCount > 0 ? `dimension = 'core_type' AND dim_label = '${coreType()}'` : `dimension = 'all'`);
  const overall = el('div');
  reactive(overall, async () => {
    const rows = await query(`SELECT engine, engine_ord, bucket, sum(count) AS n FROM ds_moveorder
      WHERE ${scope()} AND ${engineWhere()} GROUP BY ALL ORDER BY engine_ord, bucket`);
    if (!rows.length) return emptyNote('No cutoffs recorded for this selection.');
    const totals = new Map();
    rows.forEach(r => totals.set(r.engine, (totals.get(r.engine) ?? 0) + r.n));
    rows.forEach(r => { r.title = `${r.engine}: ${BUCKET_LABELS[r.bucket]} ${fmt.pct(r.n / totals.get(r.engine))}`; });
    const first = rows.filter(r => r.bucket === '0').map(r => ({ engine: r.engine, text: fmt.pct(r.n / totals.get(r.engine)) }));
    return [seriesLegend(BUCKETS.map((b, i) => ({ label: BUCKET_LABELS[b], color: BUCKET_COLORS[i] }))), plot(
      vg.barX(rows, { x: 'n', y: 'engine', fill: 'bucket', offset: 'normalize', order: BUCKETS, insetTop: 6, insetBottom: 6, tip: true,
        title: 'title' }),
      vg.text(first, { y: 'engine', x: 0, text: 'text', dx: 6, textAnchor: 'start', fill: '#FBF9F4', fontWeight: 600 }),
      vg.colorDomain(BUCKETS), vg.colorRange(BUCKET_COLORS), vg.yDomain([...totals.keys()]), vg.xTickFormat('%'),
      vg.marginLeft(140), vg.xLabel('Share of beta cutoffs'), vg.yLabel(null), vg.width(chartWidth()),
      vg.height(56 + 36 * totals.size))];
  });
  frag.append(panel({
    title: 'Which move caused the cutoff',
    caption: 'Every beta cutoff grouped by where the cutting move stood in the move ordering, one bar per engine in version order; a longer dark first segment means better ordering.',
    body: el('div', {}, coreScopeNote('This breakdown is', () => coreRowCount === 0), overall), note: sourceNote(mo),
  }));

  const dims = (await query(`SELECT DISTINCT dimension FROM ds_moveorder WHERE dimension <> 'all'`)).map(r => r.dimension);
  const available = DIMENSIONS.filter(d => dims.includes(d.value));
  if (!available.length) return frag;
  let dim = available[0].value;
  const subs = new Set();
  const line = (buckets, label) => async () => {
    const rows = await query(`SELECT engine, engine_ord, dim_value AS x, sum(count) FILTER (WHERE bucket IN (${buckets.map(b => `'${b}'`).join(', ')}))
        / nullif(sum(count), 0) AS share, sum(count) AS n
      FROM ds_moveorder WHERE dimension = '${dim}' AND ${engineWhere()} GROUP BY ALL HAVING sum(count) > 0 ORDER BY engine_ord, x`);
    if (!rows.length) return emptyNote('No data along this dimension.');
    return [engineLegend(), plot(
      vg.lineY(rows, { x: 'x', y: 'share', stroke: 'engine', z: 'engine', strokeWidth: 1.5 }),
      vg.dot(rows, { x: 'x', y: 'share', fill: 'engine', r: 2, tip: true }),
      ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), intTicks(), vg.xLabel(DIMENSIONS.find(d => d.value === dim).axis), vg.yLabel(label),
      vg.width(chartWidth('half')), vg.height(250))];
  };
  const firstBody = el('div'), lateBody = el('div');
  reactive(firstBody, line(['0'], 'Cutoffs by the first move'), [fn => subs.add(fn)]);
  reactive(lateBody, line(['4-7', '8+'], 'Cutoffs by the fifth move or later'), [fn => subs.add(fn)]);
  frag.append(el('div', { class: 'controls section-controls' }, segmented({ label: 'Along', value: dim,
    options: available.map(d => ({ value: d.value, label: d.label })), onChange: v => { dim = v; subs.forEach(fn => fn()); } })),
    coreScopeNote('Cutoffs along depth and ply are'));
  frag.append(panel({ span: 'half', title: 'First-move cutoffs',
    caption: 'Share of cutoffs made by the first move tried, along the chosen dimension; it usually falls deeper in the tree.', body: firstBody }));
  frag.append(panel({ span: 'half', title: 'Late cutoffs',
    caption: 'Share of cutoffs that needed five or more moves to be searched first; these are the expensive ones.', body: lateBody }));
  return frag;
}
