/** Pruning: SEE, delta and null-move pruning along the iterations or the tree, and per engine. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { clausePoints, engineById, engineScoped, engineWhere, filterSource } from '../engines.js';
import { coreRows, coreType, onCoreType } from '../settings.js';
import { chartWidth, coreScopeNote, dataTable, el, emptyNote, engineColors, engineLegend, engineName, fmt, intTicks, panel, plot,
  reactive, releaseCharts, segmented, sourceNote } from '../ui.js';

const RATES = [
  { value: 'see_prune_rate', title: 'SEE pruning', label: 'SEE prunes per quiescence node',
    caption: 'Captures skipped because static exchange evaluation says they lose material, per quiescence node.' },
  { value: 'delta_prune_rate', title: 'Delta pruning', label: 'Delta prunes per quiescence node',
    caption: 'Captures skipped because even winning the piece could not lift the score to alpha, per quiescence node.' },
  { value: 'nmp_rate', title: 'Null-move attempts', label: 'Null-move searches per node',
    caption: 'How often a null move was tried, per node.' },
  { value: 'nmp_success_rate', title: 'Null-move success', label: 'Share of null moves that failed high',
    caption: 'Share of null-move searches that produced a cutoff; a higher share means the attempts were well chosen.' },
];

export async function render() {
  const frag = document.createDocumentFragment();
  const [depth, tree, summary] = await Promise.all([dataset('depth'), dataset('tree'), dataset('summary')]);
  if (!depth && !tree) {
    frag.append(panel({ body: emptyNote('This snapshot has no per-depth pruning counters.') }));
    return frag;
  }
  const $sel = engineScoped();
  const coreSource = filterSource('core');
  let along = depth ? 'depth' : 'tree';
  const charts = RATES.map(r => ({ ...r, body: el('div') }));
  const plyNote = coreScopeNote('Counters by tree ply are', () => along === 'tree');
  const applyCore = () => {
    $sel.update(clausePoints(along === 'depth' ? ['core_type'] : [], along === 'depth' ? [[coreType()]] : null, { source: coreSource }));
    plyNote.update();
  };
  onCoreType(applyCore);
  const draw = () => {
    const table = along === 'depth' ? 'ds_depth' : 'ds_tree';
    const x = along === 'depth' ? 'depth' : 'ply';
    applyCore();
    for (const c of charts) {
      releaseCharts(c.body);
      c.body.replaceChildren(plot(
        vg.lineY(vg.from(table, { filterBy: $sel }), { x, y: c.value, stroke: 'engine', strokeWidth: 1.5 }),
        vg.dot(vg.from(table, { filterBy: $sel }), { x, y: c.value, fill: 'engine', r: 2, tip: true }),
        ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), intTicks(),
        vg.xLabel(along === 'depth' ? 'Iteration depth (plies)' : 'Ply from the root'), vg.yLabel(c.label),
        vg.width(chartWidth('half')), vg.height(230)));
    }
  };
  frag.append(el('div', { class: 'controls section-controls' },
    depth && tree ? segmented({ label: 'Along', value: along, onChange: v => { along = v; draw(); },
      options: [{ value: 'depth', label: 'Iteration depth' }, { value: 'tree', label: 'Tree ply' }] }) : null,
    engineLegend()), plyNote);
  for (const c of charts) frag.append(panel({ span: 'half', title: c.title, caption: c.caption, body: c.body }));
  draw();

  if (summary) {
    const table = el('div');
    reactive(table, async () => {
      const rows = await query(`SELECT engine_id, see_prune_rate, delta_prune_rate, nmp_rate, nmp_success_rate
        FROM ds_summary WHERE ${coreRows()} AND ${engineWhere()} ORDER BY engine_ord`);
      return dataTable([
        { label: 'Engine', value: r => engineName(engineById(r.engine_id)) },
        { label: 'SEE prunes per q-node', num: true, value: r => fmt.pct(r.see_prune_rate, 2) },
        { label: 'Delta prunes per q-node', num: true, value: r => fmt.pct(r.delta_prune_rate, 2) },
        { label: 'Null moves per node', num: true, value: r => fmt.pct(r.nmp_rate, 2) },
        { label: 'Null-move success', num: true, value: r => fmt.pct(r.nmp_success_rate, 1) },
      ], rows);
    });
    frag.append(panel({ title: 'Per engine, over all selected searches',
      caption: 'The same four rates computed over every selected search of each engine.', body: table, note: sourceNote(summary) }));
  }
  return frag;
}
