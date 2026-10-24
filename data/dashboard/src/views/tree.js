/** Tree depth: statistics per ply of the search tree, with a brush on where the nodes are. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { engineScoped } from '../engines.js';
import { chartWidth, coreScopeNote, el, emptyNote, engineColors, engineLegend, intTicks, panel, plot, releaseCharts, selectInput, sourceNote } from '../ui.js';

const MEASURES = [
  { value: 'nodes_mean', label: 'Nodes per search at this ply', log: true },
  { value: 'qratio', label: 'Quiescence share of nodes', pct: true },
  { value: 'fail_high_rate', label: 'Fail highs per node', pct: true },
  { value: 'fail_low_rate', label: 'Fail lows per node', pct: true },
  { value: 'fh_first_rate', label: 'Cutoffs by the first move', pct: true },
  { value: 'tt_hit_rate', label: 'TT hits per node', pct: true },
  { value: 'nmp_rate', label: 'Null-move searches per node', pct: true },
  { value: 'see_prune_rate', label: 'SEE prunes per quiescence node', pct: true },
];

export async function render() {
  const frag = document.createDocumentFragment();
  const tree = await dataset('tree');
  if (!tree) {
    frag.append(panel({ body: emptyNote('This snapshot has no per-ply tree statistics.') }));
    return frag;
  }
  const measures = MEASURES.filter(m => tree.columns.has(m.value));
  const $sel = engineScoped({ cross: true });
  frag.append(coreScopeNote('Tree statistics are'));
  frag.append(panel({
    title: 'Where the nodes are',
    caption: 'Share of each engine\u2019s nodes at every ply from the root; drag across it to focus the chart below on a range of plies.',
    body: [engineLegend(), plot(
      vg.areaY(vg.from('ds_tree', { filterBy: $sel }), { x: 'ply', y: 'node_share', fill: 'engine', fillOpacity: 0.08, curve: 'monotone-x' }),
      vg.lineY(vg.from('ds_tree', { filterBy: $sel }), { x: 'ply', y: 'node_share', stroke: 'engine', strokeWidth: 1.6, curve: 'monotone-x' }),
      vg.intervalX({ as: $sel }),
      ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), intTicks(), vg.xLabel('Ply from the root'), vg.yLabel('Share of nodes'),
      vg.width(chartWidth()), vg.height(260))],
    note: sourceNote(tree),
  }));

  let measure = measures[0]?.value;
  const body = el('div');
  const draw = () => {
    releaseCharts(body);
    const m = measures.find(o => o.value === measure);
    body.replaceChildren(plot(
      vg.lineY(vg.from('ds_tree', { filterBy: $sel }), { x: 'ply', y: measure, stroke: 'engine', strokeWidth: 1.6 }),
      vg.dot(vg.from('ds_tree', { filterBy: $sel }), { x: 'ply', y: measure, fill: 'engine', r: 2.2, tip: true }),
      ...engineColors(), vg.yScale(m.log ? 'log' : 'linear'), m.pct ? vg.yTickFormat('%') : null, vg.yGrid(true),
      intTicks(), vg.xLabel('Ply from the root'), vg.yLabel(m.label), vg.width(chartWidth()), vg.height(300)));
  };
  frag.append(panel({
    title: 'A measure along the plies',
    caption: 'The chosen measure at each ply, one line per engine; move ordering and pruning usually change character a few plies from the leaves.',
    controls: selectInput({ label: 'Measure', options: measures, value: measure, onChange: v => { measure = v; draw(); } }),
    body,
  }));
  draw();
  return frag;
}
