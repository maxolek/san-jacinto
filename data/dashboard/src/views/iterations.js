/** Iterations: iterative deepening, one line per engine along the depth of each iteration. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineScoped } from '../engines.js';
import { chartWidth, el, emptyNote, engineColors, engineLegend, intTicks, panel, plot, releaseCharts,
  selectInput, sourceNote } from '../ui.js';

export const DEPTH_MEASURES = [
  { value: 'nodes_mean', label: 'Nodes in the iteration', log: true },
  { value: 'time_ms_mean', label: 'Time in the iteration (ms)', log: true },
  { value: 'ebf', label: 'Effective branching factor' },
  { value: 'nps', label: 'Nodes per second' },
  { value: 'tt_hit_rate', label: 'TT hits per node', pct: true },
  { value: 'fail_high_rate', label: 'Fail highs per node', pct: true },
  { value: 'fh_first_rate', label: 'Cutoffs by the first move', pct: true },
  { value: 'move_change_rate', label: 'Best move changed from the previous iteration', pct: true },
  { value: 'eval_change_mean', label: 'Eval change from the previous iteration (cp)' },
  { value: 'researches_mean', label: 'Aspiration re-searches per iteration' },
];

export async function render() {
  const frag = document.createDocumentFragment();
  const depth = await dataset('depth');
  if (!depth) {
    frag.append(panel({ body: emptyNote('This snapshot has no per-iteration data.') }));
    return frag;
  }
  const measures = DEPTH_MEASURES.filter(m => depth.columns.has(m.value));
  const [{ lo, hi }] = await query('SELECT min(depth) AS lo, max(depth) AS hi FROM ds_depth');
  const $sel = engineScoped({ core: true, allRows: true });

  frag.append(el('p', { class: 'tab-intro' },
    'Each point is one iteration of iterative deepening. Later iterations are only reached by some searches, so a value at depth 14 ',
    'describes only the searches that got that far; the second chart shows how many did.'));
  const slider = vg.slider({ from: 'ds_depth', column: 'depth', as: $sel, select: 'interval', label: 'Up to depth',
    min: lo, max: hi, step: 1, value: hi });
  frag.append(el('div', { class: 'controls section-controls' }, slider));

  let measure = measures[0]?.value;
  const mainBody = el('div');
  const line = (column, { label, log, pct }, height = 300, span = 'full') => plot(
    vg.lineY(vg.from('ds_depth', { filterBy: $sel }), { x: 'depth', y: column, stroke: 'engine', strokeWidth: 1.6 }),
    vg.dot(vg.from('ds_depth', { filterBy: $sel }), { x: 'depth', y: column, fill: 'engine', r: 2.2, tip: true }),
    ...engineColors(), vg.yScale(log ? 'log' : 'linear'), log ? vg.yTicks(5) : null, pct ? vg.yTickFormat('%') : null, vg.yGrid(true),
    intTicks(), vg.xLabel('Iteration depth (plies)'), vg.yLabel(label), vg.width(chartWidth(span)), vg.height(height));
  const drawMain = () => {
    releaseCharts(mainBody);
    mainBody.replaceChildren(engineLegend(), line(measure, measures.find(m => m.value === measure)));
  };
  frag.append(panel({
    title: 'A measure along the iterations',
    caption: 'The chosen measure at each iteration depth, one line per engine; use the slider to hide the deepest iterations, which rest on few searches.',
    controls: selectInput({ label: 'Measure', options: measures, value: measure, onChange: v => { measure = v; drawMain(); } }),
    body: mainBody, note: sourceNote(depth),
  }));
  drawMain();

  frag.append(panel({ span: 'half', title: 'How many searches get that deep',
    caption: 'Share of each engine\u2019s searches that completed at least this iteration.',
    body: line('reached_share', { label: 'Share of searches', pct: true }, 240, 'half') }));
  frag.append(panel({ span: 'half', title: 'Where the time goes',
    caption: 'Cumulative share of all search time spent up to and including each iteration; a steep late rise means the last iterations dominate.',
    body: line('time_share_cum', { label: 'Share of search time', pct: true }, 240, 'half') }));
  return frag;
}
