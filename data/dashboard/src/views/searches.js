/** Searches: how depth relates to effort, and linked distributions over the search sample. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineScoped, engineWhere } from '../engines.js';
import { coreFilter, coreRows } from '../settings.js';
import { chartWidth, el, emptyNote, engineColors, engineLegend, fmt, intTicks, panel, plot, releaseCharts, selectInput,
  sourceNote, statLine, reactive } from '../ui.js';

const Y_METRICS = [
  { value: 'nodes', label: 'Nodes per search', log: true },
  { value: 'time_ms', label: 'Time per search (ms)', log: true },
  { value: 'nps', label: 'Nodes per second' },
  { value: 'tt_hit_rate', label: 'TT hits per node', pct: true },
  { value: 'qratio', label: 'Quiescence share of nodes', pct: true },
  { value: 'fh_first_rate', label: 'Cutoffs by the first move', pct: true },
  { value: 'abs_eval_cp', label: 'Absolute evaluation (cp)' },
];

export async function render() {
  const frag = document.createDocumentFragment();
  const sample = await dataset('sample');
  if (!sample || !sample.columns.has('depth')) {
    frag.append(panel({ body: emptyNote('This snapshot has no search-level data.') }));
    return frag;
  }
  const $cross = engineScoped({ cross: true, core: sample.columns.has('core_type') });
  const coreCond = () => (sample.columns.has('core_type') ? coreFilter() : 'TRUE');
  const metrics = Y_METRICS.filter(m => sample.columns.has(m.value));

  const stats = el('div');
  reactive(stats, async () => {
    const [s] = await query(`SELECT count(*)::DOUBLE AS n, avg(depth) AS depth, median(time_ms) AS time,
        sum(nodes) / nullif(sum(time_ms) / 1000, 0) AS nps FROM ds_sample WHERE ${coreCond()} AND ${engineWhere()}`);
    const [t] = await query(`SELECT sum(searches) AS n FROM ds_summary WHERE ${coreRows()} AND ${engineWhere()}`)
      .catch(() => [{}]);
    return statLine([
      { value: fmt.int(t?.n), label: 'searches logged' },
      { value: fmt.int(s.n), label: 'in the sample used here' },
      { value: fmt.num(s.depth, 2), label: 'mean completed depth' },
      { value: fmt.ms(s.time), label: 'median time per search' },
      { value: fmt.nps(s.nps), label: 'nodes per second' },
    ]);
  });
  await dataset('summary');
  frag.append(el('p', { class: 'tab-intro' },
    'These charts use a stratified sample of searches (the same number from every engine, where available) so that they stay quick. ',
    'Drag across any histogram to filter the others and the depth chart; double-click a histogram to clear its filter.'));
  frag.append(stats);

  let yMetric = metrics[0]?.value;
  const depthBody = el('div');
  const drawDepth = () => {
    releaseCharts(depthBody);
    const m = metrics.find(o => o.value === yMetric);
    const from = () => vg.from('ds_sample', { filterBy: $cross });
    depthBody.replaceChildren(engineLegend(), plot(
      vg.areaY(from(), { x: 'depth', y1: vg.quantile(yMetric, 0.25), y2: vg.quantile(yMetric, 0.75),
        fill: 'engine', fillOpacity: 0.1, curve: 'monotone-x' }),
      vg.lineY(from(), { x: 'depth', y: vg.median(yMetric), stroke: 'engine', strokeWidth: 1.6, curve: 'monotone-x' }),
      vg.dot(from(), { x: 'depth', y: vg.median(yMetric), fill: 'engine', r: 2.5, tip: true }),
      ...engineColors(), vg.yScale(m.log ? 'log' : 'linear'), m.log ? vg.yTicks(5) : null, m.pct ? vg.yTickFormat('%') : null, vg.yGrid(true),
      intTicks(), vg.xLabel('Completed depth (plies)'), vg.yLabel(m.label), vg.width(chartWidth()), vg.height(320)));
  };
  frag.append(panel({
    title: 'Depth against effort',
    caption: 'For each completed depth, the line is the median of the chosen measure over that engine\u2019s searches and the band holds the middle half, so a steeper line means each extra ply costs more.',
    controls: selectInput({ label: 'Measure', options: metrics, value: yMetric, onChange: v => { yMetric = v; drawDepth(); } }),
    body: depthBody, note: sourceNote(sample),
  }));
  drawDepth();

  const hist = (column, label, { log = false, step } = {}) => plot(
    vg.rectY(vg.from('ds_sample', { filterBy: $cross }), {
      x: step ? vg.bin(column, { step }) : vg.bin(column), y: vg.count(), fill: 'engine', inset: 0.5, tip: true }),
    vg.intervalX({ as: $cross }),
    ...engineColors(), log ? vg.xScale('log') : null, log ? vg.xTicks(5) : null, step ? intTicks() : null, vg.yGrid(true), vg.xLabel(label), vg.yLabel('Searches'),
    vg.width(chartWidth('half')), vg.height(220));
  const panels = [
    ['depth', 'Completed depth', 'Completed depth (plies)', { step: 1 }],
    ['nodes', 'Nodes per search', 'Nodes per search', { log: true }],
    ['time_ms', 'Time per search', 'Time per search (ms)', { log: true }],
    ['nps', 'Nodes per second', 'Nodes per second', {}],
  ].filter(([c]) => sample.columns.has(c));
  for (const [column, title, label, opts] of panels) {
    frag.append(panel({ span: 'half', title,
      caption: `Searches in the sample by ${title.toLowerCase()}, stacked by engine; drag to select a range.`,
      body: hist(column, label, opts) }));
  }
  return frag;
}
