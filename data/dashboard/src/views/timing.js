/** Timing: where the time inside a search goes, per profiled function and engine. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engines, engineWhere } from '../engines.js';
import { chartWidth, coreScopeNote, el, emptyNote, engineColors, engineLegend, fmt, panel, plot, reactive, selectInput, sourceNote } from '../ui.js';

const MEASURES = [
  { value: 'share', label: 'Share of search time', format: v => fmt.pct(v, 1), tick: '%',
    caption: 'Time inside each function as a share of the whole search (ROOT); functions nest, so the shares do not add up to 100%.' },
  { value: 'ms_per_search', label: 'Milliseconds per search', format: v => fmt.ms(v),
    caption: 'Average time each function takes per search: total time divided by the number of searches.' },
  { value: 'us_per_call', label: 'Microseconds per call', format: v => `${fmt.num(v, 3)} \u00b5s`,
    caption: 'Average cost of a single call: total time divided by the number of calls.' },
  { value: 'calls_per_search', label: 'Calls per search', format: v => fmt.compact(v),
    caption: 'How many times each function is called in an average search.' },
];

export async function render() {
  const frag = document.createDocumentFragment();
  const timing = await dataset('timing');
  if (!timing) {
    frag.append(panel({ body: emptyNote('This snapshot has no function timings.') }));
    return frag;
  }
  frag.append(coreScopeNote('Function timings are'));
  let measure = 'share';
  const body = el('div');
  const caption = el('p', { class: 'caption' });
  const redraw = reactive(body, async () => {
    const m = MEASURES.find(x => x.value === measure);
    caption.textContent = m.caption;
    const rows = (await query(`SELECT engine, engine_id, function, ${measure} AS v FROM ds_timing
      WHERE ${engineWhere()} ${measure === 'share' ? "AND upper(function) <> 'ROOT'" : ''} AND ${measure} IS NOT NULL ORDER BY engine_ord`))
      .map(r => ({ ...r, title: `${r.function}, ${r.engine}: ${m.format(r.v)}` }));
    if (!rows.length) return emptyNote('No timings for this selection.');
    const base = rows.filter(r => r.engine_id === engines.baseline);
    const order = [...new Set((base.length ? base : rows).sort((a, b) => b.v - a.v).map(r => r.function))];
    for (const f of new Set(rows.map(r => r.function))) if (!order.includes(f)) order.push(f);
    const labels = [...new Set(rows.map(r => r.engine))];
    return [engineLegend(), plot(
      vg.barX(rows, { x: 'v', y: 'engine', fy: 'function', fill: 'engine', insetTop: 1.5, insetBottom: 1.5, tip: true, title: 'title' }),
      ...engineColors(), vg.fyDomain(order), vg.yDomain(labels), vg.yAxis(null), vg.fyLabel(null), vg.xGrid(true),
      m.tick ? vg.xTickFormat(m.tick) : null, vg.marginLeft(130), vg.facetMarginTop(2), vg.facetMarginBottom(6),
      vg.xLabel(m.label), vg.width(chartWidth()), vg.height(60 + order.length * (labels.length * 12 + 12)))];
  });
  frag.append(panel({
    title: 'Function profile',
    caption: 'One group of bars per profiled function, one bar per engine; functions are sorted by the baseline\u2019s value.',
    controls: [selectInput({ label: 'Measure', value: measure, options: MEASURES, onChange: v => { measure = v; redraw(); } }), caption],
    body, note: sourceNote(timing),
  }));
  return frag;
}
