/** Time management: how search time is spread over iterations and over searches. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineScoped, engineWhere } from '../engines.js';
import { coreRows } from '../settings.js';
import { chartWidth, el, emptyNote, engineColors, engineLegend, fmt, intTicks, panel, plot, reactive, sourceNote, statLine } from '../ui.js';

export async function render() {
  const frag = document.createDocumentFragment();
  const [summary, depth, hist, points] = await Promise.all([dataset('summary'), dataset('depth'), dataset('hist'), dataset('sample_points')]);
  if (!summary) {
    frag.append(panel({ body: emptyNote('This snapshot has no search data.') }));
    return frag;
  }
  const stats = el('div');
  reactive(stats, async () => {
    const [s] = await query(`SELECT sum(time_ms) / nullif(sum(searches), 0) AS mean, sum(time_ms_p50 * searches) / sum(searches) AS p50,
        sum(time_ms_p90 * searches) / sum(searches) AS p90, sum(time_ms) / 3.6e6 AS hours
      FROM ds_summary WHERE ${coreRows()} AND ${engineWhere()}`);
    let last = null;
    if (depth) {
      [last] = await query(`WITH d AS (SELECT engine_id, depth, time_ms, searches,
          lead(searches) OVER (PARTITION BY engine_id ORDER BY depth) AS next FROM ds_depth WHERE ${coreRows()} AND ${engineWhere()})
        SELECT sum(time_ms * (searches - coalesce(next, 0)) / nullif(searches, 0)) / nullif(sum(time_ms), 0) AS share FROM d`);
    }
    return statLine([
      { value: fmt.ms(s.p50), label: 'typical median search time' },
      { value: fmt.ms(s.p90), label: 'typical 90th percentile' },
      { value: fmt.ms(s.mean), label: 'mean search time' },
      last ? { value: fmt.pct(last.share, 0), label: 'of time in each search\u2019s final iteration (approx.)' } : null,
      { value: fmt.num(s.hours, 1), label: 'hours of search logged' },
    ]);
  });
  frag.append(stats);

  if (depth) {
    const perIteration = el('div');
    reactive(perIteration, async () => {
      const rows = await query(`SELECT engine, depth, time_ms_mean, searches FROM ds_depth
        WHERE ${coreRows()} AND ${engineWhere()} ORDER BY engine_ord, depth`);
      return [engineLegend(), plot(
        vg.lineY(rows, { x: 'depth', y: 'time_ms_mean', stroke: 'engine', z: 'engine', strokeWidth: 1.5 }),
        vg.dot(rows, { x: 'depth', y: 'time_ms_mean', fill: 'engine', r: 2, tip: true }),
        ...engineColors(), vg.yScale('log'), vg.yTicks(5), vg.yGrid(true), intTicks(), vg.xLabel('Iteration depth (plies)'), vg.yLabel('Milliseconds in the iteration'),
        vg.width(chartWidth('half')), vg.height(260))];
    });
    frag.append(panel({ span: 'half', title: 'Time per iteration',
      caption: 'Average time spent in each iteration by the searches that reached it (log scale: equal steps mean equal ratios).',
      body: perIteration, note: sourceNote(depth) }));
    const cum = el('div');
    reactive(cum, async () => {
      const rows = await query(`SELECT engine, depth, time_share_cum FROM ds_depth WHERE ${coreRows()} AND ${engineWhere()}
        ORDER BY engine_ord, depth`);
      return plot(
        vg.lineY(rows, { x: 'depth', y: 'time_share_cum', stroke: 'engine', z: 'engine', strokeWidth: 1.5, curve: 'monotone-x' }),
        ...engineColors(), vg.yDomain([0, 1]), vg.yTickFormat('%'), vg.yGrid(true), intTicks(), vg.xLabel('Iteration depth (plies)'),
        vg.yLabel('Share of all search time, cumulative'), vg.width(chartWidth('half')), vg.height(260));
    });
    frag.append(panel({ span: 'half', title: 'Cumulative time by depth',
      caption: 'How much of all search time was spent up to each iteration depth, over every search together.', body: cum }));
  }

  if (hist) {
    const durations = el('div');
    reactive(durations, async () => {
      const rows = await query(`SELECT engine, mid_log AS x, share FROM ds_hist WHERE metric = 'time_ms' AND ${coreRows()}
        AND ${engineWhere()} ORDER BY engine_ord, bin_lo`);
      if (!rows.length) return emptyNote('No search times recorded.');
      return [engineLegend(), plot(
        vg.lineY(rows, { x: 'x', y: 'share', stroke: 'engine', z: 'engine', curve: 'step', strokeWidth: 1.4, tip: true }),
        ...engineColors(), vg.xScale('log'), vg.yTickFormat('%'), vg.yGrid(true), vg.xLabel('Time per search (ms)'),
        vg.yLabel('Share of searches'), vg.width(chartWidth('half')), vg.height(260))];
    });
    frag.append(panel({ span: 'half', title: 'How long searches take',
      caption: 'Distribution of search time for each engine on a log scale; two humps usually mean two time controls.', body: durations }));
  }

  if (points) {
    const $sel = engineScoped({ core: points.columns.has('core_type') });
    frag.append(panel({ span: 'half', title: 'Time against nodes',
      caption: 'One dot per sampled search: points on one diagonal share a speed, so the spread across diagonals is the variation in nodes per second.',
      body: plot(
        vg.dot(vg.from('ds_sample_points', { filterBy: $sel }), { x: 'time_ms', y: 'nodes', fill: 'engine', r: 1.4, fillOpacity: 0.4 }),
        ...engineColors(), vg.xScale('log'), vg.yScale('log'), vg.xGrid(true), vg.yGrid(true), vg.xLabel('Time per search (ms)'),
        vg.yLabel('Nodes per search'), vg.width(chartWidth('half')), vg.height(260)) }));
  }
  return frag;
}
