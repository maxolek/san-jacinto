/**
 * Timing tab — function-level timing breakdown with metric selection.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, fmt, metricCard, controlsBar, controlGroup, nativeSelect, summaryRow, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'time', label: 'Avg Time / Search' },
  { value: 'calls', label: 'Total Calls' },
  { value: 'per_call', label: 'Time per Call' },
];

export async function renderTiming() {
  const tables = window.__tables || [];
  if (!tables.includes('search_timings')) {
    return el('div', { class: 'panel' }, el('p', {}, 'No search_timings table found.'));
  }

  const container = el('div', {});

  // Summary stats
  const [stats] = await sql(`
    SELECT COUNT(DISTINCT "function") as n_funcs,
      SUM(total_time_ms) as total_ms,
      SUM(num_calls) as total_calls
    FROM search_timings
  `);
  container.appendChild(summaryRow({
    'Functions': stats?.n_funcs,
    'Total Time (ms)': stats?.total_ms,
    'Total Calls': stats?.total_calls,
    'Avg ms/Call': stats?.total_calls > 0 ? (stats.total_ms / stats.total_calls).toFixed(3) : null,
  }));

  // Metric selector
  let selectedMetric = 'time';
  const plotContainer = el('div', {});

  async function renderBar(metric) {
    plotContainer.innerHTML = '';
    if (metric === 'time') {
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW timing_by_func AS SELECT "function", SUM(total_time_ms)/NULLIF(SUM(num_calls), 0) as avg_ms_per_search FROM search_timings WHERE "function" <> 'ROOT' GROUP BY "function" ORDER BY avg_ms_per_search DESC LIMIT 15`);
      const bar = vg.plot(
        vg.barX(vg.from('timing_by_func'), { x: 'avg_ms_per_search', y: 'function', fill: COLORS[0], sort: { y: '-x' } }),
        vg.width(700), vg.height(380), vg.marginLeft(180),
        vg.xLabel('Avg Time (ms)'), vg.yLabel('Function'),
      );
      plotContainer.appendChild(bar);
    } else if (metric === 'calls') {
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW calls_by_func AS SELECT "function", SUM(num_calls) as total_calls FROM search_timings GROUP BY "function" ORDER BY total_calls DESC LIMIT 15`);
      const bar = vg.plot(
        vg.barX(vg.from('calls_by_func'), { x: 'total_calls', y: 'function', fill: COLORS[1], sort: { y: '-x' } }),
        vg.width(700), vg.height(380), vg.marginLeft(180),
        vg.xLabel('Total Calls'), vg.yLabel('Function'),
      );
      plotContainer.appendChild(bar);
    } else if (metric === 'per_call') {
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW timing_per_call AS SELECT "function", SUM(total_time_ms) / NULLIF(SUM(num_calls), 0) as avg_time_per_call, SUM(num_calls) as total_calls FROM search_timings GROUP BY "function"`);
      const scatter = vg.plot(
        vg.dot(vg.from('timing_per_call'), { x: 'total_calls', y: 'avg_time_per_call', fill: COLORS[2], opacity: 0.7, r: 5 }),
        vg.width(700), vg.height(350), vg.marginLeft(70),
        vg.xLabel('Calls'), vg.yLabel('Avg Time per Call (ms)'),
        vg.xScale('log'),
      );
      plotContainer.appendChild(scatter);
    }
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderBar(val);
  }, selectedMetric);

  container.appendChild(controlsBar(controlGroup('View', metricSelect)));
  container.appendChild(panel('Function Profiling', plotContainer));
  await renderBar(selectedMetric);

  return container;
}
