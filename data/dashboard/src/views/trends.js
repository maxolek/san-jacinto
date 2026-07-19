/**
 * Trends tab — engine version comparison over time.
 * Shows how key metrics evolve across versions with metric selection.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, getSearchTable, controlsBar, controlGroup, nativeSelect, summaryRow, fmt, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'avg_depth', label: 'Avg Depth' },
  { value: 'avg_nodes', label: 'Avg Nodes' },
  { value: 'avg_time_ms', label: 'Avg Time (ms)' },
  { value: 'avg_tt_hits', label: 'Avg TT Hits' },
  { value: 'avg_fail_highs', label: 'Avg Fail Highs' },
  { value: 'avg_nmp', label: 'Avg NMP' },
];

export async function renderTrends() {
  const searchTable = getSearchTable();
  const tables = window.__tables || [];

  if (!searchTable || !tables.includes('engines')) {
    return el('div', { class: 'panel' }, el('p', {}, 'Need search and engines tables for trends.'));
  }

  const container = el('div', {});

  // Get engine versions ordered by id
  const engines = await sql(`
    SELECT id, name, version, name || ' (' || version || ')' as label
    FROM engines ORDER BY id
  `);

  if (engines.length < 2) {
    return el('div', { class: 'panel' }, el('p', {}, 'Need at least 2 engine versions for trends.'));
  }

  // Summary stats
  const [agg] = await sql(`
    SELECT COUNT(*) as total_searches, COUNT(DISTINCT engine_id) as n_engines,
      AVG(completed_depth) as avg_depth, AVG(total_nodes) as avg_nodes,
      AVG(total_time_ms) as avg_time
    FROM ${searchTable}
  `);
  container.appendChild(summaryRow({
    'Total Searches': agg?.total_searches,
    'Engines': agg?.n_engines,
    'Avg Depth': agg?.avg_depth != null ? Number(agg.avg_depth).toFixed(1) : null,
    'Avg Nodes': agg?.avg_nodes,
    'Avg Time': agg?.avg_time != null ? Number(agg.avg_time).toFixed(0) + 'ms' : null,
  }));

  // Create trend aggregate view
  await coordinator().exec(`
    CREATE OR REPLACE TEMP VIEW trend_agg AS
    SELECT 
      e.name || ' (' || e.version || ')' as engine_label,
      e.id as engine_order,
      AVG(s.completed_depth) as avg_depth,
      AVG(s.total_nodes) as avg_nodes,
      AVG(s.total_time_ms) as avg_time_ms,
      AVG(s.total_tt_hits) as avg_tt_hits,
      AVG(s.total_fail_highs) as avg_fail_highs,
      AVG(s.total_nmp) as avg_nmp,
      COUNT(*) as n_searches
    FROM ${searchTable} s
    JOIN engines e ON s.engine_id = e.id
    GROUP BY e.id, e.name, e.version
    ORDER BY e.id
  `);

  // Metric selector for main trend line
  let selectedMetric = 'avg_depth';
  const plotContainer = el('div', {});

  async function renderTrendLine(metric) {
    plotContainer.innerHTML = '';
    const label = METRIC_OPTIONS.find(o => o.value === metric)?.label || metric;
    const line = vg.plot(
      vg.lineY(vg.from('trend_agg'), {
        x: 'engine_label',
        y: metric,
        stroke: COLORS[0],
        marker: true,
      }),
      vg.width(750),
      vg.height(300),
      vg.marginLeft(70),
      vg.marginBottom(70),
      vg.xLabel('Engine Version'),
      vg.yLabel(label),
      vg.xTickRotate(-30),
    );
    plotContainer.appendChild(line);
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderTrendLine(val);
  }, selectedMetric);

  container.appendChild(controlsBar(controlGroup('Metric', metricSelect)));
  container.appendChild(panel('Trend by Engine Version', plotContainer));
  await renderTrendLine(selectedMetric);

  // Grid of small multiples for all metrics
  const grid = el('div', { class: 'grid-2' });
  for (let i = 0; i < METRIC_OPTIONS.length; i++) {
    const m = METRIC_OPTIONS[i];
    const plot = vg.plot(
      vg.lineY(vg.from('trend_agg'), {
        x: 'engine_label',
        y: m.value,
        stroke: COLORS[i % COLORS.length],
        marker: true,
      }),
      vg.width(440),
      vg.height(200),
      vg.marginLeft(60),
      vg.marginBottom(60),
      vg.xLabel('Engine'),
      vg.yLabel(m.label),
      vg.xTickRotate(-30),
    );
    grid.appendChild(plotPanel(m.label, plot));
  }
  container.appendChild(grid);

  // Data volume bar chart
  const countPlot = vg.plot(
    vg.barY(vg.from('trend_agg'), {
      x: 'engine_label',
      y: 'n_searches',
      fill: COLORS[3],
    }),
    vg.width(750),
    vg.height(220),
    vg.marginLeft(60),
    vg.marginBottom(70),
    vg.xLabel('Engine Version'),
    vg.yLabel('Searches'),
    vg.xTickRotate(-30),
  );
  container.appendChild(plotPanel('Data Volume per Engine', countPlot));

  return container;
}
