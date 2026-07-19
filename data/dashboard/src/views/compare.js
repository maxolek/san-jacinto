/**
 * Compare tab — side-by-side engine comparison with metric selection and filters.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, getSearchTable, controlsBar, controlGroup, nativeSelect, summaryRow, fmt, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'completed_depth', label: 'Depth' },
  { value: 'total_nodes', label: 'Nodes' },
  { value: 'total_time_ms', label: 'Time (ms)' },
  { value: 'total_tt_hits', label: 'TT Hits' },
  { value: 'total_fail_highs', label: 'Fail Highs' },
  { value: 'total_nmp', label: 'NMP' },
];

export async function renderCompare() {
  const searchTable = getSearchTable();
  const tables = window.__tables || [];

  if (!searchTable || !tables.includes('engines')) {
    return el('div', { class: 'panel' }, el('p', {}, 'Need search and engines tables for comparison.'));
  }

  const container = el('div', {});
  const $filter = Selection.crossfilter();

  // Summary stats
  const [stats] = await sql(`
    SELECT COUNT(*) as total, COUNT(DISTINCT engine_id) as n_engines,
      AVG(completed_depth) as avg_depth, AVG(total_nodes) as avg_nodes,
      AVG(total_time_ms) as avg_time
    FROM ${searchTable}
  `);
  container.appendChild(summaryRow({
    'Searches': stats?.total,
    'Engines': stats?.n_engines,
    'Avg Depth': stats?.avg_depth != null ? Number(stats.avg_depth).toFixed(1) : null,
    'Avg Nodes': stats?.avg_nodes,
    'Avg Time': stats?.avg_time != null ? Number(stats.avg_time).toFixed(0) + 'ms' : null,
  }));

  // Engine filter
  const engineSelect = vg.menu({
    from: 'engines',
    column: 'id',
    label: 'Filter Engine',
    as: $filter,
  });

  // Metric selector for the distribution plot
  let selectedMetric = 'completed_depth';
  const plotContainer = el('div', {});

  async function renderDist(metric) {
    plotContainer.innerHTML = '';
    const label = METRIC_OPTIONS.find(o => o.value === metric)?.label || metric;
    const hist = vg.plot(
      vg.rectY(vg.from(searchTable, { filterBy: $filter }), {
        x: vg.bin(metric),
        y: vg.count(),
        fill: 'engine_id',
      }),
      vg.width(750),
      vg.height(300),
      vg.marginLeft(60),
      vg.xLabel(label),
      vg.yLabel('Count'),
      vg.colorLegend(true),
    );
    plotContainer.appendChild(hist);
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderDist(val);
  }, selectedMetric);

  container.appendChild(controlsBar(
    engineSelect,
    controlGroup('Metric', metricSelect),
  ));
  container.appendChild(panel('Metric Distribution by Engine', plotContainer));
  await renderDist(selectedMetric);

  // Side-by-side box plots (depth, nodes, time)
  const grid = el('div', { class: 'grid-2' });

  const depthBox = vg.plot(
    vg.rectY(vg.from(searchTable, { filterBy: $filter }), {
      x: 'engine_id', y: 'completed_depth', fill: COLORS[0],
    }),
    vg.width(420), vg.height(250), vg.marginLeft(60),
    vg.xLabel('Engine'), vg.yLabel('Depth'),
  );
  grid.appendChild(plotPanel('Depth by Engine', depthBox));

  const nodesBox = vg.plot(
    vg.rectY(vg.from(searchTable, { filterBy: $filter }), {
      x: 'engine_id', y: 'total_nodes', fill: COLORS[1],
    }),
    vg.width(420), vg.height(250), vg.marginLeft(60),
    vg.xLabel('Engine'), vg.yLabel('Nodes'),
  );
  grid.appendChild(plotPanel('Nodes by Engine', nodesBox));

  const timeBox = vg.plot(
    vg.rectY(vg.from(searchTable, { filterBy: $filter }), {
      x: 'engine_id', y: 'total_time_ms', fill: COLORS[2],
    }),
    vg.width(420), vg.height(250), vg.marginLeft(60),
    vg.xLabel('Engine'), vg.yLabel('Time (ms)'),
  );
  grid.appendChild(plotPanel('Time by Engine', timeBox));

  const ttPlot = vg.plot(
    vg.dot(vg.from(searchTable, { filterBy: $filter }), {
      x: 'total_tt_stores', y: 'total_tt_hits', fill: 'engine_id',
      opacity: 0.4, r: 3,
    }),
    vg.width(420), vg.height(250), vg.marginLeft(60),
    vg.xLabel('TT Stores'), vg.yLabel('TT Hits'),
    vg.colorLegend(true),
  );
  grid.appendChild(plotPanel('TT Efficiency', ttPlot));

  container.appendChild(grid);

  return container;
}
