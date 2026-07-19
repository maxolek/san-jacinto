/**
 * Iterations tab — iterative deepening analysis with metric selection and filters.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, getIterTable, controlsBar, controlGroup, nativeSelect, summaryRow, fmt, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'nodes', label: 'Nodes' },
  { value: 'eval', label: 'Eval' },
  { value: 'nps', label: 'NPS' },
  { value: 'time_ms', label: 'Time (ms)' },
  { value: 'tt_hits', label: 'TT Hits' },
  { value: 'fail_highs', label: 'Fail Highs' },
];

export async function renderIterations() {
  const iterTable = getIterTable();
  if (!iterTable) {
    return el('div', { class: 'panel' }, el('p', {}, 'No iteration data table found.'));
  }

  const container = el('div', {});

  // Summary stats
  const [stats] = await sql(`
    SELECT 
      COUNT(*) as total_rows,
      COUNT(DISTINCT search_id) as n_searches,
      MAX(depth) as max_depth,
      AVG(nodes) as avg_nodes,
      AVG(nps) as avg_nps
    FROM ${iterTable}
  `);
  container.appendChild(summaryRow({
    'Iterations': stats?.total_rows,
    'Searches': stats?.n_searches,
    'Max Depth': stats?.max_depth,
    'Avg Nodes': stats?.avg_nodes,
    'Avg NPS': stats?.avg_nps,
  }));

  // Controls
  const $filter = Selection.crossfilter();
  let selectedMetric = 'nodes';
  const plotContainer = el('div', {});

  const depthSlider = vg.slider({
    from: iterTable,
    column: 'depth',
    as: $filter,
    label: 'Max Depth',
  });

  async function renderLine(metric) {
    plotContainer.innerHTML = '';
    const label = METRIC_OPTIONS.find(o => o.value === metric)?.label || metric;
    const line = vg.plot(
      vg.lineY(vg.from(iterTable, { filterBy: $filter }), {
        x: 'depth',
        y: vg.avg(metric),
        stroke: COLORS[0],
        marker: true,
      }),
      vg.width(750),
      vg.height(320),
      vg.marginLeft(70),
      vg.xLabel('Iteration Depth'),
      vg.yLabel(label),
    );
    plotContainer.appendChild(line);
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderLine(val);
  }, selectedMetric);

  container.appendChild(controlsBar(
    depthSlider,
    controlGroup('Metric', metricSelect),
  ));
  container.appendChild(panel('Metric by Iteration Depth', plotContainer));
  await renderLine(selectedMetric);

  // Eval convergence (always shown)
  const evalLine = vg.plot(
    vg.lineY(vg.from(iterTable, { filterBy: $filter }), {
      x: 'depth',
      y: vg.avg('eval'),
      stroke: COLORS[1],
      marker: true,
    }),
    vg.width(750),
    vg.height(260),
    vg.marginLeft(70),
    vg.xLabel('Iteration Depth'),
    vg.yLabel('Avg Eval'),
  );
  container.appendChild(plotPanel('Eval Convergence', evalLine));

  // Fail highs per depth
  const failLine = vg.plot(
    vg.lineY(vg.from(iterTable, { filterBy: $filter }), {
      x: 'depth',
      y: vg.avg('fail_highs'),
      stroke: COLORS[3],
      marker: true,
    }),
    vg.width(750),
    vg.height(240),
    vg.marginLeft(70),
    vg.xLabel('Iteration Depth'),
    vg.yLabel('Avg Fail Highs'),
  );
  container.appendChild(plotPanel('Fail Highs by Iteration Depth', failLine));

  return container;
}
