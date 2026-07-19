/**
 * Tree Depth tab — search tree depth statistics with metric selection.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, getTreeTable, controlsBar, controlGroup, nativeSelect, summaryRow, fmt, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'nodes', label: 'Nodes' },
  { value: 'qnodes', label: 'QNodes' },
  { value: 'fail_highs', label: 'Fail Highs' },
  { value: 'fail_lows', label: 'Fail Lows' },
  { value: 'nmp', label: 'NMP' },
];

export async function renderTree() {
  const treeTable = getTreeTable();
  if (!treeTable) {
    return el('div', { class: 'panel' }, el('p', {}, 'No tree depth data table found.'));
  }

  const container = el('div', {});

  // Summary stats
  const [stats] = await sql(`
    SELECT COUNT(*) as total_rows, MAX(depth) as max_depth,
      AVG(nodes) as avg_nodes, AVG(qnodes) as avg_qnodes,
      SUM(fail_highs) as total_fh, SUM(nmp) as total_nmp
    FROM ${treeTable}
  `);
  container.appendChild(summaryRow({
    'Data Points': stats?.total_rows,
    'Max Depth': stats?.max_depth,
    'Avg Nodes': stats?.avg_nodes,
    'Avg QNodes': stats?.avg_qnodes,
    'Total Fail Highs': stats?.total_fh,
  }));

  const $filter = Selection.crossfilter();
  let selectedMetric = 'nodes';
  const plotContainer = el('div', {});

  async function renderLine(metric) {
    plotContainer.innerHTML = '';
    const label = METRIC_OPTIONS.find(o => o.value === metric)?.label || metric;
    const colorIdx = METRIC_OPTIONS.findIndex(o => o.value === metric);
    const line = vg.plot(
      vg.lineY(vg.from(treeTable, { filterBy: $filter }), {
        x: 'depth',
        y: vg.avg(metric),
        stroke: COLORS[colorIdx % COLORS.length],
        marker: true,
      }),
      vg.intervalX({ as: $filter }),
      vg.width(750),
      vg.height(320),
      vg.marginLeft(70),
      vg.xLabel('Tree Depth (ply)'),
      vg.yLabel('Avg ' + label),
    );
    plotContainer.appendChild(line);
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderLine(val);
  }, selectedMetric);

  container.appendChild(controlsBar(controlGroup('Metric', metricSelect)));
  container.appendChild(panel('Metric by Tree Depth', plotContainer));
  await renderLine(selectedMetric);

  // Multi-line comparison: nodes vs qnodes
  const multiLine = vg.plot(
    vg.lineY(vg.from(treeTable, { filterBy: $filter }), {
      x: 'depth', y: vg.avg('nodes'), stroke: COLORS[0], marker: true,
    }),
    vg.lineY(vg.from(treeTable, { filterBy: $filter }), {
      x: 'depth', y: vg.avg('qnodes'), stroke: COLORS[1], marker: true,
    }),
    vg.width(750),
    vg.height(280),
    vg.marginLeft(70),
    vg.xLabel('Tree Depth (ply)'),
    vg.yLabel('Avg Count'),
  );
  container.appendChild(plotPanel('Nodes vs QNodes by Depth', multiLine));

  // Fail highs/lows comparison
  const failPlot = vg.plot(
    vg.lineY(vg.from(treeTable, { filterBy: $filter }), {
      x: 'depth', y: vg.avg('fail_highs'), stroke: COLORS[3], marker: true,
    }),
    vg.lineY(vg.from(treeTable, { filterBy: $filter }), {
      x: 'depth', y: vg.avg('fail_lows'), stroke: COLORS[2], marker: true,
    }),
    vg.width(750),
    vg.height(260),
    vg.marginLeft(70),
    vg.xLabel('Tree Depth (ply)'),
    vg.yLabel('Count'),
  );
  container.appendChild(plotPanel('Fail Highs / Lows by Depth', failPlot));

  return container;
}
