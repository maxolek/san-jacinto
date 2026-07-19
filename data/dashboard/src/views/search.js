/**
 * Search tab — cross-filtered scatter plots and distributions over search data.
 * Features: metric selector for Y-axis, engine/phase/position filters, summary stats.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, getSearchTable, controlsBar, controlGroup, nativeSelect, summaryRow, fmt, COLORS } from '../util.js';

const Y_METRICS = [
  { value: 'eval', label: 'Eval (cp)' },
  { value: 'total_nodes', label: 'Total Nodes' },
  { value: 'total_time_ms', label: 'Time (ms)' },
  { value: 'nps', label: 'NPS' },
  { value: 'tt_hits', label: 'TT Hits' },
  { value: 'fail_highs', label: 'Fail Highs' },
  { value: 'qratio', label: 'Q-Ratio' },
];

export async function renderSearch() {
  const searchTable = getSearchTable();
  if (!searchTable) {
    return el('div', { class: 'panel' }, el('p', {}, 'No search table found.'));
  }

  const container = el('div', {});

  // Summary stats
  const [stats] = await sql(`
    SELECT 
      COUNT(*) as total,
      AVG(depth) as avg_depth,
      AVG(total_nodes) as avg_nodes,
      AVG(nps) as avg_nps,
      AVG(total_time_ms) as avg_time,
      AVG(tt_hit_ratio) as avg_tt_hit_ratio
    FROM ${searchTable}
  `);
  container.appendChild(summaryRow({
    'Searches': stats?.total,
    'Avg Depth': stats?.avg_depth,
    'Avg Nodes': stats?.avg_nodes,
    'Avg NPS': stats?.avg_nps,
    'Avg Time': stats?.avg_time,
    'Avg TT Hit %': stats?.avg_tt_hit_ratio != null ? (stats.avg_tt_hit_ratio * 100).toFixed(1) + '%' : null,
  }));

  // Cross-filter selection
  const $filter = Selection.crossfilter();

  // Mosaic engine menu
  const engineMenu = vg.menu({
    from: searchTable,
    column: 'engine_id',
    as: $filter,
    label: 'Engine',
  });

  // Build controls bar
  let selectedY = 'eval';
  const scatterContainer = el('div', {});

  const ySelect = nativeSelect(Y_METRICS, async (val) => {
    selectedY = val;
    await renderScatter(val);
  }, selectedY);

  container.appendChild(controlsBar(
    engineMenu,
    controlGroup('Y-Axis', ySelect),
  ));

  // Scatter plot (re-rendered on metric change)
  async function renderScatter(yMetric) {
    scatterContainer.innerHTML = '';
    const scatter = vg.plot(
      vg.dot(vg.from(searchTable, { filterBy: $filter }), {
        x: 'depth',
        y: yMetric,
        fill: 'engine_id',
        opacity: 0.5,
        r: 2,
      }),
      vg.intervalXY({ as: $filter }),
      vg.width(800),
      vg.height(380),
      vg.marginLeft(65),
      vg.xLabel('Depth'),
      vg.yLabel(Y_METRICS.find(o => o.value === yMetric)?.label || yMetric),
      vg.colorLegend(true),
    );
    scatterContainer.appendChild(scatter);
  }

  container.appendChild(panel('Depth vs Metric', scatterContainer));
  await renderScatter(selectedY);

  // Linked histograms row
  const histRow = el('div', { class: 'grid-3' });

  const depthHist = vg.plot(
    vg.rectY(vg.from(searchTable, { filterBy: $filter }), {
      x: vg.bin('depth'), y: vg.count(), fill: COLORS[0],
    }),
    vg.intervalX({ as: $filter }),
    vg.width(350), vg.height(200),
    vg.xLabel('Depth'), vg.yLabel('Count'),
  );
  histRow.appendChild(plotPanel('Depth', depthHist));

  const nodesHist = vg.plot(
    vg.rectY(vg.from(searchTable, { filterBy: $filter }), {
      x: vg.bin('total_nodes'), y: vg.count(), fill: COLORS[1],
    }),
    vg.intervalX({ as: $filter }),
    vg.width(350), vg.height(200),
    vg.xLabel('Nodes'), vg.yLabel('Count'),
  );
  histRow.appendChild(plotPanel('Nodes', nodesHist));

  const timeHist = vg.plot(
    vg.rectY(vg.from(searchTable, { filterBy: $filter }), {
      x: vg.bin('total_time_ms'), y: vg.count(), fill: COLORS[2],
    }),
    vg.intervalX({ as: $filter }),
    vg.width(350), vg.height(200),
    vg.xLabel('Time (ms)'), vg.yLabel('Count'),
  );
  histRow.appendChild(plotPanel('Search Time', timeHist));

  container.appendChild(histRow);

  return container;
}
