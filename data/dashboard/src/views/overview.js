/**
 * Overview tab — high-level summary metrics, selectable distribution plots, and filters.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, metricCard, fmt, plotPanel, getSearchTable, controlsBar, controlGroup, nativeSelect, summaryRow, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'depth', label: 'Depth' },
  { value: 'eval', label: 'Eval (cp)' },
  { value: 'total_nodes', label: 'Total Nodes' },
  { value: 'total_time_ms', label: 'Time (ms)' },
  { value: 'tt_hits', label: 'TT Hits' },
  { value: 'nps', label: 'NPS' },
  { value: 'qratio', label: 'Q-Ratio' },
];

export async function renderOverview() {
  const searchTable = getSearchTable();
  const tables = window.__tables || [];

  // Gather summary counts
  const [engineCount, gameCount, searchCount, expCount] = await Promise.all([
    tables.includes('engines') ? sql('SELECT COUNT(*) as n FROM engines') : [{ n: 0 }],
    tables.includes('game_stats') ? sql('SELECT COUNT(*) as n FROM game_stats') : [{ n: 0 }],
    searchTable ? sql(`SELECT COUNT(*) as n FROM ${searchTable}`) : [{ n: 0 }],
    tables.includes('experiments') ? sql('SELECT COUNT(*) as n FROM experiments') : [{ n: 0 }],
  ]);

  const container = el('div', {});

  // Metric cards
  const metrics = el('div', { class: 'grid-4' },
    metricCard(fmt(engineCount[0]?.n), 'Engines'),
    metricCard(fmt(gameCount[0]?.n), 'Games'),
    metricCard(fmt(searchCount[0]?.n), 'Searches'),
    metricCard(fmt(expCount[0]?.n), 'Experiments'),
  );
  container.appendChild(panel('Summary', metrics));

  if (!searchTable) {
    container.appendChild(el('div', { class: 'panel' }, el('p', {}, 'No search table found.')));
    return container;
  }

  // Summary stats for search data
  const [stats] = await sql(`
    SELECT 
      AVG(depth) as avg_depth,
      MAX(depth) as max_depth,
      AVG(eval) as avg_eval,
      AVG(total_nodes) as avg_nodes,
      AVG(total_time_ms) as avg_time
    FROM ${searchTable}
  `);
  container.appendChild(summaryRow({
    'Avg Depth': stats?.avg_depth,
    'Max Depth': stats?.max_depth,
    'Avg Eval': stats?.avg_eval,
    'Avg Nodes': stats?.avg_nodes,
    'Avg Time (ms)': stats?.avg_time,
  }));

  // Metric selector for histogram
  let selectedMetric = 'depth';
  const plotContainer = el('div', {});

  async function renderHistogram(metric) {
    const viewName = `overview_hist_${metric}`;
    const filterSQL = metric === 'eval' ? `WHERE ABS(${metric}) <= 1500` : metric === 'total_nodes' ? `WHERE ${metric} > 0` : '';
    await coordinator().exec(`CREATE OR REPLACE TEMP VIEW ${viewName} AS SELECT * FROM ${searchTable} ${filterSQL}`);
    
    plotContainer.innerHTML = '';
    const hist = vg.plot(
      vg.rectY(vg.from(viewName), { x: vg.bin(metric), y: vg.count(), fill: COLORS[0] }),
      vg.width(700),
      vg.height(280),
      vg.marginLeft(55),
      vg.xLabel(METRIC_OPTIONS.find(o => o.value === metric)?.label || metric),
      vg.yLabel('Count'),
    );
    plotContainer.appendChild(hist);
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderHistogram(val);
  }, selectedMetric);

  container.appendChild(controlsBar(
    controlGroup('Distribution Metric', metricSelect),
  ));
  container.appendChild(panel('Distribution', plotContainer));
  await renderHistogram(selectedMetric);

  // Engine breakdown
  if (tables.includes('engines')) {
    await coordinator().exec(`CREATE OR REPLACE TEMP VIEW engine_counts AS SELECT engine_id, COUNT(*) as count FROM ${searchTable} GROUP BY engine_id`);
    const enginePlot = vg.plot(
      vg.barX(
        vg.from('engine_counts'),
        { x: 'count', y: 'engine_id', fill: COLORS[1], sort: { y: '-x' } }
      ),
      vg.width(600),
      vg.height(Math.max(150, engineCount[0]?.n * 35)),
      vg.marginLeft(80),
      vg.xLabel('Searches'),
      vg.yLabel('Engine'),
    );
    container.appendChild(plotPanel('Searches by Engine', enginePlot));
  }

  return container;
}
