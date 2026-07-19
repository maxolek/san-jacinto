/**
 * Root Moves tab — analysis of root move evaluations and time allocation.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, fmt, controlsBar, controlGroup, nativeSelect, summaryRow, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'eval', label: 'Eval by Move' },
  { value: 'nodes', label: 'Nodes per Move' },
  { value: 'time', label: 'Time per Move' },
  { value: 'scatter', label: 'Time vs Eval' },
];

export async function renderRootMoves() {
  const tables = window.__tables || [];
  if (!tables.includes('root_moves')) {
    return el('div', { class: 'panel' }, el('p', {}, 'No root_moves table found.'));
  }

  const container = el('div', {});

  // Summary
  const [stats] = await sql(`
    SELECT COUNT(*) as total_entries,
      COUNT(DISTINCT search_id) as n_searches,
      AVG(eval) as avg_eval,
      AVG(nodes) as avg_nodes,
      AVG(time_ms) as avg_time
    FROM root_moves
  `);
  container.appendChild(summaryRow({
    'Root Move Entries': stats?.total_entries,
    'Searches Analyzed': stats?.n_searches,
    'Avg Eval': stats?.avg_eval != null ? Number(stats.avg_eval).toFixed(0) + ' cp' : null,
    'Avg Nodes/Move': stats?.avg_nodes,
    'Avg Time/Move': stats?.avg_time != null ? Number(stats.avg_time).toFixed(1) + ' ms' : null,
  }));

  // Metric selector
  let selectedMetric = 'eval';
  const plotContainer = el('div', {});

  async function renderChart(metric) {
    plotContainer.innerHTML = '';

    if (metric === 'eval') {
      // Distribution of root move evals
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW root_eval_dist AS SELECT * FROM root_moves WHERE eval IS NOT NULL AND eval BETWEEN -1000 AND 1000`);
      const hist = vg.plot(
        vg.rectY(vg.from('root_eval_dist'), { x: vg.bin('eval'), y: vg.count(), fill: COLORS[0] }),
        vg.width(750), vg.height(280), vg.marginLeft(55),
        vg.xLabel('Root Move Eval (cp)'), vg.yLabel('Count'),
      );
      plotContainer.appendChild(hist);
    } else if (metric === 'nodes') {
      // Top moves by avg nodes allocated
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW root_moves_by_nodes AS SELECT move, AVG(nodes) as avg_nodes, COUNT(*) as freq FROM root_moves GROUP BY move ORDER BY avg_nodes DESC LIMIT 20`);
      const bar = vg.plot(
        vg.barX(vg.from('root_moves_by_nodes'), { x: 'avg_nodes', y: 'move', fill: COLORS[1], sort: { y: '-x' } }),
        vg.width(750), vg.height(400), vg.marginLeft(80),
        vg.xLabel('Avg Nodes'), vg.yLabel('Move'),
      );
      plotContainer.appendChild(bar);
    } else if (metric === 'time') {
      // Top moves by avg time
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW root_moves_by_time AS SELECT move, AVG(time_ms) as avg_time, COUNT(*) as freq FROM root_moves GROUP BY move ORDER BY avg_time DESC LIMIT 20`);
      const bar = vg.plot(
        vg.barX(vg.from('root_moves_by_time'), { x: 'avg_time', y: 'move', fill: COLORS[3], sort: { y: '-x' } }),
        vg.width(750), vg.height(400), vg.marginLeft(80),
        vg.xLabel('Avg Time (ms)'), vg.yLabel('Move'),
      );
      plotContainer.appendChild(bar);
    } else if (metric === 'scatter') {
      // Time vs eval scatter (sample if large)
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW root_moves_sample AS SELECT * FROM root_moves WHERE eval IS NOT NULL AND time_ms IS NOT NULL USING SAMPLE 5000`);
      const scatter = vg.plot(
        vg.dot(vg.from('root_moves_sample'), { x: 'eval', y: 'time_ms', fill: COLORS[4], opacity: 0.4, r: 3 }),
        vg.width(750), vg.height(320), vg.marginLeft(60),
        vg.xLabel('Eval (cp)'), vg.yLabel('Time (ms)'),
      );
      plotContainer.appendChild(scatter);
    }
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderChart(val);
  }, selectedMetric);

  container.appendChild(controlsBar(controlGroup('View', metricSelect)));
  container.appendChild(panel('Root Move Analysis', plotContainer));
  await renderChart(selectedMetric);

  // Move frequency (most common root moves)
  await coordinator().exec(`CREATE OR REPLACE TEMP VIEW root_move_freq AS SELECT move, COUNT(*) as count FROM root_moves GROUP BY move ORDER BY count DESC LIMIT 25`);
  const freqBar = vg.plot(
    vg.barX(vg.from('root_move_freq'), { x: 'count', y: 'move', fill: COLORS[5], sort: { y: '-x' } }),
    vg.width(750), vg.height(450), vg.marginLeft(80),
    vg.xLabel('Occurrences'), vg.yLabel('Move'),
  );
  container.appendChild(plotPanel('Most Common Root Moves', freqBar));

  return container;
}
