/**
 * Pruning tab — SEE, delta, NMP, and LMR pruning efficiency analysis.
 * Shows pruning rates by depth, across versions, and absolute counts.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, fmt, controlsBar, controlGroup, nativeSelect, summaryRow, getSearchTable, getIterTable, getTreeTable, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'rates_by_depth', label: 'Pruning Rates by Depth' },
  { value: 'nmp_analysis', label: 'NMP Analysis' },
  { value: 'see_delta', label: 'SEE & Delta by Depth' },
  { value: 'by_engine', label: 'By Engine Version' },
  { value: 'tree_depth', label: 'Pruning at Tree Depth' },
];

export async function renderPruning() {
  const searchTable = getSearchTable();
  const iterTable = getIterTable();
  const treeTable = getTreeTable();

  if (!searchTable) {
    return el('div', { class: 'panel' }, el('p', {}, 'No search data table found.'));
  }

  const container = el('div', {});

  // Summary stats
  const [stats] = await sql(`
    SELECT COUNT(*) as total,
      AVG(total_see_prunes) as avg_see,
      AVG(total_delta_prunes) as avg_delta,
      AVG(total_nmp) as avg_nmp,
      AVG(total_nmp_failhigh) as avg_nmp_failhigh,
      AVG(nmp_ratio) as avg_nmp_ratio,
      AVG(CASE WHEN total_nmp > 0 THEN total_nmp_failhigh::DOUBLE / total_nmp ELSE NULL END) as avg_nmp_failhigh_rate
    FROM ${searchTable}
  `);
  container.appendChild(summaryRow({
    'Searches': stats?.total,
    'Avg SEE Prunes': stats?.avg_see != null ? Number(stats.avg_see).toFixed(1) : null,
    'Avg Delta Prunes': stats?.avg_delta != null ? Number(stats.avg_delta).toFixed(1) : null,
    'Avg NMP': stats?.avg_nmp != null ? Number(stats.avg_nmp).toFixed(1) : null,
    'NMP Fail Rate': stats?.avg_nmp_failhigh_rate != null ? (Number(stats.avg_nmp_failhigh_rate) * 100).toFixed(1) + '%' : null,
  }));

  // Metric selector
  let selectedMetric = 'rates_by_depth';
  const plotContainer = el('div', {});

  async function renderChart(metric) {
    plotContainer.innerHTML = '';

    if (metric === 'rates_by_depth' && iterTable) {
      // Pruning ratios by iteration depth
      const plot = vg.plot(
        vg.lineY(vg.from(iterTable), { x: 'depth', y: vg.avg('see_prune_ratio'), stroke: COLORS[0], marker: true }),
        vg.lineY(vg.from(iterTable), { x: 'depth', y: vg.avg('delta_prune_ratio'), stroke: COLORS[1], marker: true }),
        vg.lineY(vg.from(iterTable), { x: 'depth', y: vg.avg('nmp_ratio'), stroke: COLORS[2], marker: true }),
        vg.width(750), vg.height(320), vg.marginLeft(70),
        vg.xLabel('Iteration Depth'), vg.yLabel('Prune Ratio (per total nodes)'),
      );
      const legend = el('div', { style: { fontSize: '12px', padding: '8px 0', display: 'flex', gap: '16px' } },
        el('span', { style: { color: COLORS[0] } }, '● SEE'),
        el('span', { style: { color: COLORS[1] } }, '● Delta'),
        el('span', { style: { color: COLORS[2] } }, '● NMP'),
      );
      plotContainer.appendChild(legend);
      plotContainer.appendChild(plot);

    } else if (metric === 'rates_by_depth' && !iterTable) {
      // Fallback: by completed_depth from search table
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW prune_by_depth AS
        SELECT completed_depth as depth,
          AVG(total_see_prunes::DOUBLE / NULLIF(total_nodes, 0)) as see_ratio,
          AVG(total_delta_prunes::DOUBLE / NULLIF(total_nodes, 0)) as delta_ratio,
          AVG(total_nmp::DOUBLE / NULLIF(total_nodes, 0)) as nmp_ratio
        FROM ${searchTable}
        WHERE completed_depth BETWEEN 1 AND 30
        GROUP BY completed_depth ORDER BY depth`);
      const plot = vg.plot(
        vg.lineY(vg.from('prune_by_depth'), { x: 'depth', y: 'see_ratio', stroke: COLORS[0], marker: true }),
        vg.lineY(vg.from('prune_by_depth'), { x: 'depth', y: 'delta_ratio', stroke: COLORS[1], marker: true }),
        vg.lineY(vg.from('prune_by_depth'), { x: 'depth', y: 'nmp_ratio', stroke: COLORS[2], marker: true }),
        vg.width(750), vg.height(320), vg.marginLeft(70),
        vg.xLabel('Completed Depth'), vg.yLabel('Prune Ratio'),
      );
      plotContainer.appendChild(plot);

    } else if (metric === 'nmp_analysis') {
      // NMP attempts vs failures by depth
      if (iterTable) {
        const plot = vg.plot(
          vg.lineY(vg.from(iterTable), { x: 'depth', y: vg.avg('nmp'), stroke: COLORS[1], marker: true }),
          vg.lineY(vg.from(iterTable), { x: 'depth', y: vg.avg('nmp_failhigh'), stroke: COLORS[2], marker: true }),
          vg.width(750), vg.height(300), vg.marginLeft(70),
          vg.xLabel('Iteration Depth'), vg.yLabel('Avg Count'),
        );
        const legend = el('div', { style: { fontSize: '12px', padding: '8px 0', display: 'flex', gap: '16px' } },
          el('span', { style: { color: COLORS[1] } }, '● NMP Attempts'),
          el('span', { style: { color: COLORS[2] } }, '● NMP Failures'),
        );
        plotContainer.appendChild(legend);
        plotContainer.appendChild(plot);
      } else {
        // NMP fail ratio histogram
        await coordinator().exec(`CREATE OR REPLACE TEMP VIEW nmp_failhigh_dist AS
          SELECT total_nmp_failhigh::DOUBLE / NULLIF(total_nmp, 0) as fail_rate
          FROM ${searchTable} WHERE total_nmp > 0`);
        const hist = vg.plot(
          vg.rectY(vg.from('nmp_failhigh_dist'), { x: vg.bin('fail_rate'), y: vg.count(), fill: COLORS[2] }),
          vg.width(750), vg.height(280), vg.marginLeft(55),
          vg.xLabel('NMP Fail Rate'), vg.yLabel('Count'),
        );
        plotContainer.appendChild(hist);
      }

    } else if (metric === 'see_delta') {
      // SEE and delta prune counts by depth
      if (iterTable) {
        const plot = vg.plot(
          vg.lineY(vg.from(iterTable), { x: 'depth', y: vg.avg('see_prunes'), stroke: COLORS[0], marker: true }),
          vg.lineY(vg.from(iterTable), { x: 'depth', y: vg.avg('delta_prunes'), stroke: COLORS[3], marker: true }),
          vg.width(750), vg.height(300), vg.marginLeft(70),
          vg.xLabel('Iteration Depth'), vg.yLabel('Avg Prunes'),
        );
        const legend = el('div', { style: { fontSize: '12px', padding: '8px 0', display: 'flex', gap: '16px' } },
          el('span', { style: { color: COLORS[0] } }, '● SEE Prunes'),
          el('span', { style: { color: COLORS[3] } }, '● Delta Prunes'),
        );
        plotContainer.appendChild(legend);
        plotContainer.appendChild(plot);
      } else {
        plotContainer.appendChild(el('p', {}, 'Per-depth SEE/delta data requires iteration features table.'));
      }

    } else if (metric === 'by_engine') {
      // Pruning rates per engine version
      const tables = window.__tables || [];
      if (!tables.includes('engines')) {
        plotContainer.appendChild(el('p', {}, 'Engines table required for version comparison.'));
        return;
      }
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW prune_by_engine AS
        SELECT e.name || ' (' || e.version || ')' as engine_label,
          AVG(s.total_see_prunes::DOUBLE / NULLIF(s.total_nodes, 0)) as see_ratio,
          AVG(s.total_delta_prunes::DOUBLE / NULLIF(s.total_nodes, 0)) as delta_ratio,
          AVG(s.total_nmp::DOUBLE / NULLIF(s.total_nodes, 0)) as nmp_ratio,
          AVG(CASE WHEN s.total_nmp > 0 THEN s.total_nmp_failhigh::DOUBLE / s.total_nmp ELSE NULL END) as nmp_failhigh_rate
        FROM ${searchTable} s
        JOIN engines e ON s.engine_id = e.id
        GROUP BY e.id, e.name, e.version
        ORDER BY e.id`);
      const plot = vg.plot(
        vg.barY(vg.from('prune_by_engine'), { x: 'engine_label', y: 'nmp_ratio', fill: COLORS[2] }),
        vg.width(750), vg.height(280), vg.marginLeft(60), vg.marginBottom(70),
        vg.xLabel('Engine'), vg.yLabel('NMP Ratio'),
        vg.xTickRotate(-30),
      );
      plotContainer.appendChild(plot);

      const plot2 = vg.plot(
        vg.barY(vg.from('prune_by_engine'), { x: 'engine_label', y: 'see_ratio', fill: COLORS[0] }),
        vg.width(750), vg.height(280), vg.marginLeft(60), vg.marginBottom(70),
        vg.xLabel('Engine'), vg.yLabel('SEE Prune Ratio'),
        vg.xTickRotate(-30),
      );
      plotContainer.appendChild(plot2);

    } else if (metric === 'tree_depth' && treeTable) {
      // Pruning at each tree ply
      const plot = vg.plot(
        vg.lineY(vg.from(treeTable), { x: 'depth', y: vg.avg('see_prunes'), stroke: COLORS[0], marker: true }),
        vg.lineY(vg.from(treeTable), { x: 'depth', y: vg.avg('delta_prunes'), stroke: COLORS[3], marker: true }),
        vg.lineY(vg.from(treeTable), { x: 'depth', y: vg.avg('nmp'), stroke: COLORS[2], marker: true }),
        vg.width(750), vg.height(320), vg.marginLeft(70),
        vg.xLabel('Tree Depth (ply)'), vg.yLabel('Avg Count'),
      );
      const legend = el('div', { style: { fontSize: '12px', padding: '8px 0', display: 'flex', gap: '16px' } },
        el('span', { style: { color: COLORS[0] } }, '● SEE'),
        el('span', { style: { color: COLORS[3] } }, '● Delta'),
        el('span', { style: { color: COLORS[2] } }, '● NMP'),
      );
      plotContainer.appendChild(legend);
      plotContainer.appendChild(plot);
    } else if (metric === 'tree_depth' && !treeTable) {
      plotContainer.appendChild(el('p', {}, 'Tree depth table not found.'));
    }
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderChart(val);
  }, selectedMetric);

  container.appendChild(controlsBar(controlGroup('View', metricSelect)));
  container.appendChild(panel('Pruning Analysis', plotContainer));
  await renderChart(selectedMetric);

  return container;
}
