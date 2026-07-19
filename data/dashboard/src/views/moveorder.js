/**
 * Move Ordering tab — Fail-high index histogram analysis.
 * Shows which move index caused beta cutoffs, by depth, tree ply, and across versions.
 */
import * as vg from '@uwdata/vgplot';
import { coordinator } from '@uwdata/mosaic-core';
import { sql, el, panel, plotPanel, summaryRow, controlsBar, controlGroup, nativeSelect, getSearchTable, getIterTable, getTreeTable, COLORS } from '../util.js';

const BUCKET_LABELS = ['Idx 0 (TT)', 'Idx 1', 'Idx 2', 'Idx 3', 'Idx 4-7', 'Idx 8+'];
const BUCKET_COLS = ['fh_index_0', 'fh_index_1', 'fh_index_2', 'fh_index_3', 'fh_index_4to7', 'fh_index_8plus'];
const RATIO_COLS = BUCKET_COLS.map(c => c + '_ratio');

const VIEW_OPTIONS = [
  { value: 'overview', label: 'Overview Distribution' },
  { value: 'by_iter_depth', label: 'By Iteration Depth' },
  { value: 'by_tree_depth', label: 'By Tree Ply' },
  { value: 'by_engine', label: 'By Engine Version' },
];

export async function renderMoveOrder() {
  const searchTable = getSearchTable();
  const iterTable = getIterTable();
  const treeTable = getTreeTable();

  if (!searchTable) {
    return el('div', { class: 'panel' }, el('p', {}, 'No search data table found.'));
  }

  const container = el('div', {});

  // Summary: overall distribution
  const [stats] = await sql(`
    SELECT
      COUNT(*) as n,
      SUM(total_fh_index_0) as idx0,
      SUM(total_fh_index_1) as idx1,
      SUM(total_fh_index_2) as idx2,
      SUM(total_fh_index_3) as idx3,
      SUM(total_fh_index_4to7) as idx4to7,
      SUM(total_fh_index_8plus) as idx8p,
      SUM(total_fail_highs) as total_fh
    FROM ${searchTable}
    WHERE total_fh_index_0 IS NOT NULL
  `);

  const totalFh = Number(stats?.total_fh || 0);
  const pct = (v) => totalFh > 0 ? (Number(v || 0) / totalFh * 100).toFixed(1) + '%' : '—';

  container.appendChild(summaryRow({
    'Searches (w/ data)': stats?.n,
    'Idx 0 (TT/Hash)': pct(stats?.idx0),
    'Idx 1': pct(stats?.idx1),
    'Idx 2': pct(stats?.idx2),
    'Idx 3': pct(stats?.idx3),
    'Idx 4-7': pct(stats?.idx4to7),
    'Idx 8+': pct(stats?.idx8p),
  }));

  // View selector
  let selectedView = 'overview';
  const plotContainer = el('div', {});

  async function renderChart(view) {
    plotContainer.innerHTML = '';

    if (view === 'overview') {
      // Stacked bar of overall bucket distribution
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW fh_dist AS
        SELECT
          'Idx 0 (TT)' as bucket, SUM(total_fh_index_0)::DOUBLE / NULLIF(SUM(total_fail_highs),0) as ratio, 0 as ord
        FROM ${searchTable} WHERE total_fh_index_0 IS NOT NULL
        UNION ALL SELECT 'Idx 1', SUM(total_fh_index_1)::DOUBLE / NULLIF(SUM(total_fail_highs),0), 1 FROM ${searchTable} WHERE total_fh_index_0 IS NOT NULL
        UNION ALL SELECT 'Idx 2', SUM(total_fh_index_2)::DOUBLE / NULLIF(SUM(total_fail_highs),0), 2 FROM ${searchTable} WHERE total_fh_index_0 IS NOT NULL
        UNION ALL SELECT 'Idx 3', SUM(total_fh_index_3)::DOUBLE / NULLIF(SUM(total_fail_highs),0), 3 FROM ${searchTable} WHERE total_fh_index_0 IS NOT NULL
        UNION ALL SELECT 'Idx 4-7', SUM(total_fh_index_4to7)::DOUBLE / NULLIF(SUM(total_fail_highs),0), 4 FROM ${searchTable} WHERE total_fh_index_0 IS NOT NULL
        UNION ALL SELECT 'Idx 8+', SUM(total_fh_index_8plus)::DOUBLE / NULLIF(SUM(total_fail_highs),0), 5 FROM ${searchTable} WHERE total_fh_index_0 IS NOT NULL
      `);
      const plot = vg.plot(
        vg.barX(vg.from('fh_dist'), { x: 'ratio', y: 'bucket', fill: 'bucket', sort: { y: 'ord' } }),
        vg.width(700), vg.height(250), vg.marginLeft(100),
        vg.xLabel('Fraction of Fail-Highs'), vg.yLabel('Move Index Bucket'),
        vg.colorDomain(BUCKET_LABELS), vg.colorRange(COLORS),
      );
      plotContainer.appendChild(plotPanel('Overall Fail-High Distribution by Move Index', plot));

      // Also show by completed depth as a line chart
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW fh_by_cdepth AS
        SELECT completed_depth as depth,
          AVG(fh_index_0_ratio) as idx0,
          AVG(fh_index_1_ratio) as idx1,
          AVG(fh_index_2_ratio) as idx2,
          AVG(fh_index_3_ratio) as idx3,
          AVG(fh_index_4to7_ratio) as idx4to7,
          AVG(fh_index_8plus_ratio) as idx8plus
        FROM ${searchTable}
        WHERE fh_index_0_ratio IS NOT NULL AND completed_depth BETWEEN 1 AND 25
        GROUP BY completed_depth ORDER BY depth
      `);
      const linePlot = vg.plot(
        vg.lineY(vg.from('fh_by_cdepth'), { x: 'depth', y: 'idx0', stroke: COLORS[0], marker: true }),
        vg.lineY(vg.from('fh_by_cdepth'), { x: 'depth', y: 'idx1', stroke: COLORS[1], marker: true }),
        vg.lineY(vg.from('fh_by_cdepth'), { x: 'depth', y: 'idx2', stroke: COLORS[2], marker: true }),
        vg.lineY(vg.from('fh_by_cdepth'), { x: 'depth', y: 'idx3', stroke: COLORS[3], marker: true }),
        vg.lineY(vg.from('fh_by_cdepth'), { x: 'depth', y: 'idx4to7', stroke: COLORS[4], marker: true }),
        vg.lineY(vg.from('fh_by_cdepth'), { x: 'depth', y: 'idx8plus', stroke: COLORS[5], marker: true }),
        vg.width(750), vg.height(320), vg.marginLeft(70),
        vg.xLabel('Completed Depth'), vg.yLabel('Fraction of Fail-Highs'),
      );
      const legend = el('div', { style: { fontSize: '12px', padding: '8px 0', display: 'flex', gap: '16px', flexWrap: 'wrap' } },
        ...BUCKET_LABELS.map((lbl, i) => el('span', { style: { color: COLORS[i] } }, `● ${lbl}`))
      );
      plotContainer.appendChild(legend);
      plotContainer.appendChild(plotPanel('Move Ordering Quality by Completed Depth', linePlot));

    } else if (view === 'by_iter_depth' && iterTable) {
      // Per iteration depth
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW fh_iter AS
        SELECT depth,
          AVG(fh_index_0_ratio) as idx0,
          AVG(fh_index_1_ratio) as idx1,
          AVG(fh_index_2_ratio) as idx2,
          AVG(fh_index_3_ratio) as idx3,
          AVG(fh_index_4to7_ratio) as idx4to7,
          AVG(fh_index_8plus_ratio) as idx8plus
        FROM ${iterTable}
        WHERE fh_index_0_ratio IS NOT NULL AND depth BETWEEN 1 AND 25
        GROUP BY depth ORDER BY depth
      `);
      const plot = vg.plot(
        vg.lineY(vg.from('fh_iter'), { x: 'depth', y: 'idx0', stroke: COLORS[0], marker: true }),
        vg.lineY(vg.from('fh_iter'), { x: 'depth', y: 'idx1', stroke: COLORS[1], marker: true }),
        vg.lineY(vg.from('fh_iter'), { x: 'depth', y: 'idx2', stroke: COLORS[2], marker: true }),
        vg.lineY(vg.from('fh_iter'), { x: 'depth', y: 'idx3', stroke: COLORS[3], marker: true }),
        vg.lineY(vg.from('fh_iter'), { x: 'depth', y: 'idx4to7', stroke: COLORS[4], marker: true }),
        vg.lineY(vg.from('fh_iter'), { x: 'depth', y: 'idx8plus', stroke: COLORS[5], marker: true }),
        vg.width(750), vg.height(320), vg.marginLeft(70),
        vg.xLabel('Iteration Depth'), vg.yLabel('Fraction of Fail-Highs'),
      );
      const legend = el('div', { style: { fontSize: '12px', padding: '8px 0', display: 'flex', gap: '16px', flexWrap: 'wrap' } },
        ...BUCKET_LABELS.map((lbl, i) => el('span', { style: { color: COLORS[i] } }, `● ${lbl}`))
      );
      plotContainer.appendChild(legend);
      plotContainer.appendChild(plotPanel('Fail-High Index Distribution by Iteration Depth', plot));

    } else if (view === 'by_tree_depth' && treeTable) {
      // Per tree ply
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW fh_tree AS
        SELECT depth as ply,
          AVG(fh_index_0_ratio) as idx0,
          AVG(fh_index_1_ratio) as idx1,
          AVG(fh_index_2_ratio) as idx2,
          AVG(fh_index_3_ratio) as idx3,
          AVG(fh_index_4to7_ratio) as idx4to7,
          AVG(fh_index_8plus_ratio) as idx8plus
        FROM ${treeTable}
        WHERE fh_index_0_ratio IS NOT NULL AND depth BETWEEN 1 AND 30
        GROUP BY depth ORDER BY depth
      `);
      const plot = vg.plot(
        vg.lineY(vg.from('fh_tree'), { x: 'ply', y: 'idx0', stroke: COLORS[0], marker: true }),
        vg.lineY(vg.from('fh_tree'), { x: 'ply', y: 'idx1', stroke: COLORS[1], marker: true }),
        vg.lineY(vg.from('fh_tree'), { x: 'ply', y: 'idx2', stroke: COLORS[2], marker: true }),
        vg.lineY(vg.from('fh_tree'), { x: 'ply', y: 'idx3', stroke: COLORS[3], marker: true }),
        vg.lineY(vg.from('fh_tree'), { x: 'ply', y: 'idx4to7', stroke: COLORS[4], marker: true }),
        vg.lineY(vg.from('fh_tree'), { x: 'ply', y: 'idx8plus', stroke: COLORS[5], marker: true }),
        vg.width(750), vg.height(320), vg.marginLeft(70),
        vg.xLabel('Tree Ply'), vg.yLabel('Fraction of Fail-Highs'),
      );
      const legend = el('div', { style: { fontSize: '12px', padding: '8px 0', display: 'flex', gap: '16px', flexWrap: 'wrap' } },
        ...BUCKET_LABELS.map((lbl, i) => el('span', { style: { color: COLORS[i] } }, `● ${lbl}`))
      );
      plotContainer.appendChild(legend);
      plotContainer.appendChild(plotPanel('Fail-High Index Distribution by Tree Ply (move ordering degradation at depth)', plot));

    } else if (view === 'by_engine') {
      // Compare across engine versions
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW fh_by_engine AS
        SELECT engine_name,
          SUM(total_fh_index_0)::DOUBLE / NULLIF(SUM(total_fail_highs), 0) as idx0_ratio,
          SUM(total_fh_index_1)::DOUBLE / NULLIF(SUM(total_fail_highs), 0) as idx1_ratio,
          SUM(total_fh_index_2)::DOUBLE / NULLIF(SUM(total_fail_highs), 0) as idx2_ratio,
          SUM(total_fh_index_3)::DOUBLE / NULLIF(SUM(total_fail_highs), 0) as idx3_ratio,
          SUM(total_fh_index_4to7)::DOUBLE / NULLIF(SUM(total_fail_highs), 0) as idx4to7_ratio,
          SUM(total_fh_index_8plus)::DOUBLE / NULLIF(SUM(total_fail_highs), 0) as idx8plus_ratio
        FROM ${searchTable}
        WHERE total_fh_index_0 IS NOT NULL
        GROUP BY engine_name
      `);
      const plot = vg.plot(
        vg.barY(vg.from('fh_by_engine'), { x: 'engine_name', y: 'idx0_ratio', fill: COLORS[0] }),
        vg.barY(vg.from('fh_by_engine'), { x: 'engine_name', y: 'idx1_ratio', fill: COLORS[1], offset: 'zero' }),
        vg.width(700), vg.height(300), vg.marginLeft(70),
        vg.xLabel('Engine Version'), vg.yLabel('Idx 0 (TT) Ratio'),
      );
      // Simpler: just show the TT/hash move hit rate per engine
      const plot2 = vg.plot(
        vg.barY(vg.from('fh_by_engine'), { x: 'engine_name', y: 'idx0_ratio', fill: COLORS[0] }),
        vg.width(700), vg.height(280), vg.marginLeft(70), vg.marginBottom(60),
        vg.xLabel('Engine Version'), vg.yLabel('TT Move Cutoff Rate (idx 0 / total FH)'),
        vg.xTickRotate(-30),
      );
      plotContainer.appendChild(plotPanel('TT/Hash Move Cutoff Rate by Engine Version', plot2));

    } else {
      plotContainer.appendChild(el('p', { style: { padding: '16px' } }, 'Required data table not available for this view.'));
    }
  }

  const selectEl = nativeSelect(VIEW_OPTIONS, (val) => {
    selectedView = val;
    renderChart(val);
  }, selectedView);

  container.appendChild(controlsBar(controlGroup('View', selectEl)));
  container.appendChild(plotContainer);
  await renderChart(selectedView);

  return container;
}
