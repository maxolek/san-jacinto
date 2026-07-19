/**
 * Positions tab — performance breakdown by position type, game phase, and characteristics.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, fmt, controlsBar, controlGroup, nativeSelect, summaryRow, getSearchTable, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'by_phase', label: 'By Game Phase' },
  { value: 'by_type', label: 'By Position Type' },
  { value: 'accuracy_by_phase', label: 'Eval Accuracy by Phase' },
  { value: 'depth_by_type', label: 'Depth by Position Type' },
  { value: 'tactical_vs_positional', label: 'Tactical vs Positional' },
];

export async function renderPositions() {
  const searchTable = getSearchTable();
  if (!searchTable) {
    return el('div', { class: 'panel' }, el('p', {}, 'No search data table found.'));
  }

  const container = el('div', {});

  // Check if position features are available
  const [check] = await sql(`
    SELECT COUNT(*) as n FROM ${searchTable}
    WHERE game_phase IS NOT NULL OR position_type IS NOT NULL
  `);
  if (!check || check.n === 0) {
    container.appendChild(panel('Position Analysis',
      el('p', { style: { color: 'var(--text-sec)' } },
        'No position feature data available. Ensure position_features table is populated and search_features is rebuilt.'
      )
    ));
    return container;
  }

  // Summary
  const [stats] = await sql(`
    SELECT COUNT(*) as total,
      COUNT(DISTINCT game_phase) as n_phases,
      COUNT(DISTINCT position_type) as n_types,
      AVG(completed_depth) as avg_depth,
      AVG(total_time_ms) as avg_time
    FROM ${searchTable}
    WHERE game_phase IS NOT NULL
  `);
  container.appendChild(summaryRow({
    'Positions with Features': stats?.total,
    'Game Phases': stats?.n_phases,
    'Position Types': stats?.n_types,
    'Avg Depth': stats?.avg_depth != null ? Number(stats.avg_depth).toFixed(1) : null,
    'Avg Time': stats?.avg_time != null ? Number(stats.avg_time).toFixed(0) + ' ms' : null,
  }));

  // Metric selector
  let selectedMetric = 'by_phase';
  const plotContainer = el('div', {});

  async function renderChart(metric) {
    plotContainer.innerHTML = '';

    if (metric === 'by_phase') {
      // Distribution of searches by game phase
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW phase_dist AS
        SELECT game_phase, COUNT(*) as count, AVG(completed_depth) as avg_depth, AVG(nps) as avg_nps
        FROM ${searchTable}
        WHERE game_phase IS NOT NULL AND game_phase != ''
        GROUP BY game_phase ORDER BY count DESC`);
      const bar = vg.plot(
        vg.barX(vg.from('phase_dist'), { x: 'count', y: 'game_phase', fill: COLORS[0], sort: { y: '-x' } }),
        vg.width(750), vg.height(200), vg.marginLeft(120),
        vg.xLabel('Searches'), vg.yLabel('Game Phase'),
      );
      plotContainer.appendChild(bar);

      // Also show avg depth by phase
      const depthBar = vg.plot(
        vg.barX(vg.from('phase_dist'), { x: 'avg_depth', y: 'game_phase', fill: COLORS[4], sort: { y: '-x' } }),
        vg.width(750), vg.height(200), vg.marginLeft(120),
        vg.xLabel('Avg Depth'), vg.yLabel('Game Phase'),
      );
      plotContainer.appendChild(depthBar);

    } else if (metric === 'by_type') {
      // Distribution by position type
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW type_dist AS
        SELECT position_type, COUNT(*) as count
        FROM ${searchTable}
        WHERE position_type IS NOT NULL AND position_type != ''
        GROUP BY position_type ORDER BY count DESC LIMIT 15`);
      const bar = vg.plot(
        vg.barX(vg.from('type_dist'), { x: 'count', y: 'position_type', fill: COLORS[2], sort: { y: '-x' } }),
        vg.width(750), vg.height(350), vg.marginLeft(160),
        vg.xLabel('Searches'), vg.yLabel('Position Type'),
      );
      plotContainer.appendChild(bar);

    } else if (metric === 'accuracy_by_phase') {
      // Eval accuracy (eval_diff) by game phase — requires SF data
      const [sfCheck] = await sql(`SELECT COUNT(*) as n FROM ${searchTable} WHERE sf_eval IS NOT NULL AND game_phase IS NOT NULL`);
      if (!sfCheck || sfCheck.n === 0) {
        plotContainer.appendChild(el('p', {}, 'Requires Stockfish eval data + game phase. Run transform_positions first.'));
        return;
      }
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW acc_by_phase AS
        SELECT game_phase,
          AVG(ABS(eval_diff)) as mae,
          AVG(CASE WHEN best_move = sf_best_move THEN 1.0 ELSE 0.0 END) as match_rate,
          COUNT(*) as n
        FROM ${searchTable}
        WHERE sf_eval IS NOT NULL AND game_phase IS NOT NULL AND game_phase != ''
        GROUP BY game_phase`);
      const bar = vg.plot(
        vg.barX(vg.from('acc_by_phase'), { x: 'mae', y: 'game_phase', fill: COLORS[3], sort: { y: '-x' } }),
        vg.width(750), vg.height(200), vg.marginLeft(120),
        vg.xLabel('Mean Absolute Eval Error (cp)'), vg.yLabel('Game Phase'),
      );
      plotContainer.appendChild(bar);

      const matchBar = vg.plot(
        vg.barX(vg.from('acc_by_phase'), { x: 'match_rate', y: 'game_phase', fill: COLORS[1], sort: { y: '-x' } }),
        vg.width(750), vg.height(200), vg.marginLeft(120),
        vg.xLabel('Best Move Match Rate'), vg.yLabel('Game Phase'),
      );
      plotContainer.appendChild(matchBar);

    } else if (metric === 'depth_by_type') {
      // Avg depth and time by position type
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW depth_by_type AS
        SELECT position_type, AVG(completed_depth) as avg_depth, AVG(total_time_ms) as avg_time, COUNT(*) as n
        FROM ${searchTable}
        WHERE position_type IS NOT NULL AND position_type != ''
        GROUP BY position_type HAVING n >= 5
        ORDER BY avg_depth DESC LIMIT 15`);
      const bar = vg.plot(
        vg.barX(vg.from('depth_by_type'), { x: 'avg_depth', y: 'position_type', fill: COLORS[4], sort: { y: '-x' } }),
        vg.width(750), vg.height(350), vg.marginLeft(160),
        vg.xLabel('Avg Depth'), vg.yLabel('Position Type'),
      );
      plotContainer.appendChild(bar);

    } else if (metric === 'tactical_vs_positional') {
      // Scatter: tactical score vs positional score, colored by eval accuracy
      const [hasTact] = await sql(`SELECT COUNT(*) as n FROM ${searchTable} WHERE position_tactical_score IS NOT NULL`);
      if (!hasTact || hasTact.n === 0) {
        plotContainer.appendChild(el('p', {}, 'No tactical/positional scores available.'));
        return;
      }
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW tact_vs_pos AS
        SELECT position_tactical_score as tactical, position_positional_score as positional,
          completed_depth as depth, nps
        FROM ${searchTable}
        WHERE position_tactical_score IS NOT NULL AND position_positional_score IS NOT NULL
        USING SAMPLE 3000`);
      const scatter = vg.plot(
        vg.dot(vg.from('tact_vs_pos'), { x: 'tactical', y: 'positional', fill: 'depth', opacity: 0.5, r: 4 }),
        vg.width(600), vg.height(400), vg.marginLeft(60),
        vg.xLabel('Tactical Score'), vg.yLabel('Positional Score'),
        vg.colorLegend(true),
      );
      plotContainer.appendChild(scatter);
    }
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderChart(val);
  }, selectedMetric);

  container.appendChild(controlsBar(controlGroup('View', metricSelect)));
  container.appendChild(panel('Position Performance', plotContainer));
  await renderChart(selectedMetric);

  return container;
}
