/**
 * Eval Quality tab — Stockfish comparison, eval accuracy, and move agreement.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, fmt, metricCard, controlsBar, controlGroup, nativeSelect, summaryRow, getSearchTable, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'eval_diff', label: 'Eval Difference' },
  { value: 'move_rank', label: 'Move Rank' },
  { value: 'agreement_by_depth', label: 'Agreement by Depth' },
  { value: 'scatter', label: 'Engine vs SF Eval' },
];

export async function renderQuality() {
  const searchTable = getSearchTable();
  if (!searchTable) {
    return el('div', { class: 'panel' }, el('p', {}, 'No search data table found.'));
  }

  const container = el('div', {});

  // Check if SF data exists
  const [sfCheck] = await sql(`SELECT COUNT(*) as n FROM ${searchTable} WHERE sf_eval IS NOT NULL`);
  if (!sfCheck || sfCheck.n === 0) {
    container.appendChild(panel('Eval Quality',
      el('p', { style: { color: 'var(--text-sec)' } },
        'No Stockfish ground-truth data available. Run transform_positions.py to compute SF evaluations.'
      )
    ));
    return container;
  }

  // Summary stats
  const [stats] = await sql(`
    SELECT 
      COUNT(*) as total_with_sf,
      AVG(ABS(eval - sf_eval)) as mae,
      MEDIAN(ABS(eval - sf_eval)) as median_ae,
      AVG(CASE WHEN move = sf_best_move THEN 1.0 ELSE 0.0 END) as best_move_match_rate,
      STDDEV(eval - sf_eval) as eval_diff_std
    FROM ${searchTable}
    WHERE sf_eval IS NOT NULL AND eval IS NOT NULL
  `);

  const metrics = el('div', { class: 'grid-4' },
    metricCard(fmt(stats.total_with_sf), 'Positions w/ SF'),
    metricCard(stats.mae != null ? Number(stats.mae).toFixed(1) + ' cp' : '—', 'Mean |Eval Diff|'),
    metricCard(stats.median_ae != null ? Number(stats.median_ae).toFixed(1) + ' cp' : '—', 'Median |Eval Diff|'),
    metricCard(stats.best_move_match_rate != null ? (Number(stats.best_move_match_rate) * 100).toFixed(1) + '%' : '—', 'Best Move Match'),
  );
  container.appendChild(panel('Eval Accuracy Summary', metrics));

  container.appendChild(summaryRow({
    'Eval Diff StdDev': stats.eval_diff_std != null ? Number(stats.eval_diff_std).toFixed(1) + ' cp' : null,
  }));

  // Metric selector for charts
  let selectedMetric = 'eval_diff';
  const plotContainer = el('div', {});

  async function renderChart(metric) {
    plotContainer.innerHTML = '';

    if (metric === 'eval_diff') {
      // Eval difference distribution
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW eval_diff_view AS SELECT (eval - sf_eval) as eval_diff FROM ${searchTable} WHERE sf_eval IS NOT NULL AND eval IS NOT NULL AND ABS(eval - sf_eval) < 500`);
      const hist = vg.plot(
        vg.rectY(vg.from('eval_diff_view'), { x: vg.bin('eval_diff'), y: vg.count(), fill: COLORS[0] }),
        vg.width(750), vg.height(280), vg.marginLeft(55),
        vg.xLabel('Eval Difference (Engine - SF, centipawns)'), vg.yLabel('Count'),
      );
      plotContainer.appendChild(hist);

    } else if (metric === 'move_rank') {
      // Engine move rank distribution (does engine_move_rank exist?)
      try {
        await coordinator().exec(`CREATE OR REPLACE TEMP VIEW move_rank_dist AS
          SELECT 
            CASE 
              WHEN move = sf_best_move THEN 'Rank 1 (Best)'
              ELSE 'Not Best'
            END as rank_label,
            COUNT(*) as count
          FROM ${searchTable}
          WHERE sf_best_move IS NOT NULL AND move IS NOT NULL
          GROUP BY rank_label`);
        const bar = vg.plot(
          vg.barX(vg.from('move_rank_dist'), { x: 'count', y: 'rank_label', fill: COLORS[1] }),
          vg.width(750), vg.height(150), vg.marginLeft(140),
          vg.xLabel('Count'), vg.yLabel('Move Rank'),
        );
        plotContainer.appendChild(bar);
      } catch (e) {
        plotContainer.appendChild(el('p', {}, 'Move rank data not available.'));
      }

    } else if (metric === 'agreement_by_depth') {
      // Best move agreement rate by depth
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW agreement_by_depth AS
        SELECT completed_depth as depth, 
          AVG(CASE WHEN move = sf_best_move THEN 1.0 ELSE 0.0 END) as agreement_rate,
          COUNT(*) as n
        FROM ${searchTable}
        WHERE sf_best_move IS NOT NULL AND move IS NOT NULL AND completed_depth BETWEEN 1 AND 30
        GROUP BY completed_depth
        HAVING n >= 5
        ORDER BY depth`);
      const line = vg.plot(
        vg.lineY(vg.from('agreement_by_depth'), { x: 'depth', y: 'agreement_rate', stroke: COLORS[1], marker: true }),
        vg.width(750), vg.height(280), vg.marginLeft(60),
        vg.xLabel('Search Depth'), vg.yLabel('Best Move Agreement Rate'),
      );
      plotContainer.appendChild(line);

    } else if (metric === 'scatter') {
      // Engine eval vs SF eval scatter
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW eval_scatter AS SELECT eval, sf_eval FROM ${searchTable} WHERE sf_eval IS NOT NULL AND eval IS NOT NULL AND ABS(eval) < 1000 AND ABS(sf_eval) < 1000 USING SAMPLE 3000`);
      const scatter = vg.plot(
        vg.dot(vg.from('eval_scatter'), { x: 'sf_eval', y: 'eval', fill: COLORS[2], opacity: 0.4, r: 3 }),
        vg.ruleY([0], { stroke: '#555' }),
        vg.ruleX([0], { stroke: '#555' }),
        vg.width(500), vg.height(500), vg.marginLeft(60),
        vg.xLabel('Stockfish Eval (cp)'), vg.yLabel('Engine Eval (cp)'),
      );
      plotContainer.appendChild(scatter);
    }
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderChart(val);
  }, selectedMetric);

  container.appendChild(controlsBar(controlGroup('View', metricSelect)));
  container.appendChild(panel('Eval Quality Analysis', plotContainer));
  await renderChart(selectedMetric);

  // Blunder detection (large eval diff)
  const blunders = await sql(`
    SELECT COUNT(*) as count FROM ${searchTable}
    WHERE sf_eval IS NOT NULL AND eval IS NOT NULL AND ABS(eval - sf_eval) > 200
  `);
  if (blunders[0]?.count > 0) {
    container.appendChild(summaryRow({
      'Positions with |Eval Diff| > 200cp': blunders[0].count,
    }));
  }

  return container;
}
