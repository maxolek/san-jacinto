/**
 * Games tab — game results, openings, and outcomes analysis.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, fmt, metricCard, controlsBar, controlGroup, nativeSelect, summaryRow, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'run_time_s', label: 'Duration (s)' },
  { value: 'result', label: 'Result' },
  { value: 'termination', label: 'Termination' },
];

export async function renderGames() {
  const tables = window.__tables || [];
  if (!tables.includes('game_stats')) {
    return el('div', { class: 'panel' }, el('p', {}, 'No game_stats table found.'));
  }

  const container = el('div', {});

  // Summary metrics
  const [counts] = await sql(`
    SELECT 
      COUNT(*) as total,
      COUNT(*) FILTER (WHERE result = 'white') as white_wins,
      COUNT(*) FILTER (WHERE result = 'black') as black_wins,
      COUNT(*) FILTER (WHERE result = 'draw') as draws,
      AVG(run_time_s) as avg_time,
      COUNT(DISTINCT opening) as n_openings
    FROM game_stats
  `);

  const metrics = el('div', { class: 'grid-4' },
    metricCard(fmt(counts.total), 'Total Games'),
    metricCard(fmt(counts.white_wins), 'White Wins'),
    metricCard(fmt(counts.black_wins), 'Black Wins'),
    metricCard(fmt(counts.draws), 'Draws'),
  );
  container.appendChild(panel('Game Results', metrics));

  container.appendChild(summaryRow({
    'Avg Duration': counts?.avg_time != null ? counts.avg_time.toFixed(1) + 's' : null,
    'Unique Openings': counts?.n_openings,
    'White Win %': counts?.total > 0 ? ((counts.white_wins / counts.total) * 100).toFixed(1) + '%' : null,
    'Draw %': counts?.total > 0 ? ((counts.draws / counts.total) * 100).toFixed(1) + '%' : null,
  }));

  // Metric selector for distribution plot
  let selectedMetric = 'run_time_s';
  const plotContainer = el('div', {});

  async function renderDistribution(metric) {
    plotContainer.innerHTML = '';
    if (metric === 'run_time_s') {
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW games_with_time AS SELECT * FROM game_stats WHERE run_time_s IS NOT NULL`);
      const hist = vg.plot(
        vg.rectY(vg.from('games_with_time'), { x: vg.bin('run_time_s'), y: vg.count(), fill: COLORS[0] }),
        vg.width(700), vg.height(260), vg.marginLeft(55),
        vg.xLabel('Game Duration (seconds)'), vg.yLabel('Count'),
      );
      plotContainer.appendChild(hist);
    } else if (metric === 'result') {
      const bar = vg.plot(
        vg.barX(vg.from('game_stats'), { x: vg.count(), y: 'result', fill: 'result' }),
        vg.width(700), vg.height(150), vg.marginLeft(80),
        vg.xLabel('Count'), vg.yLabel('Result'),
        vg.colorDomain(['white', 'black', 'draw']),
        vg.colorRange([COLORS[4], COLORS[2], COLORS[0]]),
      );
      plotContainer.appendChild(bar);
    } else if (metric === 'termination') {
      const bar = vg.plot(
        vg.barX(vg.from('game_stats'), { x: vg.count(), y: 'termination', fill: COLORS[1], sort: { y: '-x' } }),
        vg.width(700), vg.height(200), vg.marginLeft(120),
        vg.xLabel('Count'), vg.yLabel('Termination'),
      );
      plotContainer.appendChild(bar);
    }
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderDistribution(val);
  }, selectedMetric);

  container.appendChild(controlsBar(controlGroup('View', metricSelect)));
  container.appendChild(panel('Distribution', plotContainer));
  await renderDistribution(selectedMetric);

  // Opening frequency (top 20)
  await coordinator().exec(`CREATE OR REPLACE TEMP VIEW top_openings AS SELECT opening, COUNT(*) as count FROM game_stats GROUP BY opening ORDER BY count DESC LIMIT 20`);
  const openingBar = vg.plot(
    vg.barX(
      vg.from('top_openings'),
      { x: 'count', y: 'opening', fill: COLORS[3], sort: { y: '-x' } }
    ),
    vg.width(700),
    vg.height(400),
    vg.marginLeft(200),
    vg.xLabel('Games'),
    vg.yLabel('Opening'),
  );
  container.appendChild(plotPanel('Top 20 Openings', openingBar));

  return container;
}
