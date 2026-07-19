/**
 * SPRT tab — experiment results and statistical testing with metric selection.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, fmt, metricCard, controlsBar, controlGroup, nativeSelect, summaryRow, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'elo', label: 'Elo Distribution' },
  { value: 'games', label: 'Games Played' },
  { value: 'book', label: 'By Opening Book' },
  { value: 'timeline', label: 'Timeline' },
];

export async function renderSprt() {
  const tables = window.__tables || [];
  if (!tables.includes('sprt_runs')) {
    return el('div', { class: 'panel' }, el('p', {}, 'No sprt_runs table found.'));
  }

  const container = el('div', {});

  // Summary stats
  const [counts] = await sql(`
    SELECT 
      COUNT(*) as total,
      COUNT(*) FILTER (WHERE result = 'H1') as h1_count,
      COUNT(*) FILTER (WHERE result = 'H0') as h0_count,
      COUNT(*) FILTER (WHERE result IS NULL OR result = '') as incomplete,
      AVG(elo_diff) as avg_elo,
      AVG(games_played) as avg_games
    FROM sprt_runs
  `);

  const metrics = el('div', { class: 'grid-4' },
    metricCard(fmt(counts.total), 'Total Tests'),
    metricCard(fmt(counts.h1_count), 'H1 (Improved)'),
    metricCard(fmt(counts.h0_count), 'H0 (No Gain)'),
    metricCard(fmt(counts.incomplete), 'Incomplete'),
  );
  container.appendChild(panel('SPRT Results', metrics));
  container.appendChild(summaryRow({
    'Avg Elo Diff': counts?.avg_elo != null ? Number(counts.avg_elo).toFixed(1) : null,
    'Avg Games': counts?.avg_games != null ? Number(counts.avg_games).toFixed(0) : null,
    'Pass Rate': counts?.total > 0 ? ((counts.h1_count / counts.total) * 100).toFixed(1) + '%' : null,
  }));

  // Metric selector for chart view
  let selectedMetric = 'elo';
  const plotContainer = el('div', {});

  async function renderChart(metric) {
    plotContainer.innerHTML = '';
    if (metric === 'elo') {
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW sprt_with_elo AS SELECT * FROM sprt_runs WHERE elo_diff IS NOT NULL`);
      const hist = vg.plot(
        vg.rectY(vg.from('sprt_with_elo'), { x: vg.bin('elo_diff'), y: vg.count(), fill: COLORS[0] }),
        vg.width(700), vg.height(260), vg.marginLeft(50),
        vg.xLabel('Elo Difference'), vg.yLabel('Count'),
      );
      plotContainer.appendChild(hist);
    } else if (metric === 'games') {
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW sprt_with_games AS SELECT * FROM sprt_runs WHERE games_played IS NOT NULL`);
      const hist = vg.plot(
        vg.rectY(vg.from('sprt_with_games'), { x: vg.bin('games_played'), y: vg.count(), fill: COLORS[1] }),
        vg.width(700), vg.height(260), vg.marginLeft(50),
        vg.xLabel('Games Played'), vg.yLabel('Count'),
      );
      plotContainer.appendChild(hist);
    } else if (metric === 'book') {
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW sprt_by_book AS SELECT opening_book, COUNT(*) as count FROM sprt_runs GROUP BY opening_book ORDER BY count DESC`);
      const bar = vg.plot(
        vg.barX(vg.from('sprt_by_book'), { x: 'count', y: 'opening_book', fill: COLORS[3], sort: { y: '-x' } }),
        vg.width(700), vg.height(220), vg.marginLeft(160),
        vg.xLabel('Tests'), vg.yLabel('Opening Book'),
      );
      plotContainer.appendChild(bar);
    } else if (metric === 'timeline') {
      const hasTime = await sql(`SELECT column_name FROM information_schema.columns WHERE table_name='sprt_runs' AND column_name='start_time_utc' LIMIT 1`).then(r => r.length > 0).catch(() => false);
      if (!hasTime) {
        plotContainer.appendChild(el('p', { class: 'panel' }, 'No start_time_utc column available.'));
        return;
      }
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW sprt_timeline AS SELECT * FROM sprt_runs WHERE start_time_utc IS NOT NULL`);
      const scatter = vg.plot(
        vg.dot(vg.from('sprt_timeline'), {
          x: 'start_time_utc', y: 'elo_diff', fill: 'result', r: 6, opacity: 0.7,
        }),
        vg.width(750), vg.height(300), vg.marginLeft(60),
        vg.xLabel('Date'), vg.yLabel('Elo Diff'),
        vg.colorDomain(['H1', 'H0']),
        vg.colorRange([COLORS[1], COLORS[2]]),
      );
      plotContainer.appendChild(scatter);
    }
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderChart(val);
  }, selectedMetric);

  container.appendChild(controlsBar(controlGroup('View', metricSelect)));
  container.appendChild(panel('SPRT Analysis', plotContainer));
  await renderChart(selectedMetric);

  return container;
}
