/**
 * Openings tab — opening explorer with win rates, depth, and frequency analysis.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, fmt, controlsBar, controlGroup, nativeSelect, summaryRow, getSearchTable, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'frequency', label: 'Opening Frequency' },
  { value: 'winrate', label: 'Win Rate by Opening' },
  { value: 'depth', label: 'Avg Depth by Opening' },
  { value: 'eco', label: 'ECO Code Distribution' },
];

export async function renderOpenings() {
  const tables = window.__tables || [];
  const searchTable = getSearchTable();

  // Need either game_stats (with opening) or search_features (with game_opening)
  const hasGames = tables.includes('game_stats');
  const hasFeatures = searchTable === 'search_features';

  if (!hasGames && !hasFeatures) {
    return el('div', { class: 'panel' }, el('p', {}, 'No game or search data with opening information found.'));
  }

  const container = el('div', {});

  // Summary
  if (hasGames) {
    const [stats] = await sql(`
      SELECT COUNT(*) as total_games,
        COUNT(DISTINCT opening) as n_openings,
        COUNT(DISTINCT opening_eco) as n_eco
      FROM game_stats WHERE opening IS NOT NULL AND opening != ''
    `);
    container.appendChild(summaryRow({
      'Games with Opening Data': stats?.total_games,
      'Unique Openings': stats?.n_openings,
      'ECO Codes': stats?.n_eco,
    }));
  }

  // Metric selector
  let selectedMetric = 'frequency';
  const plotContainer = el('div', {});

  async function renderChart(metric) {
    plotContainer.innerHTML = '';

    if (metric === 'frequency') {
      if (hasGames) {
        await coordinator().exec(`CREATE OR REPLACE TEMP VIEW opening_freq AS SELECT opening, COUNT(*) as count FROM game_stats WHERE opening IS NOT NULL AND opening != '' GROUP BY opening ORDER BY count DESC LIMIT 25`);
      } else {
        await coordinator().exec(`CREATE OR REPLACE TEMP VIEW opening_freq AS SELECT game_opening as opening, COUNT(*) as count FROM ${searchTable} WHERE game_opening IS NOT NULL AND game_opening != '' GROUP BY game_opening ORDER BY count DESC LIMIT 25`);
      }
      const bar = vg.plot(
        vg.barX(vg.from('opening_freq'), { x: 'count', y: 'opening', fill: COLORS[0], sort: { y: '-x' } }),
        vg.width(750), vg.height(500), vg.marginLeft(220),
        vg.xLabel('Games'), vg.yLabel('Opening'),
      );
      plotContainer.appendChild(bar);

    } else if (metric === 'winrate') {
      if (hasGames) {
        await coordinator().exec(`CREATE OR REPLACE TEMP VIEW opening_winrate AS
          SELECT opening,
            COUNT(*) as games,
            AVG(CASE WHEN result = 'white' THEN 1.0 WHEN result = 'draw' THEN 0.5 ELSE 0.0 END) as white_score
          FROM game_stats WHERE opening IS NOT NULL AND opening != ''
          GROUP BY opening HAVING games >= 3
          ORDER BY white_score DESC LIMIT 25`);
        const bar = vg.plot(
          vg.barX(vg.from('opening_winrate'), { x: 'white_score', y: 'opening', fill: COLORS[3], sort: { y: '-x' } }),
          vg.width(750), vg.height(500), vg.marginLeft(220),
          vg.xLabel('White Score (1=win, 0.5=draw, 0=loss)'), vg.yLabel('Opening'),
        );
        plotContainer.appendChild(bar);
      } else {
        plotContainer.appendChild(el('p', {}, 'Win rate requires game_stats table.'));
      }

    } else if (metric === 'depth') {
      if (hasFeatures) {
        await coordinator().exec(`CREATE OR REPLACE TEMP VIEW opening_depth AS
          SELECT game_opening as opening, AVG(completed_depth) as avg_depth, COUNT(*) as n
          FROM ${searchTable}
          WHERE game_opening IS NOT NULL AND game_opening != ''
          GROUP BY game_opening HAVING n >= 5
          ORDER BY avg_depth DESC LIMIT 25`);
        const bar = vg.plot(
          vg.barX(vg.from('opening_depth'), { x: 'avg_depth', y: 'opening', fill: COLORS[4], sort: { y: '-x' } }),
          vg.width(750), vg.height(500), vg.marginLeft(220),
          vg.xLabel('Avg Depth'), vg.yLabel('Opening'),
        );
        plotContainer.appendChild(bar);
      } else {
        plotContainer.appendChild(el('p', {}, 'Depth by opening requires search_features table.'));
      }

    } else if (metric === 'eco') {
      const source = hasGames ? 'game_stats' : searchTable;
      const ecoCol = hasGames ? 'opening_eco' : 'game_eco';
      try {
        await coordinator().exec(`CREATE OR REPLACE TEMP VIEW eco_dist AS
          SELECT SUBSTRING(${ecoCol}, 1, 1) as eco_letter, COUNT(*) as count
          FROM ${source}
          WHERE ${ecoCol} IS NOT NULL AND ${ecoCol} != ''
          GROUP BY eco_letter ORDER BY eco_letter`);
        const bar = vg.plot(
          vg.barY(vg.from('eco_dist'), { x: 'eco_letter', y: 'count', fill: COLORS[2] }),
          vg.width(750), vg.height(280), vg.marginLeft(60),
          vg.xLabel('ECO Category (A-E)'), vg.yLabel('Count'),
        );
        plotContainer.appendChild(bar);
      } catch (e) {
        plotContainer.appendChild(el('p', {}, 'ECO code data not available.'));
      }
    }
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderChart(val);
  }, selectedMetric);

  container.appendChild(controlsBar(controlGroup('View', metricSelect)));
  container.appendChild(panel('Opening Explorer', plotContainer));
  await renderChart(selectedMetric);

  return container;
}
