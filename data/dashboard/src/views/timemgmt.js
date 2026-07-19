/**
 * Time Management tab — time allocation analysis across iterations and moves.
 */
import { coordinator, Selection } from '@uwdata/mosaic-core';
import * as vg from '@uwdata/vgplot';
import { sql, el, panel, plotPanel, fmt, controlsBar, controlGroup, nativeSelect, summaryRow, getSearchTable, getIterTable, COLORS } from '../util.js';

const METRIC_OPTIONS = [
  { value: 'time_by_depth', label: 'Time by Depth' },
  { value: 'time_efficiency', label: 'NPS by Depth' },
  { value: 'time_distribution', label: 'Search Duration Distribution' },
  { value: 'time_vs_nodes', label: 'Time vs Nodes' },
];

export async function renderTimeMgmt() {
  const searchTable = getSearchTable();
  const iterTable = getIterTable();

  if (!searchTable) {
    return el('div', { class: 'panel' }, el('p', {}, 'No search data table found.'));
  }

  const container = el('div', {});

  // Summary
  const [stats] = await sql(`
    SELECT COUNT(*) as total,
      AVG(total_time_ms) as avg_time,
      MEDIAN(total_time_ms) as med_time,
      MAX(total_time_ms) as max_time,
      AVG(nps) as avg_nps,
      AVG(completed_depth) as avg_depth
    FROM ${searchTable}
    WHERE total_time_ms IS NOT NULL
  `);
  container.appendChild(summaryRow({
    'Searches': stats?.total,
    'Avg Time': stats?.avg_time != null ? Number(stats.avg_time).toFixed(0) + ' ms' : null,
    'Median Time': stats?.med_time != null ? Number(stats.med_time).toFixed(0) + ' ms' : null,
    'Max Time': stats?.max_time != null ? Number(stats.max_time).toFixed(0) + ' ms' : null,
    'Avg NPS': stats?.avg_nps,
    'Avg Depth': stats?.avg_depth != null ? Number(stats.avg_depth).toFixed(1) : null,
  }));

  // Metric selector
  let selectedMetric = 'time_by_depth';
  const plotContainer = el('div', {});

  async function renderChart(metric) {
    plotContainer.innerHTML = '';

    if (metric === 'time_by_depth') {
      if (iterTable) {
        // Time allocation per iteration depth
        const line = vg.plot(
          vg.lineY(vg.from(iterTable), {
            x: 'depth', y: vg.avg('time_ms'), stroke: COLORS[0], marker: true,
          }),
          vg.width(750), vg.height(300), vg.marginLeft(70),
          vg.xLabel('Iteration Depth'), vg.yLabel('Avg Time (ms)'),
        );
        plotContainer.appendChild(line);
      } else {
        // Fallback: use search-level data
        await coordinator().exec(`CREATE OR REPLACE TEMP VIEW time_by_depth_agg AS
          SELECT completed_depth as depth, AVG(total_time_ms) as avg_time, COUNT(*) as n
          FROM ${searchTable} WHERE completed_depth BETWEEN 1 AND 30
          GROUP BY completed_depth ORDER BY depth`);
        const line = vg.plot(
          vg.lineY(vg.from('time_by_depth_agg'), { x: 'depth', y: 'avg_time', stroke: COLORS[0], marker: true }),
          vg.width(750), vg.height(300), vg.marginLeft(70),
          vg.xLabel('Completed Depth'), vg.yLabel('Avg Total Time (ms)'),
        );
        plotContainer.appendChild(line);
      }

    } else if (metric === 'time_efficiency') {
      if (iterTable) {
        // NPS by iteration depth (does time allocation get more efficient at deeper depths?)
        const line = vg.plot(
          vg.lineY(vg.from(iterTable), {
            x: 'depth', y: vg.avg('nps'), stroke: COLORS[1], marker: true,
          }),
          vg.width(750), vg.height(300), vg.marginLeft(70),
          vg.xLabel('Iteration Depth'), vg.yLabel('Avg NPS'),
        );
        plotContainer.appendChild(line);
      } else {
        await coordinator().exec(`CREATE OR REPLACE TEMP VIEW nps_by_depth AS
          SELECT completed_depth as depth, AVG(nps) as avg_nps
          FROM ${searchTable} WHERE nps IS NOT NULL AND completed_depth BETWEEN 1 AND 30
          GROUP BY completed_depth ORDER BY depth`);
        const line = vg.plot(
          vg.lineY(vg.from('nps_by_depth'), { x: 'depth', y: 'avg_nps', stroke: COLORS[1], marker: true }),
          vg.width(750), vg.height(300), vg.marginLeft(70),
          vg.xLabel('Depth'), vg.yLabel('Avg NPS'),
        );
        plotContainer.appendChild(line);
      }

    } else if (metric === 'time_distribution') {
      // Histogram of search durations
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW time_dist AS SELECT total_time_ms FROM ${searchTable} WHERE total_time_ms IS NOT NULL AND total_time_ms BETWEEN 0 AND 60000`);
      const hist = vg.plot(
        vg.rectY(vg.from('time_dist'), { x: vg.bin('total_time_ms'), y: vg.count(), fill: COLORS[2] }),
        vg.width(750), vg.height(280), vg.marginLeft(55),
        vg.xLabel('Search Duration (ms)'), vg.yLabel('Count'),
      );
      plotContainer.appendChild(hist);

    } else if (metric === 'time_vs_nodes') {
      // Scatter: time vs nodes (shows efficiency)
      await coordinator().exec(`CREATE OR REPLACE TEMP VIEW time_nodes_scatter AS SELECT total_time_ms, total_nodes FROM ${searchTable} WHERE total_time_ms IS NOT NULL AND total_nodes IS NOT NULL USING SAMPLE 3000`);
      const scatter = vg.plot(
        vg.dot(vg.from('time_nodes_scatter'), { x: 'total_time_ms', y: 'total_nodes', fill: COLORS[4], opacity: 0.3, r: 3 }),
        vg.width(750), vg.height(320), vg.marginLeft(70),
        vg.xLabel('Time (ms)'), vg.yLabel('Total Nodes'),
        vg.xScale('log'), vg.yScale('log'),
      );
      plotContainer.appendChild(scatter);
    }
  }

  const metricSelect = nativeSelect(METRIC_OPTIONS, async (val) => {
    selectedMetric = val;
    await renderChart(val);
  }, selectedMetric);

  container.appendChild(controlsBar(controlGroup('View', metricSelect)));
  container.appendChild(panel('Time Management', plotContainer));
  await renderChart(selectedMetric);

  // Time fraction spent per depth (cumulative)
  if (iterTable) {
    await coordinator().exec(`CREATE OR REPLACE TEMP VIEW time_fraction AS
      SELECT depth, SUM(time_ms) as total_time FROM ${iterTable} GROUP BY depth ORDER BY depth`);
    const area = vg.plot(
      vg.areaY(vg.from('time_fraction'), { x: 'depth', y: 'total_time', fill: COLORS[0], opacity: 0.5 }),
      vg.lineY(vg.from('time_fraction'), { x: 'depth', y: 'total_time', stroke: COLORS[0] }),
      vg.width(750), vg.height(250), vg.marginLeft(70),
      vg.xLabel('Iteration Depth'), vg.yLabel('Total Time (ms)'),
    );
    container.appendChild(plotPanel('Cumulative Time by Depth', area));
  }

  return container;
}
