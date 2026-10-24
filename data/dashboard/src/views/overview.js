/** Overview: every selected engine version at a glance. */
import * as vg from '@uwdata/vgplot';
import { dataset, datasetInfo } from '../datasets.js';
import { query } from '../db.js';
import { engineById, engineWhere, engines, sqlIdList } from '../engines.js';
import { coreFilter, coreRows } from '../settings.js';
import { isLogBinned, scoreFromWDL } from '../stats.js';
import { chartWidth, dataTable, el, emptyNote, engineColors, engineLegend, engineName, fmt, intTicks, panel, plot,
  reactive, selectInput, sourceNote, statLine } from '../ui.js';

export const HIST_METRICS = [
  { value: 'depth', label: 'Completed depth', axis: 'Completed depth (plies)' },
  { value: 'nodes', label: 'Nodes per search', axis: 'Nodes per search' },
  { value: 'time_ms', label: 'Time per search', axis: 'Time per search (ms)' },
  { value: 'nps', label: 'Nodes per second', axis: 'Nodes per second' },
  { value: 'eval_cp', label: 'Evaluation', axis: 'Evaluation, side to move (cp)' },
  { value: 'tt_hit_rate', label: 'TT hit rate', axis: 'TT hits per node' },
  { value: 'qratio', label: 'Quiescence share', axis: 'Share of nodes in quiescence search' },
  { value: 'fh_first_rate', label: 'First-move cutoffs', axis: 'Share of cutoffs by the first move' },
];

export async function render() {
  const frag = document.createDocumentFragment();
  frag.append(el('p', { class: 'tab-intro' },
    'Every engine version in this snapshot at a glance. The chips above choose the versions shown on every tab; ',
    'the baseline is the version the others are measured against.'));

  const [summary, sides, sprt] = await Promise.all([dataset('summary'), dataset('game_sides'), dataset('sprt')]);
  if (!summary) {
    frag.append(panel({ body: emptyNote('This snapshot has no search data.') }));
    return frag;
  }

  const stats = el('div', { class: 'span-full' });
  reactive(stats, async () => {
    const [s] = await query(`SELECT sum(searches) AS searches, sum(nodes) / nullif(sum(time_ms) / 1000, 0) AS nps,
        sum(depth_mean * searches) / nullif(sum(searches), 0) AS depth
      FROM ds_summary WHERE ${coreRows()} AND ${engineWhere()}`);
    const [g] = sides ? await query(`SELECT count(DISTINCT game_id)::DOUBLE AS n FROM ds_game_sides WHERE ${coreFilter()} AND ${engineWhere()}`) : [{}];
    const [r] = sprt ? await query(`SELECT count(*)::DOUBLE AS n FROM ds_sprt
        WHERE candidate_engine_id IN (${sqlIdList()}) OR baseline_engine_id IN (${sqlIdList()})`) : [{}];
    return statLine([
      { value: fmt.int(s?.searches), label: 'searches' },
      { value: fmt.int(g?.n), label: 'games' },
      { value: `${engines.selected.length}`, label: `of ${engines.all.length} engine versions shown` },
      { value: fmt.int(r?.n), label: 'SPRT tests' },
      { value: fmt.num(s?.depth, 1), label: 'mean completed depth' },
      { value: fmt.nps(s?.nps).replace(' ', '\u2009'), label: 'nodes per second' },
    ]);
  });
  frag.append(stats);

  const table = el('div');
  reactive(table, async () => {
    const rows = await query(`SELECT engine_id, searches, nps, depth_mean, time_ms_p50 FROM ds_summary
      WHERE ${coreRows()} AND ${engineWhere()} ORDER BY engine_ord`);
    const games = sides ? await query(`SELECT engine_id, count(*)::DOUBLE AS n,
        count(*) FILTER (WHERE score = 1)::DOUBLE AS w, count(*) FILTER (WHERE score = 0.5)::DOUBLE AS d,
        count(*) FILTER (WHERE score = 0)::DOUBLE AS l FROM ds_game_sides WHERE ${coreFilter()} AND ${engineWhere()} GROUP BY 1`) : [];
    const byId = new Map(games.map(g => [g.engine_id, g]));
    return dataTable([
      { label: 'Engine', value: r => engineName(engineById(r.engine_id)) },
      { label: 'Searches', num: true, value: r => fmt.int(r.searches) },
      { label: 'Games', num: true, value: r => fmt.int(byId.get(r.engine_id)?.n) },
      { label: 'Score', num: true, title: 'Points per game from the engine\u2019s side, with a 95% interval', value: r => {
        const g = byId.get(r.engine_id);
        const s = g && scoreFromWDL(g.w, g.d, g.l);
        return s ? el('span', {}, fmt.pct(s.score), el('span', { class: 'sub' }, `${fmt.pct(s.lo, 0)} to ${fmt.pct(s.hi, 0)}`)) : '\u2014';
      } },
      { label: 'Nodes per second', num: true, value: r => fmt.nps(r.nps) },
      { label: 'Mean depth', num: true, value: r => fmt.num(r.depth_mean, 2) },
      { label: 'Median time', num: true, value: r => fmt.ms(r.time_ms_p50) },
    ], rows);
  });
  frag.append(panel({
    title: 'Engines at a glance',
    caption: 'One row per selected version, oldest first: score is points per game from that engine\u2019s side (the range under it is a 95% interval) and nodes per second is all nodes divided by all search time.',
    body: table, note: sourceNote(summary),
  }));

  const hist = await dataset('hist');
  if (hist) {
    let metric = 'depth';
    const body = el('div');
    const draw = async () => {
      const rows = await query(`SELECT engine, engine_id, bin_lo, bin_hi, mid_lin, mid_log, share, scale FROM ds_hist
        WHERE ${coreRows()} AND metric = '${metric}' AND ${engineWhere()} ORDER BY engine_ord, bin_lo`);
      if (!rows.length) return emptyNote('No values for this metric.');
      const log = rows[0].scale ? rows[0].scale === 'log' : isLogBinned(rows.map(r => ({ lo: r.bin_lo, hi: r.bin_hi })));
      const data = rows.map(r => ({ ...r, x: log ? r.mid_log : r.mid_lin }));
      const m = HIST_METRICS.find(o => o.value === metric);
      return [engineLegend(), plot(
        vg.lineY(data, { x: 'x', y: 'share', stroke: 'engine', z: 'engine', curve: 'step', strokeWidth: 1.5, tip: true }),
        vg.ruleY([0], { stroke: 'currentColor', strokeOpacity: 0.3 }),
        ...engineColors(), vg.xScale(log ? 'log' : 'linear'), log ? vg.xTicks(6) : metric === 'depth' ? intTicks() : null, vg.yTickFormat('%'), vg.yGrid(true),
        vg.xLabel(m.axis), vg.yLabel('Share of searches'), vg.width(chartWidth()), vg.height(300))];
    };
    const redraw = reactive(body, draw);
    const picker = selectInput({ label: 'Metric', options: HIST_METRICS, value: metric,
      onChange: v => { metric = v; redraw(); } });
    frag.append(panel({
      title: 'How searches are distributed',
      caption: 'Each line shows the share of one engine\u2019s searches in each bin, so versions with different amounts of data compare directly and lines on top of each other mean the versions behave alike.',
      controls: picker, body, note: sourceNote(datasetInfo.get('hist')),
    }));
  }

  const bars = el('div');
  reactive(bars, async () => {
    const rows = await query(`SELECT engine, searches FROM ds_summary WHERE ${coreRows()} AND ${engineWhere()} ORDER BY engine_ord`);
    return plot(
      vg.barX(rows, { x: 'searches', y: 'engine', fill: 'engine', insetTop: 6, insetBottom: 6, tip: true }),
      vg.ruleX([0], { stroke: 'currentColor', strokeOpacity: 0.4 }),
      ...engineColors(), vg.yDomain(rows.map(r => r.engine)), vg.xGrid(true), vg.marginLeft(150),
      vg.xLabel('Searches logged'), vg.yLabel(null), vg.width(chartWidth()), vg.height(40 + 30 * rows.length));
  });
  frag.append(panel({
    title: 'Searches by engine',
    caption: 'How many logged searches each version contributes; versions with few searches have wider intervals everywhere else.',
    body: bars,
  }));
  return frag;
}
