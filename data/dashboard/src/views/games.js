/** Games: results by engine, how games ended, their length and the openings played. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineById, engineWhere } from '../engines.js';
import { RESULT_COLORS } from '../palette.js';
import { coreFilter, onOpeningMode, openingLabelSQL, openingMode, setOpeningMode } from '../settings.js';
import { scoreFromWDL } from '../stats.js';
import { chartWidth, dataTable, el, emptyNote, engineColors, engineLegend, engineName, fmt, panel, plot, reactive, segmented,
  seriesLegend, sourceNote, statLine } from '../ui.js';

const SIDE_COLORS = { 'White won': '#D9D0BF', Draw: '#A89D89', 'Black won': '#3F3931' };

export function openingToggle() {
  const control = segmented({ label: 'Openings by', value: openingMode(), onChange: setOpeningMode,
    options: [{ value: 'family', label: 'Family' }, { value: 'variation', label: 'Family and variation' }] });
  // keep toggles on other tabs in step
  onOpeningMode(() => control.querySelectorAll('button').forEach((b, i) =>
    b.setAttribute('aria-checked', String((i === 0 ? 'family' : 'variation') === openingMode()))));
  return control;
}

export async function render() {
  const frag = document.createDocumentFragment();
  const sides = await dataset('game_sides');
  if (!sides) {
    frag.append(panel({ title: 'Games', body: emptyNote('This snapshot has no games.') }));
    return frag;
  }
  const stats = el('div');
  reactive(stats, async () => {
    const [g] = await query(`WITH g AS (SELECT DISTINCT game_id, result, run_time_s FROM ds_game_sides WHERE ${coreFilter()} AND ${engineWhere()})
      SELECT count(*)::DOUBLE AS n, avg((result = 'white')::INT) AS w, avg((result = 'draw')::INT) AS d,
        avg((result = 'black')::INT) AS b, median(run_time_s) AS t FROM g`);
    return statLine([
      { value: fmt.int(g.n), label: 'games with a selected engine' },
      { value: fmt.pct(g.w, 0), label: 'won by White' },
      { value: fmt.pct(g.d, 0), label: 'drawn' },
      { value: fmt.pct(g.b, 0), label: 'won by Black' },
      { value: g.t != null ? `${fmt.num(g.t, 0)} s` : '\u2014', label: 'median game length' },
    ]);
  });
  frag.append(stats);

  const score = el('div');
  reactive(score, async () => {
    const rows = await query(`SELECT engine, engine_ord, count(*) FILTER (WHERE score = 1)::DOUBLE AS w,
        count(*) FILTER (WHERE score = 0.5)::DOUBLE AS d, count(*) FILTER (WHERE score = 0)::DOUBLE AS l
      FROM ds_game_sides WHERE ${coreFilter()} AND ${engineWhere()} GROUP BY ALL ORDER BY engine_ord`);
    if (!rows.length) return emptyNote('No finished games for this selection.');
    const long = rows.flatMap(r => [['Win', r.w], ['Draw', r.d], ['Loss', r.l]].map(([outcome, n]) => ({ engine: r.engine, outcome, n })));
    const labels = rows.map(r => {
      const s = scoreFromWDL(r.w, r.d, r.l);
      return { engine: r.engine, text: s ? `${fmt.pct(s.score)} over ${fmt.int(s.n)} games (${fmt.pct(s.lo, 0)} to ${fmt.pct(s.hi, 0)})` : '' };
    });
    return [seriesLegend([{ label: 'Win', color: RESULT_COLORS.win }, { label: 'Draw', color: RESULT_COLORS.draw },
      { label: 'Loss', color: RESULT_COLORS.loss }]), plot(
      vg.barX(long, { x: 'n', y: 'engine', fill: 'outcome', offset: 'normalize', insetTop: 6, insetBottom: 6, tip: true }),
      vg.ruleX([0.5], { stroke: 'currentColor', strokeOpacity: 0.5, strokeDasharray: '2,3' }),
      vg.text(labels, { y: 'engine', text: 'text', frameAnchor: 'right', dx: 12, fill: 'currentColor', textAnchor: 'start' }),
      vg.colorDomain(['Win', 'Draw', 'Loss']), vg.colorRange([RESULT_COLORS.win, RESULT_COLORS.draw, RESULT_COLORS.loss]),
      vg.yDomain(rows.map(r => r.engine)), vg.xTickFormat('%'), vg.marginLeft(140), vg.marginRight(340),
      vg.xLabel('Share of the engine\u2019s games'), vg.yLabel(null), vg.width(chartWidth()), vg.height(50 + 34 * rows.length))];
  });
  frag.append(panel({
    title: 'Results by engine',
    caption: 'Wins, draws and losses from each engine\u2019s own side of the board, with an even score at the dotted line and the score and its 95% interval on the right.',
    body: score, note: sourceNote(sides),
  }));

  const ended = el('div');
  reactive(ended, async () => {
    const rows = (await query(`WITH g AS (SELECT DISTINCT game_id, result, coalesce(termination, 'unknown') AS termination
        FROM ds_game_sides WHERE ${coreFilter()} AND ${engineWhere()})
      SELECT termination, CASE result WHEN 'white' THEN 'White won' WHEN 'black' THEN 'Black won' ELSE 'Draw' END AS outcome,
        count(*)::DOUBLE AS n FROM g GROUP BY ALL`)).map(r => ({ ...r, how: String(r.termination).replaceAll('_', ' ') }));
    if (!rows.length) return emptyNote('No games for this selection.');
    const totals = new Map();
    rows.forEach(r => totals.set(r.how, (totals.get(r.how) ?? 0) + r.n));
    const order = [...totals.keys()].sort((a, b) => totals.get(b) - totals.get(a));
    return [seriesLegend(Object.entries(SIDE_COLORS).map(([label, color]) => ({ label, color }))), plot(
      vg.barX(rows, { x: 'n', y: 'how', fill: 'outcome', insetTop: 4, insetBottom: 4, tip: true }),
      vg.colorDomain(Object.keys(SIDE_COLORS)), vg.colorRange(Object.values(SIDE_COLORS)),
      vg.yDomain(order), vg.xGrid(true), vg.marginLeft(140), vg.xLabel('Games'), vg.yLabel(null),
      vg.width(chartWidth('half')), vg.height(50 + 28 * order.length))];
  });
  frag.append(panel({ span: 'half', title: 'How games ended',
    caption: 'Each game is counted once, by the way it finished and who won.', body: ended }));

  const length = el('div');
  reactive(length, async () => {
    const [{ hi }] = await query(`SELECT quantile_cont(run_time_s, 0.99) AS hi FROM ds_game_sides WHERE run_time_s > 0`);
    if (!(hi > 0)) return emptyNote('No game durations recorded.');
    const step = niceStep(hi / 24);
    const rows = await query(`SELECT engine, engine_ord, floor(least(run_time_s, ${hi}) / ${step}) * ${step} + ${step / 2} AS x,
        count(*)::DOUBLE AS n
      FROM ds_game_sides WHERE run_time_s > 0 AND ${coreFilter()} AND ${engineWhere()} GROUP BY ALL ORDER BY engine_ord, x`);
    const totals = new Map();
    rows.forEach(r => totals.set(r.engine, (totals.get(r.engine) ?? 0) + r.n));
    const data = rows.map(r => ({ ...r, share: r.n / totals.get(r.engine) }));
    return [engineLegend(), plot(
      vg.lineY(data, { x: 'x', y: 'share', stroke: 'engine', z: 'engine', curve: 'step', strokeWidth: 1.4, tip: true }),
      ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), vg.xLabel('Game length (seconds)'), vg.yLabel('Share of games'),
      vg.width(chartWidth('half')), vg.height(240))];
  });
  frag.append(panel({ span: 'half', title: 'Game length',
    caption: 'Wall-clock length of each engine\u2019s games, as a share of its games per bin.', body: length }));

  const faults = el('div');
  reactive(faults, async () => {
    const rows = await query(`SELECT engine_id, min(engine_ord) AS ord, count(*)::DOUBLE AS games, count(termination)::DOUBLE AS known,
        count(*) FILTER (WHERE termination = 'time' AND score = 0)::DOUBLE AS time,
        count(*) FILTER (WHERE termination = 'illegal_move' AND score = 0)::DOUBLE AS illegal,
        count(*) FILTER (WHERE termination = 'crash' AND score = 0)::DOUBLE AS crash,
        count(*) FILTER (WHERE termination = 'adjudication')::DOUBLE AS adjudicated
      FROM ds_game_sides WHERE ${coreFilter()} AND ${engineWhere()} GROUP BY 1 ORDER BY ord`);
    if (!rows.length) return emptyNote('No games for this selection.');
    if (!rows.some(r => r.known > 0)) return emptyNote('These games carry no termination. The match runner records how every game ended, including time losses and adjudications.');
    const cell = key => r => el('span', {}, fmt.int(r[key]), el('span', { class: 'sub' }, r.known ? fmt.pct(r[key] / r.known, 1) : ''));
    return dataTable([
      { label: 'Engine', value: r => engineName(engineById(r.engine_id)) },
      { label: 'Games', num: true, value: r => fmt.int(r.games) },
      { label: 'Lost on time', num: true, value: cell('time') },
      { label: 'Illegal move', num: true, value: cell('illegal') },
      { label: 'Crash or stall', num: true, value: cell('crash') },
      { label: 'Adjudicated', num: true, title: 'Games ended by the runner (resign or draw rules, move limit), whatever the result', value: cell('adjudicated') },
    ], rows);
  });
  frag.append(panel({ title: 'Endings that point at the engine',
    caption: 'Games each engine lost on time, by an illegal move or by crashing, with the share of its games with a known ending underneath; anything above zero in the first three columns is worth a look.',
    body: faults }));

  const openings = el('div');
  reactive(openings, async () => {
    const rows = await query(`WITH o AS (SELECT engine, engine_ord, ${openingLabelSQL()} AS label, count(*)::DOUBLE AS n
          FROM ds_game_sides WHERE opening IS NOT NULL AND opening NOT IN ('', '(unknown)') AND ${coreFilter()} AND ${engineWhere()} GROUP BY ALL),
        top AS (SELECT label FROM o GROUP BY label ORDER BY sum(n) DESC LIMIT 15)
      SELECT o.* FROM o JOIN top USING (label) ORDER BY engine_ord`);
    if (!rows.length) return emptyNote('No opening names recorded.');
    const totals = new Map();
    rows.forEach(r => totals.set(r.label, (totals.get(r.label) ?? 0) + r.n));
    const order = [...totals.keys()].sort((a, b) => totals.get(b) - totals.get(a));
    return [engineLegend(), plot(
      vg.barX(rows, { x: 'n', y: 'label', fill: 'engine', insetTop: 3, insetBottom: 3, tip: true }),
      ...engineColors(), vg.yDomain(order), vg.xGrid(true), vg.marginLeft(260), vg.xLabel('Games played by each engine'),
      vg.yLabel(null), vg.width(chartWidth()), vg.height(50 + 24 * order.length))];
  }, [fn => onOpeningMode(fn)]);
  frag.append(panel({
    title: 'Most played openings',
    caption: 'The fifteen openings met most often, stacked by engine; a game between two selected engines counts once for each of them.',
    controls: openingToggle(), body: openings,
  }));
  return frag;
}

export function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}
