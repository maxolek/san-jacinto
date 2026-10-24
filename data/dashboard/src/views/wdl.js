/** Eval vs WDL: how often the side to move goes on to win, draw or lose for each evaluation bucket. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineById, engineWhere, onEngines, selectedEngines } from '../engines.js';
import { RESULT_COLORS } from '../palette.js';
import { bucketWidth, fitLogistic, wdlBucket } from '../stats.js';
import { coreRows } from '../settings.js';
import { chartWidth, checkbox, dataTable, el, emptyNote, engineColors, engineLegend, engineName, fmt, panel, plot, reactive,
  segmented, selectInput, seriesLegend, sourceNote } from '../ui.js';

// The pipeline's bucket width; the chart infers the actual width from the data.
const DEFAULT_BUCKET = 25;

export async function render() {
  const frag = document.createDocumentFragment();
  const wdl = await dataset('wdl');
  if (!wdl || !wdl.rows) {
    frag.append(panel({ title: 'Eval vs WDL', body: emptyNote('This snapshot has no searches linked to finished games. '
      + 'The match runner links every engine search to its game and result; older engine-log games are not linked.') }));
    return frag;
  }
  const phases = (await query(`SELECT DISTINCT phase FROM ds_wdl WHERE phase <> 'all' ORDER BY 1`)).map(r => r.phase);
  const step = bucketWidth((await query('SELECT DISTINCT eval_bucket AS b FROM ds_wdl WHERE eval_bucket IS NOT NULL')).map(r => r.b), DEFAULT_BUCKET);
  const opts = { mode: 'single', engine: null, phase: 'all', fit: true, minGames: 5 };
  const subs = new Set();
  const changed = () => subs.forEach(fn => fn());
  const sub = fn => subs.add(fn);

  const engineSelect = el('span');
  const drawEngineSelect = () => {
    const sel = selectedEngines();
    if (!sel.some(e => e.id === opts.engine)) opts.engine = sel[sel.length - 1]?.id ?? null;
    engineSelect.replaceChildren(selectInput({ label: 'Engine', value: opts.engine,
      options: sel.map(e => ({ value: e.id, label: e.label })), onChange: v => { opts.engine = Number(v); changed(); } }));
    engineSelect.hidden = opts.mode !== 'single';
  };
  drawEngineSelect();
  onEngines(drawEngineSelect);
  const controls = [
    segmented({ label: 'Show', value: opts.mode, options: [{ value: 'single', label: 'One engine' }, { value: 'overlay', label: 'Overlay engines' }],
      onChange: v => { opts.mode = v; engineSelect.hidden = v !== 'single'; changed(); } }),
    engineSelect,
    phases.length > 1 ? selectInput({ label: 'Phase', value: 'all', onChange: v => { opts.phase = v; changed(); },
      options: [{ value: 'all', label: 'All phases' }, ...phases.map(p => ({ value: p, label: phaseLabel(p) }))] }) : null,
    selectInput({ label: 'Hide buckets under', value: 5, onChange: v => { opts.minGames = Number(v); changed(); },
      options: [1, 5, 10, 25].map(n => ({ value: n, label: `${n} game${n > 1 ? 's' : ''}` })) }),
    checkbox({ label: 'Fitted logistic', checked: opts.fit, onChange: v => { opts.fit = v; changed(); } }),
  ];

  const load = async () => {
    const who = opts.mode === 'single' && opts.engine != null ? `engine_id = ${opts.engine}` : engineWhere();
    const rows = await query(`SELECT engine_id, engine, eval_bucket, positions, games, wins, draws, losses FROM ds_wdl
      WHERE mate = 0 AND eval_bucket IS NOT NULL AND ${coreRows()} AND phase = '${opts.phase}' AND ${who}
      ORDER BY engine_ord, eval_bucket`);
    const buckets = rows.map(r => ({ ...r, ...wdlBucket(r) })).filter(r => r.positions > 0 && (r.games ?? r.positions) >= opts.minGames);
    const byEngine = new Map();
    for (const b of buckets) {
      if (!byEngine.has(b.engine_id)) byEngine.set(b.engine_id, []);
      byEngine.get(b.engine_id).push(b);
    }
    const fits = new Map([...byEngine].map(([id, bs]) => [id, fitLogistic(bs.map(b => b.eval_bucket), bs.map(b => b.pWin), bs.map(b => b.games))]));
    return { buckets, byEngine, fits };
  };

  const chart = el('div');
  reactive(chart, async () => {
    const { buckets, byEngine, fits } = await load();
    if (!buckets.length) return emptyNote('No positions for this selection.');
    const xs = buckets.map(b => b.eval_bucket);
    const xDomain = [Math.min(...xs) - step / 2, Math.max(...xs) + step / 2];
    const width = chartWidth();
    const common = [vg.xDomain(xDomain), vg.marginLeft(58), vg.marginRight(16), vg.width(width)];
    const zero = vg.ruleX([0], { stroke: 'currentColor', strokeOpacity: 0.25 });
    const half = vg.ruleY([0.5], { stroke: 'currentColor', strokeOpacity: 0.25, strokeDasharray: '2,3' });
    let main, legend;
    if (opts.mode === 'single') {
      const e = engineById(opts.engine);
      const long = buckets.flatMap(b => [
        { x: b.eval_bucket, outcome: 'Win', p: b.pWin, lo: b.win[0], hi: b.win[1] },
        { x: b.eval_bucket, outcome: 'Draw', p: b.pDraw, lo: b.draw[0], hi: b.draw[1] },
        { x: b.eval_bucket, outcome: 'Loss', p: b.pLoss, lo: b.loss[0], hi: b.loss[1] },
      ]);
      const score = buckets.map(b => ({ x: b.eval_bucket, p: b.score, lo: b.scoreCI[0], hi: b.scoreCI[1],
        title: `${fmt.cp(b.eval_bucket)}\nexpected score ${fmt.pct(b.score)} (${fmt.pct(b.scoreCI[0], 0)} to ${fmt.pct(b.scoreCI[1], 0)})\n`
          + `${fmt.int(b.positions)} positions from ${fmt.int(b.games)} games` }));
      const fit = fits.get(opts.engine);
      const curve = opts.fit && fit ? sampleCurve(fit, xDomain) : [];
      const showX50 = opts.fit && fit && inRange(fit.x50, xDomain);
      const colors = [vg.colorDomain(['Win', 'Draw', 'Loss']), vg.colorRange([RESULT_COLORS.win, RESULT_COLORS.draw, RESULT_COLORS.loss])];
      main = plot(
        vg.areaY(long, { x: 'x', y1: 'lo', y2: 'hi', fill: 'outcome', fillOpacity: 0.09 }),
        vg.lineY(long, { x: 'x', y: 'p', stroke: 'outcome', strokeWidth: 1.5 }),
        vg.lineY(score, { x: 'x', y: 'p', stroke: 'currentColor', strokeWidth: 1.4, strokeDasharray: '5,3' }),
        vg.dot(score, { x: 'x', y: 'p', r: 2, fill: 'currentColor', fillOpacity: 0.001, tip: true, title: 'title' }),
        curve.length ? vg.lineY(curve, { x: 'x', y: 'p', stroke: RESULT_COLORS.win, strokeWidth: 1, strokeDasharray: '1.5,2.5' }) : null,
        showX50 ? vg.ruleX([fit.x50], { stroke: RESULT_COLORS.win, strokeOpacity: 0.6 }) : null,
        showX50 ? vg.text([{ x: fit.x50, y: 0.97, label: `win 50% at ${fmt.cp(fit.x50)}` }], { x: 'x', y: 'y', text: 'label',
          dx: 6, textAnchor: 'start', fill: RESULT_COLORS.win, fontSize: 11 }) : null,
        zero, half, ...colors, ...common, vg.yDomain([0, 1]), vg.yTickFormat('%'), vg.yGrid(true), vg.xAxis(null),
        vg.marginBottom(8), vg.yLabel('Share of positions'), vg.height(340));
      legend = seriesLegend([
        { label: 'Win', color: RESULT_COLORS.win }, { label: 'Draw', color: RESULT_COLORS.draw },
        { label: 'Loss', color: RESULT_COLORS.loss }, { label: 'Expected score', color: 'var(--ink)', dash: true },
        ...(curve.length ? [{ label: 'Fitted win probability', color: RESULT_COLORS.win, dash: true }] : []),
      ]);
      legend.prepend(el('span', { class: 'legend-item' }, engineName(e)), el('span', { class: 'legend-item' }, '\u00b7'));
    } else {
      const score = buckets.map(b => ({ x: b.eval_bucket, engine: b.engine, p: b.score, lo: b.scoreCI[0], hi: b.scoreCI[1], games: b.games }));
      const curves = opts.fit ? [...fits].flatMap(([id, f]) => (f ? sampleCurve(f, xDomain).map(p => ({ ...p, engine: engineById(id)?.label })) : [])) : [];
      main = plot(
        vg.areaY(score, { x: 'x', y1: 'lo', y2: 'hi', fill: 'engine', fillOpacity: 0.08 }),
        vg.lineY(score, { x: 'x', y: 'p', stroke: 'engine', strokeWidth: 1.6, tip: true }),
        curves.length ? vg.lineY(curves, { x: 'x', y: 'p', stroke: 'engine', z: 'engine', strokeWidth: 1, strokeDasharray: '1.5,2.5' }) : null,
        zero, half, ...engineColors(), ...common, vg.yDomain([0, 1]), vg.yTickFormat('%'), vg.yGrid(true), vg.xAxis(null),
        vg.marginBottom(8), vg.yLabel('Expected score of the side to move'), vg.height(340));
      legend = el('div', {}, engineLegend(), curves.length ? seriesLegend([{ label: 'Fitted win probability', color: 'var(--ink-2)', dash: true }]) : null);
    }
    const bars = buckets.map(b => ({ engine: b.engine, x1: b.eval_bucket - step * 0.42, x2: b.eval_bucket + step * 0.42, games: b.games,
      title: `${b.engine}\n${fmt.cp(b.eval_bucket)}: ${fmt.int(b.games)} games, ${fmt.int(b.positions)} positions` }));
    const counts = plot(
      vg.rectY(bars, { x1: 'x1', x2: 'x2', y: 'games', fill: opts.mode === 'single' ? 'var(--ink-3)' : 'engine', fillOpacity: 0.55,
        tip: true, title: 'title' }),
      ...engineColors(), ...common, vg.marginTop(24), vg.yLabel('Games'), vg.yTicks(3), vg.yGrid(true),
      vg.xLabel(`Evaluation of the side to move (centipawns, ${step} cp buckets)`), vg.height(130));
    return [legend, main, counts];
  }, [sub]);
  frag.append(panel({
    title: 'Evaluation and the final result',
    caption: 'For each bucket of the searching side\u2019s own evaluation, the lines show how often that side went on to win, draw or lose, with 95% bands that count games rather than positions (positions from one game are not independent) and the number of games under the axis; evaluations beyond \u00b11000 cp fall in the outermost buckets and mate scores are counted separately below.',
    controls, body: chart, note: sourceNote(wdl),
  }));

  const fitTable = el('div');
  reactive(fitTable, async () => {
    const { byEngine, fits } = await load();
    const rows = [...byEngine.keys()].map(id => ({ id, fit: fits.get(id), bs: byEngine.get(id) }));
    if (!rows.length) return emptyNote('No positions for this selection.');
    return dataTable([
      { label: 'Engine', value: r => engineName(engineById(r.id)) },
      { label: '50% win at', num: true, title: 'Evaluation where the fitted win probability crosses one half',
        value: r => (r.fit ? el('span', {}, fmt.cp(r.fit.x50), el('span', { class: 'sub' }, `\u00b1 ${Math.round(1.96 * r.fit.x50SE)} cp`)) : '\u2014') },
      { label: 'Odds double every', num: true, title: 'How far the evaluation must rise to double the odds of winning (ln 2 / slope)',
        value: r => (r.fit ? `${Math.round(Math.LN2 / r.fit.b)} cp` : '\u2014') },
      { label: 'Positions', num: true, value: r => fmt.int(r.bs.reduce((s, b) => s + b.positions, 0)) },
      { label: 'Draws at 0 cp', num: true, value: r => {
        const b = r.bs.find(x => x.eval_bucket === 0);
        return b ? fmt.pct(b.pDraw) : '\u2014';
      } },
    ], rows);
  }, [sub]);
  frag.append(panel({ span: 'half', title: 'Fitted logistic',
    caption: 'A logistic curve fitted to the win share (weighted by games): the 50% point is the evaluation at which the side to move wins as often as not, and a shorter doubling distance means results follow the evaluation more sharply.',
    body: fitTable }));

  const mates = el('div');
  reactive(mates, async () => {
    const rows = await query(`SELECT engine_id, mate, sum(positions) AS positions, sum(wins) AS wins, sum(losses) AS losses,
        sum(draws) AS draws FROM ds_wdl WHERE mate <> 0 AND ${coreRows()} AND phase = '${opts.phase}' AND ${engineWhere()}
      GROUP BY ALL ORDER BY engine_id, mate DESC`);
    if (!rows.length) return emptyNote('No mate scores in this selection.');
    const ids = [...new Set(rows.map(r => r.engine_id))];
    const get = (id, m) => rows.find(r => r.engine_id === id && r.mate === m);
    return dataTable([
      { label: 'Engine', value: id => engineName(engineById(id)) },
      { label: 'Found a mate', num: true, value: id => {
        const r = get(id, 1);
        return r ? el('span', {}, `${fmt.pct(r.wins / r.positions)} won`, el('span', { class: 'sub' }, `${fmt.int(r.positions)} positions`)) : '\u2014';
      } },
      { label: 'Saw itself mated', num: true, value: id => {
        const r = get(id, -1);
        return r ? el('span', {}, `${fmt.pct(r.losses / r.positions)} lost`, el('span', { class: 'sub' }, `${fmt.int(r.positions)} positions`)) : '\u2014';
      } },
    ], ids);
  }, [sub]);
  frag.append(panel({ span: 'half', title: 'Mate scores',
    caption: 'Positions where the search reported a forced mate are kept out of the buckets above; this is how those games ended.',
    body: mates }));
  return frag;
}

const inRange = (x, [lo, hi]) => Number.isFinite(x) && x >= lo && x <= hi;
const phaseLabel = p => ({ opening: 'Opening', midgame: 'Middlegame', middlegame: 'Middlegame', endgame: 'Endgame', unknown: 'Unknown' }[p] ?? p);

function sampleCurve(fit, [lo, hi], n = 80) {
  return Array.from({ length: n + 1 }, (_, i) => {
    const x = lo + ((hi - lo) * i) / n;
    return { x, p: fit.predict(x) };
  });
}
