/** Root moves: how the search spreads its effort over the root moves, and the move it chose against Stockfish's lines. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineById, engineWhere } from '../engines.js';
import { coreFilter, coreRows } from '../settings.js';
import { LOSS_BUCKETS, cumulativeRanks, depthLabel, formatEval, lichessUrl, lineAxis, lossMean, lossShares, orderAxis,
  parseLines, rankText } from '../stockfish.js';
import { chartWidth, coreScopeNote, dataTable, el, emptyNote, engineColors, engineLegend, engineName, fmt, intTicks, panel, plot,
  reactive, segmented, sourceNote } from '../ui.js';

const PHASES = [{ value: 'all', label: 'All' }, { value: 'opening', label: 'Opening' }, { value: 'midgame', label: 'Middlegame' },
  { value: 'endgame', label: 'Endgame' }];
const PIPELINE = 'python -m data.databases.run_analytics_pipeline --stockfish PATH';
const pipeline = () => el('code', { class: 'nowrap' }, PIPELINE);
const pct = (part, whole) => (whole > 0 ? fmt.pct(part / whole) : '\u2014');
// The pre-aggregated tables are split by core type; the fallback on the raw log has only core_type 'all' rows.
const coreOf = d => (d.fromAgg ? coreRows() : `core_type = 'all'`);
// Worded tick labels (loss ranges, 'not searched') collide in a narrow half-width chart: tilt them there.
const tiltedTicks = () => (chartWidth('half') < 460 ? [vg.xTickRotate(-35), vg.marginBottom(64)] : []);

export async function render() {
  const frag = document.createDocumentFragment();
  const [index, gap, best, sf, effort, order] = await Promise.all(['root_index', 'root_gap', 'root_best', 'sf_moves',
    'sf_root_effort', 'sf_root_order'].map(name => dataset(name)));
  const hasRoot = [index, gap, best].some(d => d?.rows > 0);
  // N of "Stockfish's top N": the lines every rank refers to
  const [{ n: lines } = {}] = sf?.rows > 0
    ? await query('SELECT coalesce(max(lines), max(sf_rank))::DOUBLE AS n FROM ds_sf_moves') : [];
  const top = lines >= 1 ? lines : null;
  if (!hasRoot && !top) {
    frag.append(panel({ title: 'Root moves', body: emptyNote(['This snapshot has no root-move logs and no Stockfish analysis of '
      + 'the searched positions. DEV builds of the engine log the root moves, and the analytics pipeline adds Stockfish\u2019s '
      + 'lines when Stockfish is available: ', pipeline(), '.']) }));
    return frag;
  }
  frag.append(el('p', { class: 'tab-intro' },
    'The root-move log records, for every root move of a search\u2019s deepest complete iteration, its score and the effort spent on it. ',
    `The second part compares the move each engine played with Stockfish\u2019s ${top ? `top ${top} lines` : 'best lines'} for the `
      + 'same position: how often it was one of them, how many centipawns it gave up, and how much of the root search went to '
      + 'Stockfish\u2019s choices.'));
  if (hasRoot) appendRoot(frag, { index, gap, best });
  else frag.append(panel({ title: 'Root moves', body: emptyNote('This snapshot has no root-move logs; DEV builds of the engine record them.') }));
  if (top) await appendStockfish(frag, { sf, effort, order, top });
  else {
    frag.append(panel({ title: 'Against Stockfish', body: emptyNote(['This snapshot has no Stockfish analysis of the searched '
      + 'positions. The analytics pipeline adds it when Stockfish is available: ', pipeline(), '.']) }));
  }
  return frag;
}

/* ---------------------------------------------------------------- effort and choice at the root */
function appendRoot(frag, { index, gap, best }) {
  if ([index, gap, best].some(d => d && !d.fromAgg)) frag.append(coreScopeNote('Root-move logs are'));
  const indexBody = el('div'), gapBody = el('div'), bestBody = el('div');
  frag.append(panel({ span: 'half', title: 'Effort by move order',
    caption: 'Share of root nodes spent on the move tried first, second and so on (10+ counts every later move); good ordering puts most effort on the first move.',
    body: indexBody }));
  frag.append(panel({ span: 'half', title: 'How clear the choice was',
    caption: 'Best minus second-best root score in centipawns (capped at 500); moves after the first get only a bound from a null-window search, so this tends to understate the gap.',
    body: gapBody }));
  frag.append(panel({ title: 'Moves chosen most often',
    caption: 'The twenty moves most often returned as best move, stacked by engine.', body: bestBody }));

  const axis = orderAxis();
  reactive(indexBody, async () => {
    if (!index?.rows) return emptyNote('This snapshot has no root-move log.');
    const rows = await query(`SELECT engine, move_index, node_share FROM ds_root_index WHERE ${coreOf(index)} AND ${engineWhere()}
      ORDER BY engine_ord, move_index`);
    if (!rows.length) return emptyNote('No root moves for this selection.');
    return [engineLegend(), plot(
      vg.lineY(rows, { x: 'move_index', y: 'node_share', stroke: 'engine', z: 'engine', strokeWidth: 1.5 }),
      vg.dot(rows, { x: 'move_index', y: 'node_share', fill: 'engine', r: 2.2, tip: true }),
      ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), vg.xTicks(axis.ticks.slice(0, -1)), vg.xTickFormat(axis.format),
      vg.xLabel('Position of the move in the root ordering (0 = first)'), vg.yLabel('Share of root nodes'),
      vg.width(chartWidth('half')), vg.height(250))];
  });
  reactive(gapBody, async () => {
    if (!gap?.rows) return emptyNote('This snapshot has no root-move scores.');
    const rows = (await query(`SELECT engine, gap, share FROM ds_root_gap WHERE ${coreOf(gap)} AND ${engineWhere()} ORDER BY engine_ord, gap`))
      .map(r => ({ ...r, x: r.gap + 5 }));
    if (!rows.length) return emptyNote('No root-move scores for this selection.');
    const gaps = new Set(rows.map(r => r.gap));
    if (gaps.size === 1) {
      const g = rows[0].gap;
      return emptyNote(`Every search in this selection has a gap of ${g >= 500 ? '500 cp or more' : `${fmt.int(g)}\u2013${fmt.int(g + 9)} cp`}, so there is no distribution to draw.`);
    }
    return plot(
      vg.lineY(rows, { x: 'x', y: 'share', stroke: 'engine', z: 'engine', curve: 'step', strokeWidth: 1.4, tip: true }),
      ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), vg.xLabel('Best minus second-best root score (cp)'),
      vg.yLabel('Share of searches'), vg.width(chartWidth('half')), vg.height(250));
  });
  reactive(bestBody, async () => {
    if (!best?.rows) return emptyNote('No best moves recorded.');
    const rows = await query(`WITH m AS (SELECT engine, engine_ord, move, searches FROM ds_root_best WHERE ${coreOf(best)} AND ${engineWhere()}),
        top AS (SELECT move FROM m GROUP BY move ORDER BY sum(searches) DESC LIMIT 20)
      SELECT m.* FROM m JOIN top USING (move) ORDER BY engine_ord`);
    if (!rows.length) return emptyNote('No best moves for this selection.');
    const order = [...new Set(rows.map(r => r.move))].sort((a, b) =>
      rows.filter(r => r.move === b).reduce((s, r) => s + r.searches, 0) - rows.filter(r => r.move === a).reduce((s, r) => s + r.searches, 0));
    return [engineLegend(), plot(
      vg.barY(rows, { x: 'move', y: 'searches', fill: 'engine', insetLeft: 2, insetRight: 2, tip: true }),
      ...engineColors(), vg.xDomain(order), vg.yGrid(true), vg.xLabel('Move (long algebraic)'), vg.yLabel('Searches'),
      vg.width(chartWidth()), vg.height(240)), el('p', { class: 'note' }, sourceNote(best))];
  });
}

/* ---------------------------------------------------------------- the chosen move against Stockfish's top N */
async function appendStockfish(frag, { sf, effort, order, top }) {
  let phase = 'all';
  const subs = new Set();
  const sub = fn => subs.add(fn);
  const present = new Set((await query('SELECT DISTINCT phase FROM ds_sf_moves')).map(r => r.phase));
  const phases = PHASES.filter(p => p.value === 'all' || present.has(p.value));
  if (phases.length > 1) {
    frag.append(el('div', { class: 'controls section-controls' }, segmented({ label: 'Phase', value: phase, options: phases,
      onChange: v => { phase = v; subs.forEach(fn => fn()); } })));
  }
  const where = () => `${coreRows()} AND phase = '${phase}' AND ${engineWhere()}`;

  const agreement = el('div');
  frag.append(panel({ title: `Agreement with Stockfish\u2019s top ${top}`,
    caption: `How often the engine played Stockfish\u2019s first choice or one of its ${top} lines, and how many centipawns its move gave up against Stockfish\u2019s best line.`,
    body: agreement }));
  reactive(agreement, async () => {
    const rows = await query(`SELECT engine_id, sum(searches) AS n,
        coalesce(sum(searches) FILTER (WHERE sf_rank = 1), 0) AS top1,
        coalesce(sum(searches) FILTER (WHERE sf_rank BETWEEN 1 AND 3), 0) AS top3,
        coalesce(sum(searches) FILTER (WHERE sf_rank >= 1), 0) AS topn,
        coalesce(sum(searches) FILTER (WHERE loss_bucket IS NOT NULL), 0) AS known,
        sum(loss_sum) FILTER (WHERE loss_bucket IS NOT NULL) AS loss_sum,
        sum(loss_sq_sum) FILTER (WHERE loss_bucket IS NOT NULL) AS loss_sq_sum,
        coalesce(sum(searches) FILTER (WHERE loss_bucket >= 100), 0) AS big,
        min(depth_min) AS depth_min, max(depth_max) AS depth_max
      FROM ds_sf_moves WHERE ${where()} GROUP BY engine_id, engine_ord HAVING sum(searches) > 0 ORDER BY engine_ord`);
    if (!rows.length) return emptyNote('No analysed positions for this selection.');
    const depths = rows.flatMap(r => [r.depth_min, r.depth_max]).filter(Number.isFinite);
    const depth = depthLabel(Math.min(...depths), Math.max(...depths));
    const setup = depth ? `Stockfish searched each position to ${depth} and kept its ${top} best lines`
      : `Stockfish kept its ${top} best lines for each position`;
    return [dataTable([
      { label: 'Engine', value: r => engineName(engineById(r.engine_id)) },
      { label: 'Searches', num: true, title: 'Moves judged: one per search, so a position met in several games counts each time',
        value: r => fmt.int(r.n) },
      { label: 'Stockfish\u2019s first choice', num: true, title: 'Share of searches where the engine played Stockfish\u2019s best line',
        value: r => pct(r.top1, r.n) },
      top > 3 ? { label: 'In top 3', num: true, title: 'Share of searches where the engine\u2019s move was one of Stockfish\u2019s three best lines',
        value: r => pct(r.top3, r.n) } : null,
      { label: `In top ${top}`, num: true, title: `Share of searches where the engine\u2019s move was one of Stockfish\u2019s ${top} lines`,
        value: r => pct(r.topn, r.n) },
      { label: 'Mean loss (cp)', num: true, title: 'Centipawns given up against Stockfish\u2019s best line, with a 95% interval',
        value: r => {
          const m = lossMean(r.known, r.loss_sum, r.loss_sq_sum);
          if (!Number.isFinite(m.mean)) return '\u2014';
          return el('span', {}, fmt.num(m.mean, 1),
            Number.isFinite(m.lo) ? el('span', { class: 'sub' }, `${fmt.num(m.lo, 1)} to ${fmt.num(m.hi, 1)}`) : null);
        } },
      { label: 'Loss of 100 cp or more', num: true, title: 'Share of the searches whose loss is known', value: r => pct(r.big, r.known) },
    ].filter(Boolean), rows), el('p', { class: 'note' },
      `${setup}; agreement with its first choice alone is noisy, because the best few moves are often almost equal. ${sourceNote(sf)}`)];
  }, [sub]);

  const ranked = el('div'), cost = el('div');
  frag.append(panel({ span: 'half', title: 'Where the chosen move ranked',
    caption: `Share of searches where the engine\u2019s move was among Stockfish\u2019s first k lines, for k from 1 to ${top}.`, body: ranked }));
  frag.append(panel({ span: 'half', title: 'What the chosen move cost',
    caption: 'Share of searches by the centipawns the engine\u2019s move gave up against Stockfish\u2019s best line; weight further right means costlier moves.',
    body: cost }));
  reactive(ranked, async () => {
    const rows = await query(`SELECT engine, engine_ord, sf_rank, sum(searches) AS n FROM ds_sf_moves WHERE ${where()}
      GROUP BY ALL ORDER BY engine_ord, sf_rank`);
    const points = cumulativeRanks(rows, top).map(p => ({ ...p, title: `${p.engine}: ${fmt.pct(p.share)} within the top ${p.k}` }));
    if (!points.length) return emptyNote('No analysed positions for this selection.');
    return [engineLegend(), plot(
      vg.lineY(points, { x: 'k', y: 'share', stroke: 'engine', z: 'engine', strokeWidth: 1.5 }),
      vg.dot(points, { x: 'k', y: 'share', fill: 'engine', r: 2.2, tip: true, title: 'title' }),
      ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), vg.xDomain([0.5, top + 0.5]),
      vg.xTicks(Array.from({ length: top }, (_, i) => i + 1)), intTicks(), vg.xLabel('Stockfish\u2019s first k lines'),
      vg.yLabel('Share of searches'), vg.width(chartWidth('half')), vg.height(250))];
  }, [sub]);
  reactive(cost, async () => {
    const rows = await query(`SELECT engine, engine_ord, loss_bucket, sum(searches) AS n FROM ds_sf_moves
      WHERE loss_bucket IS NOT NULL AND ${where()} GROUP BY ALL ORDER BY engine_ord, loss_bucket`);
    const points = lossShares(rows).map(p => ({ ...p, title: `${p.engine}, ${p.label} cp: ${fmt.pct(p.share)}` }));
    if (!points.length) return emptyNote('No known losses for this selection.');
    return plot(
      vg.lineY(points, { x: 'bucket', y: 'share', stroke: 'engine', z: 'engine', strokeWidth: 1.5 }),
      vg.dot(points, { x: 'bucket', y: 'share', fill: 'engine', r: 2.2, tip: true, title: 'title' }),
      ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), vg.yZero(true), vg.xDomain([-0.5, LOSS_BUCKETS.length - 0.5]),
      vg.xTicks(LOSS_BUCKETS.map((_, i) => i)), vg.xTickFormat(i => LOSS_BUCKETS[i]?.label ?? ''), ...tiltedTicks(),
      vg.xLabel('Centipawns lost against Stockfish\u2019s best line'), vg.yLabel('Share of searches'),
      vg.width(chartWidth('half')), vg.height(250));
  }, [sub]);

  const effortBody = el('div'), orderBody = el('div');
  frag.append(panel({ span: 'half', title: 'Effort on Stockfish\u2019s moves',
    caption: 'Mean share of a search\u2019s root nodes spent on each of Stockfish\u2019s lines, and on all other root moves together.',
    body: effortBody }));
  frag.append(panel({ span: 'half', title: 'Where Stockfish\u2019s best move was searched',
    caption: 'Share of searches by the position of Stockfish\u2019s first-line move in the engine\u2019s root ordering; a high share at 0 means the engine tried it first.',
    body: orderBody }));
  const lineX = lineAxis(top), orderX = orderAxis();
  reactive(effortBody, async () => {
    if (!effort?.rows) return emptyNote('This snapshot has no root-move logs for the analysed positions.');
    const rows = await query(`SELECT engine, sf_rank, node_share FROM ds_sf_root_effort WHERE ${where()} AND searches > 0
      ORDER BY engine_ord, sf_rank`);
    if (!rows.length) return emptyNote('No root-move logs for this selection.');
    // a line no search spent nodes on has no row: its mean share is 0
    const points = fillShares(rows, r => lineX.x(r.sf_rank), r => r.node_share, lineX.ticks)
      .map(p => ({ ...p, title: `${p.engine}, ${lineX.format(p.x)}: ${fmt.pct(p.share)} of root nodes` }));
    return [engineLegend(), plot(
      vg.lineY(points.filter(p => p.x <= top), { x: 'x', y: 'share', stroke: 'engine', z: 'engine', strokeWidth: 1.5 }),
      vg.dot(points, { x: 'x', y: 'share', fill: 'engine', r: 2.2, tip: true, title: 'title' }),
      ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), vg.yZero(true), vg.xDomain(lineX.domain), vg.xTicks(lineX.ticks),
      vg.xTickFormat(lineX.format), vg.xLabel('Stockfish\u2019s line'), vg.yLabel('Mean share of root nodes'),
      vg.width(chartWidth('half')), vg.height(250))];
  }, [sub]);
  reactive(orderBody, async () => {
    if (!order?.rows) return emptyNote('This snapshot has no root-move logs for the analysed positions.');
    const rows = await query(`SELECT engine, move_index, share FROM ds_sf_root_order WHERE ${where()} ORDER BY engine_ord, move_index`);
    if (!rows.length) return emptyNote('No root-move logs for this selection.');
    const points = fillShares(rows, r => orderX.x(r.move_index), r => r.share, orderX.ticks)
      .map(p => ({ ...p, title: `${p.engine}, ${orderX.format(p.x)}: ${fmt.pct(p.share)} of searches` }));
    const cap = orderX.ticks[orderX.ticks.length - 2];
    return plot(
      vg.lineY(points.filter(p => p.x <= cap), { x: 'x', y: 'share', stroke: 'engine', z: 'engine', strokeWidth: 1.5 }),
      vg.dot(points, { x: 'x', y: 'share', fill: 'engine', r: 2.2, tip: true, title: 'title' }),
      ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), vg.yZero(true), vg.xDomain(orderX.domain), vg.xTicks(orderX.ticks),
      vg.xTickFormat(orderX.format), ...tiltedTicks(), vg.xLabel('Position in the engine\u2019s root ordering (0 = first)'), vg.yLabel('Share of searches'),
      vg.width(chartWidth('half')), vg.height(250));
  }, [sub]);

  const misses = el('div');
  frag.append(panel({ title: 'Largest misses',
    caption: 'The positions where the engine\u2019s move gave up the most against Stockfish\u2019s best line; each position opens on the Lichess analysis board.',
    body: misses }));
  reactive(misses, async () => {
    const d = await dataset('sf_misses');
    if (!d) return emptyNote('Per-position details need a full snapshot (export_public --full) or a local analytics file.');
    const rows = await query(`SELECT engine_id, fen, move, max(loss) AS worst, max(sf_rank) AS sf_rank, max(move_eval) AS move_eval,
        max(lines) AS lines, sum(searches) AS times, max(sf_lines) AS sf_lines
      FROM ds_sf_misses WHERE ${coreFilter()} AND ${phase === 'all' ? 'TRUE' : `phase = '${phase}'`} AND ${engineWhere()}
      GROUP BY engine_id, engine_ord, fen, move ORDER BY worst DESC, engine_ord, fen LIMIT 25`);
    if (!rows.length) return emptyNote('No move lost centipawns against Stockfish in this selection.');
    return [dataTable([
      { label: 'Engine', value: r => engineName(engineById(r.engine_id)) },
      { label: 'Position', value: r => el('span', {}, el('a', { href: lichessUrl(r.fen), target: '_blank', rel: 'noopener' }, r.fen),
        r.times > 1 ? el('span', { class: 'sub' }, `searched ${fmt.int(r.times)} times`) : null) },
      { label: 'Played', value: r => el('span', { class: 'nowrap' }, r.move, el('span', { class: 'sub' },
        [rankText(r.sf_rank, r.lines ?? top), formatEval(r.move_eval)].filter(Boolean).join(' \u00b7 '))) },
      { label: 'Stockfish\u2019s lines', value: r => lineList(parseLines(r.sf_lines), r.move) },
      { label: 'Loss (cp)', num: true, value: r => fmt.int(r.worst) },
    ], rows), el('p', { class: 'note' }, 'Evaluations are Stockfish\u2019s, in centipawns from the side to move (#3: mate in three, '
      + '\u2212#2: mated in two); in the loss a mate counts as 1000 cp. ', sourceNote(d))];
  }, [sub]);
}

/** One point per engine and x tick, 0 where the data has no row. */
function fillShares(rows, xOf, valueOf, ticks) {
  const byEngine = new Map();
  for (const r of rows) {
    if (!byEngine.has(r.engine)) byEngine.set(r.engine, new Map());
    const shares = byEngine.get(r.engine), x = xOf(r);
    shares.set(x, (shares.get(x) ?? 0) + (valueOf(r) ?? 0));
  }
  return [...byEngine].flatMap(([engine, shares]) => ticks.map(x => ({ engine, x, share: shares.get(x) ?? 0 })));
}

function lineList(lines, played) {
  if (!lines.length) return '\u2014';
  return el('span', {}, lines.map((l, i) => [i ? ' \u00b7 ' : null,
    el('span', { class: l.move === played ? 'nowrap sig' : 'nowrap' }, `${l.move} ${formatEval(l.eval)}`)]));
}
