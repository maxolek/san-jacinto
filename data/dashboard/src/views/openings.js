/** Openings: frequency and score by opening family, or by family and variation. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineWhere } from '../engines.js';
import { SEQUENTIAL } from '../palette.js';
import { coreFilter, coreType, onOpeningMode, openingLabelSQL } from '../settings.js';
import { scoreFromWDL } from '../stats.js';
import { chartWidth, coreScopeNote, el, emptyNote, engineColors, engineLegend, fmt, panel, plot, reactive, selectInput, seriesLegend,
  sourceNote } from '../ui.js';
import { openingToggle } from './games.js';

export async function render() {
  const frag = document.createDocumentFragment();
  const [openings, sides] = await Promise.all([dataset('openings'), dataset('game_sides')]);
  if (!openings) {
    frag.append(panel({ title: 'Openings', body: emptyNote('This snapshot has no opening names.') }));
    return frag;
  }
  const opts = { top: 20, minGames: 10 };
  const subs = new Set();
  const changed = () => subs.forEach(fn => fn());
  const sub = [fn => subs.add(fn), fn => onOpeningMode(fn)];
  // Opening aggregates are not split by core type; per-game rows are, so a selected core type is counted from those.
  const source = () => (coreType() !== 'all' && sides
    ? `(SELECT engine_id, engine, engine_ord, opening, opening_variation, opening_eco, count(*)::DOUBLE AS games,
        count(*) FILTER (WHERE score = 1)::DOUBLE AS wins, count(*) FILTER (WHERE score = 0.5)::DOUBLE AS draws,
        count(*) FILTER (WHERE score = 0)::DOUBLE AS losses
      FROM ds_game_sides WHERE ${coreFilter()} GROUP BY ALL)`
    : 'ds_openings');
  frag.append(el('div', { class: 'controls section-controls' },
    openingToggle(),
    selectInput({ label: 'Show the', value: opts.top, onChange: v => { opts.top = Number(v); changed(); },
      options: [10, 20, 40].map(n => ({ value: n, label: `${n} most played` })) }),
    selectInput({ label: 'Score needs at least', value: opts.minGames, onChange: v => { opts.minGames = Number(v); changed(); },
      options: [5, 10, 20, 50].map(n => ({ value: n, label: `${n} games` })) })),
    coreScopeNote('Opening results are', () => !sides));

  const load = () => query(`WITH o AS (SELECT engine, engine_ord, ${openingLabelSQL()} AS label, sum(games) AS games,
        sum(wins) AS wins, sum(draws) AS draws, sum(losses) AS losses
      FROM ${source()} WHERE opening IS NOT NULL AND opening NOT IN ('', '(unknown)') AND ${engineWhere()} GROUP BY ALL),
    top AS (SELECT label, sum(games) AS total FROM o GROUP BY label ORDER BY total DESC LIMIT ${opts.top})
    SELECT o.*, top.total FROM o JOIN top USING (label) ORDER BY top.total DESC, engine_ord`);

  const score = el('div');
  reactive(score, async () => {
    const rows = (await load()).filter(r => r.games >= opts.minGames).map(r => {
      const s = scoreFromWDL(r.wins, r.draws, r.losses);
      return { ...r, score: s.score, lo: s.lo, hi: s.hi,
        title: `${r.engine} in ${r.label}\nscore ${fmt.pct(s.score)} (${fmt.pct(s.lo, 0)} to ${fmt.pct(s.hi, 0)}) over ${fmt.int(r.games)} games` };
    });
    if (!rows.length) return emptyNote('No opening has enough games for this selection; lower the minimum.');
    const order = [...new Set(rows.map(r => r.label))];
    return [engineLegend(), plot(
      vg.ruleX([0.5], { stroke: 'currentColor', strokeOpacity: 0.4, strokeDasharray: '2,3' }),
      vg.ruleY(rows, { y: 'label', x1: 'lo', x2: 'hi', stroke: 'engine', strokeOpacity: 0.5 }),
      vg.dot(rows, { y: 'label', x: 'score', fill: 'engine', r: 3.5, tip: true, title: 'title' }),
      ...engineColors(), vg.yDomain(order), vg.xDomain([0, 1]), vg.xTickFormat('%'), vg.xGrid(true), vg.marginLeft(280),
      vg.xLabel('Score from the engine\u2019s side (95% interval)'), vg.yLabel(null), vg.width(chartWidth()),
      vg.height(50 + 26 * order.length))];
  }, sub);
  frag.append(panel({
    title: 'Score by opening',
    caption: 'Each dot is one engine\u2019s score in that opening from its own side with a 95% interval, openings ordered by how often they were played; dots right of the dotted line did better than even.',
    body: score, note: sourceNote(openings),
  }));

  const freq = el('div');
  reactive(freq, async () => {
    const rows = await load();
    if (!rows.length) return emptyNote('No opening names for this selection.');
    const order = [...new Set(rows.map(r => r.label))];
    return [engineLegend(), plot(
      vg.barX(rows, { x: 'games', y: 'label', fill: 'engine', insetTop: 3, insetBottom: 3, tip: true }),
      ...engineColors(), vg.yDomain(order), vg.xGrid(true), vg.marginLeft(280), vg.xLabel('Games played by each engine'),
      vg.yLabel(null), vg.width(chartWidth()), vg.height(50 + 24 * order.length))];
  }, sub);
  frag.append(panel({ title: 'How often each opening was played',
    caption: 'Games per opening stacked by engine, where a game between two selected engines counts once for each.', body: freq }));

  const eco = el('div');
  reactive(eco, async () => {
    const rows = await query(`SELECT engine, engine_ord, coalesce(upper(substr(opening_eco, 1, 1)), '?') AS letter, sum(games) AS games
      FROM ${source()} WHERE ${engineWhere()} GROUP BY ALL ORDER BY engine_ord, letter`);
    if (!rows.length || rows.every(r => r.letter === '?')) return emptyNote('No ECO codes recorded.');
    const letters = ['A', 'B', 'C', 'D', 'E', '?'].filter(l => rows.some(r => r.letter === l));
    const colors = letters.map((l, i) => (l === '?' ? '#CFC6B6' : SEQUENTIAL[Math.min(SEQUENTIAL.length - 1, i + 1)]));
    return [seriesLegend(letters.map((l, i) => ({ label: ECO_NAMES[l] ?? l, color: colors[i] }))), plot(
      vg.barX(rows, { x: 'games', y: 'engine', fill: 'letter', offset: 'normalize', insetTop: 6, insetBottom: 6, tip: true }),
      vg.colorDomain(letters), vg.colorRange(colors), vg.yDomain([...new Set(rows.map(r => r.engine))]), vg.xTickFormat('%'),
      vg.marginLeft(140), vg.xLabel('Share of the engine\u2019s games'), vg.yLabel(null), vg.width(chartWidth('half')),
      vg.height(50 + 34 * new Set(rows.map(r => r.engine)).size))];
  });
  frag.append(panel({ span: 'half', title: 'ECO volumes',
    caption: 'Share of each engine\u2019s games by ECO volume: A flank openings, B semi-open, C open, D closed, E Indian defences.', body: eco }));

  const depth = el('div');
  frag.append(panel({ span: 'half', title: 'Search depth by opening',
    caption: 'Mean completed depth of the searches played in the ten most common openings; deeper searches usually mean simpler positions.',
    body: depth }));
  reactive(depth, async () => {
    const d = await dataset('opening_depth');
    if (!d) return emptyNote('Needs searches linked to games.');
    const rows = await query(`WITH o AS (SELECT engine, engine_ord, ${openingLabelSQL()} AS label, sum(searches) AS n,
          sum(depth_mean * searches) / sum(searches) AS depth FROM ds_opening_depth WHERE opening <> '(unknown)' AND ${coreFilter()} AND ${engineWhere()} GROUP BY ALL),
        top AS (SELECT label FROM o GROUP BY label ORDER BY sum(n) DESC LIMIT 10)
      SELECT o.* FROM o JOIN top USING (label) WHERE n >= 20 ORDER BY engine_ord`);
    if (!rows.length) return emptyNote('Not enough searches per opening.');
    const totals = new Map();
    rows.forEach(r => totals.set(r.label, (totals.get(r.label) ?? 0) + r.n));
    return plot(
      vg.dot(rows, { x: 'depth', y: 'label', fill: 'engine', r: 3.5, tip: true }),
      ...engineColors(), vg.yDomain([...totals.keys()].sort((a, b) => totals.get(b) - totals.get(a))), vg.xGrid(true),
      vg.marginLeft(200), vg.xLabel('Mean completed depth (plies)'), vg.yLabel(null),
      vg.width(chartWidth('half')), vg.height(50 + 24 * totals.size));
  }, sub);
  return frag;
}

const ECO_NAMES = { A: 'A flank', B: 'B semi-open', C: 'C open', D: 'D closed', E: 'E Indian', '?': 'unknown' };
