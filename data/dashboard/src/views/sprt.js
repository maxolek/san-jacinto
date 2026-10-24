/** SPRT: sequential tests between versions, with pentanomial statistics when they were recorded. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineById, sqlIdList } from '../engines.js';
import { interpolate, RESULT_COLORS } from '../palette.js';
import { eloFromPentanomial, eloFromWDL, los } from '../stats.js';
import { chartWidth, coreScopeNote, dataTable, el, emptyNote, engineName, fmt, panel, plot, reactive, seriesLegend, sourceNote, statLine } from '../ui.js';
import { sprtElo } from './compare.js';

const RESULTS = ['pass', 'fail', 'inconclusive'];
const normalise = r => {
  const v = String(r ?? '').toLowerCase();
  return v === 'h1' || v === 'pass' || v === 'passed' ? 'pass' : v === 'h0' || v === 'fail' || v === 'failed' ? 'fail' : v ? 'inconclusive' : 'running';
};
const hasPenta = r => ['penta_ll', 'penta_ld', 'penta_dd', 'penta_wl', 'penta_wd', 'penta_ww'].every(k => r[k] != null);
const PAIR_OUTCOMES = [
  { label: 'Lost both', color: RESULT_COLORS.loss, count: r => r.penta_ll },
  { label: 'Lost one, drew one', color: interpolate([RESULT_COLORS.loss, RESULT_COLORS.draw], 0.5), count: r => r.penta_ld },
  { label: 'Two draws, or a win and a loss', color: RESULT_COLORS.draw, count: r => r.penta_dd + r.penta_wl },
  { label: 'Won one, drew one', color: interpolate([RESULT_COLORS.draw, RESULT_COLORS.win], 0.5), count: r => r.penta_wd },
  { label: 'Won both', color: RESULT_COLORS.win, count: r => r.penta_ww },
];
const RECENT_PAIR_RUNS = 20;
const NO_PAIRS = 'None of these tests recorded results per game pair. The match runner records them (pairs, pentanomial counts, nElo) for every SPRT; older runs only have win, draw and loss totals.';

export async function render() {
  const frag = document.createDocumentFragment();
  const sprt = await dataset('sprt');
  if (!sprt || !sprt.rows) {
    frag.append(panel({ title: 'SPRT', body: emptyNote('No SPRT runs in this snapshot.') }));
    return frag;
  }
  const load = async () => (await query(`SELECT * FROM ds_sprt
      WHERE candidate_engine_id IN (${sqlIdList()}) OR baseline_engine_id IN (${sqlIdList()})
      ORDER BY start_ms NULLS LAST, id`)).map(r => {
    const e = sprtElo(r);
    const wdl = r.candidate_wins != null ? eloFromWDL(r.candidate_wins, r.candidate_draws ?? 0, r.candidate_losses ?? 0) : null;
    return { ...r, outcome: normalise(r.result), elo: e.elo ?? wdl?.elo, lo: e.lo ?? wdl?.lo, hi: e.hi ?? wdl?.hi,
      date: Number.isFinite(r.start_ms) ? new Date(r.start_ms) : null, short: shortPair(r),
      pair: `${engineById(r.candidate_engine_id)?.label ?? r.candidate_engine_id} vs ${engineById(r.baseline_engine_id)?.label ?? r.baseline_engine_id}` };
  }).map(r => ({ ...r, title: `${r.pair}\n${r.date ? fmt.date(r.start_ms) : `run ${r.id}`}: ${fmt.signed(r.elo, 1)} Elo `
    + `(${fmt.signed(r.lo, 1)} to ${fmt.signed(r.hi, 1)})\n${r.outcome}, ${fmt.int(r.games_played)} games` }));

  frag.append(coreScopeNote('Test verdicts are'));
  const stats = el('div');
  reactive(stats, async () => {
    const rows = await load();
    const count = k => rows.filter(r => r.outcome === k).length;
    return statLine([
      { value: fmt.int(rows.length), label: 'tests involving the selected engines' },
      { value: fmt.int(count('pass')), label: 'passed (H1 accepted)' },
      { value: fmt.int(count('fail')), label: 'failed (H0 accepted)' },
      { value: fmt.int(count('inconclusive') + count('running')), label: 'inconclusive or running' },
      { value: fmt.int(rows.reduce((s, r) => s + (r.games_played ?? 0), 0)), label: 'games played' },
    ]);
  });
  frag.append(stats);

  const timeline = el('div');
  reactive(timeline, async () => {
    const rows = (await load()).filter(r => Number.isFinite(r.elo));
    if (!rows.length) return emptyNote('No finished tests with an Elo estimate.');
    const dated = rows.every(r => r.date);
    const x = dated ? 'date' : 'id';
    const colors = [vg.colorDomain([...RESULTS, 'running']), vg.colorRange([RESULT_COLORS.win, RESULT_COLORS.loss, RESULT_COLORS.inconclusive, RESULT_COLORS.draw])];
    return [seriesLegend([{ label: 'Passed', color: RESULT_COLORS.win }, { label: 'Failed', color: RESULT_COLORS.loss },
      { label: 'Inconclusive', color: RESULT_COLORS.inconclusive }]), plot(
      vg.ruleY([0], { stroke: 'currentColor', strokeOpacity: 0.35 }),
      vg.ruleX(rows, { x, y1: 'lo', y2: 'hi', stroke: 'outcome', strokeWidth: 1.5 }),
      vg.dot(rows, { x, y: 'elo', fill: 'outcome', r: 4, tip: true, title: 'title' }),
      vg.text(rows, { x, y: 'hi', text: 'short', dy: -8, fontSize: 10.5, fill: 'currentColor' }),
      ...colors, dated ? vg.xScale('utc') : null, vg.yGrid(true), vg.xLabel(dated ? 'Start of the test' : 'Run'),
      vg.xInsetLeft(40), vg.xInsetRight(40), vg.yLabel('Elo of the candidate (95% interval)'), vg.width(chartWidth()), vg.height(320))];
  });
  frag.append(panel({
    title: 'Tests over time',
    caption: 'Each test is placed at its start time (from the experiments table); the vertical line is the 95% interval for the candidate\u2019s Elo against its baseline, and the colour is the SPRT verdict.',
    body: timeline, note: sourceNote(sprt),
  }));

  const pairs = el('div');
  reactive(pairs, async () => {
    const runs = (await load()).filter(hasPenta).reverse().slice(0, RECENT_PAIR_RUNS);
    if (!runs.length) return emptyNote(NO_PAIRS);
    const seen = new Map();
    const labelled = runs.map(r => {
      const base = `${shortPair(r)}, ${r.date ? fmt.date(r.start_ms) : `run ${r.id}`}`;
      const k = (seen.get(base) ?? 0) + 1;
      seen.set(base, k);
      return { r, label: k > 1 ? `${base} (${k})` : base };
    });
    const long = labelled.flatMap(({ r, label }) => {
      const total = PAIR_OUTCOMES.reduce((s, o) => s + o.count(r), 0);
      return PAIR_OUTCOMES.map((o, i) => ({ run: label, outcome: o.label, n: o.count(r),
        title: `${label}\n${o.label}: ${fmt.int(o.count(r))} of ${fmt.int(total)} pairs (${fmt.pct(o.count(r) / total, 0)})`
          + (i === 2 ? `\n${fmt.int(r.penta_dd)} with two draws, ${fmt.int(r.penta_wl)} with a win and a loss` : '') }));
    });
    const labels = PAIR_OUTCOMES.map(o => o.label);
    return [seriesLegend(PAIR_OUTCOMES.map(o => ({ label: o.label, color: o.color }))), plot(
      vg.barX(long, { x: 'n', y: 'run', fill: 'outcome', offset: 'normalize', order: labels, insetTop: 3, insetBottom: 3,
        tip: true, title: 'title' }),
      vg.colorDomain(labels), vg.colorRange(PAIR_OUTCOMES.map(o => o.color)), vg.yDomain(labelled.map(l => l.label)),
      vg.xTickFormat('%'), vg.xGrid(true), vg.marginLeft(220), vg.xLabel('Share of the test\u2019s game pairs'), vg.yLabel(null),
      vg.width(chartWidth()), vg.height(50 + 24 * labelled.length))];
  });
  frag.append(panel({
    title: 'Game pairs',
    caption: `Each test\u2019s game pairs (one opening played twice with colours swapped), split by the points the candidate took from the pair, for the ${RECENT_PAIR_RUNS} most recent tests that recorded pairs: green on the right is pairs the candidate won outright, and the grey middle is pairs that came out even.`,
    body: pairs,
  }));

  const table = el('div');
  reactive(table, async () => {
    const rows = (await load()).reverse();
    if (!rows.length) return emptyNote('No tests involve the selected engines.');
    const penta = rows.some(hasPenta);
    const columns = [
      { label: 'Date', value: r => el('span', { class: 'nowrap' }, r.date ? fmt.date(r.start_ms) : `run ${r.id}`) },
      { label: 'Candidate', value: r => engineName(engineById(r.candidate_engine_id)) },
      { label: 'Baseline', value: r => engineName(engineById(r.baseline_engine_id)) },
      { label: 'Result', value: r => el('span', {}, el('span', { class: `tag ${r.outcome}` }, r.outcome),
        el('span', { class: 'sub' }, [r.llr != null ? `LLR ${fmt.num(r.llr, 2)}` : null,
          r.elo0 != null ? `Elo H0 ${r.elo0} / H1 ${r.elo1}` : null]
          .filter(Boolean).join(' \u00b7 '))) },
      { label: 'Games', num: true, value: r => el('span', {}, fmt.int(r.games_played),
        r.pairs != null ? el('span', { class: 'sub' }, `${fmt.int(r.pairs)} pairs`) : null) },
      { label: 'W\u2013D\u2013L', num: true, value: r => (r.candidate_wins != null ? `${r.candidate_wins}\u2013${r.candidate_draws}\u2013${r.candidate_losses}` : '\u2014') },
    ];
    columns.push({ label: 'Pentanomial', num: true, title: 'Pairs scoring 0, \u00bd, 1 (draw-draw + win-loss), 1\u00bd and 2 points',
      value: r => (hasPenta(r) ? el('span', {}, [r.penta_ll, r.penta_ld, r.penta_dd + r.penta_wl, r.penta_wd, r.penta_ww].join(' \u00b7 '),
        el('span', { class: 'sub' }, `${r.penta_dd} DD, ${r.penta_wl} WL`)) : '\u2014') });
    columns.push(
      { label: 'Elo', num: true, value: r => (Number.isFinite(r.elo) ? el('span', {}, fmt.signed(r.elo, 1),
        el('span', { class: 'sub' }, Number.isFinite(r.lo) ? `${fmt.signed(r.lo, 1)} to ${fmt.signed(r.hi, 1)}` : '')) : '\u2014') },
    );
    columns.push({ label: 'nElo', num: true, title: 'Normalised Elo: the score difference in units of its own spread',
      value: r => (r.nelo != null ? el('span', {}, fmt.signed(r.nelo, 1), el('span', { class: 'sub' },
        r.nelo_low != null ? `${fmt.signed(r.nelo_low, 1)} to ${fmt.signed(r.nelo_high, 1)}` : '')) : '\u2014') });
    columns.push(
      // sprt_runs.los is stored in percent (tests/sprt.py); the fallback computes a fraction.
      { label: 'LOS', num: true, title: 'Likelihood of superiority',
        value: r => fmt.pct(r.los != null ? r.los / 100 : los(r.candidate_wins, r.candidate_losses), 1) },
      { label: 'Book and time', value: r => el('span', {}, r.time_control ?? '', el('span', { class: 'sub' }, shortBook(r.opening_book))) },
    );
    return [penta ? null : el('p', { class: 'tab-note' }, 'No test here recorded pair counts, so its Elo interval comes from the win, draw and loss totals.'),
      dataTable(columns, rows)];
  });
  frag.append(panel({
    title: 'All tests',
    caption: 'Newest first; the pentanomial column counts opening pairs by the candidate\u2019s points over two games with colours swapped, and Elo intervals come from those pairs when they were recorded.',
    body: table,
  }));

  const cores = el('div');
  reactive(cores, async () => {
    const rows = (await load()).filter(r => r.core_summary);
    const out = [];
    for (const r of rows) {
      let parsed;
      try { parsed = JSON.parse(r.core_summary); } catch { continue; }
      for (const [core, v] of Object.entries(parsed ?? {})) {
        const p = Array.isArray(v?.pentanomial) && v.pentanomial.length === 5 ? eloFromPentanomial(v.pentanomial) : null;
        const core2 = core === 'P' ? 'P-cores' : core === 'E' ? 'E-cores' : core;
        const elo = v?.elo ?? p?.elo, pairs = v?.pairs ?? p?.pairs;
        const name = `${shortPair(r)}, ${r.date ? fmt.date(r.start_ms).replace(/ \d{4}$/, '') : `run ${r.id}`}`;
        out.push({ pair: name, id: r.id, core: core2, pairs, elo, lo: v?.elo_low ?? p?.lo, hi: v?.elo_high ?? p?.hi,
          title: `${name} on ${core2}: ${fmt.signed(elo, 1)} Elo over ${fmt.int(pairs)} pairs` });
      }
    }
    if (!out.length) {
      return emptyNote('None of these tests has a per-core-type breakdown. The match runner records one for every SPRT whose games ran on pinned P- or E-cores.');
    }
    const facets = ['P-cores', 'E-cores', ...new Set(out.map(o => o.core))].filter((c, i, a) => a.indexOf(c) === i && out.some(o => o.core === c));
    return plot(
      vg.ruleX([0], { stroke: 'currentColor', strokeOpacity: 0.35 }),
      vg.ruleY(out, { y: 'pair', fy: 'core', x1: 'lo', x2: 'hi', stroke: 'currentColor', strokeOpacity: 0.6 }),
      vg.dot(out, { y: 'pair', fy: 'core', x: 'elo', fill: 'var(--accent)', r: 3.5, tip: true, title: 'title' }),
      vg.fyDomain(facets), vg.yDomain([...new Set(out.map(o => o.pair))]), vg.marginRight(80),
      vg.xGrid(true), vg.marginLeft(200), vg.fyLabel(null), vg.xLabel('Elo of the candidate (95% interval)'), vg.yLabel(null),
      vg.width(chartWidth()), vg.height(60 + out.length * 22 + 40));
  });
  frag.append(panel({
    title: 'Per core type',
    caption: 'The same tests split by the core type the games ran on; similar results on both mean the verdict does not depend on the hardware.',
    body: cores,
  }));
  return frag;
}

const short = label => String(label ?? '').replace(/^Tomahawk\s+/i, '');
function shortPair(r) {
  return `${short(engineById(r.candidate_engine_id)?.label ?? r.candidate_engine_id)} vs ${short(engineById(r.baseline_engine_id)?.label ?? r.baseline_engine_id)}`;
}
function shortBook(book) {
  return book ? String(book).split(/[\\/]/).pop() : '';
}
