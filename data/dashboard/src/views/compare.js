/** Compare: each selected version against the baseline, with intervals, quantile boxes and CDFs. */
import * as vg from '@uwdata/vgplot';
import { dataset, datasetInfo } from '../datasets.js';
import { query } from '../db.js';
import { engineById, engines, engineWhere, sqlIdList } from '../engines.js';
import { cumulative, difference, eloFromPentanomial, isLogBinned, quantileFromHist, ratioSE, scoreFromWDL } from '../stats.js';
import { coreFilter, coreRows, coreType } from '../settings.js';
import { chartWidth, dataTable, el, emptyNote, engineColors, engineLegend, engineName, fmt, panel, plot, reactive,
  selectInput, sourceNote } from '../ui.js';

const DIST_METRICS = [
  { value: 'depth', label: 'Completed depth', axis: 'Completed depth (plies)' },
  { value: 'time_ms', label: 'Time per search', axis: 'Time per search (ms)' },
  { value: 'nodes', label: 'Nodes per search', axis: 'Nodes per search' },
  { value: 'nps', label: 'Nodes per second', axis: 'Nodes per second (per search)' },
];

const ratio = (y, x, yy, xy, xx) => r => ({
  v: r[y] != null && r[x] ? r[y] / r[x] : null,
  se: ratioSE({ sumY: r[y], sumX: r[x], sumYY: r[yy], sumXY: r[xy], sumXX: r[xx], n: r.searches }),
});

/**
 * Rows of the comparison table: how to read a value and its standard error from a summary row.
 * Exact sums of squares exist when the summary was computed in the browser; otherwise the errors
 * come from the search sample (fillSampleErrors).
 */
export const METRICS = [
  { label: 'Nodes per second', title: 'All nodes divided by all search time', format: v => fmt.nps(v),
    get: r => ({ v: r.nps, se: r.nodes_sq != null ? ratioSE({ sumY: r.nodes, sumX: r.time_ms, sumYY: r.nodes_sq, sumXY: r.nodes_time,
      sumXX: r.time_sq, n: r.searches }) * 1000 : r.nps_se }) },
  { label: 'Mean depth', format: v => fmt.num(v, 2), get: r => ({ v: r.depth_mean, se: r.depth_sd / Math.sqrt(r.searches) }) },
  { label: 'Median time', title: 'Median search time; medians are shown without an interval', format: v => fmt.ms(v),
    get: r => ({ v: r.time_ms_p50, se: NaN }) },
  { label: 'Mean nodes', format: v => fmt.compact(v), get: r => ({ v: r.nodes_mean, se: r.nodes_sd / Math.sqrt(r.searches) }) },
  { label: 'TT hit rate', title: 'Transposition-table hits per node', format: v => fmt.pct(v, 1),
    get: r => (r.tt_hits != null ? ratio('tt_hits', 'nodes', 'tt_hits_sq', 'tt_hits_nodes', 'nodes_sq')(r) : { v: r.tt_hit_rate, se: r.tt_hit_rate_se }) },
  { label: 'First-move cutoffs', title: 'Share of beta cutoffs produced by the first move searched', format: v => fmt.pct(v, 1),
    get: r => (r.fh0 != null ? ratio('fh0', 'fh', 'fh0_sq', 'fh0_fh', 'fh_sq')(r) : { v: r.fh_first_rate, se: r.fh_first_rate_se }) },
  { label: 'Quiescence share', title: 'Share of nodes spent in quiescence search', format: v => fmt.pct(v, 1),
    get: r => (r.qnodes_sq != null ? ratio('qnodes', 'nodes', 'qnodes_sq', 'qnodes_nodes', 'nodes_sq')(r) : { v: r.qratio, se: r.qratio_se }) },
  { label: 'Null-move success', title: 'Null-move searches that failed high', format: v => fmt.pct(v, 1),
    get: r => (r.nmp_fh != null ? ratio('nmp_fh', 'nmp', 'nmp_fh_sq', 'nmp_fh_nmp', 'nmp_sq')(r) : { v: r.nmp_success_rate, se: NaN }) },
];

export async function render() {
  const frag = document.createDocumentFragment();
  frag.append(el('p', { class: 'tab-intro' },
    'Each selected version against the ', el('strong', {}, 'baseline'), ' chosen above. Differences carry 95% intervals; ',
    'a difference in dark type has an interval that excludes zero.'));
  const [summary, hist, sides, sprt] = await Promise.all([dataset('summary'), dataset('hist'), dataset('game_sides'), dataset('sprt')]);
  if (!summary) {
    frag.append(panel({ body: emptyNote('This snapshot has no search data to compare.') }));
    return frag;
  }
  const table = el('div');
  reactive(table, async () => {
    const core = coreType();
    const rows = await query(`SELECT * FROM ds_summary WHERE ${coreRows()} AND ${engineWhere()} ORDER BY engine_ord`);
    if (!rows.length) return emptyNote('No searches for this selection.');
    await fillSampleErrors(rows, core);
    const base = rows.find(r => r.engine_id === engines.baseline) ?? rows[0];
    const games = sides ? await query(`SELECT engine_id, count(*) FILTER (WHERE score = 1)::DOUBLE AS w,
        count(*) FILTER (WHERE score = 0.5)::DOUBLE AS d, count(*) FILTER (WHERE score = 0)::DOUBLE AS l
      FROM ds_game_sides WHERE ${engineWhere()} AND ${coreFilter()} GROUP BY 1`) : [];
    const scoreBy = new Map(games.map(g => [g.engine_id, scoreFromWDL(g.w, g.d, g.l)]));
    const runs = sprt ? await query(`SELECT * FROM ds_sprt WHERE candidate_engine_id IN (${sqlIdList()})
      AND baseline_engine_id IN (${sqlIdList()}) ORDER BY start_ms DESC NULLS LAST, id DESC`) : [];
    const diffLine = (d, scale = 'rel') => {
      const est = scale === 'rel' ? d.rel : d.diff;
      if (!Number.isFinite(est)) return null;
      const f = scale === 'rel' ? fmt.signedPct : v => `${fmt.signed(v * 100)} pts`;
      const half = scale === 'rel' ? (d.relHi - d.relLo) / 2 : (d.hi - d.lo) / 2;
      const text = Number.isFinite(half) ? `${f(est)} \u00b1 ${(half * 100).toFixed(1)}` : f(est);
      return el('span', { class: `sub${d.significant ? ' sig' : ''}` }, text);
    };
    const specs = [
      { label: 'Searches', cell: r => fmt.int(r.searches) },
      ...METRICS.map(m => ({ label: m.label, title: m.title, cell: r => {
        const a = m.get(r), b = m.get(base);
        const value = a.v == null ? '\u2014' : m.format(a.v);
        if (r === base) return el('span', {}, value, el('span', { class: 'sub base' }, 'baseline'));
        return el('span', {}, value, diffLine(difference(a.v, a.se, b.v, b.se)));
      } })),
    ];
    if (scoreBy.size) {
      specs.push({ label: 'Score', title: 'Points per game from the engine\u2019s side; the difference is in percentage points', cell: r => {
        const s = scoreBy.get(r.engine_id), b = scoreBy.get(base.engine_id);
        if (!s) return '\u2014';
        if (r === base || !b) return el('span', {}, fmt.pct(s.score), el('span', { class: 'sub base' }, r === base ? 'baseline' : `${fmt.int(s.n)} games`));
        return el('span', {}, fmt.pct(s.score), diffLine(difference(s.score, s.se, b.score, b.se), 'abs'));
      } });
    }
    if (runs.length) {
      specs.push({ label: 'SPRT against the baseline', title: 'Most recent SPRT between this version and the baseline, from this version\u2019s side', cell: r => {
        if (r === base) return '';
        const run = runs.find(x => (x.candidate_engine_id === r.engine_id && x.baseline_engine_id === base.engine_id)
          || (x.candidate_engine_id === base.engine_id && x.baseline_engine_id === r.engine_id));
        if (!run) return el('span', { class: 'sub' }, 'no test');
        const e = sprtElo(run);
        const sign = run.candidate_engine_id === r.engine_id ? 1 : -1;
        const [lo, hi] = sign > 0 ? [e.lo, e.hi] : [-e.hi, -e.lo];
        return el('span', {}, Number.isFinite(e.elo) ? `${fmt.signed(sign * e.elo, 1)} Elo` : '\u2014', el('span', { class: 'sub' },
          Number.isFinite(lo) ? `${fmt.signed(lo, 1)} to ${fmt.signed(hi, 1)} \u00b7 ${run.result ?? 'running'}` : run.result ?? ''));
      } });
    }
    const columns = [
      { label: 'Metric', value: s => el('span', { title: s.title ?? null }, s.label) },
      ...rows.map(r => ({ num: true, value: s => s.cell(r),
        label: el('span', {}, engineName(engineById(r.engine_id)), r === base ? el('span', { class: 'sub base' }, 'baseline') : null) })),
    ];
    return dataTable(columns, specs);
  });
  frag.append(panel({
    title: 'Key metrics against the baseline',
    caption: 'One column per engine, with the change against the baseline and its 95% margin under each value (dark type means the change is clear of zero); nodes per second is total nodes over total search time, not an average of per-search rates.',
    body: table,
    note: `${sourceNote(summary)} Intervals treat searches as independent, so read them as a lower bound on the real uncertainty.`,
  }));

  if (hist) {
    let metric = 'depth';
    const metricSubs = new Set();
    const picker = selectInput({ label: 'Distribution of', options: DIST_METRICS, value: metric,
      onChange: v => { metric = v; metricSubs.forEach(fn => fn()); } });
    const loadBins = async () => {
      const rows = await query(`SELECT engine, bin_lo, bin_hi, count, scale FROM ds_hist
        WHERE metric = '${metric}' AND ${coreRows()} AND ${engineWhere()} ORDER BY engine_ord, bin_lo`);
      const byEngine = new Map();
      for (const r of rows) {
        if (!byEngine.has(r.engine)) byEngine.set(r.engine, []);
        byEngine.get(r.engine).push({ lo: r.bin_lo, hi: r.bin_hi, count: r.count });
      }
      const log = rows[0]?.scale ? rows[0].scale === 'log' : isLogBinned(rows.map(r => ({ lo: r.bin_lo, hi: r.bin_hi })));
      return { byEngine, log };
    };
    const axis = () => DIST_METRICS.find(o => o.value === metric).axis;
    const boxes = el('div');
    reactive(boxes, async () => {
      const { byEngine, log } = await loadBins();
      if (!byEngine.size) return emptyNote('No values for this metric.');
      const f = formatFor(metric);
      const data = [...byEngine].map(([engine, bins]) => {
        const q = p => quantileFromHist(bins, p, { log });
        const d = { engine, p10: q(0.1), p25: q(0.25), p50: q(0.5), p75: q(0.75), p90: q(0.9) };
        return { ...d, title: `${engine}\nmedian ${f(d.p50)}\nmiddle half ${f(d.p25)} to ${f(d.p75)}` };
      });
      return plot(
        vg.ruleY(data, { y: 'engine', x1: 'p10', x2: 'p90', stroke: 'engine', strokeWidth: 1.2 }),
        vg.barX(data, { y: 'engine', x1: 'p25', x2: 'p75', fill: 'engine', fillOpacity: 0.2, stroke: 'engine', strokeWidth: 1,
          insetTop: 8, insetBottom: 8, className: 'box' }),
        vg.tickX(data, { y: 'engine', x: 'p50', stroke: 'engine', strokeWidth: 2.5, insetTop: 5, insetBottom: 5, tip: true, title: 'title' }),
        ...engineColors(), vg.yDomain(data.map(d => d.engine)), vg.xScale(log ? 'log' : 'linear'), log ? vg.xTicks(5) : null,
        vg.xGrid(true), vg.marginLeft(140), vg.xLabel(axis()), vg.yLabel(null), vg.width(chartWidth('half')),
        vg.height(60 + 42 * data.length));
    }, [fn => metricSubs.add(fn)]);
    const cdf = el('div');
    reactive(cdf, async () => {
      const { byEngine, log } = await loadBins();
      if (!byEngine.size) return emptyNote('No values for this metric.');
      const data = [...byEngine].flatMap(([engine, bins]) => cumulative(bins).map(p => ({ engine, ...p })));
      return [engineLegend(), plot(
        vg.lineY(data, { x: 'x', y: 'share', stroke: 'engine', z: 'engine', strokeWidth: 1.6, tip: true }),
        vg.ruleY([0.5], { stroke: 'currentColor', strokeOpacity: 0.35, strokeDasharray: '3,3' }),
        ...engineColors(), vg.xScale(log ? 'log' : 'linear'), log ? vg.xTicks(5) : null, vg.yDomain([0, 1]), vg.yTickFormat('%'),
        vg.yGrid(true), vg.xLabel(axis()), vg.yLabel('Searches at or below'), vg.width(chartWidth('half')), vg.height(300))];
    }, [fn => metricSubs.add(fn)]);
    frag.append(el('div', { class: 'controls section-controls' }, picker));
    frag.append(panel({ span: 'half', title: 'Spread by engine',
      caption: 'Boxes cover the middle half of searches, the thick tick is the median and the whiskers reach the 10th and 90th percentiles.',
      body: boxes, note: sourceNote(datasetInfo.get('hist')) }));
    frag.append(panel({ span: 'half', title: 'Cumulative distribution',
      caption: 'The height of a line at any value is the share of that engine\u2019s searches at or below it; a line further right means larger values throughout.',
      body: cdf }));
  }
  return frag;
}

function formatFor(metric) {
  return metric === 'time_ms' ? fmt.ms : metric === 'nps' ? fmt.nps : metric === 'depth' ? v => fmt.num(v, 1) : v => fmt.compact(v);
}

/** Elo and interval of an SPRT row: measured values first, the pentanomial counts otherwise. */
export function sprtElo(run) {
  const penta = ['penta_ll', 'penta_ld', 'penta_dd', 'penta_wl', 'penta_wd', 'penta_ww'].every(k => run[k] != null)
    ? eloFromPentanomial([run.penta_ll, run.penta_ld, run.penta_dd + run.penta_wl, run.penta_wd, run.penta_ww]) : null;
  return { elo: run.elo_diff ?? penta?.elo, lo: run.elo_low ?? penta?.lo, hi: run.elo_high ?? penta?.hi };
}

/**
 * When the snapshot's summary has no sums of squares (pre-aggregated tables), estimate standard errors
 * from the search sample: the per-search spread comes from the sample, the sample size from the full data.
 * For a ratio of sums R = sum(y) / sum(x) the delta-method variance is sum((y - R x)^2) / (n - 1).
 */
export async function fillSampleErrors(rows, core) {
  if (rows.every(r => r.nodes_sq != null && r.depth_sd != null)) return;
  const sample = await dataset('sample');
  const has = c => sample?.columns.has(c);
  if (!has('nodes') || !has('time_ms')) return;
  const coreCond = core === 'all' || !has('core_type') ? 'TRUE' : `core_type = '${core}'`;
  const values = rows.map(r => `(${r.engine_id}, ${num(r.nodes / r.time_ms)}, ${num(r.tt_hit_rate)}, ${num(r.qratio)}, ${num(r.fh_first_rate)})`);
  const spread = (rate, weight, key) => (has(rate)
    ? `sum(power(${weight}, 2) * power(${rate} - r.${key}, 2)) / nullif(count(${rate}) - 1, 0)` : 'NULL');
  const stats = await query(`WITH r(engine_id, nps_r, tt_r, q_r, fh_r) AS (VALUES ${values.join(', ')})
    SELECT s.engine_id, avg(time_ms) AS t_mean, sum(power(nodes - r.nps_r * time_ms, 2)) / nullif(count(*) - 1, 0) AS nps_s2,
      avg(nodes) AS n_mean, ${spread('tt_hit_rate', 'nodes', 'tt_r')} AS tt_s2, ${spread('qratio', 'nodes', 'q_r')} AS q_s2,
      ${has('fail_high_rate') ? 'avg(fail_high_rate * nodes)' : 'NULL'} AS fh_mean,
      ${has('fail_high_rate') ? spread('fh_first_rate', 'fail_high_rate * nodes', 'fh_r') : 'NULL'} AS fh_s2,
      ${has('depth') ? 'stddev_samp(depth)' : 'NULL'} AS depth_sd, stddev_samp(nodes) AS nodes_sd
    FROM ds_sample s JOIN r USING (engine_id) WHERE ${coreCond} AND time_ms > 0 GROUP BY 1`);
  const byId = new Map(stats.map(s => [s.engine_id, s]));
  const se = (s2, mean, n) => (s2 > 0 && mean > 0 && n > 1 ? Math.sqrt(s2 / n) / mean : NaN);
  for (const r of rows) {
    const s = byId.get(r.engine_id);
    if (!s) continue;
    r.nps_se ??= se(s.nps_s2, s.t_mean, r.searches) * 1000;
    r.tt_hit_rate_se ??= se(s.tt_s2, s.n_mean, r.searches);
    r.qratio_se ??= se(s.q_s2, s.n_mean, r.searches);
    r.fh_first_rate_se ??= se(s.fh_s2, s.fh_mean, r.searches);
    r.depth_sd ??= s.depth_sd;
    r.nodes_sd ??= s.nodes_sd;
  }
}

const num = v => (Number.isFinite(v) ? String(v) : 'NULL');
