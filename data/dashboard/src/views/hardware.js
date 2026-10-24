/** Hardware: speed against the measured clock, knps per GHz, and P-core against E-core. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineById, engines, engineWhere } from '../engines.js';
import { coreType, onCoreType } from '../settings.js';
import { decomposeNps } from '../stats.js';
import { chartWidth, dataTable, el, emptyNote, engineColors, engineLegend, engineName, fmt, panel, plot, reactive,
  sourceNote, statLine } from '../ui.js';

const CORE_LABEL = { P: 'P-core', E: 'E-core', '-': 'not pinned', mixed: 'moved between cores', unknown: 'unknown core' };
const coreLabel = c => CORE_LABEL[c] ?? c;

export async function render() {
  const frag = document.createDocumentFragment();
  const env = await dataset('env');
  if (!env || !env.rows) {
    frag.append(panel({ title: 'Hardware',
      body: emptyNote('This snapshot has no measured clock speeds. Games recorded by the match runner carry env_mhz_mean and env_core_type; older data does not.') }));
    return frag;
  }
  frag.append(el('p', { class: 'tab-intro' },
    'Speed depends on the code and on the processor it ran on. ', el('strong', {}, 'Knps per GHz'),
    ' divides speed by the measured clock, so it compares code on equal footing. P-cores and E-cores still differ in how much work ',
    'they do per clock tick; that gap is a property of the hardware, not of the engine.'));
  const allCores = el('p', { class: 'tab-note' }, 'This tab always shows every core type side by side, so the Cores filter above does not narrow it.');
  const showAllCores = () => { allCores.hidden = coreType() === 'all'; };
  showAllCores();
  onCoreType(showAllCores);
  frag.append(allCores);

  const perCore = async () => query(`SELECT engine_id, engine, core_type, sum(searches) AS searches, sum(nodes) AS nodes,
      sum(time_ms) AS time_ms, sum(mhz_time) AS mhz_time FROM ds_env WHERE ${engineWhere()} GROUP BY ALL ORDER BY min(engine_ord), core_type`);

  const stats = el('div');
  reactive(stats, async () => {
    const rows = await perCore();
    const by = c => rows.filter(r => r.core_type === c);
    const clock = c => {
      const rs = by(c);
      const t = rs.reduce((s, r) => s + r.time_ms, 0);
      return t ? rs.reduce((s, r) => s + r.mhz_time, 0) / t : NaN;
    };
    const total = rows.reduce((s, r) => s + r.time_ms, 0);
    const ratios = [...new Set(rows.map(r => r.engine_id))].map(id => {
      const p = rows.find(r => r.engine_id === id && r.core_type === 'P'), e = rows.find(r => r.engine_id === id && r.core_type === 'E');
      return p && e ? (e.nodes / e.mhz_time) / (p.nodes / p.mhz_time) : NaN;
    }).filter(Number.isFinite);
    return statLine([
      { value: Number.isFinite(clock('P')) ? `${(clock('P') / 1000).toFixed(2)} GHz` : '\u2014', label: 'mean P-core clock, weighted by time' },
      { value: Number.isFinite(clock('E')) ? `${(clock('E') / 1000).toFixed(2)} GHz` : '\u2014', label: 'mean E-core clock' },
      { value: total ? fmt.pct(by('P').reduce((s, r) => s + r.time_ms, 0) / total, 0) : '\u2014', label: 'of search time on P-cores' },
      { value: ratios.length ? fmt.pct(ratios.reduce((a, b) => a + b, 0) / ratios.length, 0) : '\u2014',
        label: 'E-core work per clock, as a share of P-core' },
    ]);
  });
  frag.append(stats);

  const scatter = el('div');
  reactive(scatter, async () => {
    const rows = (await query(`SELECT engine, core_type, mhz_mean, nps, searches FROM ds_env WHERE ${engineWhere()} AND searches > 0
      ORDER BY engine_ord, core_type, mhz_mean`)).map(r => ({ ...r, ghz: r.mhz_mean / 1000, mnps: r.nps / 1e6, core: coreLabel(r.core_type),
      series: `${r.engine} ${r.core_type}`,
      title: `${r.engine} on ${coreLabel(r.core_type)}\n${(r.mhz_mean / 1000).toFixed(2)} GHz: ${(r.nps / 1e6).toFixed(2)} Mnps over ${fmt.int(r.searches)} searches` }));
    if (!rows.length) return emptyNote('No clock measurements for this selection.');
    return [engineLegend(), plot(
      vg.lineY(rows, { x: 'ghz', y: 'mnps', stroke: 'engine', z: 'series', strokeWidth: 1, strokeOpacity: 0.6 }),
      vg.dot(rows, { x: 'ghz', y: 'mnps', stroke: 'engine', fill: 'engine', fillOpacity: 0.25, symbol: 'core', r: 'searches', tip: true,
        title: 'title' }),
      ...engineColors(), vg.symbolDomain(['P-core', 'E-core', 'not pinned', 'moved between cores', 'unknown core']),
      vg.symbolRange(['circle', 'square', 'triangle', 'diamond', 'cross']),
      vg.rRange([1.5, 5]), vg.xGrid(true), vg.yGrid(true), vg.xLabel('Measured clock during the game (GHz)'),
      vg.yLabel('Million nodes per second'), vg.width(chartWidth()), vg.height(340)),
    el('p', { class: 'legend' }, 'Circles are P-cores, squares E-cores, triangles unpinned games; larger marks hold more searches.')];
  });
  frag.append(panel({
    title: 'Speed against clock',
    caption: 'Nodes per second in each 100 MHz band of the measured clock, one line per engine and core type: a line that sits higher at the same clock is faster code.',
    body: scatter, note: sourceNote(env),
  }));

  const perClock = el('div');
  reactive(perClock, async () => {
    const rows = (await perCore()).filter(r => r.mhz_time > 0).map(r => {
      const k = (r.nodes / (r.time_ms / 1000)) / 1000 / (r.mhz_time / r.time_ms / 1000);
      return { ...r, core: coreLabel(r.core_type), k, ktext: `${Math.round(k)}`, title: `${r.engine} on ${coreLabel(r.core_type)}: ${Math.round(k)} knps per GHz` };
    });
    if (!rows.length) return emptyNote('No clock measurements for this selection.');
    const labels = [...new Set(rows.map(r => r.engine))];
    const facets = ['P-core', 'E-core', 'not pinned', ...new Set(rows.map(r => r.core))].filter((c, i, a) => a.indexOf(c) === i && rows.some(r => r.core === c));
    return plot(
      vg.ruleY(rows, { y: 'engine', fy: 'core', x1: 0, x2: 'k', stroke: 'engine', strokeOpacity: 0.35 }),
      vg.dot(rows, { y: 'engine', fy: 'core', x: 'k', fill: 'engine', r: 4.5, tip: true, title: 'title' }),
      vg.text(rows, { y: 'engine', fy: 'core', x: 'k', text: 'ktext', dx: 10, textAnchor: 'start', fill: 'currentColor' }),
      ...engineColors(), vg.yDomain(labels), vg.fyDomain(facets), vg.xGrid(true), vg.marginLeft(150), vg.marginRight(90), vg.fyLabel(null),
      vg.xLabel('Thousand nodes per second per GHz of measured clock'), vg.yLabel(null), vg.width(chartWidth()),
      vg.height(60 + rows.length * 26 + 30));
  });
  frag.append(panel({
    title: 'Knps per GHz, by core type',
    caption: 'Speed divided by the clock it ran at, by core type: differences within a group are the code, and the gap between groups is how much less an E-core does per tick.',
    body: perClock,
  }));

  const table = el('div');
  reactive(table, async () => {
    const rows = await perCore();
    const ids = [...new Set(rows.map(r => r.engine_id))];
    const of = id => rows.filter(r => r.engine_id === id);
    const base = of(engines.baseline).length ? engines.baseline : ids[0];
    const out = ids.map(id => {
      const rs = of(id);
      const nodes = rs.reduce((s, r) => s + r.nodes, 0), time = rs.reduce((s, r) => s + r.time_ms, 0);
      const mhz = rs.reduce((s, r) => s + r.mhz_time, 0) / time;
      const p = rs.find(r => r.core_type === 'P'), e = rs.find(r => r.core_type === 'E');
      return { id, nps: nodes / (time / 1000), mhz, pShare: (p?.time_ms ?? 0) / time,
        ep: p && e ? (e.nodes / e.mhz_time) / (p.nodes / p.mhz_time) : NaN,
        split: id === base ? null : decomposeNps(rs, of(base)) };
    });
    return dataTable([
      { label: 'Engine', value: r => engineName(engineById(r.id)) },
      { label: 'Nodes per second', num: true, value: r => fmt.nps(r.nps) },
      { label: 'Change', num: true, title: 'Nodes per second against the baseline', value: r => (r.split ? fmt.signedPct(r.split.total) : el('span', { class: 'base' }, 'baseline')) },
      { label: 'From the code', num: true, title: 'Change in knps per GHz on the baseline\u2019s mix of core types and clocks',
        value: r => (r.split ? el('span', { class: 'sig' }, fmt.signedPct(r.split.code)) : '') },
      { label: 'From the hardware', num: true, title: 'What remains: a different clock or a different share of time on each core type',
        value: r => (r.split ? fmt.signedPct(r.split.hardware) : '') },
      { label: 'Time on P-cores', num: true, value: r => fmt.pct(r.pShare, 0) },
      { label: 'Mean clock', num: true, value: r => `${(r.mhz / 1000).toFixed(2)} GHz` },
      { label: 'E-core per clock', num: true, title: 'E-core knps per GHz as a share of P-core knps per GHz for the same engine', value: r => fmt.pct(r.ep, 0) },
    ], out);
  });
  frag.append(panel({
    title: 'Where a speed difference comes from',
    caption: 'The change in nodes per second against the baseline, split into the part the code explains (knps per GHz on the baseline\u2019s mix of core types and clocks) and the part the hardware explains (clock and time on each core type); the two parts multiply to the total.',
    body: table,
  }));
  return frag;
}
