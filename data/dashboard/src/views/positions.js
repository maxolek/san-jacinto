/** Positions: search behaviour by game phase and position type. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { query } from '../db.js';
import { engineScoped, engineWhere } from '../engines.js';
import { coreRows } from '../settings.js';
import { chartWidth, el, emptyNote, engineColors, engineLegend, fmt, panel, plot, reactive, selectInput, sourceNote } from '../ui.js';

const VIEWS = [
  { value: 'searches', label: 'Searches', axis: 'Searches', format: fmt.int },
  { value: 'depth_mean', label: 'Mean depth', axis: 'Mean completed depth (plies)', format: v => fmt.num(v, 2) },
  { value: 'time_mean', label: 'Mean time', axis: 'Mean search time (ms)', format: fmt.ms },
  { value: 'nps', label: 'Nodes per second', axis: 'Nodes per second', format: fmt.nps },
  { value: 'sf_mae', label: 'Eval error vs Stockfish', axis: 'Mean absolute difference from Stockfish (cp)', format: v => `${fmt.num(v, 0)} cp` },
  { value: 'sf_match', label: 'Best move agrees with Stockfish', axis: 'Share of best moves that match', format: v => fmt.pct(v), pct: true },
];
const PHASES = ['opening', 'midgame', 'middlegame', 'endgame', 'unknown'];

export async function render() {
  const frag = document.createDocumentFragment();
  const positions = await dataset('positions');
  if (!positions) {
    frag.append(panel({ body: emptyNote('This snapshot has no position features. Run the position step of the analytics pipeline.') }));
    return frag;
  }
  const [sf] = await query("SELECT sum(sf_n) AS n FROM ds_positions WHERE core_type = 'all'");
  const views = VIEWS.filter(v => !v.value.startsWith('sf_') || sf?.n > 0);
  let view = 'depth_mean';
  const subs = new Set();
  const grouped = (dim, title, caption) => {
    const body = el('div');
    reactive(body, async () => {
      const v = views.find(x => x.value === view);
      const agg = view === 'searches' ? 'sum(searches)' : view === 'nps' ? 'sum(nps * searches) / sum(searches)'
        : `sum(${view} * searches) / nullif(sum(CASE WHEN ${view} IS NOT NULL THEN searches END), 0)`;
      const rows = (await query(`SELECT engine, ${dim} AS g, ${agg} AS v, sum(searches) AS n FROM ds_positions
        WHERE ${coreRows()} AND ${engineWhere()} GROUP BY ALL HAVING v IS NOT NULL ORDER BY min(engine_ord)`))
        .map(r => ({ ...r, title: `${r.engine}, ${r.g}: ${v.format(r.v)} (${fmt.int(r.n)} searches)` }));
      if (!rows.length) return emptyNote('No data for this selection.');
      const groups = [...new Set(rows.map(r => r.g))].sort((a, b) => rank(a) - rank(b) || String(a).localeCompare(String(b)));
      const labels = [...new Set(rows.map(r => r.engine))];
      return plot(
        vg.barX(rows, { x: 'v', y: 'engine', fy: 'g', fill: 'engine', insetTop: 1.5, insetBottom: 1.5, tip: true, title: 'title' }),
        ...engineColors(), vg.fyDomain(groups), vg.yDomain(labels), vg.yAxis(null), vg.fyLabel(null), vg.xGrid(true),
        v.pct ? vg.xTickFormat('%') : null, vg.marginLeft(110), vg.facetMarginBottom(6), vg.xLabel(v.axis),
        vg.width(chartWidth('half')), vg.height(50 + groups.length * (labels.length * 12 + 14)));
    }, [fn => subs.add(fn)]);
    return panel({ span: 'half', title, caption, body });
  };
  frag.append(el('div', { class: 'controls section-controls' },
    selectInput({ label: 'Show', value: view, options: views, onChange: v => { view = v; subs.forEach(fn => fn()); } }), engineLegend()));
  frag.append(grouped('game_phase', 'By game phase', 'The chosen measure for each phase of the game, one bar per engine.'));
  frag.append(grouped('position_type', 'By position type', 'The same measure for tactical and positional positions, as classified by the position step.'));

  const pts = await dataset('positions_points');
  if (pts) {
    const $sel = engineScoped({ core: true });
    frag.append(panel({ title: 'Tactical against positional score',
      caption: 'Each dot is a sampled position placed by its tactical and positional feature scores; colour shows the engine that searched it.',
      body: plot(
        vg.dot(vg.from('ds_positions_points', { filterBy: $sel }), { x: 'tactical', y: 'positional', fill: 'engine', r: 1.6, fillOpacity: 0.45 }),
        ...engineColors(), vg.xGrid(true), vg.yGrid(true), vg.xLabel('Tactical score'), vg.yLabel('Positional score'),
        vg.width(chartWidth()), vg.height(340)),
      note: sourceNote(positions) }));
  }
  return frag;
}

const rank = g => {
  const i = PHASES.indexOf(String(g).toLowerCase());
  return i < 0 ? 99 : i;
};
