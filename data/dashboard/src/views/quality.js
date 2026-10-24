/** Eval quality: the engine's evaluation and best move against Stockfish. */
import * as vg from '@uwdata/vgplot';
import { dataset } from '../datasets.js';
import { sfPerspective } from '../defs-engine.js';
import { query } from '../db.js';
import { engineById, engineScoped, engineWhere } from '../engines.js';
import { coreRows } from '../settings.js';
import { chartWidth, dataTable, el, emptyNote, engineColors, engineLegend, engineName, fmt, intTicks, panel, plot,
  reactive, sourceNote } from '../ui.js';

export async function render() {
  const frag = document.createDocumentFragment();
  const quality = await dataset('quality');
  if (!quality) {
    frag.append(panel({ title: 'Eval quality',
      body: emptyNote('No Stockfish evaluations in this snapshot. The position step of the analytics pipeline adds them when Stockfish is available.') }));
    return frag;
  }
  const perspective = await sfPerspective();
  frag.append(el('p', { class: 'tab-intro' },
    'Stockfish\u2019s evaluation of the same positions serves as a reference. Both scores are compared from the side to move',
    perspective === 'white' ? ' (this snapshot stores Stockfish\u2019s score from White\u2019s side, so it is flipped for Black to move).' : '.'));

  const table = el('div');
  reactive(table, async () => {
    const rows = await query(`SELECT engine_id, sum(n) AS n, sum(mae * n) / sum(n) AS mae, sum(bias * n) / sum(n) AS bias,
        sum(match_rate * match_n) / nullif(sum(match_n), 0) AS match, sum(blunders) AS blunders
      FROM ds_quality WHERE ${coreRows()} AND ${engineWhere()} GROUP BY 1 ORDER BY 1`);
    return dataTable([
      { label: 'Engine', value: r => engineName(engineById(r.engine_id)) },
      { label: 'Positions', num: true, value: r => fmt.int(r.n) },
      { label: 'Mean absolute difference', num: true, value: r => `${fmt.num(r.mae, 1)} cp` },
      { label: 'Mean difference', num: true, title: 'Positive: the engine is more optimistic than Stockfish', value: r => fmt.cp(r.bias) },
      { label: 'Same best move', num: true, value: r => fmt.pct(r.match) },
      { label: 'Off by more than 200 cp', num: true, value: r => fmt.pct(r.blunders / r.n) },
    ], rows);
  });
  frag.append(panel({ title: 'Agreement with Stockfish',
    caption: 'How far the engine\u2019s evaluation is from Stockfish\u2019s on the same positions, and how often both pick the same move.',
    body: table, note: sourceNote(quality) }));

  const hist = el('div');
  reactive(hist, async () => {
    const d = await dataset('quality_hist');
    const rows = d ? await query(`SELECT engine, (bin_lo + bin_hi) / 2 AS x, share FROM ds_quality_hist WHERE ${coreRows()} AND ${engineWhere()} ORDER BY engine_ord, bin_lo`) : [];
    if (!rows.length) return emptyNote('No differences to show.');
    return [engineLegend(), plot(
      vg.lineY(rows, { x: 'x', y: 'share', stroke: 'engine', z: 'engine', curve: 'step', strokeWidth: 1.4, tip: true }),
      vg.ruleX([0], { stroke: 'currentColor', strokeOpacity: 0.35 }),
      ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), vg.xLabel('Engine minus Stockfish (cp, capped at \u00b1500)'),
      vg.yLabel('Share of positions'), vg.width(chartWidth('half')), vg.height(250))];
  });
  frag.append(panel({ span: 'half', title: 'Size of the differences',
    caption: 'Distribution of the engine\u2019s evaluation minus Stockfish\u2019s; a narrow peak at zero is good, a shifted peak is a bias.', body: hist }));

  const byDepth = el('div');
  reactive(byDepth, async () => {
    const rows = await query(`SELECT engine, depth, match_rate, n FROM ds_quality WHERE n >= 10 AND ${coreRows()} AND ${engineWhere()}
      ORDER BY engine_ord, depth`);
    if (!rows.length) return emptyNote('Not enough positions per depth.');
    return plot(
      vg.lineY(rows, { x: 'depth', y: 'match_rate', stroke: 'engine', z: 'engine', strokeWidth: 1.5 }),
      vg.dot(rows, { x: 'depth', y: 'match_rate', fill: 'engine', r: 2.2, tip: true }),
      ...engineColors(), vg.yTickFormat('%'), vg.yGrid(true), intTicks(), vg.xLabel('Completed depth (plies)'), vg.yLabel('Same best move as Stockfish'),
      vg.width(chartWidth('half')), vg.height(250));
  });
  frag.append(panel({ span: 'half', title: 'Agreement by depth',
    caption: 'Share of positions where the engine\u2019s best move equals Stockfish\u2019s, by the depth the engine reached.', body: byDepth }));

  const pts = await dataset('quality_points');
  if (pts) {
    const $sel = engineScoped({ core: true });
    frag.append(panel({ title: 'Engine against Stockfish',
      caption: 'Each dot is a sampled position; dots on the diagonal are positions where both engines agree on the score.',
      body: plot(
        vg.ruleY([0], { stroke: 'currentColor', strokeOpacity: 0.25 }), vg.ruleX([0], { stroke: 'currentColor', strokeOpacity: 0.25 }),
        vg.dot(vg.from('ds_quality_points', { filterBy: $sel }), { x: 'sf_eval', y: 'eval', fill: 'engine', r: 1.5, fillOpacity: 0.4 }),
        ...engineColors(), vg.xGrid(true), vg.yGrid(true), vg.xLabel('Stockfish (cp, side to move)'), vg.yLabel('Engine (cp, side to move)'),
        vg.width(chartWidth()), vg.height(380)) }));
  }
  return frag;
}
