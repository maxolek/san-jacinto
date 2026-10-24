import test from 'node:test';
import assert from 'node:assert/strict';
import { bucketWidth, cumulative, decomposeNps, difference, eloFromPentanomial, eloFromWDL, fitLogistic, isLogBinned, los,
  mergeCounts, quantileFromHist, ratioSE, scoreFromWDL, scoreToElo, wdlBucket, wilson } from '../src/stats.js';

const close = (a, b, eps = 1e-6, msg) => assert.ok(Math.abs(a - b) <= eps, `${msg ?? ''} expected ${b}, got ${a}`);

test('wilson interval matches the textbook values and stays inside [0, 1]', () => {
  const [lo, hi] = wilson(0.5, 100);
  close(lo, 0.4038, 1e-4); close(hi, 0.5962, 1e-4);
  const [lo0, hi0] = wilson(0, 20);
  assert.equal(lo0, 0); close(hi0, 0.1611, 1e-4);
  assert.ok(wilson(1, 5)[1] <= 1);
  assert.ok(wilson(0.5, 0).every(Number.isNaN));
});

test('wdl bucket uses games, not positions, as the sample size', () => {
  const b = wdlBucket({ positions: 400, games: 40, wins: 200, draws: 100, losses: 100 });
  close(b.pWin, 0.5); close(b.pDraw, 0.25); close(b.pLoss, 0.25); close(b.score, 0.625);
  const byPositions = wdlBucket({ positions: 400, games: 400, wins: 200, draws: 100, losses: 100 });
  const width = r => r[1] - r[0];
  assert.ok(width(b.win) > 2.5 * width(byPositions.win), 'fewer games give a wider interval');
  assert.deepEqual(b.win, wilson(0.5, 40));
  // expected score interval from the per-game variance of 1 / 0.5 / 0 outcomes
  const variance = 0.5 * 0.375 ** 2 + 0.25 * 0.125 ** 2 + 0.25 * 0.625 ** 2;
  close(b.scoreCI[1] - b.score, 1.959963984540054 * Math.sqrt(variance / 40), 1e-9);
  assert.equal(wdlBucket({ positions: 0 }), null);
  // games never exceed positions
  assert.equal(wdlBucket({ positions: 10, games: 50, wins: 5, draws: 5, losses: 0 }).games, 10);
});

test('bucket width is read from the data, not assumed', () => {
  assert.equal(bucketWidth([-50, 0, 25, 100, 1000]), 25);
  assert.equal(bucketWidth([-1000, -950, -900]), 50);
  assert.equal(bucketWidth([0, 0, 50, null, NaN]), 50);
  assert.equal(bucketWidth([75], 25), 25);
  assert.ok(Number.isNaN(bucketWidth([])));
});

test('mergeCounts sums buckets across phases', () => {
  const rows = [
    { eval_bucket: 0, phase: 'opening', positions: 10, games: 4, wins: 3, draws: 5, losses: 2 },
    { eval_bucket: 0, phase: 'endgame', positions: 6, games: 3, wins: 1, draws: 1, losses: 4 },
    { eval_bucket: 25, phase: 'opening', positions: 2, games: 1, wins: 2, draws: 0, losses: 0 },
  ];
  const merged = mergeCounts(rows, r => r.eval_bucket);
  assert.equal(merged.length, 2);
  assert.deepEqual([merged[0].positions, merged[0].wins, merged[0].draws, merged[0].losses, merged[0].games], [16, 4, 6, 6, 7]);
});

test('logistic fit recovers the 50% point of a known curve', () => {
  const xs = [], ps = [], ws = [];
  for (let x = -400; x <= 400; x += 25) {
    xs.push(x); ps.push(1 / (1 + Math.exp(-(x - 80) / 90))); ws.push(50);
  }
  const fit = fitLogistic(xs, ps, ws);
  assert.ok(fit.converged);
  close(fit.x50, 80, 0.5);
  close(1 / fit.b, 90, 0.5);
  close(fit.predict(80), 0.5, 1e-3);
  assert.ok(fit.x50SE > 0 && fit.x50SE < 20);
  assert.equal(fitLogistic([1, 2], [0.5, 0.6], [1, 1]), null);
});

test('difference and ratio standard errors', () => {
  const d = difference(110, 3, 100, 4);
  close(d.diff, 10); close(d.rel, 0.1);
  close(d.hi - d.lo, 2 * 1.959963984540054 * 5, 1e-9);
  assert.equal(d.significant, true);
  assert.equal(difference(101, 3, 100, 4).significant, false);
  // ratio of sums with no spread in the per-item ratio has zero error
  const exact = ratioSE({ sumY: 20, sumX: 10, sumYY: 100, sumXY: 50, sumXX: 25, n: 4 });
  close(exact, 0, 1e-12);
  const noisy = ratioSE({ sumY: 20, sumX: 10, sumYY: 116, sumXY: 52, sumXX: 26, n: 4 });
  assert.ok(noisy > 0);
  assert.ok(Number.isNaN(ratioSE({ sumY: 1, sumX: 1, n: 1 })));
});

test('quantiles and CDF from histogram bins', () => {
  const lin = [{ lo: 0, hi: 10, count: 50 }, { lo: 10, hi: 20, count: 50 }];
  close(quantileFromHist(lin, 0.5), 10);
  close(quantileFromHist(lin, 0.25), 5);
  close(quantileFromHist(lin, 0.75), 15);
  const log = [{ lo: 1, hi: 10, count: 10 }, { lo: 10, hi: 100, count: 10 }];
  close(quantileFromHist(log, 0.25, { log: true }), Math.sqrt(10));
  assert.equal(isLogBinned(log.concat([{ lo: 100, hi: 1000, count: 1 }])), true);
  assert.equal(isLogBinned([{ lo: 0, hi: 1 }, { lo: 1, hi: 2 }, { lo: 2, hi: 3 }]), false);
  const cdf = cumulative(lin);
  assert.deepEqual(cdf.map(p => [p.x, p.share]), [[0, 0], [10, 0.5], [20, 1]]);
  assert.ok(Number.isNaN(quantileFromHist([], 0.5)));
});

test('Elo from scores, game counts and pentanomial pairs', () => {
  close(scoreToElo(0.5), 0);
  close(scoreToElo(0.75), 190.848501887865, 1e-9);
  const s = scoreFromWDL(30, 40, 30);
  close(s.score, 0.5);
  assert.ok(s.lo < 0.5 && s.hi > 0.5);
  const e = eloFromWDL(60, 20, 20);
  assert.ok(e.elo > 0 && e.lo < e.elo && e.hi > e.elo);
  const p = eloFromPentanomial([10, 20, 40, 20, 10]);
  close(p.score, 0.5); close(p.elo, 0);
  assert.equal(p.pairs, 100);
  close(los(10, 10), 0.5);
  assert.ok(los(60, 40) > 0.95);
});

test('NPS change splits into a code part and a hardware part', () => {
  // baseline: 400 knps/GHz on P-cores at 4 GHz, 250 on E-cores at 3 GHz, equal time on both
  const rows = (kP, kE, fP, fE, tP, tE) => [
    { core_type: 'P', time_ms: tP, nodes: kP * fP * tP, mhz_time: fP * 1000 * tP },
    { core_type: 'E', time_ms: tE, nodes: kE * fE * tE, mhz_time: fE * 1000 * tE },
  ];
  const base = rows(400, 250, 4, 3, 1000, 1000);
  const sameHardware = decomposeNps(rows(440, 275, 4, 3, 1000, 1000), base);
  close(sameHardware.code, 0.1, 1e-9); close(sameHardware.hardware, 0, 1e-9); close(sameHardware.total, 0.1, 1e-9);
  const sameCode = decomposeNps(rows(400, 250, 4, 3, 3000, 1000), base);
  close(sameCode.code, 0, 1e-9); assert.ok(sameCode.hardware > 0);
  close((1 + sameCode.code) * (1 + sameCode.hardware), 1 + sameCode.total, 1e-9);
  const none = decomposeNps([{ core_type: 'P', time_ms: 1, nodes: 1, mhz_time: 0 }], base);
  assert.ok(Number.isNaN(none.code));
});
