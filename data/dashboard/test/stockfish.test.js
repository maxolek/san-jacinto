import test from 'node:test';
import assert from 'node:assert/strict';
import { LOSS_BUCKETS, cumulativeRanks, depthLabel, formatEval, lichessUrl, lineAxis, lossMean, lossShares, orderAxis,
  parseLines, rankText } from '../src/stockfish.js';

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `expected ${b}, got ${a}`);

test('evaluations are signed centipawns, mates count moves from the plies of the engine scale', () => {
  assert.equal(formatEval(35), '+35');
  assert.equal(formatEval(-120), '\u2212120');
  assert.equal(formatEval(0), '0');
  assert.equal(formatEval(-0.4), '0');
  // Stockfish's mate in 3 is stored as 100000 - 5 plies; being mated in 2 as -(100000 - 4)
  assert.equal(formatEval(99995), '#3');
  assert.equal(formatEval(99999), '#1');
  assert.equal(formatEval(-99996), '\u2212#2');
  assert.equal(formatEval(-99998), '\u2212#1');
  assert.equal(formatEval(null), '\u2014');
  assert.equal(formatEval(NaN), '\u2014');
});

test('Lichess links put underscores between the FEN fields', () => {
  assert.equal(lichessUrl('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'),
    'https://lichess.org/analysis/rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR_b_KQkq_e3_0_1');
  assert.equal(lichessUrl(' 8/8/8/8/8/8/8/K6k  w - - 0 60 '), 'https://lichess.org/analysis/8/8/8/8/8/8/8/K6k_w_-_-_0_60');
});

test('depth and rank labels', () => {
  assert.equal(depthLabel(10, 10), 'depth 10');
  assert.equal(depthLabel(10, 12), 'depth 10\u201312');
  assert.equal(depthLabel(null, 12), 'depth 12');
  assert.equal(depthLabel(null, undefined), null);
  assert.equal(rankText(1, 5), 'line 1');
  assert.equal(rankText(0, 5), 'outside the top 5');
  assert.equal(rankText(0, null), 'outside the lines');
  assert.equal(rankText(null, 5), '');
});

test('mean loss and its interval come from the count, sum and sum of squares', () => {
  // losses 0, 10, 20, 30
  const m = lossMean(4, 60, 1400);
  close(m.mean, 15);
  const se = Math.sqrt(500 / 3 / 4);
  close(m.lo, 15 - 1.959963984540054 * se);
  close(m.hi, 15 + 1.959963984540054 * se);
  // losses 0, 0, 0, 40: the normal interval would dip below zero
  const skewed = lossMean(4, 40, 1600);
  assert.equal(skewed.lo, 0);
  assert.ok(skewed.hi > 10);
  const one = lossMean(1, 25, 625);
  close(one.mean, 25);
  assert.ok(Number.isNaN(one.lo) && Number.isNaN(one.hi));
  assert.ok(Number.isNaN(lossMean(0, 0, 0).mean));
});

test('cumulative rank shares count moves outside the lines in the denominator', () => {
  const rows = [
    { engine: 'A', sf_rank: 1, n: 50 }, { engine: 'A', sf_rank: 2, n: 20 }, { engine: 'A', sf_rank: 0, n: 30 },
    { engine: 'B', sf_rank: 3, n: 10 }, { engine: 'B', sf_rank: 1, n: 30 },
    { engine: 'C', sf_rank: 0, n: 0 },
  ];
  const points = cumulativeRanks(rows, 3);
  assert.deepEqual(points.filter(p => p.engine === 'A').map(p => p.share), [0.5, 0.7, 0.7]);
  assert.deepEqual(points.filter(p => p.engine === 'B').map(p => p.share), [0.75, 0.75, 1]);
  assert.equal(points.filter(p => p.engine === 'C').length, 0);
});

test('loss shares cover every bucket in order, and odd edges land in the bucket that holds them', () => {
  assert.deepEqual(LOSS_BUCKETS.map(b => b.label),
    ['0', '1\u20139', '10\u201324', '25\u201349', '50\u201399', '100\u2013199', '200\u2013399', '400+']);
  const rows = [{ engine: 'A', loss_bucket: 0, n: 60 }, { engine: 'A', loss_bucket: 10, n: 25 }, { engine: 'A', loss_bucket: 12, n: 5 },
    { engine: 'A', loss_bucket: 800, n: 10 }];
  const shares = lossShares(rows);
  assert.equal(shares.length, LOSS_BUCKETS.length);
  assert.deepEqual(shares.map(s => s.share), [0.6, 0, 0.3, 0, 0, 0, 0, 0.1]);
  assert.deepEqual(shares.map(s => s.bucket), [0, 1, 2, 3, 4, 5, 6, 7]);
});

test('axes put the leftover category apart from the ordered values', () => {
  const lines = lineAxis(5);
  assert.deepEqual(lines.ticks, [1, 2, 3, 4, 5, 7]);
  assert.equal(lines.x(0), 7);
  assert.equal(lines.x(3), 3);
  assert.deepEqual(lines.ticks.map(lines.format), ['#1', '#2', '#3', '#4', '#5', 'other']);
  assert.equal(lines.format(6), '');
  const order = orderAxis();
  assert.equal(order.x(-1), 12);
  assert.equal(order.x(14), 10);
  assert.deepEqual(order.ticks.map(order.format), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+', 'not searched']);
  assert.equal(order.format(0.5), '');
});

test('Stockfish lines are parsed from rank:move:eval items in rank order', () => {
  assert.deepEqual(parseLines('2:d2d4:30,1:e2e4:35,3:g1f3:,4:c2c4:-99996'), [
    { rank: 1, move: 'e2e4', eval: 35 }, { rank: 2, move: 'd2d4', eval: 30 }, { rank: 3, move: 'g1f3', eval: null },
    { rank: 4, move: 'c2c4', eval: -99996 }]);
  assert.deepEqual(parseLines(null), []);
  assert.deepEqual(parseLines(''), []);
});
