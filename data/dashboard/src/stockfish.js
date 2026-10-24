/** Pure helpers for comparing the engine's moves with Stockfish's lines (Root moves tab). No DOM, no database. */
import { meanInterval } from './stats.js';

const MINUS = '\u2212';
// Engine-scale evaluations: a mate is +/-(MATE - plies to mate), and |eval| >= MATE_BOUND marks one.
const MATE = 100000;
const MATE_BOUND = 99000;
const finite = v => typeof v === 'number' && Number.isFinite(v);

/** Loss buckets of agg_sf_moves by their lower edge in centipawns, in order. */
export const LOSS_BUCKETS = [0, 1, 10, 25, 50, 100, 200, 400].map((lo, i, all) => ({
  lo, label: i === all.length - 1 ? `${lo}+` : lo === 0 ? '0' : `${lo}\u2013${all[i + 1] - 1}`,
}));

/** Evaluation from the side to move: signed centipawns, or #n / −#n for mate in n moves (n = ceil(plies / 2)). */
export function formatEval(value) {
  if (!finite(value)) return '\u2014';
  if (Math.abs(value) >= MATE_BOUND) return `${value < 0 ? MINUS : ''}#${Math.ceil((MATE - Math.abs(value)) / 2)}`;
  const cp = Math.round(value);
  return cp > 0 ? `+${cp}` : cp < 0 ? `${MINUS}${-cp}` : '0';
}

/** The position on the Lichess analysis board. */
export const lichessUrl = fen => `https://lichess.org/analysis/${encodeURI(String(fen).trim().split(/\s+/).join('_'))}`;

/** "depth 10", or "depth 10–12" when the analyses used several depths; null without a depth. */
export function depthLabel(min, max) {
  const values = [min, max].filter(finite);
  if (!values.length) return null;
  const lo = Math.min(...values), hi = Math.max(...values);
  return lo === hi ? `depth ${lo}` : `depth ${lo}\u2013${hi}`;
}

/** Where the engine's move stood among Stockfish's lines: "line 2", "outside the top 5", or '' when unknown. */
export function rankText(rank, lines) {
  if (!finite(rank) || rank < 0) return '';
  if (rank >= 1) return `line ${rank}`;
  return finite(lines) && lines > 0 ? `outside the top ${lines}` : 'outside the lines';
}

/**
 * Mean loss with a 95% interval from a count, a sum and a sum of squares. Losses are never negative,
 * so the interval stops at 0; with fewer than two positions there is no interval.
 */
export function lossMean(n, sum, sumSq) {
  if (!finite(n) || n <= 0 || !finite(sum)) return { mean: NaN, lo: NaN, hi: NaN };
  const mean = sum / n;
  const sd = finite(sumSq) && n > 1 ? Math.sqrt(Math.max(0, (sumSq - n * mean * mean) / (n - 1))) : NaN;
  const [lo, hi] = meanInterval(mean, sd, n);
  return { mean, lo: Math.max(0, lo), hi };
}

function byEngine(rows) {
  const groups = new Map();
  for (const r of rows) {
    if (!groups.has(r.engine)) groups.set(r.engine, []);
    groups.get(r.engine).push(r);
  }
  return groups;
}
const total = rows => rows.reduce((s, r) => s + (finite(r.n) ? r.n : 0), 0);

/**
 * Share of positions whose move was within Stockfish's first k lines, for k = 1..lines, per engine.
 * rows: { engine, sf_rank, n } where sf_rank 0 is a move outside the lines; every row counts in the denominator.
 */
export function cumulativeRanks(rows, lines) {
  const out = [];
  for (const [engine, list] of byEngine(rows)) {
    const all = total(list);
    if (!(all > 0)) continue;
    let within = 0;
    for (let k = 1; k <= lines; k++) {
      within += total(list.filter(r => Number(r.sf_rank) === k));
      out.push({ engine, k, share: within / all, within, total: all });
    }
  }
  return out;
}

/**
 * Share of positions per loss bucket, per engine, with every bucket present (0 when never reached).
 * rows: { engine, loss_bucket, n }; an unexpected edge counts in the bucket that contains it.
 */
export function lossShares(rows) {
  const out = [];
  for (const [engine, list] of byEngine(rows)) {
    const all = total(list);
    if (!(all > 0)) continue;
    const counts = LOSS_BUCKETS.map(() => 0);
    for (const r of list) {
      const edge = Number(r.loss_bucket);
      let i = -1;
      LOSS_BUCKETS.forEach((b, j) => { if (b.lo <= edge) i = j; });
      if (i >= 0 && finite(r.n)) counts[i] += r.n;
    }
    LOSS_BUCKETS.forEach((b, i) => out.push({ engine, bucket: i, label: b.label, n: counts[i], share: counts[i] / all }));
  }
  return out;
}

/** x positions for Stockfish's lines 1..lines, with every other root move two steps further right. */
export function lineAxis(lines) {
  const other = lines + 2;
  return {
    ticks: [...Array.from({ length: lines }, (_, i) => i + 1), other],
    domain: [0.5, other + 0.5],
    x: rank => (rank >= 1 ? rank : other),
    format: v => (v === other ? 'other' : Number.isInteger(v) && v >= 1 && v <= lines ? `#${v}` : ''),
  };
}

/** x positions for the root ordering: 0 to cap (cap and later counted together) and 'not searched' two steps further right. */
export function orderAxis(cap = 10) {
  const none = cap + 2;
  return {
    ticks: [...Array.from({ length: cap + 1 }, (_, i) => i), none],
    domain: [-0.5, none + 0.5],
    x: index => (index < 0 ? none : Math.min(index, cap)),
    format: v => (v === none ? 'not searched' : v === cap ? `${cap}+` : Number.isInteger(v) && v >= 0 && v < cap ? String(v) : ''),
  };
}

/** Stockfish's lines from "rank:move:eval" items joined by commas, in rank order. */
export function parseLines(text) {
  if (!text) return [];
  return String(text).split(',').map(item => {
    const [rank, move, value] = item.split(':');
    return { rank: Number(rank), move, eval: value ? Number(value) : null };
  }).filter(l => l.move && finite(l.rank)).sort((a, b) => a.rank - b.rank);
}
