/** Small, dependency-free statistics used by the charts. Every function is pure. */

export const Z95 = 1.959963984540054;
const finite = v => typeof v === 'number' && Number.isFinite(v);

/** Wilson score interval for a proportion p observed on an effective sample size n. */
export function wilson(p, n, z = Z95) {
  if (!finite(p) || !finite(n) || n <= 0) return [NaN, NaN];
  const q = Math.max(0, Math.min(1, p));
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const centre = (q + z2 / (2 * n)) / denom;
  const half = (z * Math.sqrt(q * (1 - q) / n + z2 / (4 * n * n))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

/** Mean +- z standard errors. */
export function meanInterval(mean, sd, n, z = Z95) {
  if (!finite(mean) || !finite(sd) || !finite(n) || n < 2) return [NaN, NaN];
  const se = sd / Math.sqrt(n);
  return [mean - z * se, mean + z * se];
}

/**
 * Standard error of a ratio of sums R = sum(y) / sum(x) over n items (delta method),
 * from sum(y^2), sum(x*y) and sum(x^2). Used for NPS = sum(nodes) / sum(time).
 */
export function ratioSE({ sumY, sumX, sumYY, sumXY, sumXX, n }) {
  if (![sumY, sumX, sumYY, sumXY, sumXX, n].every(finite) || n < 2 || sumX === 0) return NaN;
  const r = sumY / sumX;
  const ss = Math.max(0, sumYY - 2 * r * sumXY + r * r * sumXX);
  return Math.sqrt((ss / (n - 1)) * n) / Math.abs(sumX);
}

/** Difference a - b and relative difference a / b - 1 with z-intervals, assuming independent estimates. */
export function difference(a, seA, b, seB, z = Z95) {
  const diff = a - b;
  const seDiff = Math.sqrt((finite(seA) ? seA * seA : NaN) + (finite(seB) ? seB * seB : NaN));
  const rel = b !== 0 ? a / b - 1 : NaN;
  const seRel = b !== 0 ? Math.sqrt((seA / b) ** 2 + ((a * seB) / (b * b)) ** 2) : NaN;
  const out = { diff, rel, lo: diff - z * seDiff, hi: diff + z * seDiff, relLo: rel - z * seRel, relHi: rel + z * seRel };
  out.significant = finite(out.lo) && finite(out.hi) && (out.lo > 0 || out.hi < 0);
  return out;
}

/** Width of evenly spaced buckets: the smallest gap between distinct bucket values, or `fallback` with fewer than two. */
export function bucketWidth(values, fallback = NaN) {
  const sorted = [...new Set(values.filter(Number.isFinite))].sort((a, b) => a - b);
  let step = Infinity;
  for (let i = 1; i < sorted.length; i++) step = Math.min(step, sorted[i] - sorted[i - 1]);
  return Number.isFinite(step) && step > 0 ? step : fallback;
}

/**
 * Win/draw/loss probabilities and expected score for one eval bucket, from the side to move.
 * Intervals use games (not positions) as the sample size: positions of one game are not independent.
 */
export function wdlBucket({ wins = 0, draws = 0, losses = 0, positions, games }, z = Z95) {
  const n = positions ?? wins + draws + losses;
  if (!(n > 0)) return null;
  const pWin = wins / n, pDraw = draws / n, pLoss = losses / n;
  const score = pWin + pDraw / 2;
  const eff = Math.max(1, Math.min(n, games ?? n));
  const variance = pWin * (1 - score) ** 2 + pDraw * (0.5 - score) ** 2 + pLoss * score ** 2;
  const se = Math.sqrt(variance / eff);
  return {
    positions: n, games: eff, pWin, pDraw, pLoss, score,
    win: wilson(pWin, eff, z), draw: wilson(pDraw, eff, z), loss: wilson(pLoss, eff, z),
    scoreCI: [Math.max(0, score - z * se), Math.min(1, score + z * se)],
  };
}

/**
 * Sum rows that share a key (e.g. the same eval bucket over several phases or core types).
 * Games are summed as well, which over-counts games that span several groups; callers prefer
 * pre-aggregated 'all' rows when they exist.
 */
export function mergeCounts(rows, keyOf, fields = ['positions', 'games', 'wins', 'draws', 'losses']) {
  const out = new Map();
  for (const row of rows) {
    const key = keyOf(row);
    const acc = out.get(key) ?? out.set(key, { ...row, ...Object.fromEntries(fields.map(f => [f, 0])) }).get(key);
    for (const f of fields) acc[f] += Number(row[f] ?? 0);
  }
  return [...out.values()];
}

/**
 * Weighted logistic fit p(x) = 1 / (1 + exp(-(a + b x))) to observed proportions (Newton-Raphson
 * on the binomial likelihood with fractional outcomes). Returns x50 = -a / b, the x where p = 0.5,
 * and its delta-method standard error. x is rescaled internally for stability.
 */
export function fitLogistic(xs, ps, ws, { maxIter = 60 } = {}) {
  const pts = xs.map((x, i) => [x, ps[i], ws[i]]).filter(([x, p, w]) => finite(x) && finite(p) && finite(w) && w > 0);
  if (pts.length < 3) return null;
  const scale = Math.max(1, ...pts.map(([x]) => Math.abs(x)));
  const W = pts.reduce((s, [, , w]) => s + w, 0);
  const pbar = Math.min(0.99, Math.max(0.01, pts.reduce((s, [, p, w]) => s + p * w, 0) / W));
  let a = Math.log(pbar / (1 - pbar)), b = 0, converged = false, h = null;
  for (let it = 0; it < maxIter; it++) {
    let g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
    for (const [x0, p, w] of pts) {
      const x = x0 / scale;
      const mu = 1 / (1 + Math.exp(-(a + b * x)));
      const v = w * mu * (1 - mu);
      g0 += w * (p - mu); g1 += w * (p - mu) * x;
      h00 += v; h01 += v * x; h11 += v * x * x;
    }
    const det = h00 * h11 - h01 * h01;
    if (!(det > 1e-12)) break;
    const da = (h11 * g0 - h01 * g1) / det, db = (h00 * g1 - h01 * g0) / det;
    a += da; b += db;
    h = { h00, h01, h11, det };
    if (Math.abs(da) < 1e-9 && Math.abs(db) < 1e-9) { converged = true; break; }
    if (!finite(a) || !finite(b) || Math.abs(b) > 1e4) break;
  }
  if (!h || !finite(a) || !finite(b) || b === 0) return null;
  const bx = b / scale;
  const x50 = -a / bx;
  // covariance of (a, b_scaled) is the inverse Hessian; transform to x50 by the delta method.
  const va = h.h11 / h.det, vb = h.h00 / h.det, vab = -h.h01 / h.det;
  const x50s = -a / b;
  const varX50s = (va + x50s * x50s * vb + 2 * x50s * vab) / (b * b);
  const se = Math.sqrt(Math.max(0, varX50s)) * scale;
  return { a, b: bx, x50, x50SE: se, scale: 1 / bx, converged, predict: x => 1 / (1 + Math.exp(-(a + bx * x))) };
}

/** Quantile from histogram bins [{lo, hi, count}], interpolating inside a bin (geometrically for log bins). */
export function quantileFromHist(bins, q, { log = false } = {}) {
  const sorted = bins.filter(b => b.count > 0).sort((x, y) => x.lo - y.lo);
  const total = sorted.reduce((s, b) => s + b.count, 0);
  if (!total) return NaN;
  const target = Math.max(0, Math.min(1, q)) * total;
  let cum = 0;
  for (const b of sorted) {
    if (cum + b.count >= target) {
      const f = b.count ? (target - cum) / b.count : 0;
      if (log && b.lo > 0 && b.hi > 0) return b.lo * (b.hi / b.lo) ** f;
      return b.lo + (b.hi - b.lo) * f;
    }
    cum += b.count;
  }
  return sorted[sorted.length - 1].hi;
}

/** True when bin edges grow geometrically (constant hi / lo), i.e. the metric was binned on a log scale. */
export function isLogBinned(bins) {
  const r = bins.filter(b => b.lo > 0 && b.hi > b.lo).map(b => b.hi / b.lo);
  if (r.length < 3) return false;
  const w = bins.filter(b => b.hi > b.lo).map(b => b.hi - b.lo);
  const constRatio = Math.max(...r) / Math.min(...r) < 1.02;
  const constWidth = Math.max(...w) / Math.min(...w) < 1.02;
  return constRatio && !constWidth;
}

/** Cumulative share per bin edge, for CDF lines. */
export function cumulative(bins) {
  const sorted = bins.filter(b => b.count > 0).sort((x, y) => x.lo - y.lo);
  const total = sorted.reduce((s, b) => s + b.count, 0);
  let cum = 0;
  const out = sorted.length ? [{ x: sorted[0].lo, share: 0 }] : [];
  for (const b of sorted) {
    cum += b.count;
    out.push({ x: b.hi, share: cum / total });
  }
  return out;
}

export function scoreToElo(score) {
  if (!finite(score) || score <= 0 || score >= 1) return score >= 1 ? Infinity : score <= 0 ? -Infinity : NaN;
  return -400 * Math.log10(1 / score - 1);
}

/** Points per game and its 95% interval from game counts (trinomial model). */
export function scoreFromWDL(wins, draws, losses, z = Z95) {
  const n = wins + draws + losses;
  if (!(n > 0)) return null;
  const s = (wins + draws / 2) / n;
  const variance = (wins * (1 - s) ** 2 + draws * (0.5 - s) ** 2 + losses * s * s) / n;
  const se = Math.sqrt(variance / n);
  return { score: s, se, lo: Math.max(0, s - z * se), hi: Math.min(1, s + z * se), n };
}

/** Elo and 95% interval from game counts (trinomial model). */
export function eloFromWDL(wins, draws, losses, z = Z95) {
  const r = scoreFromWDL(wins, draws, losses, z);
  if (!r) return null;
  return { score: r.score, elo: scoreToElo(r.score), lo: scoreToElo(r.score - z * r.se), hi: scoreToElo(r.score + z * r.se), n: r.n };
}

/** Elo and 95% interval from pentanomial pair counts [LL, LD, DD+WL, WD, WW]. */
export function eloFromPentanomial(counts, z = Z95) {
  const n = counts.reduce((s, c) => s + c, 0);
  if (!(n > 0)) return null;
  const v = [0, 0.25, 0.5, 0.75, 1];
  const s = counts.reduce((acc, c, i) => acc + c * v[i], 0) / n;
  const variance = counts.reduce((acc, c, i) => acc + c * (v[i] - s) ** 2, 0) / n;
  const se = Math.sqrt(variance / n);
  return { score: s, elo: scoreToElo(s), lo: scoreToElo(s - z * se), hi: scoreToElo(s + z * se), pairs: n };
}

/**
 * Split an NPS difference into a code part and a hardware part.
 * rows: [{core_type, nodes, time_ms, mhz_time}] per engine, where mhz_time = sum(clock MHz * time).
 * The code part evaluates the engine's knps-per-GHz on the baseline's core-type mix and clock;
 * the hardware part is what remains (clock and share of time on each core type).
 */
export function decomposeNps(engineRows, baseRows) {
  const prep = rows => {
    const map = new Map();
    let nodes = 0, time = 0;
    for (const r of rows) {
      if (!(r.time_ms > 0) || !(r.nodes > 0)) continue;
      nodes += r.nodes; time += r.time_ms;
      const mhz = r.mhz_time > 0 ? r.mhz_time / r.time_ms : NaN;
      map.set(r.core_type, { time: r.time_ms, nps: r.nodes / (r.time_ms / 1000), mhz });
    }
    return { map, nps: time > 0 ? nodes / (time / 1000) : NaN, time };
  };
  const e = prep(engineRows), b = prep(baseRows);
  if (!(e.nps > 0) || !(b.nps > 0)) return null;
  const common = [...b.map.keys()].filter(c => e.map.has(c) && finite(e.map.get(c).mhz) && finite(b.map.get(c).mhz));
  const total = e.nps / b.nps;
  if (!common.length) return { total: total - 1, code: NaN, hardware: NaN, common };
  const bt = common.reduce((s, c) => s + b.map.get(c).time, 0);
  let num = 0, den = 0;
  for (const c of common) {
    const w = b.map.get(c).time / bt, fb = b.map.get(c).mhz;
    const ke = e.map.get(c).nps / e.map.get(c).mhz, kb = b.map.get(c).nps / fb;
    num += w * ke * fb; den += w * kb * fb;
  }
  const code = num / den;
  return { total: total - 1, code: code - 1, hardware: total / code - 1, common };
}

/** Error function (Abramowitz and Stegun 7.1.26), for LOS. */
export function erf(x) {
  const s = Math.sign(x), t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    * Math.exp(-x * x);
  return s * y;
}

/** Likelihood of superiority from wins and losses (draws carry no information). */
export function los(wins, losses) {
  if (!(wins + losses > 0)) return NaN;
  return 0.5 * (1 + erf((wins - losses) / Math.sqrt(2 * (wins + losses))));
}
