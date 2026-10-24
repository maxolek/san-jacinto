#!/usr/bin/env python3
"""Estimate an engine's strength on Stockfish's UCI_Elo scale.

Plays opening pairs against Stockfish with UCI_LimitStrength=true and adapts
UCI_Elo as it learns, so most games are played near the level where the
candidate scores 50% (where each game carries the most information). The
answer is the UCI_Elo value at which the candidate is expected to score 50%.

Model: a pair at Stockfish level X scores s in {0, .25, .5, .75, 1} for the
candidate, with E[s] = 1 / (1 + 10^(k (X - R) / 400)). R, the level where the
candidate scores 50%, is the answer; k is the local scale of UCI_Elo (1 means
calibrated; at short time controls it is often well below 1). Both are fitted
by maximum (quasi-)likelihood with weak priors that only matter before the data
identify them; levels follow a three-point design around R (half at R, a
quarter each where the candidate should score ~70% / 30%). The 95% interval
uses the robust (sandwich) covariance of the pair scores, so draws, the
correlation inside color-reversed pairs and an uncertain scale all widen it.
Pairs enter the fit in the order they started, so short decisive games are not
over-represented when the run stops.

Stockfish's UCI_Elo is anchored to CCRL Blitz ratings (roughly +/-100 Elo, per
the Stockfish source); the mapping depends on time control and hardware, so
compare estimates only at the same settings. The default 60+0.6 is a common
blitz setting. Games on P-cores and E-cores are also estimated separately.

Examples:
  python -m tests.strength_estimator --engine tomahawk.exe --stockfish stockfish.exe
  python -m tests.strength_estimator --engine tomahawk.exe --stockfish stockfish.exe --tc 10+0.1 --target-ci 40
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import environment  # noqa: E402
from utils.match_runner import (EngineSpec, JobSpec, MatchRunner, describe_slots, plan_slots,  # noqa: E402
                                print_line, probe_uci, tc_arguments)
from utils.openings import OpeningBook, OpeningRegister  # noqa: E402
from utils.pentanomial import PairTally, Z95  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "logs" / "strength_results"
LN10_400 = math.log(10) / 400


def expected_score(rating, level, slope=1.0):
    return 1 / (1 + 10 ** (slope * (level - rating) / 400))


class StrengthModel:
    """Fractional-logistic fit of pair scores against Stockfish levels.

    E[s] = 1 / (1 + 10^(k (X - R) / 400)). R is the level where the candidate
    scores 50%; k is the local scale of Stockfish's UCI_Elo relative to real
    Elo (k < 1: the scale is compressed, as it is at short time controls).
    k has a log-normal prior centred on 1 (sd 0.5), R a very weak normal prior,
    so both are defined before the data can identify them. The covariance is
    the sandwich estimate, so the interval for R widens when k is small or
    poorly known instead of pretending the scale is calibrated.
    """

    def __init__(self, prior_mean=2000.0, prior_sd=600.0, log_slope_sd=0.5):
        self.prior_mean, self.prior_sd, self.log_slope_sd = prior_mean, prior_sd, log_slope_sd
        self.observations = []  # (level, pair mean score)
        self._cache = None

    def add(self, level, score):
        self.observations.append((float(level), float(score)))
        self._cache = None

    def _terms(self, rating, theta, with_priors=True):
        k = math.exp(theta)
        if with_priors:
            grad = [-(rating - self.prior_mean) / self.prior_sd ** 2, -theta / self.log_slope_sd ** 2]
            info = [[1 / self.prior_sd ** 2, 0.0], [0.0, 1 / self.log_slope_sd ** 2]]
        else:
            grad = [0.0, 0.0]
            info = [[0.0, 0.0], [0.0, 0.0]]
        meat = [[0.0, 0.0], [0.0, 0.0]]
        for level, score in self.observations:
            z = LN10_400 * k * (rating - level)
            p = 1 / (1 + math.exp(-max(-60.0, min(60.0, z))))
            d_rating = LN10_400 * k          # dz/dR
            d_theta = z                      # dz/dtheta
            residual = score - p
            g = (residual * d_rating, residual * d_theta)
            w = p * (1 - p)
            grad[0] += g[0]
            grad[1] += g[1]
            info[0][0] += w * d_rating ** 2
            info[0][1] += w * d_rating * d_theta
            info[1][1] += w * d_theta ** 2
            for i in range(2):
                for j in range(2):
                    meat[i][j] += g[i] * g[j]
        info[1][0] = info[0][1]
        return grad, info, meat

    @staticmethod
    def _inverse(m):
        det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
        return [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]

    def fit_full(self):
        """(R, standard error of R, slope k, standard error of log k)."""
        if self._cache is not None:
            return self._cache
        rating, theta = self.prior_mean, 0.0
        for _ in range(200):
            grad, info, _ = self._terms(rating, theta)
            inv = self._inverse(info)
            step_r = inv[0][0] * grad[0] + inv[0][1] * grad[1]
            step_t = inv[1][0] * grad[0] + inv[1][1] * grad[1]
            rating += max(-300.0, min(300.0, step_r))
            theta += max(-0.5, min(0.5, step_t))
            if abs(step_r) < 1e-5 and abs(step_t) < 1e-7:
                break
        if not self.observations:
            self._cache = (rating, None, math.exp(theta), None)
            return self._cache
        # Priors regularise the estimate, but the error bars come from the data alone
        # (sandwich); only if the data cannot identify both parameters yet (e.g. all
        # pairs at one level) is the prior information used to keep the matrix invertible.
        _, info, meat = self._terms(rating, theta, with_priors=False)
        det = info[0][0] * info[1][1] - info[0][1] * info[1][0]
        if det <= 1e-12 * max(1e-300, info[0][0] * info[1][1]):
            _, info, meat = self._terms(rating, theta)
        inv = self._inverse(info)
        n = len(self.observations)
        correction = n / (n - 1) if n > 1 else 1.0
        cov = [[sum(inv[i][a] * meat[a][b] * inv[b][j] for a in range(2) for b in range(2)) * correction
                for j in range(2)] for i in range(2)]
        self._cache = (rating, math.sqrt(max(cov[0][0], 0.0)), math.exp(theta), math.sqrt(max(cov[1][1], 0.0)))
        return self._cache

    def fit(self):
        """(estimate, standard error) of R; standard error None before any data."""
        rating, se, _, _ = self.fit_full()
        return rating, se

    @property
    def slope(self):
        return self.fit_full()[2]

    @property
    def pairs(self):
        return len(self.observations)

    def mixed(self):
        """True once both a candidate win and a loss (pair score above and below .5) were seen."""
        scores = [s for _, s in self.observations]
        return any(s > 0.5 for s in scores) and any(s < 0.5 for s in scores)


class EstimatorController:
    def __init__(self, args, candidate, stockfish, book, slots, run_dir, level_range):
        self.args, self.candidate, self.stockfish = args, candidate, stockfish
        self.book, self.slots, self.run_dir = book, slots, Path(run_dir)
        self.min_level, self.max_level = level_range
        self.model = StrengthModel(prior_mean=args.start_elo)
        self.by_core = {}      # core type -> StrengthModel
        self.by_level = {}     # level -> PairTally
        self.register = OpeningRegister()
        self.rng = random.Random(args.seed)
        self.max_pairs = args.max_games // 2
        self.planned = 0
        self.next_global = 0
        self.global_of = {}
        self.pairs = {}        # global index -> dict(level, core, opening, games)
        self.fed = 0
        self.voided = set()
        self.started = time.time()
        self.last_status = 0.0
        self.dirty = False
        self.decision = None
        self.failures = 0
        self.games_file = open(self.run_dir / "games.jsonl", "w", encoding="utf-8")
        self.games = 0

    # -- design ---------------------------------------------------------------

    def estimate(self):
        rating, se = self.model.fit()
        return rating, se

    def choose_level(self):
        """Three-point design around the estimate: half the pairs at R, a quarter each at
        R -/+ D where the candidate is expected to score about 70% / 30%. The outer points
        identify the scale slope k at a small cost in precision for R."""
        rating, se = self.estimate()
        slope = self.model.slope
        if not self.model.mixed():
            # Until both wins and losses are seen the fit is only the prior pulled by one side:
            # move in steps, like a bisection that has not found its bracket yet.
            scores = [s for _, s in self.model.observations]
            if scores and all(s >= 0.5 for s in scores):
                rating = max(lvl for lvl, _ in self.model.observations) + 150
            elif scores:
                rating = min(lvl for lvl, _ in self.model.observations) - 150
            slope = 1.0
        offset = min(400.0, 147.0 / max(slope, 0.2))  # logit(0.7) / (ln10/400 * k)
        u = self.rng.random()
        level = rating + (-offset if u < 0.25 else offset if u > 0.75 else 0.0)
        level += self.rng.uniform(-self.args.spread, self.args.spread)
        return int(max(self.min_level, min(self.max_level, round(level / 10) * 10)))

    def next_job(self, slot):
        if self.decision or self.planned >= self.max_pairs:
            return None
        if self.failures >= max(3, len(self.slots)):
            return None
        size = max(1, min(self.args.pairs_per_job, self.max_pairs - self.planned))
        self.planned += size
        level = self.choose_level()
        opponent = EngineSpec(self.stockfish.name, self.stockfish.cmd, self.stockfish.args,
                              {**self.stockfish.options, "UCI_LimitStrength": "true", "UCI_Elo": str(level)},
                              workdir=self.stockfish.workdir, pin_option=self.stockfish.pin_option)
        return JobSpec((self.candidate, opponent), self.book.take(size), self.book, meta={"level": level})

    # -- results --------------------------------------------------------------

    def on_pair_start(self, job, pair_no):
        index = self.next_global
        self.next_global += 1
        self.global_of[(job.id, pair_no)] = index
        self.pairs[index] = dict(level=job.spec.meta["level"], core=job.slot.core_type,
                                 opening=job.spec.openings[pair_no - 1], games={})
        return True

    def on_game(self, game):
        index = self.global_of.get((game.job.id, game.pair_no))
        if index is None:
            return
        score = game.score_for(self.candidate.name)
        pair = self.pairs[index]
        record = dict(pair=index + 1, job=game.job.id, job_game=game.game_no, level=pair["level"],
                      white=game.white, black=game.black, result=game.result, reason=game.reason,
                      candidate_score=score, seconds=round(game.seconds, 2),
                      finished_utc=datetime.fromtimestamp(game.finished, timezone.utc).isoformat(), **game.env)
        self.games_file.write(json.dumps(record) + "\n")
        self.games_file.flush()
        self.games += 1
        if not self.args.quiet and not self.decision:
            env = game.env
            ghz = f"{env['mhz_mean'] / 1000:.1f}GHz" if env.get("mhz_mean") else "  -   "
            print_line(f"  SF {pair['level']:4d}  {env['core_type']} cpu{env['cpu'] if env['cpu'] is not None else '-':<3} "
                       f"{ghz}  {game.white[:9]:>9s}-{game.black[:9]:<9s} {game.result:<7s} "
                       f"{game.reason[:32]:<32s} {game.seconds:6.1f}s")
        if score is None:
            return
        if game.env.get("core_mixed"):
            self.voided.add(index)  # never count a game that may have mixed core types
            self._consume()
            return
        pair["games"][game.game_no % 2 or 2] = score
        self._consume()

    def _consume(self):
        """Feed completed pairs to the model in start order."""
        while self.fed < self.next_global:
            if self.fed in self.voided:
                self.fed += 1
                continue
            pair = self.pairs[self.fed]
            if len(pair["games"]) < 2:
                break
            s1, s2 = pair["games"][1], pair["games"][2]
            self.model.add(pair["level"], (s1 + s2) / 2)
            self.by_core.setdefault(pair["core"], StrengthModel(self.args.start_elo)).add(pair["level"], (s1 + s2) / 2)
            self.by_level.setdefault(pair["level"], PairTally()).add_pair(s1, s2)
            self.register.record_pair(pair["opening"], (s1, s2), pair["core"])
            self.fed += 1
            self.dirty = True
        self._check_stop()

    def _check_stop(self):
        if self.decision or self.model.pairs < self.args.min_games // 2 or not self.model.mixed():
            return
        rating, se = self.estimate()
        half = Z95 * se
        if half <= self.args.target_ci:
            self.decision = "precision reached"
        elif rating + 2 * se < self.min_level:
            self.decision = f"below Stockfish's range ({self.min_level})"
        elif rating - 2 * se > self.max_level:
            self.decision = f"above Stockfish's range ({self.max_level})"
        if self.decision:
            self.print_status(force=True)

    def on_job_end(self, job):
        unstarted = job.pairs - job.pairs_started
        voided = 0
        for pair_no in range(1, job.pairs_started + 1):
            index = self.global_of.get((job.id, pair_no))
            if index is not None and len(self.pairs[index]["games"]) < 2:
                self.voided.add(index)
                voided += 1
        self.planned -= unstarted + voided
        self._consume()
        if job.killed:
            return
        if job.returncode or not job.games:
            self.failures += 1
            tail = job.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-6:]
            print_line(f"[STRENGTH] job {job.id} exited with code {job.returncode}:\n    " + "\n    ".join(tail))
            if self.failures >= max(3, len(self.slots)) and self.decision is None:
                self.decision = "stopped: jobs keep failing (see the job logs)"
                print_line(f"[STRENGTH] {self.decision}")
        else:
            self.failures = 0

    def on_warning(self, job, text):
        print_line(f"[cutechess] {text}")

    def finished(self):
        if self.decision is None and len(self.voided) > max(20, self.max_pairs // 20):
            self.decision = f"stopped: {len(self.voided)} pairs had to be excluded (see warnings / job logs)"
            print_line(f"[STRENGTH] {self.decision}")
        return self.decision is not None

    def tick(self):
        self.print_status()

    # -- reporting ------------------------------------------------------------

    def print_status(self, force=False):
        now = time.monotonic()
        if not force and (not self.dirty or now - self.last_status < self.args.status_every):
            return
        self.last_status, self.dirty = now, False
        if not self.model.pairs:
            return
        rating, se = self.estimate()
        cores = "  ".join(f"{core}: {m.fit()[0]:.0f}({m.pairs})" for core, m in sorted(self.by_core.items()))
        precision = f"+/-{Z95 * se:.0f}" if self.model.mixed() else "(bracketing)"
        print_line(f"[STRENGTH] {self.model.pairs} pairs | UCI_Elo ~ {rating:.0f} {precision} "
                   f"(target +/-{self.args.target_ci:g}) | scale {self.model.slope:.2f} | {cores}")

    def summary(self):
        rating, se, slope, slope_se = self.model.fit_full()
        out = dict(estimate=round(rating, 1), stderr=round(se, 2) if se else None,
                   ci_low=round(rating - Z95 * se, 1) if se else None,
                   ci_high=round(rating + Z95 * se, 1) if se else None,
                   scale=round(slope, 3), scale_ci=[round(slope * math.exp(-Z95 * slope_se), 3),
                                                    round(slope * math.exp(Z95 * slope_se), 3)] if slope_se else None,
                   pairs=self.model.pairs, games=2 * self.model.pairs, decision=self.decision,
                   by_core={}, by_level=[])
        for core, model in sorted(self.by_core.items()):
            r, s = model.fit()
            out["by_core"][core] = dict(estimate=round(r, 1), ci=round(Z95 * s, 1) if s else None, pairs=model.pairs)
        for level, tally in sorted(self.by_level.items()):
            t = tally.summary()
            implied = level + t["raw_elo"] if t["raw_elo"] is not None else None
            out["by_level"].append(dict(level=level, pairs=t["pairs"], wins=t["wins"], draws=t["draws"],
                                        losses=t["losses"], score=t["raw_score"], implied=implied,
                                        expected=expected_score(rating, level, slope)))
        return out

    def report(self, summary, elapsed):
        s = summary
        lines = ["", "=" * 72, f"STRENGTH ESTIMATE   ({s['games']} games, {elapsed / 60:.1f} min)", "=" * 72]
        if s["stderr"]:
            lines.append(f"  Stockfish UCI_Elo equivalent: {s['estimate']:.0f}   "
                         f"95% CI [{s['ci_low']:.0f}, {s['ci_high']:.0f}]   ({s['decision'] or 'stopped'})")
            if s["scale_ci"]:
                lines.append(f"  UCI_Elo scale here: 1 UCI_Elo point = {s['scale']:.2f} Elo "
                             f"(95% CI {s['scale_ci'][0]:.2f}-{s['scale_ci'][1]:.2f}; 1.00 = calibrated)")
        else:
            lines.append("  Not enough decisive results for an estimate yet.")
        for core, e in s["by_core"].items():
            ci = f" +/-{e['ci']:.0f}" if e["ci"] else ""
            lines.append(f"    on {core}-cores: {e['estimate']:.0f}{ci}  ({e['pairs']} pairs)")
        lines += ["", f"    {'SF level':>8s} {'pairs':>6s} {'+W =D -L':>12s} {'score':>7s} {'model':>7s} {'implied':>8s}"]
        for row in s["by_level"]:
            implied = f"{row['implied']:.0f}" if row["implied"] is not None else "-"
            lines.append(f"    {row['level']:>8d} {row['pairs']:>6d} {'+%d =%d -%d' % (row['wins'], row['draws'], row['losses']):>12s} "
                         f"{row['score']:>7.1%} {row['expected']:>7.1%} {implied:>8s}")
        lines += ["", "  Note: UCI_Elo is anchored to CCRL Blitz (about +/-100) at longer time controls;",
                  "  at short ones the scale is compressed (scale < 1). Compare estimates only at equal TC/hardware."]
        if self.register.rows:
            lines += ["", "  Openings by family:", self.register.format_table(limit=8, level="family")]
        print_line("\n".join(lines))

    def close(self):
        self.games_file.close()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", required=True, help="Engine to rate")
    p.add_argument("--engine-option", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--stockfish", default=os.environ.get("STOCKFISH"), help="Stockfish executable ($STOCKFISH)")
    p.add_argument("--stockfish-option", action="append", default=[], metavar="NAME=VALUE",
                   help="Extra Stockfish options (Threads=1 and Hash=16 by default)")
    p.add_argument("--cutechess-cli", default=None)
    tc = p.add_argument_group("time control (choose one; default tc 60+0.6)")
    tc.add_argument("--tc", default=None)
    tc.add_argument("--time", type=float, default=None, help="Seconds per move")
    tc.add_argument("--depth", type=int, default=None)
    e = p.add_argument_group("estimation")
    e.add_argument("--start-elo", type=float, default=2000, help="First Stockfish level / prior mean")
    e.add_argument("--target-ci", type=float, default=30, help="Stop when the 95%% CI half-width is below this")
    e.add_argument("--min-games", type=int, default=100)
    e.add_argument("--max-games", type=int, default=1200)
    e.add_argument("--spread", type=float, default=50, help="Random offset of each job's level around the estimate")
    e.add_argument("--seed", type=int, default=None)
    c = p.add_argument_group("CPU")
    c.add_argument("--concurrency", type=int, default=None,
                   help="Cores to use = concurrent games (default: every physical core but one)")
    b = p.add_argument_group("openings / run")
    b.add_argument("--book", default=PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn")
    b.add_argument("--book-depth", type=int, default=16)
    b.add_argument("--pairs-per-job", type=int, default=2)
    b.add_argument("--maxmoves", type=int, default=150)
    b.add_argument("--status-every", type=float, default=30)
    b.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.stockfish:
        raise SystemExit("Pass --stockfish or set STOCKFISH")
    if args.tc is None and args.time is None and args.depth is None:
        args.tc = "60+0.6"
    tc_args = tc_arguments(depth=args.depth, movetime=args.time, tc=args.tc)

    def options(items):
        out = {}
        for item in items:
            name, sep, value = item.partition("=")
            if not sep:
                raise SystemExit(f"expected NAME=VALUE, got {item!r}")
            out[name] = value
        return out

    candidate = EngineSpec.from_path("Candidate", args.engine, options(args.engine_option)).resolve()
    stockfish = EngineSpec.from_path("Stockfish", args.stockfish,
                                     {"Threads": "1", "Hash": "16", **options(args.stockfish_option)}).resolve()
    info = probe_uci(stockfish.cmd, stockfish.args)
    elo_line = info["options"].get("UCI_Elo", "")
    if "UCI_LimitStrength" not in info["options"] or not elo_line:
        raise SystemExit(f"{args.stockfish} has no UCI_LimitStrength / UCI_Elo options")
    lo = int(elo_line.split(" min ")[1].split()[0]) if " min " in elo_line else 1320
    hi = int(elo_line.split(" max ")[1].split()[0]) if " max " in elo_line else 3190
    args.start_elo = min(max(args.start_elo, lo), hi)

    slots = plan_slots(concurrency=args.concurrency)
    book = OpeningBook(args.book, plies=args.book_depth, seed=args.seed)
    run_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    machine = environment.machine_info()
    meta = dict(engine=args.engine, stockfish=args.stockfish, stockfish_name=info["name"],
                uci_elo_range=[lo, hi], time_control=args.tc or (f"{args.time}s/move" if args.time else f"depth {args.depth}"),
                machine=machine, slots=[vars(s) | {"label": s.label} for s in slots],
                book=str(args.book), book_seed=book.seed)
    (run_dir / "config.json").write_text(json.dumps(dict(arguments=vars(args), meta=meta), indent=2, default=str),
                                         encoding="utf-8")
    print_line(f"[STRENGTH] {args.engine} vs {info['name']} (UCI_Elo {lo}-{hi}), {meta['time_control']}")
    print_line(f"[STRENGTH] machine {machine['cpu_model']}; {describe_slots(slots)}")
    print_line(f"[STRENGTH] target 95% CI +/-{args.target_ci:g} Elo, {args.min_games}-{args.max_games} games; "
               f"results {run_dir}")

    controller = EstimatorController(args, candidate, stockfish, book, slots, run_dir, (lo, hi))
    runner = MatchRunner(args.cutechess_cli, slots, run_dir, tc_args, plies=args.book_depth,
                         extra_args=["-maxmoves", str(args.maxmoves)])
    started = time.time()
    try:
        runner.run(controller)
    except KeyboardInterrupt:
        print_line("\n[STRENGTH] interrupted; estimate from the games so far")
    finally:
        controller.close()
        summary = controller.summary()
        summary["meta"] = meta
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        controller.register.save(run_dir, run_meta=dict(run=run_dir.name, candidate=args.engine,
                                                        baseline=info["name"], time_control=meta["time_control"]))
        controller.report(summary, time.time() - started)
    return summary


if __name__ == "__main__":
    main()
