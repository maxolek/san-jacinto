#!/usr/bin/env python3
"""SPRT between two engines with fastchess, one game per CPU core, pentanomial statistics.

Every concurrent game runs on its own physical core (see utils/match_runner.py):
the fastchess process of each game is started on that core's CPU and engines
with a UCI `Affinity` option are told the same CPU, so both engines of a game
alternate on one core. A game is therefore always P-core vs P-core or E-core vs
E-core, and the whole machine can be used. On Windows each fastchess process and
its engines also run in a job object whose affinity limit is the slot, so an
engine cannot leave it (old Tomahawk builds pin themselves to the P-cores at
startup); on Linux such an engine's games are found by the runner's guard and
excluded. Every game reports its core type,
CPU and measured clock. --concurrency is the number of games at once (default:
every physical core but one); --threads N gives each game N CPUs of one core
type and both engines option.Threads=N. CPUs another run holds (utils/cpu_slots
lock files) are left alone; --no-pin leaves placement to the OS.

Per-move limit: exactly one of --tc, --time, --depth or --nodes. --nodes needs an
engine that honours `go nodes` (checked before the run). --tb adds Syzygy
adjudication (WDL files are enough). fastchess reports the --maxmoves limit as
"Draw by adjudication".

All statistics use opening pairs (pentanomial model): LLR, Elo and normalized
Elo with 95% intervals, LOS, and the same per core type and per opening.
Pairs are tested in the order they started, never the order they finished.

Results go to <results-dir>/<run>/ (default logs/sprt_results). The raw database
gets the SPRT row, every played game (games, source 'match', with its core type,
CPU and clock), the pairwise result (match_results, source 'sprt'), and with
--log the engines' telemetry: each side from its own log folder
<logroot>/<run>/<side>/, attributed to its own build and linked to the played
games by position. Telemetry that fails to load is kept for
`--retry-ingest <logroot>/<run>`.

Examples:
  python -m tests.sprt --engine-a new.exe --engine-b base.exe --tc 10+0.1 --elo0 0 --elo1 5
  python -m tests.sprt --engine-a new.exe --engine-b base.exe --tc 10+0.1 --concurrency 10
  python -m tests.sprt --engine-a new.exe --engine-b base.exe --nodes 20000 --concurrency 4 --no-db
  python -m tests.sprt --retry-ingest logs/sprt_logs/20261025_101129_354256
"""
import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import environment, uci_probe  # noqa: E402
from utils.fastchess_match import FASTCHESS, limit_label, tb_args  # noqa: E402,F401  (FASTCHESS: tests/release.py)
from utils.fixed_match import require_node_limit  # noqa: E402
from utils.match_runner import (EngineSpec, JobSpec, MatchRunner, describe_slots, find_fastchess,  # noqa: E402
                                plan_slots, print_line, probe_uci, release_slots, tc_arguments,
                                termination_class)
from utils.openings import OpeningBook, OpeningRegister, classify, display_name  # noqa: E402
from utils.pentanomial import FINISHED_GAME, LABELS, PairTally, PentanomialSPRT  # noqa: E402

LOGS_DIR = PROJECT_ROOT / "logs"
SPRT_LOG_DIR = LOGS_DIR / "sprt_logs"
RESULTS_DIR = LOGS_DIR / "sprt_results"
OPENING_HISTORY = LOGS_DIR / "opening_register.csv"
# (log folder of the side, its engine name in the run)
SIDES = (("candidate", "Candidate"), ("baseline", "Baseline"))


def parse_fastchess_output(output, candidate_name="Candidate", elo0=0, elo1=10,
                           alpha=0.05, beta=0.05):
    """Rebuild pentanomial evidence from a single fastchess transcript (game numbers in order)."""
    state = PentanomialSPRT(elo0, elo1, alpha, beta)
    for line in output.splitlines():
        match = FINISHED_GAME.search(line)
        if match:
            number, white, black, result = match.groups()
            white = "Candidate" if white == candidate_name else white
            black = "Candidate" if black == candidate_name else black
            state.add_game(int(number), white, black, result)
    return state.snapshot()


# ----------------------------------------------------------------------------
# Presentation helpers
# ----------------------------------------------------------------------------

def fmt_elo(value, low=None, high=None):
    if value is None:
        return "   n/a"
    if low is None:
        return f"{value:+6.1f}"
    return f"{value:+6.1f} +/-{(high - low) / 2:4.1f}"


def fmt_ghz(mhz):
    return f"{mhz / 1000:.1f}GHz" if mhz else "  -   "


class SprtController:
    """Schedules opening pairs on the slots and turns finished games into SPRT evidence."""

    def __init__(self, args, engines, book, slots, run_dir, plotter=None):
        self.args, self.engines, self.book, self.slots = args, engines, book, slots
        self.run_dir = Path(run_dir)
        self.plotter = plotter
        self.state = PentanomialSPRT(args.elo0, args.elo1, args.alpha, args.beta)
        self.max_pairs = args.max_games // 2
        # Pairs excluded (unfinished, or an engine off its core) are replayed, but only up to a
        # limit: if many are excluded something is systematically wrong and the run must stop.
        self.max_excluded = max(20, self.max_pairs // 20)
        self.stop_reason = None
        self.planned = 0
        self.next_global = 0
        self.global_of = {}         # (job id, pair no) -> global pair index (start order)
        self.pairs = {}             # global index -> dict(job, pair_no, opening, games={1: score, 2: score})
        self.fed = 0                # pairs already added to the breakdowns (same order as the SPRT)
        self.tainted = set()        # global indexes of pairs with a game that mixed core types
        self.by_core = {}           # core type -> PairTally
        self.core_time = {}         # core type -> [seconds, games, mhz sum, mhz count]
        self.register = OpeningRegister()
        self.terminations = {}      # (class, engine at fault or None) -> count
        self.failures = 0
        self.started = time.time()
        self.last_status = 0.0
        self.status_dirty = False
        self.games_file = open(self.run_dir / "games.jsonl", "w", encoding="utf-8")
        self.stats_file = open(self.run_dir / "stats.jsonl", "w", encoding="utf-8")
        self.finished_games = 0

    # -- scheduling -----------------------------------------------------------

    def next_job(self, slot):
        if self.state.result != "inconclusive" or self.planned >= self.max_pairs or self.stop_reason:
            return None
        remaining = self.max_pairs - self.planned
        size = max(1, min(self.args.pairs_per_job, math.ceil(remaining / len(self.slots))))
        self.planned += size
        return JobSpec(self.engines, self.book.take(size), self.book)

    def on_pair_start(self, job, pair_no):
        index = self.next_global
        self.next_global += 1
        self.global_of[(job.id, pair_no)] = index
        self.pairs[index] = dict(job=job, pair_no=pair_no, opening=job.spec.openings[pair_no - 1], games={})
        return True

    def excluded(self):
        return len(self.state.voided)

    def finished(self):
        if self.stop_reason is None and self.excluded() > self.max_excluded:
            self.stop_reason = (f"{self.excluded()} pairs had to be excluded (limit {self.max_excluded}); "
                                "see the warnings above and the job logs")
            print_line(f"[SPRT] stopping: {self.stop_reason}")
        return self.state.result != "inconclusive" or self.stop_reason is not None

    # -- results --------------------------------------------------------------

    def on_game(self, game):
        index = self.global_of.get((game.job.id, game.pair_no))
        if index is None:
            return
        number = 2 * index + (1 if game.game_no % 2 else 2)
        score = game.score_for("Candidate")
        pair = self.pairs[index]
        details = classify(pair["opening"])
        env = game.env
        term, fault_side = termination_class(game.reason)
        fault = None
        if fault_side:
            fault = game.white if fault_side == "white" else game.black
        self.terminations[(term, fault)] = self.terminations.get((term, fault), 0) + 1
        record = dict(run=self.run_dir.name, game=number, pair=index + 1, job=game.job.id, job_game=game.game_no,
                      white=game.white, black=game.black, result=game.result, reason=game.reason,
                      termination=term, candidate_score=score, seconds=round(game.seconds, 2),
                      started_utc=datetime.fromtimestamp(game.started, timezone.utc).isoformat(),
                      finished_utc=datetime.fromtimestamp(game.finished, timezone.utc).isoformat(),
                      eco=details[0], opening=details[1], variation=details[2], **env)
        self.games_file.write(json.dumps(record) + "\n")
        self.games_file.flush()
        self.finished_games += 1
        if not self.args.quiet and self.state.result == "inconclusive":
            opening = display_name(details)
            print_line(f"  #{number:<6d} {env['core_type']} cpu{env['cpu'] if env['cpu'] is not None else '-':<3} "
                       f"{fmt_ghz(env.get('mhz_mean'))}  {game.white[:9]:>9s}-{game.black[:9]:<9s} "
                       f"{game.result:<7s} {term:<21s} {game.seconds:6.1f}s  {opening[:44]}")
        if score is None:
            return
        if env.get("core_mixed"):
            # Never count a game that may have been P-core against E-core.
            if index not in self.tainted:
                self.tainted.add(index)
                self.state.void_pair(index)
                self._feed_breakdowns()
                print_line(f"[SPRT] pair {index + 1} excluded: an engine left its {env['core_type']}-core "
                           f"(cpu {env['cpu']}) during game {number}")
            return
        pair["games"][game.game_no % 2 or 2] = (score, game.seconds, env.get("mhz_mean"), env["core_type"])
        updates = self.state.add_game(number, game.white, game.black, game.result)
        self._feed_breakdowns()
        for snapshot in updates:
            self.stats_file.write(json.dumps(self._enrich(snapshot), allow_nan=False) + "\n")
            if self.plotter:
                self.plotter.update(snapshot)
        if updates:
            self.stats_file.flush()
            self.status_dirty = True
            if self.state.result != "inconclusive":
                self.print_status(force=True)

    def _feed_breakdowns(self):
        """Per-core and per-opening tallies see exactly the pairs the SPRT has tested."""
        while self.fed < self.state.next_pair:
            pair = self.pairs.get(self.fed)
            self.fed += 1
            if not pair or len(pair["games"]) < 2 or (self.fed - 1) in self.tainted:
                continue  # voided
            (s1, t1, m1, core), (s2, t2, m2, _) = pair["games"][1], pair["games"][2]
            self.by_core.setdefault(core, PairTally()).add_pair(s1, s2)
            timing = self.core_time.setdefault(core, [0.0, 0, 0.0, 0])
            timing[0] += t1 + t2
            timing[1] += 2
            for mhz in (m1, m2):
                if mhz:
                    timing[2] += mhz
                    timing[3] += 1
            self.register.record_pair(pair["opening"], (s1, s2), core)

    def on_job_end(self, job):
        unstarted = job.pairs - job.pairs_started
        voided = 0
        for pair_no in range(1, job.pairs_started + 1):
            index = self.global_of.get((job.id, pair_no))
            if index is None:
                continue
            if index in self.tainted:
                voided += 1
            elif len(self.pairs[index]["games"]) < 2:
                voided += 1
                self.state.void_pair(index)
        self._feed_breakdowns()
        self.planned -= unstarted + voided
        if job.killed:
            return
        if job.returncode or not job.games:
            self.failures += 1
            tail = job.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
            print_line(f"[SPRT] job {job.id} on {job.slot.label} exited with code {job.returncode} "
                       f"after {len(job.games)} games; last output:\n    " + "\n    ".join(tail))
            if self.failures >= max(3, len(self.slots)) and self.stop_reason is None:
                self.stop_reason = f"{self.failures} jobs in a row failed (see the job logs)"
                print_line(f"[SPRT] stopping: {self.stop_reason}")
        else:
            self.failures = 0
        if voided:
            print_line(f"[SPRT] job {job.id}: {voided} pair(s) excluded from the test "
                       "(unfinished, or an engine left its core)")

    def on_warning(self, job, text):
        print_line(f"[fastchess] {text}")

    def tick(self):
        if self.plotter:
            self.plotter._pump_events()
        self.print_status()

    # -- reporting ------------------------------------------------------------

    def _enrich(self, snapshot):
        snapshot = dict(snapshot)
        snapshot["by_core"] = {core: t.summary() for core, t in sorted(self.by_core.items())}
        snapshot["elapsed_s"] = round(time.time() - self.started, 1)
        return snapshot

    def rate(self):
        minutes = (time.time() - self.started) / 60
        return self.finished_games / minutes if minutes > 0 else 0.0

    def print_status(self, force=False):
        now = time.monotonic()
        if not force and (not self.status_dirty or now - self.last_status < self.args.status_every):
            return
        self.last_status, self.status_dirty = now, False
        s = self.state.snapshot()
        if not s["pairs"]:
            return
        progress = s["llr_progress"]
        direction = "H1" if s["llr"] >= 0 else "H0"
        cores = "  ".join(f"{core}:{fmt_elo(t.summary()['elo_diff']).strip()}({t.pairs})"
                          for core, t in sorted(self.by_core.items()))
        eta = f"  ~{s['eta_pairs']:,} pairs to go" if s["eta_pairs"] is not None else ""
        print_line(
            f"[SPRT] {s['pairs']:,} pairs | LLR {s['llr']:+.2f} [{s['lbound']:+.2f},{s['ubound']:+.2f}] "
            f"{abs(progress):.0%} to {direction} | Elo {fmt_elo(s['elo_diff'], s['elo_low'], s['elo_high'])} "
            f"| nElo {fmt_elo(s['nelo'], s['nelo_low'], s['nelo_high'])} | LOS {s['los']:.1f}% | "
            f"penta {'-'.join(map(str, s['pentanomial']))} | {cores} | {self.rate():.1f} games/min{eta}")

    def report(self, stats, elapsed):
        s = stats
        lines = ["", "=" * 78,
                 f"SPRT {s['result'].upper()}   ({s['pairs']:,} pairs = {s['games_played']:,} games, "
                 f"{elapsed / 60:.1f} min, {self.rate():.1f} games/min)", "=" * 78]
        if s["pairs"]:
            lines += [
                f"  LLR       {s['llr']:+.3f}   bounds [{s['lbound']:+.3f}, {s['ubound']:+.3f}]   "
                f"H0: Elo {s['elo0']:+g}  H1: Elo {s['elo1']:+g}  alpha {s['alpha']} beta {s['beta']}",
                f"  Elo       {s['elo_diff']:+.2f}   95% CI [{s['elo_low']:+.2f}, {s['elo_high']:+.2f}]   "
                f"LOS {s['los']:.2f}%",
                f"  nElo      {s['nelo']:+.2f}   95% CI [{s['nelo_low']:+.2f}, {s['nelo_high']:+.2f}]",
                f"  score     {s['score']:.4f} per game   pair sd {s['pair_stdev']:.4f}"
                + (f"   pair/game variance ratio {s['variance_ratio']:.3f}" if s["variance_ratio"] else "")
                + ("  (pairing removes {:.0%} of the variance)".format(1 - s['variance_ratio'])
                   if s['variance_ratio'] and s['variance_ratio'] < 1 else ""),
                "  pairs     " + "   ".join(f"{label} {n}" for label, n in zip(LABELS, s["pentanomial"]))
                + f"   (DD {s['pentanomial_detail']['DD']}, WL {s['pentanomial_detail']['WL']})",
                f"  games     +{s['candidate_wins']} ={s['candidate_draws']} -{s['candidate_losses']}   "
                f"as white +{s['candidate_white_wins']} ={s['candidate_white_draws']} -{s['candidate_white_losses']}"
                f"   as black +{s['candidate_black_wins']} ={s['candidate_black_draws']} -{s['candidate_black_losses']}"
                f"   draws {s['draw_ratio']:.1f}%",
            ]
            if s["voided_pairs"]:
                lines.append(f"  excluded  {s['voided_pairs']} pair(s): unfinished, or an engine left its core"
                             f" ({s.get('excluded_pairs', 0)} of them)")
            if s.get("repinned_games"):
                lines.append(f"  pinning   {s['repinned_games']} game(s) started with an engine outside its core; "
                             "it was moved back before the game")
            lines += ["", "  By core type (both engines of a game on the same core):",
                      f"    {'core':4s} {'pairs':>6s} {'Elo':>16s} {'nElo':>7s} {'penta LL-LD-DD/WL-WD-WW':>24s} "
                      f"{'s/game':>7s} {'clock':>7s}"]
            for core, tally in sorted(self.by_core.items()):
                t = tally.summary()
                seconds, games, mhz_sum, mhz_n = self.core_time.get(core, [0, 0, 0, 0])
                lines.append(f"    {core:4s} {t['pairs']:>6d} {fmt_elo(t['elo_diff'], t['elo_low'], t['elo_high']):>16s} "
                             f"{t['nelo']:>+7.1f} {'-'.join(map(str, t['pentanomial'])):>24s} "
                             f"{seconds / max(games, 1):>7.1f} {fmt_ghz(mhz_sum / mhz_n if mhz_n else None):>7s}")
        if self.terminations:
            lines += ["", "  Terminations:"]
            for (term, fault), count in sorted(self.terminations.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {count:6d}  {term}{f'  ({fault})' if fault else ''}")
            bad = {k: v for k, v in self.terminations.items() if k[1]}
            if bad:
                lines.append("    ! time forfeits / crashes / illegal moves above count as losses")
        if self.register.rows:
            lines += ["", "  Openings by family:", self.register.format_table(limit=10, level="family"),
                      "", "  Openings by variation:", self.register.format_table(limit=10, level="variation")]
        print_line("\n".join(lines))

    def close(self):
        self.games_file.close()
        self.stats_file.close()


# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------

def _raw_db(db=None):
    from data.etl.paths import RAW_DB
    return Path(db or RAW_DB)


def ingest_telemetry(cnxn, run_logs, ids):
    """Load one run's engine logs (<run_logs>/candidate/ and baseline/), each side attributed to its own build.

    `ids`: the run's engine and experiment ids, as in its engines.json, and (when the
    runner's games were stored) its run_dir / run_id. An engine game is linked to the
    played game (games row, source 'match') its searches were made in; only an engine
    game that matches no played game gets a row of its own (source 'engine'). Each side
    is one transaction, recorded in telemetry_loads, so a retry loads each side exactly
    once. data/etl/telemetry.py, which also loads runs of more engines (tests/tournament.py).
    Returns True if everything was loaded.
    """
    from data.etl.telemetry import ingest_telemetry as ingest
    return ingest(cnxn, run_logs, ids)


def retry_ingest(run_logs, db=None):
    """Load a run's preserved telemetry with the engine ids recorded for it (engines.json)."""
    from data.etl.telemetry import retry_ingest as retry
    return retry(run_logs, db=db)


def upload_logs(args, fastchess_stats, runtime=None, run_dir=None, run_meta=None, run_logs=None):
    """Store the run: SPRT row with its played games, pairwise result, then (--log) telemetry."""
    from data import etl, ratings

    db_path = _raw_db(getattr(args, "db", None))
    if not db_path.exists():
        print(f"[DATA] Database {db_path} not found; skipping upload (set CHESS_RAW_DB or pass --db).")
        return None
    print(f"[DATA] Uploading to {db_path} ...")
    cnxn = etl.connect_raw(db_path)  # waits for another run's write instead of failing after 5 s
    try:
        etl.ensure_schema(cnxn)
        # engines are identified by their build (binary + net hash), not their version string;
        # the baseline first: it is usually the canonical build of its version (legacy rows)
        print("[DATA] Identifying engine builds...")
        baseline_engine_id = etl.resolve_engine(cnxn, args.engine_b)
        candidate_engine_id = etl.resolve_engine(cnxn, args.engine_a)
        sprt_id = etl.start_experiment(cnxn, "SPRT", candidate_engine_id, comparison_engine_id=baseline_engine_id,
                                       info=json.dumps(run_meta or {}, default=str))

        # The results first: telemetry below can fail and be retried, they cannot. Every game the
        # runner played, with its authoritative result, opening, moves and the environment it ran
        # in (cpu, P/E core, measured clock), goes in with the SPRT row (one transaction).
        match_games = {}
        if run_dir is not None:
            match_games = etl.bulk_log_match_games(cnxn, run_dir, sprt_id, {"Candidate": candidate_engine_id,
                                                                            "Baseline": baseline_engine_id},
                                                   commit=False)
        args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        etl.log_sprt(cnxn, sprt_id, candidate_engine_id, baseline_engine_id,
                     **{**args_dict, **fastchess_stats}, runtime=runtime)
        print(f"[DATA] Logged {len(match_games)} games with their core type, cpu and clock")
        # pairwise result; rating fits leave SPRT results out unless asked (early stopping biases them)
        limit = dict(tc=args.tc, depth=args.depth, time=args.time, nodes=getattr(args, "nodes", None))
        if candidate_engine_id == baseline_engine_id:
            print(f"[DATA] Candidate and baseline are the same build (engine id {candidate_engine_id}): no "
                  "match_results row (a build against itself would only distort the rating fits)")
        else:
            ratings.record_snapshot(cnxn, source="sprt", engine_a=candidate_engine_id, engine_b=baseline_engine_id,
                                    snapshot=fastchess_stats,
                                    tc_category=ratings.category(**limit, threads=getattr(args, "threads", 1)),
                                    time_control=limit_label(**limit), experiment_id=sprt_id)
        etl.update_experiment(cnxn, sprt_id, {"end_time_utc": datetime.now(timezone.utc).isoformat()})

        # this run's engine logs only
        run_logs = Path(run_logs) if run_logs else None
        if not args.log:
            print("[DATA] Engine logging was disabled, skipping JSONL ingestion.")
        elif run_logs and run_logs.is_dir():
            ids = dict(candidate=candidate_engine_id, baseline=baseline_engine_id, experiment=sprt_id, loaded=[],
                       run_dir=str(Path(run_dir).resolve()) if run_dir else None,
                       run_id=Path(run_dir).name if run_dir else None)
            (run_logs / "engines.json").write_text(json.dumps(ids), encoding="utf-8")  # a retry attributes the logs
            if ingest_telemetry(cnxn, run_logs, ids):
                shutil.rmtree(run_logs, ignore_errors=True)
                print(f"[DATA] Telemetry loaded for SPRT {sprt_id}.")
            else:
                print(f"[DATA] Log files preserved; retry with: python -m tests.sprt --retry-ingest {run_logs}")
        print(f"[DATA] Logged SPRT experiment {sprt_id}.")
    finally:
        cnxn.close()
    return sprt_id


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def parse_params(values):
    params = {}
    for item in values or []:
        name, sep, value = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"--param expects NAME=VALUE, got {item!r}")
        params[name] = value
    return params


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine-a", help="Candidate engine path")
    p.add_argument("--engine-b", help="Baseline engine path")
    p.add_argument("--param", dest="param_list", action="append", metavar="NAME=VALUE",
                   help="UCI option for the candidate only (repeatable), e.g. tuned values")
    p.add_argument("--fastchess", default=None,
                   help="fastchess executable (default: $FASTCHESS, libs/fastchess, or fastchess on PATH)")
    p.add_argument("--retry-ingest", metavar="RUN_LOG_DIR",
                   help="Only load the preserved telemetry of an earlier run (<logroot>/<run>) into the database")

    tc = p.add_argument_group("per-move limit (choose one)")
    tc.add_argument("--depth", type=int, default=None, help="Depth per move")
    tc.add_argument("--time", type=float, default=None, help="Seconds per move")
    tc.add_argument("--tc", type=str, default=None, help="Time control (e.g. 10+0.1)")
    tc.add_argument("--nodes", type=int, default=None,
                    help="Nodes per move: hardware-independent, for eval/NNUE changes (the engine must honour "
                         "go nodes)")

    s = p.add_argument_group("SPRT")
    s.add_argument("--elo0", type=float, default=0)
    s.add_argument("--elo1", type=float, default=10)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--beta", type=float, default=0.05)
    s.add_argument("--max-games", type=int, default=1000, help="Stop inconclusive after this many games")

    c = p.add_argument_group("CPU")
    c.add_argument("--concurrency", type=int, default=None,
                   help="Concurrent games (default: every physical core but one, divided by --threads). "
                        "The runner pins each game to its own core(s): P-cores first, then E-cores")
    c.add_argument("--threads", type=int, default=1,
                   help="Search threads per engine (option.Threads); each pinned game gets this many cpus "
                        "of one core type")
    c.add_argument("--no-pin", action="store_true", help="Leave the placement of games to the OS")

    b = p.add_argument_group("openings")
    b.add_argument("--book", default=PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn",
                   help="Opening book (.pgn or .epd)")
    b.add_argument("--book-depth", type=int, default=16, help="Plies used from each book line")
    b.add_argument("--seed", type=int, default=None, help="Book shuffle seed (default: random, recorded)")

    a = p.add_argument_group("adjudication")
    a.add_argument("--maxmoves", type=int, default=100,
                   help="Adjudicate a draw after this many moves (fastchess: 'Draw by adjudication')")
    a.add_argument("--tb", default=None,
                   help="Syzygy tablebase dir(s) for adjudication (';'-separated on Windows); WDL files are enough")
    a.add_argument("--tbpieces", type=int, default=None, help="Only adjudicate positions with at most this many pieces")

    r = p.add_argument_group("run")
    r.add_argument("--pairs-per-job", type=int, default=4,
                   help="Opening pairs per fastchess process (amortizes engine start-up)")
    r.add_argument("--status-every", type=float, default=15, help="Seconds between status lines")
    r.add_argument("--quiet", action="store_true", help="No per-game lines")
    r.add_argument("--log", action="store_true", help="Engine telemetry logging (uploaded to the database)")
    r.add_argument("--plot", action="store_true", help="Show live SPRT plots (LLR, Elo)")
    r.add_argument("--no-db", action="store_true", help="Do not upload results to the database")
    r.add_argument("--db", default=None, help="Raw SQLite database (default: CHESS_RAW_DB or platform default)")
    r.add_argument("--logroot", default=SPRT_LOG_DIR,
                   help="Root of the engine telemetry logs (each run logs into <logroot>/<run>/<side>/)")
    r.add_argument("--results-dir", default=RESULTS_DIR,
                   help="Where run folders go (default logs/sprt_results); the opening register CSV "
                        "(opening_register.csv) goes into its parent folder")
    return p


def resolve_args(args=None):
    """Accept argv, None (sys.argv) or a Namespace from release.py (missing fields get defaults)."""
    parser = build_parser()
    if isinstance(args, argparse.Namespace):
        defaults = vars(parser.parse_args([]))
        merged = {**defaults, **{k: v for k, v in vars(args).items() if v is not None or k not in defaults}}
        args = argparse.Namespace(**merged)
    else:
        args = parser.parse_args(args)
    if not args.retry_ingest and not (args.engine_a and args.engine_b):
        parser.error("--engine-a and --engine-b are required")
    params = dict(getattr(args, "params", None) or {})
    params.update(parse_params(getattr(args, "param_list", None)))
    args.params = params
    return args


def _declared_threads(engine):
    """(declared, max) of the engine's Threads option, read with the runner's own probe."""
    line = probe_uci(engine.cmd, engine.args)["options"].get("Threads")
    if line is None:
        return False, None
    found = re.search(r"\bmax\s+(\d+)", line)
    return True, int(found.group(1)) if found else None


def require_threads(engine, threads):
    """Refuse a thread count the engine would silently ignore (fastchess only warns)."""
    if not engine.args:
    #    uci_probe.require_threads(engine.cmd, threads)
        return
    declared, most = _declared_threads(engine)  # a script run by an interpreter: uci_probe starts executables
    if not declared:
        raise ValueError(f"{engine.name} declares no Threads option, so it cannot search with {threads} threads")
    if most is not None and most < threads:
        raise ValueError(f"{engine.name} accepts at most Threads={most}, not {threads}")


def require_options(engine, values):
    """Refuse candidate options (--param) fastchess would skip with only a warning: an undeclared
    option (names are case-sensitive) or a value the option refuses would leave the candidate
    on its defaults, and the SPRT would silently test something else."""
    problems = uci_probe.option_problems(probe_uci(engine.cmd, engine.args)["options"], values)
    if problems:
        where = engine.args[-1] if engine.args else engine.cmd  # a script's path, not its interpreter's
        raise ValueError(f"--param for the candidate ({where}): " + "; ".join(problems))


def normalize_legacy_check_defaults(engine):
    """Wrap engines whose UCI check options use the non-standard defaults 0/1."""
    declarations = probe_uci(engine.cmd, engine.args)["options"].values()
    invalid_defaults = any(
        re.search(r"\btype check\b.*\bdefault\s+[01](?:\s|$)", line)
        for line in declarations
    )
    if not invalid_defaults:
        return

    executable = os.path.abspath(engine.cmd)
    engine.cmd = sys.executable
    engine.args = (str(PROJECT_ROOT / "utils" / "uci_check_compat.py"), executable, *engine.args)
    engine.workdir = engine.workdir or os.path.dirname(executable)
    engine.pin_option = None
    print_line(f"[SPRT] {engine.name}: normalizing legacy UCI check-option defaults through compatibility wrapper")


def main(args=None):
    args = resolve_args(args)
    if args.retry_ingest:
        return retry_ingest(args.retry_ingest, db=args.db)
    if args.max_games < 2 or args.max_games % 2:
        raise ValueError("--max-games must be a positive even number (complete opening pairs)")
    if args.concurrency is not None and args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    # Threads: --threads for both engines; tuned params may give the candidate its own
    params = dict(args.params)
    threads_b = args.threads or 1
    threads_a = int(params.pop("Threads", threads_b))
    if min(threads_a, threads_b) < 1:
        raise ValueError("--threads must be positive")
    threads = max(threads_a, threads_b)
    # everything that can be refused is checked before cpus are claimed or folders created
    tc_args = tc_arguments(depth=args.depth, movetime=args.time, tc=args.tc, nodes=args.nodes)
    extra_args = ["-maxmoves", str(args.maxmoves)] + tb_args(args.tb, args.tbpieces)
    fastchess = find_fastchess(args.fastchess)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_logs = Path(args.logroot).resolve() / run_id  # this run's engine logs only, one folder per side

    def side_options(side, threads):
        return {"log_dir": str(run_logs / side), "uci_logging": str(bool(args.log)).lower(), "Threads": threads}

    candidate = EngineSpec.from_path("Candidate", args.engine_a, {**side_options("candidate", threads_a), **params})
    baseline = EngineSpec.from_path("Baseline", args.engine_b, side_options("baseline", threads_b))
    normalize_legacy_check_defaults(candidate)
    normalize_legacy_check_defaults(baseline)
    candidate.resolve()
    baseline.resolve()
    require_options(candidate, params)
    if args.nodes is not None:
        require_node_limit([candidate, baseline], label="SPRT")
    for engine, count in ((candidate, threads_a), (baseline, threads_b)):
        if count > 1:
            require_threads(engine, count)
    book = OpeningBook(args.book, plies=args.book_depth, seed=args.seed)

    slots = plan_slots(concurrency=args.concurrency, threads=threads, pin=not args.no_pin)
    try:
        return _run(args, slots, candidate, baseline, book, fastchess, tc_args, extra_args, run_id, run_logs,
                    threads)
    finally:
        release_slots(slots)


def _run(args, slots, candidate, baseline, book, fastchess, tc_args, extra_args, run_id, run_logs, threads):
    results_dir = Path(args.results_dir)
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True)
    if args.log:
        for side, _ in SIDES:
            (run_logs / side).mkdir(parents=True, exist_ok=True)
    machine = environment.machine_info()
    pinned = slots[0].cpu is not None
    hard = pinned and os.name == "nt"  # each job in a job object with the slot as its affinity limit
    mask = "job affinity limit + process mask" if hard else "process mask"
    limit = dict(tc=args.tc, depth=args.depth, time=args.time, nodes=args.nodes)
    meta = dict(run=run_dir.name, candidate=args.engine_a, baseline=args.engine_b,
                time_control=limit_label(**limit), threads=threads, fastchess=fastchess,
                machine=machine, slots=[vars(s) | {"label": s.label} for s in slots],
                pinning={e.name: (f"Affinity option + {mask}" if e.pin_option and pinned else
                                  mask if pinned else "none") for e in (candidate, baseline)},
                book=str(args.book), book_openings=len(book), book_seed=book.seed, model="pentanomial-logistic",
                engine_logs=str(run_logs) if args.log else None)

    print_line(f"[SPRT] Candidate {args.engine_a}\n[SPRT] Baseline  {args.engine_b}")
    print_line(f"[SPRT] H0 Elo {args.elo0:+g}  H1 Elo {args.elo1:+g}  alpha {args.alpha}  beta {args.beta}  "
               f"max {args.max_games:,} games  {meta['time_control']}"
               + (f"  {threads} threads" if threads > 1 else ""))
    print_line(f"[SPRT] book {Path(args.book).name}: {len(book):,} openings, {args.book_depth} plies, seed {book.seed}")
    print_line(f"[SPRT] machine {machine['cpu_model']}; {machine['topology']}; power {machine['power_source'] or '?'}")
    print_line(f"[SPRT] {describe_slots(slots)}")
    if pinned:
        print_line("[SPRT] each game runs on one core type: P-cores only play P-cores, E-cores only E-cores"
                   + ("; engines cannot leave their slot (job affinity limit)" if hard else ""))
    for engine in (candidate, baseline):
        if pinned and not engine.pin_option:
            print_line(f"[SPRT] note: {engine.name} has no Affinity option; its process mask keeps it on the core")
    print_line(f"[SPRT] engine logging {'enabled: ' + str(run_logs) if args.log else 'disabled'}")
    print_line(f"[SPRT] fastchess {fastchess}")
    print_line(f"[SPRT] results {run_dir}")

    (run_dir / "config.json").write_text(json.dumps(dict(arguments=vars(args), meta=meta), indent=2,
                                                    default=str), encoding="utf-8")
    plotter = None
    if args.plot:
        from utils.sprt_plot import LivePlotter

        def label(engine):
            info = probe_uci(engine.cmd, engine.args)
            return {"version": info.get("version") or info.get("name") or engine.name}
        plotter = LivePlotter(args.elo0, args.elo1, args.alpha, args.beta, label(candidate), label(baseline),
                              args.time, args.depth, args.tc or (f"{args.nodes} nodes / move" if args.nodes else None))

    controller = SprtController(args, (candidate, baseline), book, slots, run_dir, plotter)
    runner = MatchRunner(fastchess, slots, run_dir, tc_args, plies=args.book_depth, extra_args=extra_args)
    start_time = time.time()
    failure = None
    try:
        runner.run(controller)
    except KeyboardInterrupt:
        failure = "interrupted"
        print_line("\n[SPRT] interrupted; writing results so far")
    finally:
        controller.close()
        run_time = time.time() - start_time
        stats = controller._enrich(controller.state.snapshot())
        stats["error"] = failure or controller.stop_reason
        stats["games_per_minute"] = round(controller.rate(), 2)
        stats["repinned_games"] = runner.repinned_games
        stats["mixed_core_games"] = runner.mixed_games
        stats["excluded_pairs"] = len(controller.tainted)
        stats.update(run_id=run_dir.name, book_seed=book.seed, slots=len(slots), threads=threads,
                     env_host=machine.get("host"), env_cpu_model=machine.get("cpu_model"))
        stats["terminations"] = [dict(termination=t, fault=f, count=n)
                                 for (t, f), n in sorted(controller.terminations.items(), key=lambda kv: -kv[1])]
        (run_dir / "summary.json").write_text(json.dumps(stats, indent=2, allow_nan=False, default=str),
                                              encoding="utf-8")
        controller.register.save(run_dir, run_meta=meta, history_csv=results_dir.parent / OPENING_HISTORY.name)
        controller.report(stats, run_time)

    if plotter:
        plotter.finalize(stats, run_dir / "sprt.png")
    if failure is None and controller.stop_reason is None and not args.no_db:
        upload_logs(args, fastchess_stats=stats, runtime=run_time, run_dir=run_dir, run_meta=meta, run_logs=run_logs)
    if not args.log:
        shutil.rmtree(run_logs, ignore_errors=True)  # engines may create their log folder with logging off
    elif run_logs.is_dir():
        print_line(f"[SPRT] engine logs kept in {run_logs}")
    print_line(f"[SPRT] {stats['result']}: {stats['pairs']} pairs, LLR {stats['llr']:+.4f}  ({run_dir})")
    if plotter:
        plotter.show_final()
    if failure:
        raise KeyboardInterrupt
    if controller.stop_reason:
        # an error, like a stopped SPSA: not an inconclusive test that release gating would read as "not accepted"
        raise RuntimeError(f"SPRT stopped after {stats['pairs']} pairs: {controller.stop_reason} (not uploaded; "
                           f"results in {run_dir})")
    return {"accepted": stats["result"] == "pass", "elo": stats["elo_diff"], "games": stats["games_played"],
            "result": stats["result"], "run_dir": str(run_dir)}


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"[SPRT] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
