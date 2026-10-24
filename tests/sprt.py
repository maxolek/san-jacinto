#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys
import sqlite3
import re
from pathlib import Path
import time
from datetime import datetime, timezone
import platform
import threading
import json
import shutil
import signal
import queue

from utils.openings import OpeningBook, OpeningRegister, classify, display_name  # noqa: E402
from utils.pentanomial import FINISHED_GAME, LABELS, PairTally, PentanomialSPRT  # noqa: E402

from utils import environment  # noqa: E402
from utils.match_runner import (EngineSpec, JobSpec, MatchRunner, describe_slots, plan_slots,  # noqa: E402
                                print_line, tc_arguments)
from utils.fastchess_match import (
    FASTCHESS, Side, limit_args, limit_label, prepare_match, run_fastchess, run_match, stop_fastchess, tb_args
)

system = platform.system()

# paths
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# log paths
LOGS_DIR = PROJECT_ROOT / "logs"
SPRT_LOG_DIR = LOGS_DIR / "sprt_logs"

def parse_fastchess_output(output, candidate_name="Candidate", elo0=0, elo1=10,
                           alpha=0.05, beta=0.05):
    """Rebuild pentanomial evidence from game results, never fastchess summaries."""
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

def termination_class(reason):
    """Coarse class of a cutechess termination text, and the side at fault if any."""
    text = (reason or "").lower()
    side = "white" if text.startswith("white") else "black" if text.startswith("black") else None
    for key, label in (("loses on time", "time forfeit"), ("disconnects", "crash"),
                       ("connection stalls", "stall"), ("illegal move", "illegal move"),
                       ("resigns", "resignation"), ("mates", "mate"), ("stalemate", "stalemate"),
                       ("3-fold", "repetition"), ("fifty", "50-move rule"),
                       ("insufficient", "insufficient material"), ("maximal game length", "move limit"),
                       ("adjudication", "adjudication")):
        if key in text:
            return label, (side if label in ("time forfeit", "crash", "stall", "illegal move") else None)
    return (text or "unknown"), None


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
        if self.failures >= max(3, len(self.slots)):
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
            if self.failures >= max(3, len(self.slots)):
                print_line("[SPRT] too many failed jobs, not starting new ones")
        else:
            self.failures = 0
        if voided:
            print_line(f"[SPRT] job {job.id}: {voided} pair(s) excluded from the test "
                       "(unfinished, or an engine left its core)")

    def on_warning(self, job, text):
        print_line(f"[cutechess] {text}")

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



def ingest_telemetry(cnxn, run_logs, ids):
    """Load one run's engine logs (<run_logs>/<side>/), each side attributed to its own build.

    `ids`: the run's engine and experiment ids, as in its engines.json. Every run has
    its own log folder, so logs of other, aborted or concurrent runs are never mixed in.
    Each side is one transaction, loaded completely or not at all (even if the process
    is killed). It records the load in telemetry_loads, whose key refuses a second load
    of the same side, and the side's logs are then removed, so a retry loads each side
    exactly once. Returns True if everything was loaded.
    """
    from data import etl
    run_logs = Path(run_logs)
    etl.ensure_schema(cnxn)  # e.g. a migration added since the run (a retry): no transaction is open yet
    ok = True
    for side, opponent in (("candidate", "baseline"), ("baseline", "candidate")):
        log_dir = run_logs / side
        if side in ids.setdefault("loaded", []) or not log_dir.is_dir():
            continue
        print(f"[DATA] Logging {side} telemetry ...")
        try:
            # first, in the side's transaction: a side loaded before (e.g. the process stopped
            # right after that commit) fails here, before any of its rows are added again
            cnxn.execute("INSERT INTO telemetry_loads (experiment_id, side) VALUES (?, ?)", (ids["experiment"], side))
            etl.consolidate_instance_logs(log_dir)
            paths = etl.get_jsonl_paths(log_dir)
            game_map = {}
            if paths["game"].exists():
                game_map = etl.bulk_log_game(cnxn, paths["game"], ids["experiment"], second_engine_id=ids[opponent],
                                             engine_id=ids[side], commit=False)
            if paths["search"].exists():
                etl.bulk_log_search_and_timing(
                    cnxn,
                    paths["search"],
                    game_map,
                    engine_id=ids[side],
                    timing_path=paths["timing"] if paths["timing"].exists() else None,
                    root_moves_path=paths["root_moves"] if paths["root_moves"].exists() else None,
                    commit=False,
                )
            cnxn.commit()
        except sqlite3.IntegrityError:
            cnxn.rollback()
            print(f"[DATA] {side} telemetry of experiment {ids['experiment']} was loaded already")
        except Exception as e:
            cnxn.rollback()
            print(f"[DATA] JSONL ingestion failed ({side}): {e}")
            ok = False
            continue
        ids["loaded"].append(side)
        (run_logs / "engines.json").write_text(json.dumps(ids), encoding="utf-8")
        shutil.rmtree(log_dir, ignore_errors=True)
    return ok


def retry_ingest(run_logs):
    """Load a run's preserved telemetry with the engine ids recorded for it (engines.json)."""
    from data import etl
    run_logs = Path(run_logs)
    ids = json.loads((run_logs / "engines.json").read_text(encoding="utf-8"))
    cnxn = etl.connect_raw()
    try:
        if ingest_telemetry(cnxn, run_logs, ids):
            shutil.rmtree(run_logs, ignore_errors=True)
            print(f"[DATA] Telemetry of experiment {ids['experiment']} loaded.")
    finally:
        cnxn.close()



def upload_logs(args, fastchess_stats, runtime=None, run_logs=None):
    from data import etl, ratings
    print("[DATA] Preparing to upload logs to database...")

    cnxn = etl.connect_raw()  # waits for another run's write instead of failing after 5 s
    try:
        # engines are identified by their build (binary + net hash), not their version string;
        # the baseline first: it is usually the canonical build of its version (legacy rows)
        print("[DATA] Identifying engine builds...")
        baseline_engine_id = etl.resolve_engine(cnxn, args.engine_b)
        candidate_engine_id = etl.resolve_engine(cnxn, args.engine_a)

        # log sprt experiment
        sprt_id = etl.start_experiment(
            cnxn,
            "SPRT",
            candidate_engine_id,
            comparison_engine_id = baseline_engine_id
        )

        # The results first: telemetry below can fail and be retried, they cannot.
        # log sprt experiment details in db.sprt (always, regardless of --log)
        args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        etl.log_sprt(
            cnxn,
            sprt_id,  # experiment_id
            candidate_engine_id,
            baseline_engine_id,
            **{**args_dict, **fastchess_stats},
            runtime=runtime
        )
        # pairwise result; rating fits leave SPRT results out unless asked (early stopping biases them)
        limit = dict(tc=args.tc, depth=args.depth, time=args.time, nodes=getattr(args, "nodes", None))
        ratings.record_snapshot(cnxn, source="sprt", engine_a=candidate_engine_id, engine_b=baseline_engine_id,
                                snapshot=fastchess_stats,
                                tc_category=ratings.category(**limit, threads=getattr(args, "threads", 1)),
                                time_control=limit_label(**limit), experiment_id=sprt_id)
        etl.update_experiment(
            cnxn,
            sprt_id,
            {"end_time_utc": datetime.now(timezone.utc).isoformat()}
        )

        # This run's engine logs only (engines log games even with --log off)
        run_logs = Path(run_logs) if run_logs else None
        if not args.log:
            print("[DATA] Logging was disabled, skipping JSONL ingestion.")
            if run_logs:
                shutil.rmtree(run_logs, ignore_errors=True)
        elif run_logs and run_logs.is_dir():
            ids = dict(candidate=candidate_engine_id, baseline=baseline_engine_id, experiment=sprt_id, loaded=[])
            (run_logs / "engines.json").write_text(json.dumps(ids), encoding="utf-8")  # a retry attributes the logs
            if ingest_telemetry(cnxn, run_logs, ids):
                shutil.rmtree(run_logs, ignore_errors=True)
                print(f"[DATA] Logging completed for SPRT {sprt_id}.")
            else:
                print(f"[DATA] Log files preserved; retry with: python -m tests.sprt --retry-ingest {run_logs}")
    finally:
        cnxn.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SPRT runner using cutechess-cli")

    # Engines
    p.add_argument("--engine-a", required=True, help="Candidate engine path")
    p.add_argument("--engine-b", required=True, help="Baseline engine path")
    p.add_argument("--params", type=dict, help="Dictionary of parameter and their values to test against the current version of the engine")
    p.add_argument("--retry-ingest", metavar="RUN_LOG_DIR", 
        help="Only load the preserved telemetry of an earlier run (<logroot>/<run>) into the database")

    # fastchess-cli path
    p.add_argument(
        "--fastchess",
        default=str(FASTCHESS),
        help="Path to fastchess.exe"
    )

    # Time control (choose ONE)
    p.add_argument("--nodes", type=int, default=None, help="Nodes per move")
    p.add_argument("--depth", type=int, default=None, help="Depth per move")
    p.add_argument("--time", type=float, default=None, help="Seconds per move")
    p.add_argument("--tc", type=str, default=None, help="Time control (e.g. 0+1)")

    # SPRT parameters
    p.add_argument("--elo0", type=int, default=0)
    p.add_argument("--elo1", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--max-games", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=2, help="Number of concurrent games")
    p.add_argument("--threads", type=int, default=1, help="Threads per engine (option.Threads); each pinned game gets this many cpu cores")
    p.add_argument("--no-pin", action="store_true", 
                   help="Do not pin each game to its own CPU cores (default: pin)")

    # Opening book
    p.add_argument("--book", default= PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn" , help="Opening book file")
    p.add_argument("--book-depth", type=int, default=16) # 8 full moves

    # adjudication
    p.add_argument("--tb", default=None,
                   help="Syzygy tablebase dir(s) for adjudication (';'-separated on Windows); WDL files are enough")
    p.add_argument("--tbpieces", type=int, default=None, help="Only adjudicate positions with at most this many pieces")

    # Logging
    p.add_argument('--log', action="store_true", help="Flag to turn on logging for candidate engine")
    p.add_argument('--plot', action='store_true', help='Show live SPRT plots (LLR, Elo)')
    p.add_argument(
        "--logroot",
        default=SPRT_LOG_DIR,
        help="Root directory for SPRT logs (each run logs into <logroot>/<run>/<side>/)"
    )

    args = p.parse_args(argv)
    if not args.retry_ingest and not (args.engine_a and args.engine_b):
        p.error("--engine-a and --engine-b are required")
    return args



def main(args=None):
    if args is None:
        args = parse_args()
    if getattr(args, "retry_ingest", None):
        return retry_ingest(args.retry_ingest)
    state = PentanomialSPRT(args.elo0, args.elo1, args.alpha, args.beta)
    if args.max_games < 2 or args.max_games % 2:
        raise ValueError("--max-games must be a positive even number (complete opening pairs)")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")

    # Validate TC
    nodes = getattr(args, "nodes", None)
    limits = limit_args(depth=args.depth, time=args.time, tc=args.tc, nodes=nodes)

    engine_a = os.path.abspath(args.engine_a)
    engine_b = os.path.abspath(args.engine_b)

    # Threads: --threads for both engines; tuned params may give the candidate its own
    params = dict(args.params or {})
    threads_b = getattr(args, "threads", 1)
    threads_a = int(params.pop("Threads", threads_b))
    if min(threads_a, threads_b) < 1:
        raise ValueError("--threads must be positive")

    # logging // options
    print(f"[SPRT] Engine logging: {'enabled' if args.log else 'disabled'}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = LOGS_DIR / "sprt_results" / run_id
    run_logs = Path(args.logroot).resolve() / run_id  # this run's engine logs only (engines run in their own dir)

    def side_options(side):
        return [
            f"option.log_dir={run_logs / side}",
            f"option.uci_logging={str(args.log).lower()}",
        ]

    options_a = side_options("candidate") + [f"options.{name}={value}" for name, value in params.items()]
    match = prepare_match(
        Side(engine_a, options_a, threads_a),
        Side(engine_b, side_options("baseline"), threads_b),
        limits=limits, pairs=args.max_games // 2, concurrency=args.concurrency,
        run_dir=run_dir, fastchess=args.fastchess, book=args.book, book_depth=args.book_depth,
        pin=not getattr(args, "no_pin", False), maxmoves=100,
        extra=tb_args(getattr(args, "tb", None), getattr(args, "tbpieces", None)),
        meta=dict(arguments=vars(args), model="pentanomial-logistic")
    )
    print(f"[SPRT] Results: {run_dir}")
    print("[SPRT] Launching fastchess:")

    # launch
    start_time = time.time()
    plotter = None 
    if args.plot:
        from data import etl
        from utils.sprt_plot import LivePlotter
        plotter = LivePlotter(args.elo0, args.elo1, args.alpha, args.beta, 
                              etl.probe_engine_metadata(args.engine_a),
                              etl.probe_engine_metadata(args.engine_b),
                              args.time, args.depth, args.tc or (f"{nodes} nodes / move" if nodes else None)
                            )
    stats = run_match(match, state, plotter)
    run_time = time.time() - start_time 
    if plotter: plotter.finalize(stats, run_dir / "sprt.png")

    # finalize
    upload_logs(args, fastchess_stats=stats, runtime=run_time, run_logs=run_logs)

    print(f"[SPRT] {stats['result']}: {stats['pairs']} pairs, LLR {stats['llr']:+.4f}")
    if plotter:
        plotter.show_final()

    return {
        "accepted": stats['result'] == "pass",
        "elo": stats['elo_diff'],
        "games": stats['games_played']
    }

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[SPRT] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
