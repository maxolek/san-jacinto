"""Fixed-length matches on the match runner: one or more pairings, every pair counted.

A *pairing* is two engines (utils.match_runner.EngineSpec) playing a set number of
opening pairs; the first engine has White in the first game of each pair and its
score is the one reported (like the SPRT's Candidate). An engine name stands for one
engine (command, arguments, options) in all pairings of a match. A fixed match plays one or
more pairings with utils/match_runner.py (one fastchess job per CPU slot, pinned in
a job object, every game with its slot, CPU, core type and measured clock), with no
early stopping, so every Elo is a fixed-sample estimate. Used by tests/smp.py (one
pairing per ladder step) and nnue/registry.py `match`; CMA-ES (one pairing per
candidate) and tournaments (one per pair of engines) can play a generation or a
whole schedule in one call.

    result = run_fixed_match([Pairing(a, b, 200), Pairing(c, d, 200, key="c-d")],
                             tc_args=tc_arguments(tc="10+0.1"), book="book.pgn", results_dir="logs/x")
    result["pairings"]["a vs b"]["elo_diff"]

Scheduling. A job plays a few pairs (pairs_per_job) of one pairing. An idle slot gets
the pairing furthest behind (share of its openings handed out; on a tie the one with
the fewest pairs on the slot's core type), so all pairings advance together and a
slow drift of the machine (clock, temperature, background load) does not favour one.
Near the end jobs get smaller, so the last pairs are spread over the slots.

Openings. Each pairing takes its openings from its own book (Pairing.book) or the
match's book when the controller is created. Pairings sharing one OpeningBook get
different openings; pairings with books of their own shuffled with the same seed
play the same openings (e.g. the candidates of one CMA-ES generation).

Statistics. Every pairing has its own utils.pentanomial.PentanomialSPRT
(sequential=False: every pair counts), fed in the order its pairs *started*, never
the order they finished (that would depend on the results), which also checks that
the two games of a pair reversed colours; a PairTally per core type (P/E) counts the
same pairs. A pair with an unfinished game (its job crashed or was stopped) or a game
in which an engine was found off its core (core_mixed) is excluded and its opening
played again, up to max_excluded pairs (like the SPRT); more stop the match, as do
jobs that fail in a row (max(3, slots)). run_fixed_match turns that stop_reason into
MatchStopped and Ctrl-C into MatchInterrupted (a KeyboardInterrupt); both carry the
results so far (.result) and leave summary.json behind.

Run folder <results_dir>/<run_name or time>/: config.json ({"meta": ...}, as the SPRT's),
games.jsonl (one record per finished game with the SPRT's fields, so
data/etl/match_games.py can store them, plus the pairing: `pairing`, `pairing_pair`,
`pairing_game`; `game` and `pair` number all pairs of the run in start order),
games.pgn and jobs/ (the runner's), summary.json (the result).
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from utils import environment, uci_probe
from utils.match_runner import (EngineSpec, JobSpec, MatchRunner, describe_slots, find_fastchess, plan_slots,
                                print_line, probe_uci, release_slots, termination_class)
from utils.openings import OpeningBook, classify, display_name
from utils.pentanomial import LABELS, PairTally, PentanomialSPRT

CANDIDATE, BASELINE = "Candidate", "Baseline"  # the identities PentanomialSPRT counts for


@dataclass
class Pairing:
    """Two engines and how many opening pairs they play; `first` is White in each pair's first game.

    key: the pairing's label in the results (default "<first> vs <second>"), unique in a match.
    book: its own opening book (default: the match's). meta: anything, copied to its result.
    """
    first: EngineSpec
    second: EngineSpec
    pairs: int
    key: str | None = None
    book: object = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.key is None:
            self.key = f"{self.first.name} vs {self.second.name}"

    @classmethod
    def of(cls, item):
        """A Pairing, or a tuple (first, second, pairs[, key])."""
        if isinstance(item, cls):
            return item
        if isinstance(item, (tuple, list)) and 3 <= len(item) <= 4:
            return cls(*item)
        raise TypeError(f"a pairing is a Pairing or (first, second, pairs[, key]), not {item!r}")


def _engine_identity(engine):
    """What makes two EngineSpecs the same engine: its command, arguments and options."""
    return os.path.normcase(os.path.abspath(engine.cmd)), tuple(engine.args), dict(engine.options)


def normalize_pairings(pairings):
    """[Pairing], checked: distinct engine names within a pairing, unique string keys, pairs >= 0,
    and one engine per name across the pairings (results, stored games and telemetry folders go
    by the name)."""
    out = [Pairing.of(p) for p in pairings]
    if not out:
        raise ValueError("a match needs at least one pairing")
    keys = set()
    named = {}  # engine name -> (its EngineSpec, the key of the first pairing it plays in)
    for p in out:
        if not isinstance(p.key, str):
            raise TypeError(f"pairing keys are strings, not {p.key!r}")
        if p.key in keys:
            raise ValueError(f"two pairings are called {p.key!r}: give them distinct keys")
        keys.add(p.key)
        if p.first.name == p.second.name:
            raise ValueError(f"pairing {p.key!r}: both engines are called {p.first.name!r} (fastchess tells "
                             "the sides apart by name)")
        if not isinstance(p.pairs, int) or isinstance(p.pairs, bool) or p.pairs < 0:
            raise ValueError(f"pairing {p.key!r}: pairs must be a non-negative integer, not {p.pairs!r}")
        for engine in (p.first, p.second):
            known, where = named.setdefault(engine.name, (engine, p.key))
            if known is not engine and _engine_identity(known) != _engine_identity(engine):
                raise ValueError(f"pairings {where!r} and {p.key!r}: two different engines are called "
                                 f"{engine.name!r} ({known.cmd} {list(known.args)} {known.options} and {engine.cmd} "
                                 f"{list(engine.args)} {engine.options}); results, stored games and telemetry go by "
                                 "the name: give them distinct names")
    return out


class MatchStopped(RuntimeError):
    """The match stopped before it was complete (failing jobs, too many excluded pairs).

    .result: what run_fixed_match would have returned, with the pairs played so far."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


class MatchInterrupted(KeyboardInterrupt):
    """Ctrl-C: the jobs were stopped and the results so far written. .result: as MatchStopped's."""

    def __init__(self, result):
        super().__init__("interrupted")
        self.result = result


# ----------------------------------------------------------------------------
# Engine checks (before any cpu is claimed or folder created)
# ----------------------------------------------------------------------------

def engine_threads(engine):
    return int(engine.options.get("Threads", 1))


def require_options(engine, values):
    """Refuse option values fastchess would skip with only a warning (undeclared, out of range, ...)."""
    problems = uci_probe.option_problems(probe_uci(engine.cmd, engine.args)["options"], values)
    if problems:
        where = engine.args[-1] if engine.args else engine.cmd  # a script's path, not its interpreter's
        raise ValueError(f"{engine.name} ({where}): " + "; ".join(problems))


def require_threads(engine, threads):
    """Refuse a thread count the engine would silently ignore (fastchess only warns).

    Executables are checked with uci_probe.require_threads; scripts run by an
    interpreter (uci_probe starts executables only) by their answer to `uci`."""
    if not engine.args:
        uci_probe.require_threads(engine.cmd, threads)
    else:
        require_options(engine, {"Threads": threads})


def require_node_limit(engines, label="MATCH"):
    """Refuse engines that ignore `go nodes` (uci_probe); scripts run by an interpreter are not checked."""
    uci_probe.require_node_limit([e.cmd for e in engines if not e.args])
    for engine in engines:
        if engine.args:
            print_line(f"[{label}] warning: could not check that {engine.name} ({engine.args[-1]}) honours go nodes")


def tc_label(tc_args):
    """The time control label of fastchess -each keys (utils.fastchess_match.limit_label's format)."""
    keys = dict(arg.split("=", 1) for arg in tc_args if "=" in arg)
    for name in ("depth", "nodes"):
        if name in keys:
            return f"{name}={keys[name]}"
    if "st" in keys:
        return f"st={float(keys['st']):g}"
    return keys.get("tc") or " ".join(tc_args)


# ----------------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------------

class _PairingState:
    def __init__(self, index, pairing, book):
        self.index, self.pairing, self.key = index, pairing, pairing.key
        self.book = pairing.book if pairing.book is not None else book
        if self.book is None and pairing.pairs:
            raise ValueError(f"pairing {pairing.key!r} has no opening book (and the match has none)")
        self.queue = deque(self.book.take(pairing.pairs) if pairing.pairs else ())  # openings still to hand out
        self.on_core = {}       # core type -> pairs handed to slots of that type
        self.names = {pairing.first.name: CANDIDATE, pairing.second.name: BASELINE}
        self.state = PentanomialSPRT(0, 10, sequential=False)
        self.started = 0        # pairs started: the next pair's index in this pairing's start order
        self.pairs = {}         # local index -> dict(job, pair_no, opening, games={1|2: (score, s, mhz, core)})
        self.fed = 0            # pairs already added to the breakdowns (same order as the state)
        self.tainted = set()    # local indexes of pairs with a game that may have mixed core types
        self.by_core = {}       # core type -> PairTally
        self.core_time = {}     # core type -> [seconds, games, mhz sum, mhz count]

    def progress(self):
        return (self.pairing.pairs - len(self.queue)) / self.pairing.pairs if self.pairing.pairs else 1.0


class FixedMatchController:
    """MatchRunner controller: plays every pairing's openings, replays voided pairs, counts in start order.

    The runner calls next_job / on_pair_start / on_game / on_job_end / on_warning / finished / tick
    (see utils/match_runner.py). result(...) gives the per-pairing snapshots.
    """

    def __init__(self, pairings, book, slots, run_dir, *, pairs_per_job=4, quiet=False, status_every=30.0,
                 label="MATCH", max_excluded=None, max_failures=None):
        pairings = normalize_pairings(pairings)
        if pairs_per_job < 1:
            raise ValueError("pairs_per_job must be at least 1")
        self.slots = list(slots)
        self.run_dir = Path(run_dir)
        self.pairs_per_job, self.quiet, self.status_every, self.label = pairs_per_job, quiet, status_every, label
        self.pairings = [_PairingState(i, p, book) for i, p in enumerate(pairings)]
        total = sum(p.pairs for p in pairings)
        # Excluded pairs are replayed, but only up to a limit: if many are excluded something is
        # systematically wrong and the match must stop (as the SPRT does).
        self.max_excluded = max(20, total // 20) if max_excluded is None else max_excluded
        self.max_failures = max(3, len(self.slots)) if max_failures is None else max_failures
        self.stop_reason = None
        self.failures = 0           # jobs in a row that failed
        self.next_global = 0        # the next pair's index in the start order of the whole run
        self.index_of = {}          # (job id, pair no) -> (pairing state, local index, global index)
        self.terminations = {}      # (class, engine at fault or None) -> count
        self.finished_games = 0
        self.started = time.time()
        self.last_status = time.monotonic()
        self.status_dirty = False
        self.games_file = open(self.run_dir / "games.jsonl", "w", encoding="utf-8")

    # -- scheduling -----------------------------------------------------------

    def next_job(self, slot):
        if self.stop_reason:
            return None
        waiting = [p for p in self.pairings if p.queue]
        if not waiting:
            return None
        core = getattr(slot, "core_type", "-")
        p = min(waiting, key=lambda p: (p.progress(), p.on_core.get(core, 0) / p.pairing.pairs, p.index))
        remaining = sum(len(q.queue) for q in self.pairings)
        size = max(1, min(self.pairs_per_job, len(p.queue), math.ceil(remaining / max(1, len(self.slots)))))
        openings = [p.queue.popleft() for _ in range(size)]
        p.on_core[core] = p.on_core.get(core, 0) + size
        return JobSpec((p.pairing.first, p.pairing.second), openings, p.book,
                       meta=dict(pairing=p.key, pairing_index=p.index))

    def on_pair_start(self, job, pair_no):
        p = self.pairings[job.spec.meta["pairing_index"]]
        local, number = p.started, self.next_global
        p.started += 1
        self.next_global += 1
        self.index_of[(job.id, pair_no)] = (p, local, number)
        p.pairs[local] = dict(job=job, pair_no=pair_no, opening=job.spec.openings[pair_no - 1], games={})
        return True

    def excluded(self):
        return sum(len(p.state.voided) for p in self.pairings)

    def finished(self):
        if self.stop_reason is None and self.excluded() > self.max_excluded:
            self.stop_reason = (f"{self.excluded()} pairs had to be excluded (limit {self.max_excluded}); "
                                "see the warnings above and the job logs")
            print_line(f"[{self.label}] stopping: {self.stop_reason}")
        return self.stop_reason is not None

    # -- results --------------------------------------------------------------

    def on_game(self, game):
        entry = self.index_of.get((game.job.id, game.pair_no))
        if entry is None:
            return
        p, local, number = entry
        half = 1 if game.game_no % 2 else 2
        score = game.score_for(p.pairing.first.name)
        pair = p.pairs[local]
        details = classify(pair["opening"])
        env = game.env
        term, fault_side = termination_class(game.reason)
        fault = (game.white if fault_side == "white" else game.black) if fault_side else None
        self.terminations[(term, fault)] = self.terminations.get((term, fault), 0) + 1
        record = dict(run=self.run_dir.name, game=2 * number + half, pair=number + 1, job=game.job.id,
                      job_game=game.game_no, white=game.white, black=game.black, result=game.result,
                      reason=game.reason, termination=term, candidate_score=score, seconds=round(game.seconds, 2),
                      started_utc=datetime.fromtimestamp(game.started, timezone.utc).isoformat(),
                      finished_utc=datetime.fromtimestamp(game.finished, timezone.utc).isoformat(),
                      eco=details[0], opening=details[1], variation=details[2],
                      pairing=p.key, pairing_pair=local + 1, pairing_game=2 * local + half, **env)
        self.games_file.write(json.dumps(record, default=str) + "\n")
        self.games_file.flush()
        self.finished_games += 1
        if not self.quiet:
            mhz, cpu = env.get("mhz_mean"), env.get("cpu")
            where = p.key if len(self.pairings) > 1 else display_name(details)
            print_line(f"  #{2 * number + half:<6d} {env.get('core_type', '-')} "
                       f"cpu{cpu if cpu is not None else '-':<3} "
                       f"{f'{mhz / 1000:.1f}GHz' if mhz else '  -   '}  {game.white[:9]:>9s}-{game.black[:9]:<9s} "
                       f"{game.result:<7s} {term:<21s} {game.seconds:6.1f}s  {where[:40]}")
        if score is None or local in p.tainted:
            return  # unfinished (the pair is voided when its job ends), or the pair is excluded already
        if env.get("core_mixed"):
            # Never count a game that may have been P-core against E-core.
            p.tainted.add(local)
            p.state.void_pair(local)
            self._feed(p)
            print_line(f"[{self.label}] {p.key}: pair {local + 1} excluded, an engine left its "
                       f"{env.get('core_type')}-core (cpu {env.get('cpu')}) during game {2 * number + half}")
            return
        pair["games"][half] = (score, game.seconds, env.get("mhz_mean"), env.get("core_type", "-"))
        # PentanomialSPRT knows Candidate and Baseline; it raises if a pair did not reverse colours
        if p.state.add_game(2 * local + half, p.names.get(game.white, game.white),
                            p.names.get(game.black, game.black), game.result):
            self._feed(p)
            self.status_dirty = True

    def _feed(self, p):
        """The per-core tallies see exactly the pairs the pairing's state has counted."""
        while p.fed < p.state.next_pair:
            local = p.fed
            p.fed += 1
            pair = p.pairs.get(local)
            if not pair or len(pair["games"]) < 2 or local in p.tainted:
                continue  # voided
            (s1, t1, m1, core), (s2, t2, m2, _) = pair["games"][1], pair["games"][2]
            p.by_core.setdefault(core, PairTally()).add_pair(s1, s2)
            timing = p.core_time.setdefault(core, [0.0, 0, 0.0, 0])
            timing[0] += t1 + t2
            timing[1] += 2
            for mhz in (m1, m2):
                if mhz:
                    timing[2] += mhz
                    timing[3] += 1

    def on_job_end(self, job):
        p = self.pairings[job.spec.meta["pairing_index"]]
        replay = []
        for pair_no in range(1, job.pairs_started + 1):
            entry = self.index_of.get((job.id, pair_no))
            if entry is None:
                continue
            local = entry[1]
            if local in p.tainted:
                replay.append(p.pairs[local]["opening"])
            elif len(p.pairs[local]["games"]) < 2:
                replay.append(p.pairs[local]["opening"])
                p.state.void_pair(local)
        self._feed(p)
        voided = len(replay)
        replay += job.spec.openings[job.pairs_started:]  # never started
        if replay:
            # played again soon, with the same openings: the pairing still plays the openings it was given
            p.queue.extendleft(reversed(replay))
            core = getattr(job.slot, "core_type", "-")
            p.on_core[core] = p.on_core.get(core, 0) - len(replay)
        if job.killed:
            return
        if job.returncode or not job.games:
            self.failures += 1
            try:
                tail = job.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
            except OSError:
                tail = []
            print_line(f"[{self.label}] job {job.id} ({p.key}) on {job.slot.label} exited with code {job.returncode} "
                       f"after {len(job.games)} games; last output:\n    " + "\n    ".join(tail))
            if self.failures >= self.max_failures and self.stop_reason is None:
                self.stop_reason = f"{self.failures} jobs in a row failed (see the job logs)"
                print_line(f"[{self.label}] stopping: {self.stop_reason}")
        else:
            self.failures = 0
        if voided:
            print_line(f"[{self.label}] job {job.id} ({p.key}): {voided} pair(s) excluded and played again "
                       "(unfinished, or an engine left its core)")

    def on_warning(self, job, text):
        print_line(f"[fastchess] {text}")

    def tick(self):
        self.print_status()

    # -- reporting ------------------------------------------------------------

    def pairs_done(self):
        return sum(p.state.pairs_tested for p in self.pairings)

    def rate(self):
        minutes = (time.time() - self.started) / 60
        return self.finished_games / minutes if minutes > 0 else 0.0

    def print_status(self, force=False):
        now = time.monotonic()
        if not force and (not self.status_every or not self.status_dirty or now - self.last_status < self.status_every):
            return
        self.last_status, self.status_dirty = now, False
        total = sum(p.pairing.pairs for p in self.pairings)
        parts = []
        for p in self.pairings[:4]:
            s = p.state.snapshot()
            elo = (f"{s['elo_diff']:+.1f} +/-{(s['elo_high'] - s['elo_low']) / 2:.1f}"
                   if s["elo_diff"] is not None else "n/a")
            parts.append(f"{p.key} {elo} ({s['pairs']})")
        more = f" | +{len(self.pairings) - 4} more" if len(self.pairings) > 4 else ""
        print_line(f"[{self.label}] {self.pairs_done():,}/{total:,} pairs | " + " | ".join(parts) + more
                   + f" | {self.rate():.1f} games/min")

    def pairing_result(self, p):
        """The pairing's PentanomialSPRT snapshot with its key, engines, core breakdown and exclusions."""
        s = p.state.snapshot()
        by_core = {}
        for core, tally in sorted(p.by_core.items()):
            seconds, games, mhz_sum, mhz_n = p.core_time.get(core, [0.0, 0, 0.0, 0])
            by_core[core] = dict(tally.summary(), seconds_per_game=round(seconds / games, 2) if games else None,
                                 mhz_mean=round(mhz_sum / mhz_n) if mhz_n else None)
        s.update(key=p.key, first=p.pairing.first.name, second=p.pairing.second.name,
                 pairs_requested=p.pairing.pairs, complete=s["pairs"] >= p.pairing.pairs,
                 excluded_pairs=len(p.tainted), by_core=by_core, meta=dict(p.pairing.meta))
        return s

    def result(self, runner=None, elapsed=None, error=None):
        pairings = {p.key: self.pairing_result(p) for p in self.pairings}
        out = dict(run_id=self.run_dir.name, run_dir=str(self.run_dir), pairings=pairings,
                   pairs=sum(s["pairs"] for s in pairings.values()),
                   pairs_requested=sum(p.pairing.pairs for p in self.pairings),
                   games=self.finished_games, error=error or self.stop_reason, stop_reason=self.stop_reason,
                   interrupted=error == "interrupted",
                   elapsed_s=round(time.time() - self.started if elapsed is None else elapsed, 1),
                   games_per_minute=round(self.rate(), 2), slots=len(self.slots),
                   voided_pairs=self.excluded(), excluded_pairs=sum(len(p.tainted) for p in self.pairings),
                   terminations=[dict(termination=t, fault=f, count=n) for (t, f), n in
                                 sorted(self.terminations.items(), key=lambda kv: -kv[1])])
        if runner is not None:
            out.update(repinned_games=runner.repinned_games, mixed_core_games=runner.mixed_games)
        return out

    def close(self):
        self.games_file.close()


def format_report(result, label="MATCH"):
    """Console table of a run_fixed_match result: per pairing Elo, nElo, games and per core type."""
    def elo(s, key="elo_diff", low="elo_low", high="elo_high"):
        return f"{s[key]:+7.1f} +/-{(s[high] - s[low]) / 2:5.1f}" if s.get(key) is not None else f"{'n/a':>15s}"

    minutes = result["elapsed_s"] / 60
    lines = ["", "=" * 78,
             f"{label} {'STOPPED' if result['error'] else 'DONE'}   ({result['pairs']:,}/{result['pairs_requested']:,} "
             f"pairs, {result['games']:,} games, {minutes:.1f} min, {result['games_per_minute']:.1f} games/min)",
             "=" * 78,
             f"  {'pairing':24s} {'pairs':>9s} {'Elo':>15s} {'nElo':>15s} {'+W =D -L':>14s}  "
             f"penta {'-'.join(l.replace(' / ', '/') for l in LABELS)}"]
    for s in result["pairings"].values():
        wdl = f"+{s['candidate_wins']} ={s['candidate_draws']} -{s['candidate_losses']}"
        lines.append(f"  {s['key'][:24]:24s} {s['pairs']:>4d}/{s['pairs_requested']:<4d} {elo(s)} "
                     f"{elo(s, 'nelo', 'nelo_low', 'nelo_high')} {wdl:>14s}  {'-'.join(map(str, s['pentanomial']))}")
        if len(s["by_core"]) > 1:
            lines.append("  " + " " * 24 + "  by core: " + "  ".join(
                f"{core} {t['elo_diff']:+.1f} ({t['pairs']})" for core, t in s["by_core"].items()
                if t["elo_diff"] is not None))
    if result.get("voided_pairs"):
        lines.append(f"  excluded and replayed: {result['voided_pairs']} pair(s) (unfinished, or an engine left its "
                     f"core: {result.get('excluded_pairs', 0)})")
    if result["error"]:
        lines.append(f"  stopped: {result['error']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Running a match
# ----------------------------------------------------------------------------

def _book_info(book):
    if book is None:
        return None
    return dict(path=str(getattr(book, "path", "")), openings=len(book) if hasattr(book, "__len__") else None,
                seed=getattr(book, "seed", None), plies=getattr(book, "plies", None))


def run_fixed_match(pairings, *, tc_args, results_dir, book=None, plies=None, seed=None, concurrency=None,
                    threads=None, pin=True, slots=None, fastchess=None, run_name=None, extra_args=(),
                    pairs_per_job=4, quiet=False, status_every=30.0, report=True, label="MATCH",
                    time_control=None, meta=None, max_excluded=None, show_setup=True):
    """Play every pairing's opening pairs on the match runner; returns the result dict below.

    pairings: [Pairing or (first EngineSpec, second EngineSpec, pairs[, key])]. Engines are
      resolved here; an engine's option Threads > 1 is checked against what it declares,
      and a nodes limit against engines that ignore `go nodes`.
    tc_args: fastchess -each keys (utils.match_runner.tc_arguments / fastchess_match.limit_args).
    book: OpeningBook or a .pgn/.epd path (shuffled with `seed`), for pairings without their own.
    plies: book plies fastchess plays (default: the book's). extra_args: e.g. ["-maxmoves", "100"]
      + fastchess_match.tb_args(...).
    slots: planned slots (utils.match_runner.plan_slots) to play on, kept by the caller; by
      default slots of `threads` cpus (default: the most Threads of any engine) are planned
      with plan_slots(concurrency, threads=, pin=) and released when the match ends.
    results_dir / run_name: the run folder is results_dir/run_name (default: the time).
    quiet: no line per game. status_every: seconds between status lines (0: none). report: the
      final table. label: prefix of the console lines. time_control: label stored with the games
      (default from tc_args). meta: stored in config.json's meta. max_excluded: see the module doc.
    show_setup: print the machine, slot and fastchess lines of the run header; False for runs
      repeated on the same slots (CMA-ES generations after the first).

    Returns dict(run_id, run_dir, pairings={key: PentanomialSPRT snapshot (elo_diff, elo_low/high,
    nelo..., pentanomial, pentanomial_detail, candidate_wins/draws/losses, pairs, ...) + key, first,
    second, pairs_requested, complete, excluded_pairs, by_core={core: PairTally summary +
    seconds_per_game, mhz_mean}, meta}, pairs, pairs_requested, games, error, stop_reason,
    interrupted, elapsed_s, games_per_minute, slots, threads, time_control, voided_pairs,
    excluded_pairs, repinned_games, mixed_core_games, terminations, book). The candidate is each
    pairing's first engine. Raises MatchStopped / MatchInterrupted (with .result) when the match
    could not be completed.
    """
    pairings = normalize_pairings(pairings)
    tc_args = list(tc_args)
    # everything that can be refused is checked before cpus are claimed or folders created
    fastchess = find_fastchess(fastchess)
    if isinstance(book, (str, os.PathLike)):
        book = OpeningBook(book, plies=plies or 16, seed=seed)
    if book is None and any(p.book is None and p.pairs for p in pairings):
        raise ValueError("an opening book is needed: pass book=, or give every pairing its own")
    plies = plies or getattr(book if book is not None else pairings[0].book, "plies", None) or 16
    engines = list({id(e): e for p in pairings for e in (p.first, p.second)}.values())
    for engine in engines:
        engine.resolve()
    most = max(engine_threads(e) for e in engines)
    threads = threads or most
    if most > threads:
        raise ValueError(f"an engine searches with {most} threads, but games get {threads} cpu(s)")
    for engine in engines:
        if engine_threads(engine) > 1:
            require_threads(engine, engine_threads(engine))
    if any(arg.startswith("nodes=") for arg in tc_args):
        require_node_limit(engines)
    if slots is not None:
        if slots and slots[0].cpu is not None and min(len(s.cpus) for s in slots) < threads:
            raise ValueError(f"the slots have fewer than the {threads} cpus each game needs")
        return _play(pairings, slots, book, fastchess, tc_args, plies, threads, results_dir, run_name, extra_args,
                     pairs_per_job, quiet, status_every, report, label, time_control, meta, max_excluded, show_setup)
    slots = plan_slots(concurrency=concurrency, threads=threads, pin=pin)
    try:
        return _play(pairings, slots, book, fastchess, tc_args, plies, threads, results_dir, run_name, extra_args,
                     pairs_per_job, quiet, status_every, report, label, time_control, meta, max_excluded, show_setup)
    finally:
        release_slots(slots)


def _play(pairings, slots, book, fastchess, tc_args, plies, threads, results_dir, run_name, extra_args,
          pairs_per_job, quiet, status_every, report, label, time_control, meta, max_excluded, show_setup):
    run_dir = Path(results_dir) / (run_name or datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True)
    machine = environment.machine_info()
    pinned = bool(slots) and slots[0].cpu is not None
    hard = pinned and os.name == "nt"  # each job in a job object with the slot as its affinity limit
    mask = "job affinity limit + process mask" if hard else "process mask"
    engines = list({id(e): e for p in pairings for e in (p.first, p.second)}.values())
    time_control = time_control or tc_label(tc_args)
    run_meta = {**(meta or {}),
                **dict(run=run_dir.name, time_control=time_control, threads=threads, fastchess=fastchess,
                       machine=machine, slots=[vars(s) | {"label": s.label} for s in slots],
                       pinning={e.name: (f"Affinity option + {mask}" if e.pin_option and pinned else
                                         mask if pinned else "none") for e in engines},
                       book=_book_info(book), plies=plies, pairs_per_job=pairs_per_job, model="pentanomial-logistic",
                       pairings=[dict(key=p.key, pairs=p.pairs, book=_book_info(p.book), meta=p.meta,
                                      **{side: dict(name=e.name, cmd=e.cmd, args=list(e.args), options=e.options)
                                         for side, e in (("first", p.first), ("second", p.second))})
                                 for p in pairings])}
    (run_dir / "config.json").write_text(json.dumps(dict(meta=run_meta), indent=2, default=str), encoding="utf-8")

    total = sum(p.pairs for p in pairings)
    print_line(f"[{label}] {len(pairings)} pairing(s), {total:,} pairs ({2 * total:,} games), {time_control}"
               + (f", {threads} cpus per game" if threads > 1 else ""))
    for p in pairings[:8]:
        print_line(f"[{label}]   {p.key}: {p.pairs} pairs ({p.first.name} has White first)")
    if len(pairings) > 8:
        print_line(f"[{label}]   ... {len(pairings) - 8} more")
    books = {}  # one line per distinct book (a CMA-ES generation's pairings have copies of one shuffle)
    for p in pairings:
        b = p.book if p.book is not None else book
        if b is not None:
            info = _book_info(b)
            line = f"book {Path(info['path']).name}: {info['openings']} openings, {plies} plies, seed {info['seed']}"
            books[line] = books.get(line, 0) + 1
    for line, count in books.items():
        print_line(f"[{label}] {line}" + (f" ({count} pairings)" if count > 1 else ""))
    if show_setup:
        print_line(f"[{label}] machine {machine.get('cpu_model')}; {machine.get('topology')}")
        print_line(f"[{label}] {describe_slots(slots)}")
        for engine in engines:
            if pinned and not engine.pin_option:
                print_line(f"[{label}] note: {engine.name} has no Affinity option; its process mask keeps it on "
                           "the core")
        print_line(f"[{label}] fastchess {fastchess}")
    print_line(f"[{label}] results {run_dir}")

    controller = FixedMatchController(pairings, book, slots, run_dir, pairs_per_job=pairs_per_job, quiet=quiet,
                                      status_every=status_every, label=label, max_excluded=max_excluded)
    runner = MatchRunner(fastchess, slots, run_dir, tc_args, plies=plies, extra_args=list(extra_args))
    started = time.time()
    failure = None
    try:
        runner.run(controller)
    except KeyboardInterrupt:
        failure = "interrupted"
        print_line(f"\n[{label}] interrupted; writing the results so far")
    except BaseException as e:
        failure = f"{type(e).__name__}: {e}"
        raise
    finally:
        controller.close()
        result = controller.result(runner, time.time() - started, failure)
        result.update(threads=threads, time_control=time_control, book=_book_info(book))
        (run_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                              encoding="utf-8")
        if report:
            print_line(format_report(result, label))
    if failure == "interrupted":
        raise MatchInterrupted(result)
    if controller.stop_reason:
        raise MatchStopped(f"{label} stopped after {result['pairs']} of {result['pairs_requested']} pairs: "
                           f"{controller.stop_reason} (results in {run_dir})", result)
    return result
