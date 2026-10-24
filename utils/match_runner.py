"""Concurrent cutechess games, one game per CPU core.

Every concurrent game gets a *slot*: one logical CPU of one physical core. A
slot runs one cutechess-cli process at a time (a *job* of a few opening pairs,
concurrency 1), so the two engines of a game take turns on that CPU and never
compete with another game. Engines are pinned twice:

  1. the job's cutechess process is started with the slot's CPU as its affinity,
     which Windows and Linux pass on to the engines it starts (so engines
     without an Affinity option, e.g. Stockfish, stay on the slot), and
  2. engines that declare the UCI option "Affinity" also get
     `setoption name Affinity value <cpu>` (Tomahawk's own thread pinning).

The OS scheduler never runs a process outside its affinity mask, so a game is
always P-core vs P-core or E-core vs E-core, never mixed. As a safety net the
runner checks the engine processes at the start and end of every game; a game
where an engine was found off its core is flagged (`core_mixed`) and SPRT
excludes its pair. Every finished game carries its slot, CPU, core type and the
clock that CPU actually ran at while the game was played.

The only placement setting is how many cores to use (see plan_slots).

Controllers decide what to play (SPRT, SPSA, strength estimation):
    next_job(slot) -> JobSpec | None        what the idle slot plays next
    on_pair_start(job, pair_no) -> bool     False aborts the job
    on_game(game: GameResult)               each finished game
    on_job_end(job)                         the job's process exited or was killed
    on_warning(job, text)                   a cutechess warning (first occurrence)
    finished() -> bool                      stop everything now
    tick()                                  called several times per second
"""
from __future__ import annotations

import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from utils import cpu_topology, environment

STARTED = re.compile(r"Started game (\d+) of (\d+) \((.+?) vs (.+?)\)")
FINISHED = re.compile(r"Finished game (\d+) \((.+?) vs (.+?)\):\s*(\S+)(?:\s+\{(.*)\})?")
WARNING = re.compile(r"^(Warning|Critical|Fatal):\s*(.*)")


# ----------------------------------------------------------------------------
# Slots
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Slot:
    index: int
    cpu: int | None       # logical CPU; None = placement left to the OS
    core_type: str = "-"  # "P", "E", or "-" when not pinned
    cluster: int = -1     # L2 cluster of the core
    core: int = -1        # physical core (its first logical CPU)

    @property
    def label(self):
        return f"{self.core_type}{self.index:02d}" if self.cpu is None else f"{self.core_type}/cpu{self.cpu:02d}"


def _core_order(topo, cores):
    """Fastest type first; within a type, round-robin over L2 clusters (like the engine)."""
    ordered = []
    for speed in sorted({c.speed for c in cores}, reverse=True):
        clusters = {}
        for core in (c for c in cores if c.speed == speed):
            clusters.setdefault(core.cluster if core.cluster >= 0 else id(core), []).append(core)
        groups = list(clusters.values())
        for i in range(max(len(g) for g in groups)):
            ordered.extend(g[i] for g in groups if i < len(g))
    return ordered


def plan_slots(concurrency=None, topo=None):
    """One slot per concurrent game, placed automatically.

    `concurrency` is how many cores the run may use (default: every physical
    core but one, leaving room for the OS). Cores are filled fastest type
    first (P-cores, then E-cores spread over their L2 clusters), one logical
    CPU per physical core. Only a request above the number of physical cores
    also uses SMT siblings, i.e. puts two games on one P-core.
    """
    topo = topo or cpu_topology.topology()
    if not topo.supported:  # e.g. macOS: the OS places the games
        count = concurrency or max(1, (os.cpu_count() or 2) - 1)
        return [Slot(i, None, "-") for i in range(count)]
    allowed = set(topo.allowed) or {cpu for core in topo.cores for cpu in core.cpus}
    ordered = _core_order(topo, [c for c in topo.cores if allowed & set(c.cpus)])
    chosen = []
    for core in ordered:
        usable = [cpu for cpu in core.cpus if cpu in allowed]
        # Leave logical CPU 0 (interrupt handling on Windows) idle when its sibling is free.
        chosen.append((usable[-1] if 0 in usable and len(usable) > 1 else usable[0], core))
    physical = len(chosen)
    for core in ordered:
        for cpu in core.cpus:
            if cpu in allowed and all(cpu != c for c, _ in chosen):
                chosen.append((cpu, core))
    if concurrency is None:
        concurrency = max(1, physical - 1)
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if concurrency > len(chosen):
        raise ValueError(f"concurrency {concurrency} exceeds the {len(chosen)} logical CPUs available "
                         f"({topo.describe()})")
    return [Slot(i, cpu, topo.core_type(core), core.cluster, core.cpus[0])
            for i, (cpu, core) in enumerate(chosen[:concurrency])]


def describe_slots(slots):
    if not slots or slots[0].cpu is None:
        return f"{len(slots)} concurrent games (CPU placement left to the OS on this platform)"
    counts = {}
    for s in slots:
        counts[s.core_type] = counts.get(s.core_type, 0) + 1
    kinds = " + ".join(f"{n} on {t}-cores" for t, n in sorted(counts.items(), key=lambda kv: kv[0] != "P"))
    shared = len(slots) - len({s.core for s in slots})
    return (f"{len(slots)} concurrent games: {kinds} ({', '.join(f'cpu{s.cpu}' for s in slots)})"
            + (f", {shared} sharing a P-core via SMT" if shared > 0 else ""))


# ----------------------------------------------------------------------------
# Engines
# ----------------------------------------------------------------------------

_PROBES = {}


def probe_uci(cmd, args=(), timeout=20.0):
    """{'name': ..., 'options': {name: line}} from the engine's `uci` answer (cached)."""
    key = (os.path.abspath(cmd), tuple(args))
    if key in _PROBES:
        return _PROBES[key]
    proc = subprocess.Popen([cmd, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1,
                            cwd=os.path.dirname(os.path.abspath(cmd)) or None)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)],
                     daemon=True).start()
    info = {"name": None, "version": None, "options": {}}
    try:
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                raise RuntimeError(f"{cmd} did not answer 'uci' within {timeout:.0f}s")
            if line is None:
                raise RuntimeError(f"{cmd} exited before 'uciok'")
            line = line.strip()
            if line.startswith("id name "):
                info["name"] = line[8:].strip()
            elif line.startswith("id version "):
                info["version"] = line[11:].strip()
            elif line.startswith("option name "):
                name = re.match(r"option name (.+?) type ", line)
                if name:
                    info["options"][name.group(1)] = line
            elif line == "uciok":
                break
    finally:
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
    _PROBES[key] = info
    return info


@dataclass
class EngineSpec:
    name: str
    cmd: str
    args: tuple = ()
    options: dict = field(default_factory=dict)
    workdir: str | None = None
    pin_option: str | None = None  # UCI option that receives the slot's CPU; None = detect "Affinity"

    @classmethod
    def from_path(cls, name, path, options=None, **kwargs):
        """An engine executable, or a Python UCI engine script (run with this interpreter)."""
        path = str(path)
        if path.lower().endswith(".py"):
            script = os.path.abspath(path)
            return cls(name, sys.executable, (script,), dict(options or {}),
                       workdir=kwargs.pop("workdir", os.path.dirname(script)), **kwargs)
        return cls(name, path, (), dict(options or {}), **kwargs)

    def resolve(self):
        """Check the executable and detect the pinning option once."""
        self.cmd = os.path.abspath(self.cmd) if os.path.exists(self.cmd) else (shutil.which(self.cmd) or self.cmd)
        if not os.path.exists(self.cmd):
            raise FileNotFoundError(f"Engine not found: {self.cmd}")
        if self.pin_option is None:
            options = probe_uci(self.cmd, self.args)["options"]
            self.pin_option = next((o for o in options if o.lower() == "affinity"), "")
        return self

    def cutechess_args(self, slot):
        out = ["-engine", f"name={self.name}", f"cmd={self.cmd}",
               f"dir={self.workdir or os.path.dirname(self.cmd) or '.'}"]
        out += [f"arg={a}" for a in self.args]
        options = dict(self.options)
        if slot.cpu is not None and self.pin_option:
            options[self.pin_option] = str(slot.cpu)
        out += [f"option.{k}={v}" for k, v in options.items()]
        return out


# ----------------------------------------------------------------------------
# Jobs and results
# ----------------------------------------------------------------------------

@dataclass
class JobSpec:
    engines: tuple            # (first, second): `first` has White in the first game of each pair
    openings: list            # utils.openings.Opening, one per pair
    book: object = None       # utils.openings.OpeningBook that writes the job's book file
    meta: dict = field(default_factory=dict)


@dataclass
class GameResult:
    job: "Job"
    pair_no: int              # 1-based within the job
    game_no: int              # 1-based within the job
    white: str
    black: str
    result: str               # "1-0", "0-1", "1/2-1/2" or "*"
    reason: str
    started: float            # epoch seconds
    finished: float
    env: dict

    @property
    def opening(self):
        return self.job.spec.openings[self.pair_no - 1]

    @property
    def seconds(self):
        return self.finished - self.started

    def score_for(self, name):
        """1, 0.5, 0 for the engine called `name`, None if unfinished."""
        if self.result == "1/2-1/2":
            return 0.5
        if self.result not in ("1-0", "0-1"):
            return None
        return float((self.result == "1-0") == (self.white == name))


class Job:
    def __init__(self, job_id, slot, spec, directory):
        self.id, self.slot, self.spec = job_id, slot, spec
        self.directory = Path(directory)
        fmt = getattr(spec.book, "format", "pgn")
        self.book = self.directory / f"job{job_id:05d}_book.{fmt}"
        self.pgn = self.directory / f"job{job_id:05d}.pgn"
        self.log_path = self.directory / f"job{job_id:05d}.log"
        self.proc = None
        self.log = None
        self.started_at = time.time()
        self.game_start = {}
        self.games = []
        self.pairs_started = 0
        self.pids = set()
        self.killed = False
        self.kill_reason = None
        self.returncode = None
        self.repinned = {}       # game number -> [(pid, mask it had)] fixed at game start
        self.escapes = 0         # engine processes found outside the slot

    @property
    def pairs(self):
        return len(self.spec.openings)

    def pairs_finished(self):
        done = {}
        for g in self.games:
            done.setdefault(g.pair_no, 0)
            done[g.pair_no] += 1
        return sorted(p for p, n in done.items() if n == 2)


def _kill_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def find_cutechess(path=None):
    candidates = [path, os.environ.get("CUTECHESS_CLI"), shutil.which("cutechess-cli"),
                  r"C:\Program Files (x86)\Cute Chess\cutechess-cli.exe",
                  r"C:\Program Files\Cute Chess\cutechess-cli.exe",
                  str(Path(__file__).resolve().parent.parent / "libs" / "cutechess" / "build" / "cutechess-cli")]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "cutechess-cli not found. Pass --cutechess-cli, set CUTECHESS_CLI, or put it on PATH "
        f"(looked for: {', '.join(c for c in candidates if c)})")


class MatchRunner:
    """Keeps every slot busy with jobs from a controller and reports finished games."""

    def __init__(self, cutechess, slots, out_dir, tc_args, plies=16, extra_args=(), recover=True,
                 sample_clocks=True, clock_interval=1.0):
        self.cutechess = find_cutechess(cutechess)
        self.slots = list(slots)
        self.out_dir = Path(out_dir)
        self.jobs_dir = self.out_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.tc_args = list(tc_args)
        self.plies = plies
        self.extra_args = list(extra_args)
        self.recover = recover
        self.events = queue.Queue()
        self.running = {}   # slot index -> Job
        self.jobs = []
        self.next_id = 1
        self.stopping = False
        self.warnings = set()
        self.repinned_games = 0   # games whose engines had to be moved back onto their core at the start
        self.mixed_games = 0      # games where an engine was found off its core at the end
        self.sampler = (environment.ClockSampler(interval=clock_interval)
                        if sample_clocks and any(s.cpu is not None for s in self.slots) else None)
        self.games_pgn = self.out_dir / "games.pgn"
        self.cleanup = cpu_topology.KillOnCloseJob()

    # -- lifecycle ------------------------------------------------------------

    def run(self, controller):
        if self.sampler:
            self.sampler.start()
        try:
            while True:
                if not self.stopping:
                    for slot in self.slots:
                        if slot.index not in self.running:
                            spec = controller.next_job(slot)
                            if spec is not None:
                                self._launch(slot, spec)
                if not self.running:
                    break
                try:
                    job_id, stamp, line = self.events.get(timeout=0.1)
                except queue.Empty:
                    pass
                else:
                    self._handle(controller, job_id, stamp, line)
                    # Drain whatever else arrived without starving tick().
                    for _ in range(256):
                        try:
                            job_id, stamp, line = self.events.get_nowait()
                        except queue.Empty:
                            break
                        self._handle(controller, job_id, stamp, line)
                controller.tick()
                if not self.stopping and controller.finished():
                    self.stop("finished")
        except BaseException:
            self.stop("interrupted")
            raise
        finally:
            for job in list(self.running.values()):
                _kill_tree(job.proc)
            if self.sampler:
                self.sampler.stop()

    def stop(self, reason="stopped"):
        """Kill every running job; their unfinished games are reported through on_job_end."""
        self.stopping = True
        for job in list(self.running.values()):
            self.kill(job, reason)

    def kill(self, job, reason):
        if job.proc and job.proc.poll() is None:
            job.killed, job.kill_reason = True, reason
            _kill_tree(job.proc)

    # -- jobs -----------------------------------------------------------------

    def command(self, slot, spec, book_path, pgn_path):
        first, second = spec.engines
        fmt = "epd" if str(book_path).endswith(".epd") else "pgn"
        cmd = [self.cutechess, *first.cutechess_args(slot), *second.cutechess_args(slot),
               "-each", "proto=uci", *self.tc_args,
               "-games", "2", "-rounds", str(len(spec.openings)), "-repeat",
               "-concurrency", "1",
               "-openings", f"file={book_path}", f"format={fmt}", "order=sequential",
               f"plies={self.plies}",
               "-pgnout", str(pgn_path)]
        if self.recover:
            cmd.append("-recover")
        return cmd + self.extra_args

    def _launch(self, slot, spec):
        job = Job(self.next_id, slot, spec, self.jobs_dir)
        self.next_id += 1
        if spec.book is not None:
            spec.book.write(spec.openings, job.book)
        else:
            job.book.write_text("\n".join(o.text for o in spec.openings), encoding="utf-8")
        cmd = self.command(slot, spec, job.book, job.pgn)
        job.log = open(job.log_path, "w", encoding="utf-8")
        job.log.write(" ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        popen = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     text=True, bufsize=1, encoding="utf-8", errors="replace")
        if os.name == "nt":
            popen["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen["start_new_session"] = True
        with cpu_topology.launch_affinity([slot.cpu] if slot.cpu is not None else None):
            job.proc = subprocess.Popen(cmd, **popen)
        if os.name == "nt":
            self.cleanup.add(job.proc._handle)
        job.started_at = time.time()
        self.running[slot.index] = job
        self.jobs.append(job)
        threading.Thread(target=self._reader, args=(job,), daemon=True).start()
        return job

    def _reader(self, job):
        try:
            for line in job.proc.stdout:
                self.events.put((job.id, time.time(), line.rstrip("\r\n")))
        finally:
            self.events.put((job.id, time.time(), None))

    def _job(self, job_id):
        for job in self.running.values():
            if job.id == job_id:
                return job
        return None

    def environment(self, job, start, end):
        env = dict(slot=job.slot.index, slot_label=job.slot.label, cpu=job.slot.cpu,
                   core_type=job.slot.core_type, cluster=job.slot.cluster,
                   concurrency=len(self.running), pinned=job.slot.cpu is not None,
                   engine_pids=sorted(job.pids))
        env["power_source"], env["battery_percent"] = environment.power_status()
        if self.sampler:
            env.update(self.sampler.window(job.slot.cpu, start, end))
        return env

    def guard(self, job):
        """Check that every process of the job is still on its slot's CPU.

        Both engines of a game must run on one core so that a game is never
        P-core against E-core. The OS keeps processes inside their mask, so this
        only finds something if a process changed its own affinity. Returns
        [(pid, mask it had)] for processes found outside the slot; they are
        moved back immediately.
        """
        if job.slot.cpu is None or job.proc is None or job.proc.poll() is not None:
            return []  # nothing to check once cutechess (and with it the job's engines) has exited
        offenders = []
        job.pids.update(environment.child_pids(job.proc.pid))
        for pid in sorted(job.pids):
            mask = cpu_topology.process_affinity(pid)
            if mask is None or mask == [job.slot.cpu]:
                continue
            offenders.append((pid, mask))
            cpu_topology.set_process_affinity(pid, [job.slot.cpu])
        job.escapes += len(offenders)
        return offenders

    def _handle(self, controller, job_id, stamp, line):
        job = self._job(job_id)
        if job is None:
            return
        if line is None:
            job.returncode = job.proc.wait()
            job.log.close()
            self._append_pgn(job)
            del self.running[job.slot.index]
            controller.on_job_end(job)
            return
        job.log.write(line + "\n")
        match = STARTED.search(line)
        if match:
            number = int(match.group(1))
            job.game_start[number] = stamp
            fixed = self.guard(job)
            if fixed:
                job.repinned[number] = fixed
                self.repinned_games += 1
                if self.repinned_games == 1:
                    controller.on_warning(job, f"engine process(es) outside slot {job.slot.label} at the start of "
                                               f"game {number}: " + ", ".join(
                                                   f"pid {pid} on cpus {cpu_topology.format_cpu_list(mask)}"
                                                   for pid, mask in fixed) +
                                          "; moved back to its core (further cases are counted in the summary)")
            if number % 2 == 1:
                job.pairs_started += 1
                if controller.on_pair_start(job, (number + 1) // 2) is False:
                    self.kill(job, "not needed")
            return
        match = FINISHED.search(line)
        if match:
            number = int(match.group(1))
            start = job.game_start.get(number, job.started_at)
            env = self.environment(job, start, stamp)
            left = self.guard(job)
            # An engine found off its core at the end may have played part of the game on a
            # different core type: controllers must not count such a game.
            env["core_mixed"] = bool(left)
            env["repinned_at_start"] = len(job.repinned.get(number, ()))
            if left:
                self.mixed_games += 1
                controller.on_warning(job, f"engine process(es) left slot {job.slot.label} during game {number} "
                                           "(the game may mix core types and is excluded)")
            game = GameResult(job=job, pair_no=(number + 1) // 2, game_no=number,
                              white=match.group(2), black=match.group(3), result=match.group(4),
                              reason=(match.group(5) or "").strip(), started=start, finished=stamp,
                              env=env)
            job.games.append(game)
            controller.on_game(game)
            return
        match = WARNING.match(line)
        if match:
            text = match.group(2).strip()
            key = re.sub(r"\d+", "#", text)
            if key not in self.warnings:
                self.warnings.add(key)
                controller.on_warning(job, text)

    def _append_pgn(self, job):
        """Collect the job's games into games.pgn, tagged with where they were played.

        cutechess writes the games of a job (concurrency 1) in the order they were
        played, so the n-th game in the file is job game n (JobGame tag).
        """
        if not job.pgn.exists():
            return
        text = job.pgn.read_text(encoding="utf-8", errors="replace")
        counter = iter(range(1, 1_000_000))

        def tags(match):
            return (match.group(1) + f'[Job "{job.id}"]\n[JobGame "{next(counter)}"]\n'
                    f'[Slot "{job.slot.label}"]\n[CoreType "{job.slot.core_type}"]\n'
                    f'[Cpu "{job.slot.cpu if job.slot.cpu is not None else "-"}"]\n')
        tagged = re.sub(r'(?m)^(\[Event [^\n]*\]\n)', tags, text)
        with open(self.games_pgn, "a", encoding="utf-8") as out:
            out.write(tagged.rstrip("\n") + "\n\n")


def tc_arguments(depth=None, movetime=None, tc=None, timemargin=30):
    """cutechess time-control arguments for exactly one of depth / seconds per move / tc."""
    given = [x is not None for x in (depth, movetime, tc)]
    if sum(given) != 1:
        raise ValueError("Specify exactly one of depth, time per move, or time control")
    if depth is not None:
        return [f"depth={depth}", "tc=inf"]
    if movetime is not None:
        return [f"st={movetime}", f"timemargin={timemargin}"]
    return [f"tc={tc}"]


def print_line(text, stream=None):
    stream = stream or sys.stdout
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        stream.write(text.encode("ascii", "replace").decode() + "\n")
    stream.flush()
