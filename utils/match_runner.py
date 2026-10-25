"""Concurrent fastchess games, one game per CPU core.

Every concurrent game gets a *slot*: one logical CPU of one physical core (with
--threads N, N CPUs of one core type). A slot runs one fastchess process at a
time (a *job* of a few opening pairs, -concurrency 1), so the two engines of a
game take turns on that CPU and never compete with another game. Engines are
pinned three ways:

  1. the job's fastchess process is started with the slot's CPUs as its process
     affinity mask, which Windows and Linux pass on to the engines it starts (so
     engines without an Affinity option, e.g. Stockfish, stay on the slot),
  2. engines that declare the UCI option "Affinity" also get
     `option.Affinity=<cpus>` (Tomahawk's own thread pinning), and
  3. on Windows the fastchess process is started suspended and put in a job
     object of its own whose affinity limit is the slot (JOB_OBJECT_LIMIT_AFFINITY)
     before it runs; every engine it starts joins that job, and no process in it
     can widen its affinity (an old build pinning itself to the P-cores at
     startup keeps running on the slot). Linux has no unprivileged equivalent:
     there an engine can change its own affinity, and the guard below excludes
     its games.

fastchess's own -use-affinity is never used: on Windows it places engines with
CPU sets, which the process masks checked below cannot see.

The OS scheduler never runs a process outside its affinity mask, so a game is
always P-core vs P-core or E-core vs E-core, never mixed. As a safety net the
runner checks the engine processes at the start and end of every game, and
(without a job affinity limit) ten times a second while a job's engines start up
(fastchess starts them inside the job's first game). An engine found off its
slot before a game is moved back (the game counts as repinned); a game in which
one was found off its slot while playing is flagged (`core_mixed`) and SPRT
excludes its pair. Every finished game carries its slot, CPU, core type, whether
the slot was a hard limit (`affinity_limited`) and the clock that CPU actually
ran at while the game was played.

Slots are claimed with the lock files of utils.cpu_slots, so concurrent runs
(of any tool: they all play on this runner) never pin games to the same CPU.
The claims die with the process; release_slots() gives them back earlier.

fastchess numbers the games of a job 1, 2, ... in start order; games 2k+1 and
2k+2 play the job's k-th opening with colours reversed (always -games 2: fastchess
swaps -games and -rounds when -games > 2).

Controllers decide what to play (SPRT, SPSA, strength estimation, the fixed
matches of utils/fixed_match.py):
    next_job(slot) -> JobSpec | None        what the idle slot plays next
    on_pair_start(job, pair_no) -> bool     False aborts the job
    on_game(game: GameResult)               each finished game
    on_job_end(job)                         the job's process exited or was killed
    on_warning(job, text)                   a fastchess warning, the first of its kind (warning_key);
                                            MatchRunner.warning_count counts every one
    finished() -> bool                      stop everything now
    tick()                                  called several times per second

A finished job keeps its results, not its process: the fastchess process, its
output pipe and its job object are released when the job ends, so a long run
(thousands of jobs) holds no handles for the jobs it has played.
"""
from __future__ import annotations

import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from utils import cpu_slots, cpu_topology, environment
from utils.fastchess_match import LIBS_FASTCHESS

STARTED = re.compile(r"Started game (\d+) of (\d+) \((.+?) vs (.+?)\)")
FINISHED = re.compile(r"Finished game (\d+) \((.+?) vs (.+?)\):\s*(\S+)(?:\s+\{(.*)\})?")
# fastchess writes "Warning; ..." (and "Warning: ..." for a few configuration notes)
WARNING = re.compile(r"^(Warning|Critical|Fatal)[:;]\s*(.*)")
UCI_MOVE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")
STARTUP_WATCH_S = 2.0  # how long the engines of a job's first game are watched while they start
# fastchess has exited but the job's output is still open (an engine it started outlives it and holds
# the inherited pipe): after EXIT_GRACE_S what is left of the job is killed, after END_TIMEOUT_S the
# job is ended without waiting for its output any longer
EXIT_GRACE_S = 2.0
END_TIMEOUT_S = 10.0
CREATE_SUSPENDED = 0x00000004  # Windows process creation flag


def warning_key(text):
    """The kind of a fastchess warning: its text with moves and numbers replaced.

    "Bestmove does not match beginning of last PV - move e2e4 from Stockfish" and
    "... - move g8f6 from Stockfish" are one kind, reported to the controller once.
    """
    return re.sub(r"\d+", "#", UCI_MOVE.sub("<move>", text))


# Substrings of fastchess's termination texts (libs/fastchess/app/src/matchmaking/match/match.hpp,
# e.g. "White mates", "Draw by 3-fold repetition", "Black loses on time (100ms overrun)") -> class.
# Older cutechess texts ("resigns", "maximal game length") are kept for transcripts of earlier runs.
# fastchess reports the -maxmoves limit as "Draw by adjudication", like -draw adjudication.
TERMINATIONS = (
    ("syzygytb", "tablebase"), ("loses on time", "time forfeit"), ("disconnects", "crash"),
    ("connection stalls", "stall"), ("illegal move", "illegal move"), ("resigns", "resignation"),
    ("mates", "mate"), ("stalemate", "stalemate"), ("3-fold", "repetition"), ("fifty", "50-move rule"),
    ("insufficient", "insufficient material"), ("maximal game length", "move limit"),
    ("adjudication", "adjudication"), ("interrupted", "interrupted"),
)
FAULTS = ("time forfeit", "crash", "stall", "illegal move")


def termination_class(reason):
    """Coarse class of a fastchess termination text (GameResult.reason), and the side at fault if any.

    The classes are what games.jsonl's `termination` holds and data/etl/match_games.py stores.
    """
    text = (reason or "").lower()
    side = "white" if text.startswith("white") else "black" if text.startswith("black") else None
    for key, label in TERMINATIONS:
        if key in text:
            return label, (side if label in FAULTS else None)
    return (text or "unknown"), None


# ----------------------------------------------------------------------------
# Slots
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Slot:
    index: int
    cpu: int | None       # (first) logical CPU; None = placement left to the OS
    core_type: str = "-"  # "P", "E", or "-" when not pinned
    cluster: int = -1     # L2 cluster of the (first) core
    core: int = -1        # physical core (its first logical CPU)
    cpus: tuple = ()      # every logical CPU of the slot (several for multi-threaded games)

    def __post_init__(self):
        if not self.cpus and self.cpu is not None:
            object.__setattr__(self, "cpus", (self.cpu,))

    @property
    def label(self):
        if self.cpu is None:
            return f"{self.core_type}{self.index:02d}"
        if len(self.cpus) > 1:
            return f"{self.core_type}/cpu{','.join(map(str, self.cpus))}"
        return f"{self.core_type}/cpu{self.cpu:02d}"


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


def plan_slots(concurrency=None, topo=None, threads=1, pin=True, claim=True):
    """One slot per concurrent game, placed automatically.

    `concurrency` is how many games run at once (default: every physical core
    but one, leaving room for the OS, divided by `threads`). Cores are filled
    fastest type first (P-cores, then E-cores spread over their L2 clusters),
    one logical CPU per physical core; a slot of `threads` CPUs takes that many
    cores of one type. Only a request above the number of physical cores also
    uses SMT siblings, i.e. puts two games on one P-core.

    CPUs held by another run (utils.cpu_slots lock files) are skipped, and a core
    with a CPU held elsewhere is used only after every free core; with claim=True
    the slots' CPUs are claimed for this process (release_slots). pin=False, or a
    platform without affinity (macOS), leaves placement to the OS.
    """
    if threads < 1:
        raise ValueError("threads must be at least 1")
    topo = topo or cpu_topology.topology()
    if not pin or not topo.supported:  # e.g. macOS: the OS places the games
        count = concurrency or max(1, ((os.cpu_count() or 2) - 1) // threads)
        if count < 1:
            raise ValueError("concurrency must be at least 1")
        return [Slot(i, None, "-") for i in range(count)]
    allowed = set(topo.allowed) or {cpu for core in topo.cores for cpu in core.cpus}
    cores = [c for c in topo.cores if allowed & set(c.cpus)]
    requested = concurrency
    for _ in range(5):  # another run may claim a cpu between our check and our claim: plan again
        busy = {cpu for cpu in sorted(allowed) if cpu_slots.held_elsewhere(cpu)} if claim else set()
        free = allowed - busy
        ordered = _core_order(topo, [c for c in cores if free & set(c.cpus)])
        chosen = []
        for core in ordered:
            if busy & set(core.cpus):
                continue  # another run plays on this core: only its free sibling, and only later
            usable = [cpu for cpu in core.cpus if cpu in free]
            # Leave logical CPU 0 (interrupt handling on Windows) idle when its sibling is free.
            chosen.append((usable[-1] if 0 in usable and len(usable) > 1 else usable[0], core))
        whole = len(chosen)
        for core in ordered:
            for cpu in core.cpus:
                if cpu in free and all(cpu != c for c, _ in chosen):
                    chosen.append((cpu, core))
        # slots of `threads` cpus, each of one core type, in that order
        groups, pending = [], {}
        for position, (cpu, core) in enumerate(chosen):
            kind = topo.core_type(core)
            pending.setdefault(kind, []).append((position, cpu, core))
            if len(pending[kind]) == threads:
                groups.append(pending.pop(kind))
        count = requested
        if count is None:
            own_cores = sum(all(p < whole for p, _, _ in g) for g in groups)
            count = max(1, min(own_cores, (len(cores) - 1) // threads))
        if count < 1:
            raise ValueError("concurrency must be at least 1")
        if count > len(groups):
            held = f"; other runs hold cpus {cpu_topology.format_cpu_list(busy)}" if busy else ""
            raise ValueError(f"{count} concurrent games x {threads} cpu(s) exceed the {len(free)} logical CPUs "
                             f"available ({topo.describe()}{held}): lower --concurrency or --threads, or --no-pin")
        picked = groups[:count]
        if claim:
            before = cpu_slots.claimed()
            taken, lost = [], False
            for _, cpu, _ in (entry for group in picked for entry in group):
                if not cpu_slots.claim(cpu):
                    lost = True
                    break
                taken.append(cpu)
            if lost:  # another run took a cpu meanwhile: give back what this attempt took, plan again
                for cpu in taken:
                    if cpu not in before:
                        cpu_slots.unclaim(cpu)
                continue
        return [Slot(i, group[0][1], topo.core_type(group[0][2]), group[0][2].cluster, group[0][2].cpus[0],
                     tuple(cpu for _, cpu, _ in group))
                for i, group in enumerate(picked)]
    raise ValueError("could not claim cpus for the games: other runs are starting at the same time; try again")


def release_slots(slots):
    """Give back the cpu claims of `slots` (also done when the process exits)."""
    for slot in slots or ():
        for cpu in slot.cpus:
            cpu_slots.unclaim(cpu)


def describe_slots(slots, topo=None):
    if not slots or slots[0].cpu is None:
        return f"{len(slots)} concurrent games (CPU placement left to the OS)"
    counts = {}
    for s in slots:
        counts[s.core_type] = counts.get(s.core_type, 0) + 1
    kinds = " + ".join(f"{n} on {t}-cores" for t, n in sorted(counts.items(), key=lambda kv: kv[0] != "P"))
    topo = topo or cpu_topology.topology()
    core_of = {cpu: core.cpus[0] for core in topo.cores for cpu in core.cpus}
    cpus = [cpu for s in slots for cpu in s.cpus]
    shared = len(cpus) - len({core_of.get(cpu, cpu) for cpu in cpus})
    return (f"{len(slots)} concurrent games: {kinds} ({', '.join('cpu' + '+'.join(map(str, s.cpus)) for s in slots)})"
            + (f", {shared} cpu(s) sharing a P-core via SMT" if shared > 0 else ""))


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
    reader = threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)],
                              daemon=True)
    reader.start()
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
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        try:
            proc.stdin.close()
        except OSError:
            pass
        # the reader ends at EOF once the engine has exited; only then is its pipe closed
        reader.join(timeout=2)
        if not reader.is_alive():
            proc.stdout.close()
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
        """Check the executable, that fastchess can start it, and detect the pinning option once."""
        self.cmd = os.path.abspath(self.cmd) if os.path.exists(self.cmd) else (shutil.which(self.cmd) or self.cmd)
        if not os.path.exists(self.cmd):
            raise FileNotFoundError(f"Engine not found: {self.cmd}")
        fastchess_path(self.cmd)  # a path fastchess cannot start fails here, before any game
        if self.pin_option is None:
            options = probe_uci(self.cmd, self.args)["options"]
            self.pin_option = next((o for o in options if o.lower() == "affinity"), "")
        return self

    def fastchess_args(self, slot, shared=False):
        """The `-engine` block of one fastchess job on `slot`.

        A pinned slot's CPUs go to the engine's Affinity option; unpinned
        multi-threaded engines sharing the machine get Affinity=none (auto would
        pin every game's threads to the same CPUs). `dir` (the engine's working
        directory) may contain spaces; the command is given by fastchess_path.
        """
        out = ["-engine", f"name={self.name}", f"cmd={fastchess_path(self.cmd)}",
               f"dir={self.workdir or os.path.dirname(self.cmd) or '.'}"]
        if self.args:  # one string: fastchess appends it to the command line as is
            out.append("args=" + (subprocess.list2cmdline(self.args) if os.name == "nt" else shlex.join(self.args)))
        options = dict(self.options)
        if self.pin_option:
            if slot.cpu is not None:
                options[self.pin_option] = ",".join(map(str, slot.cpus))
            elif shared and int(options.get("Threads", 1)) > 1:
                options.setdefault(self.pin_option, "none")
        out += [f"option.{k}={v}" for k, v in options.items()]
        return out


def fastchess_path(path):
    """`path` in a form fastchess can start on Windows.

    fastchess puts the engine path unquoted at the start of the command line it
    gives CreateProcess (engine/process/process_win.hpp: "{dir/cmd} {args}"), and
    quoting `cmd` does not help (it is joined to `dir` first). A path with a space
    is therefore ambiguous: for C:\\a b\\engine.exe Windows runs C:\\a.exe if that
    exists, and otherwise the engine sees a split command line. The 8.3 short form
    of the path has no spaces; where the volume has no short names this raises
    ValueError. Unchanged on other platforms (fastchess passes the path as argv[0]).
    """
    path = str(path)
    if os.name != "nt" or " " not in path:
        return path
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    k32.GetShortPathNameW.restype = ctypes.c_uint32
    buffer = ctypes.create_unicode_buffer(32768)
    length = k32.GetShortPathNameW(path, buffer, len(buffer))
    short = buffer.value if 0 < length < len(buffer) else ""
    if not short or " " in short:
        raise ValueError(f"fastchess cannot start an engine from a path with a space on Windows: {path} "
                         "(this volume has no 8.3 short names to avoid it); move the engine to a folder "
                         "without spaces")
    return short


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
        self.config = self.directory / f"job{job_id:05d}_fastchess.json"  # fastchess's own state file
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
        self.watch = None        # (game number, until) while the job's engines start up
        self.watched_at = 0.0
        self.tainted = set()     # game numbers in which an engine was found off the slot while playing
        self.container = None    # its Windows job object (kill on close, slot affinity limit)
        self.limited = False     # the OS enforces the slot: no process of the job can leave it
        self.output_ended = False  # the reader reached the end of fastchess's output
        self.exited_at = None    # (monotonic) when fastchess was first seen exited with its output still open
        self.leftovers_killed = False
        self.end_forced = False

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


def find_fastchess(path=None):
    """The fastchess executable: `path`, $FASTCHESS, the libs/fastchess build, or fastchess on PATH.

    The order of utils.fastchess_match.FASTCHESS, the tools' --fastchess default."""
    candidates = [path, os.environ.get("FASTCHESS"), str(LIBS_FASTCHESS), shutil.which("fastchess")]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "fastchess not found. Pass --fastchess, set FASTCHESS, build libs/fastchess, or put it on PATH "
        f"(looked for: {', '.join(str(c) for c in candidates if c)})")


class MatchRunner:
    """Keeps every slot busy with jobs from a controller and reports finished games."""

    def __init__(self, fastchess, slots, out_dir, tc_args, plies=16, extra_args=(), recover=True,
                 sample_clocks=True, clock_interval=1.0):
        self.fastchess = find_fastchess(fastchess)
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
        self.jobs = []      # every job, finished ones without their process (see _handle)
        self.next_id = 1
        self.stopping = False
        self.warnings = {}  # warning_key -> how many warnings of that kind the jobs printed
        self.repinned_games = 0   # games whose engines had to be moved back onto their core at the start
        self.mixed_games = 0      # games where an engine was found off its core at the end
        self.sampler = (environment.ClockSampler(interval=clock_interval)
                        if sample_clocks and any(s.cpu is not None for s in self.slots) else None)
        self.games_pgn = self.out_dir / "games.pgn"

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
                self._watch_startups()
                self._check_exits(controller)
                controller.tick()
                if not self.stopping and controller.finished():
                    self.stop("finished")
        except BaseException:
            self.stop("interrupted")
            raise
        finally:
            for job in list(self.running.values()):
                _kill_tree(job.proc)
                self._kill_leftovers(job)
            if self.sampler:
                self.sampler.stop()

    def stop(self, reason="stopped"):
        """Kill every running job; their unfinished games are reported through on_job_end."""
        self.stopping = True
        for job in list(self.running.values()):
            self.kill(job, reason)

    def kill(self, job, reason):
        if job.proc is None:
            return
        if job.proc.poll() is None:
            job.killed, job.kill_reason = True, reason
            _kill_tree(job.proc)
        elif not job.output_ended:
            self._kill_leftovers(job)  # fastchess is gone but something it started holds its output open

    def warning_count(self, text=""):
        """How many warnings the jobs printed that start like `text` (of its kind: see warning_key)."""
        prefix = warning_key(text)
        return sum(count for key, count in self.warnings.items() if key.startswith(prefix))

    def _kill_leftovers(self, job):
        """Kill what the job's fastchess left behind; the engines it started hold its output pipe.

        On Windows closing the job object kills every process still in it; on POSIX the
        job's process group outlives fastchess while an engine of it runs. Without a job
        object on Windows nothing is known to kill: _check_exits ends the job anyway.
        """
        if job.container:
            job.container.close()
        elif os.name != "nt" and job.proc is not None and not job.output_ended:
            try:
                os.killpg(job.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _check_exits(self, controller):
        """Jobs whose fastchess has exited while their output stays open.

        Engines inherit fastchess's output, so an engine that outlives fastchess (it
        crashed or was killed, and the engine ignores the end of its input) keeps it
        open: the reader would never see its end and the job would run for ever.
        After EXIT_GRACE_S such leftovers are killed, which ends the output; should it
        still be open after END_TIMEOUT_S the job is ended without it.
        """
        now = time.monotonic()
        for job in list(self.running.values()):
            if job.output_ended or job.proc.poll() is None:
                continue
            if job.exited_at is None:
                job.exited_at = now
                continue
            if now - job.exited_at > EXIT_GRACE_S and not job.leftovers_killed:
                job.leftovers_killed = True
                self._kill_leftovers(job)
            if now - job.exited_at > END_TIMEOUT_S and not job.end_forced:
                job.end_forced = True
                controller.on_warning(job, f"fastchess of job {job.id} exited {END_TIMEOUT_S:.0f}s ago but its "
                                           "output is still open; the job is ended without it")
                self.events.put((job.id, time.time(), None))

    # -- jobs -----------------------------------------------------------------

    def command(self, slot, spec, book_path, pgn_path, config_path=None):
        first, second = spec.engines
        fmt = "epd" if str(book_path).endswith(".epd") else "pgn"
        shared = len(self.slots) > 1
        config_path = config_path or Path(pgn_path).with_suffix(".json")
        cmd = [self.fastchess,
               # fastchess saves its state on exit: keep it with the job, not in the cwd
               "-config", f"outname={config_path}",
               *first.fastchess_args(slot, shared), *second.fastchess_args(slot, shared),
               "-each", "proto=uci", *self.tc_args,
               # always 2 games per round (one opening, colours reversed); fastchess swaps
               # -games and -rounds when -games > 2
               "-games", "2", "-rounds", str(len(spec.openings)),
               "-concurrency", "1",
               "-openings", f"file={book_path}", f"format={fmt}", "order=sequential",
               f"plies={self.plies}",
               "-pgnout", f"file={pgn_path}"]
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
        cmd = self.command(slot, spec, job.book, job.pgn, job.config)
        job.log = open(job.log_path, "w", encoding="utf-8")
        job.log.write(" ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        popen = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     text=True, bufsize=1, encoding="utf-8", errors="replace", cwd=str(self.jobs_dir))
        if os.name == "nt":
            # suspended until it is in its job object, so it and every engine it starts are inside
            popen["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED
        else:
            popen["start_new_session"] = True
        with cpu_topology.launch_affinity(list(slot.cpus) if slot.cpu is not None else None):
            job.proc = subprocess.Popen(cmd, **popen)
        if os.name == "nt":
            try:
                # kill on close, and the slot as a hard affinity limit nothing in the job can widen
                job.container = cpu_topology.KillOnCloseJob(list(slot.cpus) if slot.cpu is not None else None)
                job.limited = job.container.add(job.proc._handle) and job.container.affinity is not None
            except OSError:
                job.container, job.limited = None, False
            try:
                resumed = cpu_topology.resume_process(job.proc.pid)
            except OSError:
                resumed = 0
            if not resumed:
                _kill_tree(job.proc)
                job.proc.wait()
                job.log.close()
                if job.container:
                    job.container.close()
                raise RuntimeError(f"could not start fastchess for job {job.id}: it did not resume")
        job.started_at = time.time()
        self.running[slot.index] = job
        self.jobs.append(job)
        threading.Thread(target=self._reader, args=(job,), daemon=True).start()
        return job

    def _reader(self, job):
        stream = job.proc.stdout  # the job may drop its process before this thread ends (_check_exits)
        try:
            for line in stream:
                self.events.put((job.id, time.time(), line.rstrip("\r\n")))
        finally:
            job.output_ended = True
            try:
                stream.close()  # one pipe per job: never left open for the rest of a long run
            except OSError:
                pass
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
                   affinity_limited=job.limited, engine_pids=sorted(job.pids))
        if len(job.slot.cpus) > 1:
            env["cpus"] = list(job.slot.cpus)
        env["power_source"], env["battery_percent"] = environment.power_status()
        if self.sampler:
            env.update(self.sampler.window(list(job.slot.cpus) if len(job.slot.cpus) > 1 else job.slot.cpu,
                                           start, end))
        return env

    def guard(self, job):
        """Check that every process of the job is still on its slot's CPUs.

        Both engines of a game must run on one core type so that a game is never
        P-core against E-core. The OS keeps processes inside their mask, so this
        only finds something if a process changed its own affinity. Returns
        [(pid, mask it had)] for processes found outside the slot; they are
        moved back immediately. Only the job's current processes are checked
        (an engine restarted by -recover gets a new pid; an old pid may since
        belong to an unrelated process).
        """
        if job.slot.cpu is None or job.proc is None or job.proc.poll() is not None:
            return []  # nothing to check once fastchess (and with it the job's engines) has exited
        offenders = []
        allowed = set(job.slot.cpus)
        current = environment.child_pids(job.proc.pid)
        job.pids.update(current)
        for pid in sorted(current):
            mask = cpu_topology.process_affinity(pid)
            if mask is None or (mask and set(mask) <= allowed):
                continue
            offenders.append((pid, mask))
            cpu_topology.set_process_affinity(pid, list(job.slot.cpus))
        job.escapes += len(offenders)
        return offenders

    def _repinned(self, controller, job, number, fixed):
        """Engine processes moved back onto the slot before (or as) game `number` started."""
        if number not in job.repinned:
            self.repinned_games += 1
        job.repinned.setdefault(number, []).extend(fixed)
        if self.repinned_games == 1 and len(job.repinned[number]) == len(fixed):
            controller.on_warning(job, f"engine process(es) outside slot {job.slot.label} at the start of "
                                       f"game {number}: " + ", ".join(
                                           f"pid {pid} on cpus {cpu_topology.format_cpu_list(mask or [])}"
                                           for pid, mask in fixed) +
                                  "; moved back to its core (further cases are counted in the summary)")

    def _watch_startups(self):
        """Check the engines of jobs whose first game just started, ten times a second.

        Only for jobs without a job-object affinity limit (Linux, or where Windows
        refused the limit). fastchess starts the engines inside a job's first game, so
        the check at its "Started game" line sees none. An engine that moves itself
        off its slot while starting up (e.g. an old build pinning itself to fixed
        cores) may already have searched there: it is moved back at once, so it does
        not disturb other slots' games for the rest of the game, and the game is
        excluded like any game in which an engine was found off its slot
        (core_mixed). Later games of the job are checked before they start, as usual.
        """
        now = time.time()
        for job in list(self.running.values()):
            if job.watch is None:
                continue
            number, deadline = job.watch
            if now > deadline:
                job.watch = None
                continue
            if now - job.watched_at < 0.1:
                continue
            job.watched_at = now
            if self.guard(job):
                job.tainted.add(number)

    def _handle(self, controller, job_id, stamp, line):
        job = self._job(job_id)
        if job is None:
            return
        if line is None:
            job.returncode = job.proc.wait()
            if job.container:
                job.container.close()  # fastchess has exited: anything it left behind dies with the job
            job.log.close()
            self._append_pgn(job)
            del self.running[job.slot.index]
            job.proc = None  # releases the process handle; the reader closed the pipe
            controller.on_job_end(job)
            return
        job.log.write(line + "\n")
        match = STARTED.search(line)
        if match:
            number = int(match.group(1))
            job.game_start[number] = stamp
            fixed = self.guard(job)
            if fixed:
                self._repinned(controller, job, number, fixed)
            if job.watch is None and not job.games and not job.limited:
                # fastchess starts a job's engines inside its first game, after this line: keep
                # checking them while they start up (see _watch_startups); not needed where the
                # job object's affinity limit already keeps every process on the slot
                job.watch = (number, stamp + STARTUP_WATCH_S)
            if number % 2 == 1:
                job.pairs_started += 1
                if controller.on_pair_start(job, (number + 1) // 2) is False:
                    self.kill(job, "not needed")
            return
        match = FINISHED.search(line)
        if match:
            number = int(match.group(1))
            start = job.game_start.get(number, job.started_at)
            if job.watch and job.watch[0] == number:
                job.watch = None
            left = self.guard(job)  # first, so that the game's environment lists every engine process
            env = self.environment(job, start, stamp)
            # An engine found off its core during the game (the start-up watch) or at its end may
            # have played part of the game on a different core type: controllers must not count it.
            mixed = bool(left) or number in job.tainted
            env["core_mixed"] = mixed
            env["repinned_at_start"] = len(job.repinned.get(number, ()))
            if mixed:
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
            key = warning_key(text)
            self.warnings[key] = self.warnings.get(key, 0) + 1
            if self.warnings[key] == 1:
                controller.on_warning(job, text)

    def _append_pgn(self, job):
        """Collect the job's games into games.pgn, tagged with where they were played.

        fastchess writes no job numbering of its own: its [Round] is the job's
        opening pair (1-based) and the pair's first game has the first engine as
        White, so a game is job game 2 * (Round - 1) + (1 if White is the first
        engine else 2) (JobGame tag). With -concurrency 1 the file is also in play
        order, which numbers games without a usable Round.
        """
        if not job.pgn.exists():
            return
        text = job.pgn.read_text(encoding="utf-8", errors="replace")
        first = job.spec.engines[0].name
        cpu = ",".join(map(str, job.slot.cpus)) if job.slot.cpu is not None else "-"
        out = []
        for position, chunk in enumerate((c for c in re.split(r"(?m)^(?=\[Event )", text) if c.strip()), 1):
            number = position
            pair = re.search(r'(?m)^\[Round "(\d+)"\]', chunk)
            white = re.search(r'(?m)^\[White "(.*)"\]', chunk)
            if pair and white:
                number = 2 * (int(pair.group(1)) - 1) + (1 if white.group(1) == first else 2)
            tags = (f'[Job "{job.id}"]\n[JobGame "{number}"]\n[Slot "{job.slot.label}"]\n'
                    f'[CoreType "{job.slot.core_type}"]\n[Cpu "{cpu}"]\n')
            out.append(re.sub(r"^(\[Event [^\n]*\]\n)", lambda m: m.group(1) + tags, chunk, count=1).rstrip("\n"))
        if out:
            with open(self.games_pgn, "a", encoding="utf-8") as f:
                f.write("\n\n".join(out) + "\n\n")


def tc_arguments(depth=None, movetime=None, tc=None, nodes=None, timemargin=30):
    """fastchess time-control keys for exactly one of depth / seconds per move / tc / nodes per move."""
    given = [x is not None for x in (depth, movetime, tc, nodes)]
    if sum(given) != 1:
        raise ValueError("Specify exactly one of depth, time per move, time control, or nodes per move")
    if depth is not None:
        return [f"depth={depth}", "tc=inf"]
    if movetime is not None:
        return [f"st={movetime}", f"timemargin={timemargin}"]
    if nodes is not None:
        return [f"nodes={nodes}", "tc=inf"]
    return [f"tc={tc}"]


def print_line(text, stream=None):
    stream = stream or sys.stdout
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        stream.write(text.encode("ascii", "replace").decode() + "\n")
    stream.flush()
