"""Give every game its own CPUs, in the engine's own placement order.

Tomahawk (affinity.cpp) reports on `uci` the order in which `Affinity=auto` uses the CPUs:

    info string cpu topology 16 logical cpus: 4 performance + 8 efficiency cores (smt); auto order 0,2,4,6,8,12,...

one logical CPU per physical core first, fastest cores first, SMT siblings last. A slot is
`threads` consecutive CPUs of that order, and each concurrent game gets a slot of its own:
both engines of the game run only on the slot's CPUs and take turns on them, and the engine
places its search threads inside the slot (it never leaves the CPUs it was given).

  threads == 1   one fastchess process with -use-affinity <one cpu per game slot>; fastchess
                 keeps each slot's engines on its cpu
  threads  > 1   fastchess 1.8.2 pins only one cpu per game, so every slot runs its own
                 fastchess process (-concurrency 1) started with the slot as its affinity,
                 which the engines it starts inherit

Pinning needs Windows or Linux, and engines that report their order: builds without the
Affinity option don't, and pin themselves at startup, which would defeat the slots.

Slots only use cpus this process may run on (start /affinity, taskset restrict a run), and
each pinned run claims its cpus with a lock file, so concurrent runs pin to different cpus;
the locks are released with release() or when the process exits.
"""
import os
import re
import subprocess
import sys
import tempfile
from collections import namedtuple
from contextlib import contextmanager

Topology = namedtuple("Topology", "order cores")  # placement order of all logical cpus, physical cores
TOPOLOGY = re.compile(r"cpu topology \d+ logical cpus: (?:(\d+) performance \+ (\d+) efficiency cores|(\d+) cores)"
                      r".*?; auto order ([\d,\-]+)")
SUPPORTED = sys.platform == "win32" or sys.platform.startswith("linux")
_claims = {}  # cpu -> lock file descriptor held by this process


def parse_cpus(text):
    """'0-3,8,6' -> [0, 1, 2, 3, 8, 6], the format affinity.cpp prints."""
    cpus = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        cpus += range(int(first), int(last or first) + 1)
    return cpus


def format_cpus(cpus):
    return ",".join(map(str, cpus))


def engine_topology(engine, timeout=15):
    """The engine's Topology from its `uci` reply, or None if it doesn't report one."""
    engine = os.path.abspath(engine)
    try:
        reply = subprocess.run([engine], input="uci\nquit\n", capture_output=True, text=True,
                               timeout=timeout, cwd=os.path.dirname(engine)).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in reply.splitlines():
        match = TOPOLOGY.search(line)
        if match:
            performance, efficiency, cores, order = match.groups()
            return Topology(parse_cpus(order), int(cores) if cores else int(performance) + int(efficiency))
    return None


def plan(engines, threads, games, enabled=True, log=print):
    """CPUs of each of `games` concurrent game slots of `threads` cpus, or None to leave games unpinned.

    The slots' cpus are claimed for this process until release().
    """
    if not enabled:
        return None
    if not SUPPORTED:
        log(f"[AFFINITY] not pinning games: no cpu affinity on {sys.platform}")
        return None
    topology = None
    for engine in dict.fromkeys(os.path.abspath(e) for e in engines):
        found = engine_topology(engine)
        if found is None:
            log(f"[AFFINITY] warning: not pinning games: {engine} does not report its cpu order. A build "
                "without the Affinity option pins itself to fixed cpus, so games are not isolated and the "
                "two sides may not get equal cpus.")
            return None
        topology = topology or found
    need = threads * games
    allowed = _allowed_cpus()
    order = [cpu for cpu in topology.order if cpu in allowed]
    if need > len(order):
        raise ValueError(f"{games} concurrent games x {threads} thread(s) need {need} cpus, but this process may use "
                         f"{len(order)}: lower the concurrency or threads, or run unpinned (--no-pin)")
    held = set(_claims)
    claimed = []
    for cpu in order:
        if len(claimed) == need:
            break
        if _claim(cpu):
            claimed.append(cpu)
    if len(claimed) < need:
        for cpu in claimed:
            if cpu not in held:
                _unclaim(cpu)
        raise ValueError(f"{games} concurrent games x {threads} thread(s) need {need} cpus, but other pinned runs "
                         f"hold all but {len(claimed)}: wait for them, lower the concurrency, or run unpinned (--no-pin)")
    if any(topology.order.index(cpu) >= topology.cores for cpu in claimed):
        log("[AFFINITY] warning: some game slots use second hyperthreads (SMT), sharing a core with another slot")
    slots = [claimed[i * threads:(i + 1) * threads] for i in range(games)]
    log("[AFFINITY] game slots: " + " | ".join(format_cpus(s) for s in slots))
    return slots


def release():
    """Give back the cpus this process claimed for its game slots."""
    for cpu in list(_claims):
        _unclaim(cpu)


def physical_slots(engine, threads):
    """How many slots of `threads` cpus fit on the physical cores this process may use (no SMT sharing)."""
    topology = engine_topology(engine) if SUPPORTED else None
    if topology is None:
        return 1
    allowed = _allowed_cpus()
    cores = [cpu for cpu in topology.order[:topology.cores] if cpu in allowed]
    return max(1, len(cores) // threads)


def _lock_path(cpu):
    user = f"-{os.getuid()}" if hasattr(os, "getuid") else ""  # %TEMP% is already per user on Windows
    folder = os.path.join(tempfile.gettempdir(), f"san-jacinto-cpus{user}")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"cpu{cpu}.lock")


def _claim(cpu):
    """Lock `cpu` for this process unless another run holds it (a lock dies with its process)."""
    if cpu in _claims:
        return True
    try:
        fd = os.open(_lock_path(cpu), os.O_RDWR | os.O_CREAT)
    except OSError as e:
        raise ValueError(f"cannot create cpu lock files ({e}); run unpinned (--no-pin)") from e
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return False
    _claims[cpu] = fd
    return True


def _unclaim(cpu):
    fd = _claims.pop(cpu)
    if sys.platform == "win32":
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    os.close(fd)


def _kernel32():
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_size_t)]
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    return ctypes, k32


def _process_mask():
    """This process's affinity mask (Windows)."""
    ctypes, k32 = _kernel32()
    mask, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return mask.value


def _allowed_cpus():
    """The cpus this process may run on."""
    if sys.platform == "win32":
        mask = _process_mask()
        return {cpu for cpu in range(mask.bit_length()) if mask >> cpu & 1}
    return set(os.sched_getaffinity(0))


def fastchess_args(slots):
    """fastchess options pinning each game to its one-cpu slot (single-threaded games)."""
    if slots and any(len(s) != 1 for s in slots):
        raise ValueError("fastchess -use-affinity pins one cpu per game; run multi-cpu slots as separate processes")
    return ["-use-affinity", format_cpus([s[0] for s in slots])] if slots else []


def split_rounds(pairs, slots):
    """Opening pairs for each slot's process: as even as possible, the first slots one more."""
    count = min(slots, pairs)
    return [pairs // count + (i < pairs % count) for i in range(count)]


def global_game(local, slot, slots):
    """Number of a slot process's `local` game when `slots` processes interleave their pairs.

    Pair k of slot i is pair k * slots + i overall, so games 2g+1 and 2g+2 remain one
    opening pair with colors reversed (with split_rounds, pairs 0..total-1 are all used).
    """
    pair, half = divmod(local - 1, 2)
    return 2 * (pair * slots + slot) + half + 1


@contextmanager
def _inherited_affinity(cpus):
    """Restrict this process to `cpus` for the block; processes started inside inherit it."""
    if sys.platform == "win32":
        ctypes, k32 = _kernel32()
        if max(cpus) >= 8 * ctypes.sizeof(ctypes.c_size_t):
            raise ValueError("pinning cpus beyond the first processor group is not supported; use --no-pin")
        handle = k32.GetCurrentProcess()
        previous = _process_mask()
        if not k32.SetProcessAffinityMask(handle, sum(1 << cpu for cpu in cpus)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            k32.SetProcessAffinityMask(handle, previous)
    else:
        previous = os.sched_getaffinity(0)
        os.sched_setaffinity(0, cpus)
        try:
            yield
        finally:
            os.sched_setaffinity(0, previous)


def popen_pinned(cmd, cpus, **kwargs):
    """subprocess.Popen whose process, and every process it starts, runs only on `cpus`."""
    with _inherited_affinity(cpus):
        return subprocess.Popen(cmd, **kwargs)
