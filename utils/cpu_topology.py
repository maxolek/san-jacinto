"""Physical core layout of this machine, used to pin concurrent games.

Hybrid CPUs (Intel P/E cores, ARM big.LITTLE) have cores of different speed.
A game is only fair if both engines run on the same kind of core, so the match
runner gives every concurrent game one physical core and labels it with its
type: "P" for the fastest core type and "E" for slower ones. On homogeneous
CPUs every core is a "P" core.

Detection mirrors Tomahawk's own `Affinity` logic: Windows reports an
efficiency class per core (GetLogicalProcessorInformationEx), Linux reports
`cpu_capacity` (ARM) or lists Intel performance cores in
/sys/devices/cpu_core/cpus. Elsewhere (macOS) nothing can be pinned.

Pinning is inherited: `launch_affinity(cpus)` temporarily restricts this
process (Windows) or thread (Linux) so that a process started inside the block,
and every engine it starts, begins life on those CPUs.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import struct
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")


@dataclass(frozen=True)
class Core:
    cpus: tuple        # logical CPUs of this physical core, ascending
    speed: int = 0     # higher is faster (Windows efficiency class / Linux capacity)
    cluster: int = -1  # id of a shared L2 cache (E-core clusters); -1 = unknown


@dataclass(frozen=True)
class Topology:
    cores: tuple = ()
    supported: bool = False              # CPU pinning works on this platform
    allowed: tuple = field(default=())   # logical CPUs this process may use

    @property
    def fastest(self):
        return max((c.speed for c in self.cores), default=0)

    @property
    def hybrid(self):
        return len({c.speed for c in self.cores}) > 1

    def core_type(self, core):
        """'P' for cores within 10% of the fastest type, else 'E' (same rule as the engine)."""
        return "P" if core.speed * 10 >= self.fastest * 9 else "E"

    def type_of_cpu(self, cpu):
        for core in self.cores:
            if cpu in core.cpus:
                return self.core_type(core)
        return "?"

    def describe(self):
        if not self.cores:
            return f"{os.cpu_count() or 1} logical cpus (topology unknown, no pinning)"
        logical = sum(len(c.cpus) for c in self.cores)
        p = sum(self.core_type(c) == "P" for c in self.cores)
        e = len(self.cores) - p
        smt = any(len(c.cpus) > 1 for c in self.cores)
        kinds = f"{p} performance + {e} efficiency cores" if e else f"{p} cores"
        return f"{logical} logical cpus: {kinds}{' (smt)' if smt else ''}"


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------

_RELATION_PROCESSOR_CORE = 0
_RELATION_CACHE = 2
_RELATION_ALL = 0xFFFF


def _windows_cores():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    length = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformationEx(_RELATION_ALL, None, ctypes.byref(length))
    if not length.value:
        return []
    buf = ctypes.create_string_buffer(length.value)
    if not k32.GetLogicalProcessorInformationEx(_RELATION_ALL, buf, ctypes.byref(length)):
        return []
    raw = buf.raw
    cores, l2 = [], []
    offset = 0
    while offset < length.value:
        relationship, size = struct.unpack_from("<II", raw, offset)
        if size == 0:
            break
        if relationship == _RELATION_PROCESSOR_CORE:
            # PROCESSOR_RELATIONSHIP: Flags, EfficiencyClass, Reserved[20], GroupCount, GroupMask[]
            efficiency = raw[offset + 9]
            mask, group = struct.unpack_from("<QH", raw, offset + 32)  # a core lives in one group
            cpus = tuple(64 * group + bit for bit in range(64) if mask >> bit & 1)
            cores.append((cpus, efficiency))
        elif relationship == _RELATION_CACHE and raw[offset + 8] == 2:
            # CACHE_RELATIONSHIP: Level at +8, GroupMask at +40
            mask, group = struct.unpack_from("<QH", raw, offset + 40)
            l2.append((group, mask))
        offset += size
    result = []
    for cpus, efficiency in cores:
        cluster = -1
        if cpus:
            first = cpus[0]
            for index, (group, mask) in enumerate(l2):
                if group == first // 64 and mask >> (first % 64) & 1:
                    cluster = index
                    break
        result.append(Core(cpus=cpus, speed=efficiency, cluster=cluster))
    return result


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def parse_cpu_list(text):
    """'0-3,8 10' -> [0, 1, 2, 3, 8, 10] (the Linux sysfs / engine Affinity syntax)."""
    cpus = []
    for part in text.replace(",", " ").split():
        first, _, last = part.partition("-")
        cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def format_cpu_list(cpus):
    """[0, 1, 2, 3, 8] -> '0-3,8'."""
    cpus = sorted(set(cpus))
    parts, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        parts.append(f"{cpus[i]}-{cpus[j]}" if j >= i + 2 else ",".join(map(str, cpus[i:j + 1])))
        i = j + 1
    return ",".join(parts)


def _linux_cores():
    root = Path("/sys/devices/system/cpu")
    try:
        online = parse_cpu_list(_read(root / "online"))
    except ValueError:
        return []
    try:
        intel_p = set(parse_cpu_list(_read("/sys/devices/cpu_core/cpus")))
    except ValueError:
        intel_p = set()
    by_first = {}
    for cpu in online:
        base = root / f"cpu{cpu}"
        try:
            siblings = parse_cpu_list(_read(base / "topology/thread_siblings_list")) or [cpu]
        except ValueError:
            siblings = [cpu]
        capacity = _read(base / "cpu_capacity")
        if capacity.isdigit():
            speed = int(capacity)
        else:
            speed = 1 if cpu in intel_p else 0
        cluster = -1
        for index in range(8):
            cache = base / f"cache/index{index}"
            level = _read(cache / "level")
            if not level:
                break
            if level == "2":
                try:
                    shared = parse_cpu_list(_read(cache / "shared_cpu_list"))
                except ValueError:
                    shared = []
                cluster = shared[0] if shared else -1
                break
        entry = by_first.setdefault(min(siblings), {"cpus": set(), "speed": speed, "cluster": cluster})
        entry["cpus"].add(cpu)
    return [Core(cpus=tuple(sorted(v["cpus"])), speed=v["speed"], cluster=v["cluster"])
            for _, v in sorted(by_first.items())]


def allowed_cpus():
    """Logical CPUs this process may run on (empty if unknown)."""
    if IS_WINDOWS:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        if k32.GetProcessAffinityMask(ctypes.c_void_p(k32.GetCurrentProcess()),
                                      ctypes.byref(process_mask), ctypes.byref(system_mask)):
            return [bit for bit in range(64) if process_mask.value >> bit & 1]
        return []
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return []


_TOPOLOGY = None


def topology(refresh=False):
    """Detected once per process (pass refresh=True to re-read the allowed CPUs)."""
    global _TOPOLOGY
    if _TOPOLOGY is None or refresh:
        cores = []
        try:
            cores = _windows_cores() if IS_WINDOWS else _linux_cores() if IS_LINUX else []
        except (OSError, ValueError, struct.error):
            cores = []
        cores = [c for c in cores if c.cpus]
        _TOPOLOGY = Topology(cores=tuple(cores), supported=bool(cores) and (IS_WINDOWS or IS_LINUX),
                             allowed=tuple(allowed_cpus()))
    return _TOPOLOGY


# ----------------------------------------------------------------------------
# Pinning by inheritance
# ----------------------------------------------------------------------------

_LAUNCH_LOCK = threading.Lock()


def _set_windows_mask(mask):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not k32.SetProcessAffinityMask(ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.c_size_t(mask)):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask")


@contextlib.contextmanager
def launch_affinity(cpus):
    """Start processes inside this block restricted to `cpus` (None/empty: no change).

    Child processes inherit the affinity of the process (Windows) or thread
    (Linux) that starts them, and cutechess passes its own affinity on to the
    engines it starts, so an engine never runs outside its slot even before it
    has read its options. Other threads of this process are briefly restricted
    too on Windows, which is harmless for the match runner's reader threads.
    """
    if not cpus or not (IS_WINDOWS or hasattr(os, "sched_setaffinity")):
        yield
        return
    with _LAUNCH_LOCK:
        if IS_WINDOWS:
            previous = allowed_cpus()
            _set_windows_mask(sum(1 << cpu for cpu in cpus if cpu < 64))
            try:
                yield
            finally:
                _set_windows_mask(sum(1 << cpu for cpu in previous) or (1 << 64) - 1)
        else:
            previous = os.sched_getaffinity(0)
            os.sched_setaffinity(0, set(cpus))  # the calling thread, inherited by fork()
            try:
                yield
            finally:
                os.sched_setaffinity(0, previous)


class KillOnCloseJob:
    """Windows job object whose processes all die when this Python process exits.

    Every cutechess process (and the engines it starts, which join the job
    automatically) is assigned to it, so a crashed or killed runner never leaves
    engines behind to steal CPU time from the next test. No-op elsewhere.
    """

    KILL_ON_JOB_CLOSE = 0x2000
    LIMIT_AFFINITY = 0x10

    class _Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32)]

    class _Extended(ctypes.Structure):
        pass

    _Extended._fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", ctypes.c_uint64 * 6),
                          ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                          ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    def __init__(self, cpus=None):
        self.handle = None
        self.affinity = None
        if not IS_WINDOWS:
            return
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = ctypes.c_void_p
        k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            return
        info = self._Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        if k32.SetInformationJobObject(ctypes.c_void_p(handle), 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.handle = handle
            self.k32 = k32
        else:
            k32.CloseHandle(ctypes.c_void_p(handle))
        limits = []
        if cpus and all(0 <= cpu < 8 * ctypes.sizeof(ctypes.c_size_t) for cpu in cpus):
            limits.append((self.KILL_ON_JOB_CLOSE | self.LIMIT_AFFINITY, sum(1 << cpu for cpu in set(cpus))))
        limits.append((self.KILL_ON_JOB_CLOSE, 0))
        for flags, mask in limits:
            info = self._Extended()
            info.BasicLimitInformation.LimitFlags = flags
            info.BasicLimitInformation.Affinity = mask 
            if k32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                self.handle, self.k32 = handle, k32
                self.affinity = sorted(set(cpus)) if flags & self.LIMIT_AFFINITY else None 
                return 
        k32.CloseHandle(handle)


    def add(self, process_handle):
        if self.handle:
            return bool(self.k32.AssignProcessToJobObject(self.handle),int(process_handle))
        return False

    def close(self):
        """close the job: any process still in it is killed"""
        if self.handle:
            self.k32.CloseHandle(self.handle)
            self.handle = None 


def resume_process(pid):
    """resume every thread of a process started suspended (windows)
    returns how many were resumed
    """
    if not IS_WINDOWS: return 0

    class THREADENTRY32(ctypes.Structure):
        _fields_ = [("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong), ("th32ThreadID", ctypes.c_ulong),
                    ("th32OwnerProcessID", ctypes.c_ulong), ("tpBasePri", ctypes.c_long),
                    ("tpDeltaPri", ctypes.c_long), ("dwFlags", ctypes.c_ulong)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    k32.OpenThread.restype = ctypes.c_void_p
    k32.Thread32First.argtypes = k32.Thread32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(THREADENTRY32)]
    k32.ResumeThjread.argtypes = [ctypes.c_void_p]
    k32.ResumeThread.restype = ctypes.c_ulong
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    snapshot = k32.CreateToolhelp32Snapshot(0x4, 0)
    if not snapshot or snapshot == ctypes.c_void_p(-1).value: 
        raise ctypes.WinError(ctypes.get_last_error())
    resumed = 0
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        ok = k32.Thread32First(snapshot, ctypes.byref(entry))
        while ok:
            if entry.th32OwnerProcessID == pid:
                thread = k32.OpenThread(0x0002, False, entry.th32ThreadID)
                if thread:
                    if k32.ResumeThread(thread) != 0xFFFFFFFF:
                        resumed += 1
                    k32.CloseHandle(thread)
                ok = k32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snapshot)
    return resumed


def process_affinity(pid):
    """CPUs a running process may use (None if it cannot be read).

    On Linux every thread has its own mask, so this is the union over the
    process's threads (a thread pinned elsewhere shows up here).
    """
    if IS_LINUX:
        cpus = set()
        try:
            tids = [int(t) for t in os.listdir(f"/proc/{pid}/task")]
        except OSError:
            tids = [pid]
        for tid in tids:
            try:
                cpus |= os.sched_getaffinity(tid)
            except OSError:
                continue
        return sorted(cpus) or None
    try:
        import psutil
        return sorted(psutil.Process(pid).cpu_affinity())
    except ImportError:
        pass
    except Exception:
        return None
    if IS_WINDOWS:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
            if k32.GetProcessAffinityMask(ctypes.c_void_p(handle), ctypes.byref(process_mask),
                                          ctypes.byref(system_mask)):
                return [bit for bit in range(64) if process_mask.value >> bit & 1]
        finally:
            k32.CloseHandle(ctypes.c_void_p(handle))
    return None


def set_process_affinity(pid, cpus):
    """Restrict every thread of a running process to `cpus`. False if it could not be done."""
    if IS_LINUX:
        ok = False
        try:
            tids = [int(t) for t in os.listdir(f"/proc/{pid}/task")]
        except OSError:
            tids = [pid]
        for tid in tids:
            try:
                os.sched_setaffinity(tid, set(cpus))
                ok = True
            except OSError:
                continue
        return ok
    if IS_WINDOWS:
        # SetProcessAffinityMask also moves every thread of the process onto the new mask.
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = k32.OpenProcess(0x0200 | 0x1000, False, pid)  # SET_INFORMATION | QUERY_LIMITED
        if not handle:
            return False
        try:
            return bool(k32.SetProcessAffinityMask(ctypes.c_void_p(handle),
                                                   ctypes.c_size_t(sum(1 << c for c in cpus if c < 64))))
        finally:
            k32.CloseHandle(ctypes.c_void_p(handle))
    return False
