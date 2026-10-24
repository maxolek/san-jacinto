"""Where and on what hardware a game was played.

Speed differences between engine versions only mean something if the
compute was the same. This module labels every game with its environment:
machine, CPU model, the logical CPU and core type it ran on, and the clock
it actually ran at (sampled continuously while the game was played), so the
analytics can separate code effects from compute effects (e.g. NPS per GHz,
or comparing versions only within one core type).

Clock sampling:
  Windows  PDH counters "\\Processor Information(*)\\% Processor Performance"
           x "Processor Frequency" (the "Speed" Task Manager shows, including
           turbo), plus "% Processor Time" for utilization.
  Linux    /sys/devices/system/cpu/cpuN/cpufreq/scaling_cur_freq and /proc/stat.
Elsewhere clocks are reported as unknown (None).
"""
from __future__ import annotations

import bisect
import ctypes
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from utils import cpu_topology

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")


# ----------------------------------------------------------------------------
# Static machine information
# ----------------------------------------------------------------------------

def cpu_model():
    if IS_WINDOWS:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif IS_LINUX:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or platform.processor()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or "unknown"


def power_status():
    """('ac' | 'battery' | None, battery percent or None)."""
    if IS_WINDOWS:
        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        status = SYSTEM_POWER_STATUS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            source = {0: "battery", 1: "ac"}.get(status.ACLineStatus)
            percent = status.BatteryLifePercent if status.BatteryLifePercent <= 100 else None
            return source, percent
    elif IS_LINUX:
        supplies = Path("/sys/class/power_supply")
        if supplies.is_dir():
            online = [p for p in supplies.iterdir() if (p / "online").exists()]
            source = None
            if online:
                source = "ac" if any((p / "online").read_text().strip() == "1" for p in online) else "battery"
            capacity = [p / "capacity" for p in supplies.iterdir() if (p / "capacity").exists()]
            percent = int(capacity[0].read_text().strip()) if capacity else None
            return source, percent
    return None, None


def base_mhz_by_cpu():
    """Nominal (base) clock of every logical CPU in MHz, where the OS reports it."""
    if IS_LINUX:
        out = {}
        for cpu in cpu_topology.topology().allowed or range(os.cpu_count() or 1):
            text = cpu_topology._read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/base_frequency") or \
                cpu_topology._read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq")
            if text.isdigit():
                out[cpu] = int(text) / 1000
        return out
    return {}


def machine_info():
    """One dict describing the machine, stored with every run."""
    topo = cpu_topology.topology()
    source, percent = power_status()
    p_cores = sum(topo.core_type(c) == "P" for c in topo.cores)
    return dict(
        host=socket.gethostname(),
        os=platform.platform(),
        python=platform.python_version(),
        cpu_model=cpu_model(),
        logical_cpus=os.cpu_count(),
        physical_cores=len(topo.cores) or None,
        p_cores=p_cores if topo.cores else None,
        e_cores=len(topo.cores) - p_cores if topo.cores else None,
        topology=topo.describe(),
        power_source=source,
        battery_percent=percent,
    )


def child_pids(pid):
    """All descendant process ids of `pid` (best effort; empty if unknown or gone)."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:  # psutil checks creation times, so a reused parent PID is not mistaken for a parent
            return [p.pid for p in psutil.Process(pid).children(recursive=True)]
        except Exception:
            return []
    if IS_WINDOWS:
        return _windows_descendants(pid)
    if IS_LINUX:
        parents = {}
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                try:
                    fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                    parents[int(entry.name)] = int(fields[1])
                except (OSError, IndexError, ValueError):
                    continue
        return _descendants(pid, parents)
    return []


def _descendants(pid, parents, created=None):
    """Walk the parent map from `pid`. `created(pid)` (optional) returns a creation time:
    a "child" older than its parent is a process whose parent PID was reused, not a child."""
    children = {}
    for child, parent in parents.items():
        if child != parent:
            children.setdefault(parent, []).append(child)
    out, stack, seen = [], [pid], {pid}
    while stack:
        parent = stack.pop()
        for child in children.get(parent, []):
            if child in seen:
                continue
            if created is not None:
                born, parent_born = created(child), created(parent)
                if born is None or parent_born is None or born < parent_born:
                    continue
            seen.add(child)
            out.append(child)
            stack.append(child)
    return out


def _windows_creation_time(pid, cache):
    if pid in cache:
        return cache[pid]
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    value = None
    if handle:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        if k32.GetProcessTimes(ctypes.c_void_p(handle), *[ctypes.byref(t) for t in times]):
            value = times[0].value
        k32.CloseHandle(ctypes.c_void_p(handle))
    cache[pid] = value
    return value


def _windows_descendants(pid):
    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", ctypes.c_ulong), ("cntUsage", ctypes.c_ulong),
                    ("th32ProcessID", ctypes.c_ulong), ("th32DefaultHeapID", ctypes.c_size_t),
                    ("th32ModuleID", ctypes.c_ulong), ("cntThreads", ctypes.c_ulong),
                    ("th32ParentProcessID", ctypes.c_ulong), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", ctypes.c_ulong), ("szExeFile", ctypes.c_wchar * 260)]
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    snapshot = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        return []
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    parents = {}
    ok = k32.Process32FirstW(ctypes.c_void_p(snapshot), ctypes.byref(entry))
    while ok:
        parents[entry.th32ProcessID] = entry.th32ParentProcessID
        ok = k32.Process32NextW(ctypes.c_void_p(snapshot), ctypes.byref(entry))
    k32.CloseHandle(ctypes.c_void_p(snapshot))
    # Fresh creation times per walk (the cache is per call): PIDs are reused over a long run.
    cache = {}
    return _descendants(pid, parents, created=lambda p: _windows_creation_time(p, cache))


# ----------------------------------------------------------------------------
# Clock sampling
# ----------------------------------------------------------------------------

class _PdhReader:
    """Effective MHz and utilization of every logical CPU through PDH (Windows)."""

    PDH_FMT_DOUBLE = 0x00000200
    PDH_FMT_NOCAP100 = 0x00008000
    PDH_MORE_DATA = 0x800007D2

    class _Value(ctypes.Structure):
        _fields_ = [("CStatus", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]

    class _Item(ctypes.Structure):
        pass

    _Item._fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _Value)]

    def __init__(self):
        self.pdh = ctypes.WinDLL("pdh")
        self.query = ctypes.c_void_p()
        if self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.query)):
            raise OSError("PdhOpenQueryW failed")
        self.counters = {}
        for key, path in (("perf", r"\Processor Information(*)\% Processor Performance"),
                          ("base", r"\Processor Information(*)\Processor Frequency"),
                          ("util", r"\Processor Information(*)\% Processor Time")):
            handle = ctypes.c_void_p()
            if self.pdh.PdhAddEnglishCounterW(self.query, path, None, ctypes.byref(handle)):
                raise OSError(f"PdhAddEnglishCounterW failed for {path}")
            self.counters[key] = handle
        self.pdh.PdhCollectQueryData(self.query)

    def _values(self, handle):
        fmt = self.PDH_FMT_DOUBLE | self.PDH_FMT_NOCAP100
        size, count = ctypes.c_ulong(0), ctypes.c_ulong(0)
        status = self.pdh.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(size), ctypes.byref(count), None)
        if (status & 0xFFFFFFFF) != self.PDH_MORE_DATA or not size.value:
            return {}
        buf = ctypes.create_string_buffer(size.value)
        if self.pdh.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(size), ctypes.byref(count), buf):
            return {}
        items = ctypes.cast(buf, ctypes.POINTER(self._Item))
        out = {}
        for i in range(count.value):
            name = items[i].szName or ""
            group, _, index = name.partition(",")
            if not (group.isdigit() and index.isdigit()) or items[i].FmtValue.CStatus not in (0, 1):
                continue
            out[64 * int(group) + int(index)] = items[i].FmtValue.doubleValue
        return out

    def read(self):
        """{cpu: (effective MHz, utilization %)} since the previous call."""
        self.pdh.PdhCollectQueryData(self.query)
        perf, base, util = (self._values(self.counters[k]) for k in ("perf", "base", "util"))
        return {cpu: (base[cpu] * perf[cpu] / 100 if cpu in base else None, util.get(cpu))
                for cpu in perf}

    def close(self):
        self.pdh.PdhCloseQuery(self.query)


class _LinuxReader:
    def __init__(self):
        self.previous = self._stat()

    @staticmethod
    def _stat():
        out = {}
        try:
            for line in Path("/proc/stat").read_text().splitlines():
                name, *fields = line.split()
                if name.startswith("cpu") and name[3:].isdigit():
                    values = list(map(int, fields))
                    idle = values[3] + (values[4] if len(values) > 4 else 0)
                    out[int(name[3:])] = (sum(values), idle)
        except OSError:
            pass
        return out

    def read(self):
        now = self._stat()
        out = {}
        for cpu, (total, idle) in now.items():
            old_total, old_idle = self.previous.get(cpu, (total, idle))
            busy = total - old_total
            util = 100 * (1 - (idle - old_idle) / busy) if busy > 0 else None
            freq = cpu_topology._read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
            out[cpu] = (int(freq) / 1000 if freq.isdigit() else None, util)
        self.previous = now
        return out

    def close(self):
        pass


class ClockSampler:
    """Background thread recording (time, MHz, utilization) for every logical CPU.

    `window(cpu, start, end)` summarizes the samples taken while a game ran on
    that CPU. Samples older than `history_s` are dropped.
    """

    def __init__(self, interval=1.0, history_s=4 * 3600):
        self.interval = interval
        self.samples = {}  # cpu -> deque[(t, mhz, util)]
        self.maxlen = max(16, int(history_s / interval))
        self.lock = threading.Lock()
        self.reader = None
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        try:
            self.reader = _PdhReader() if IS_WINDOWS else _LinuxReader() if IS_LINUX else None
        except Exception as exc:  # counters unavailable (e.g. disabled perf counters)
            self.error = str(exc)
            self.reader = None
        if self.reader is not None:
            self.thread = threading.Thread(target=self._run, name="clock-sampler", daemon=True)
            self.thread.start()
        return self

    @property
    def available(self):
        return self.reader is not None

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                values = self.reader.read()
            except Exception as exc:
                self.error = str(exc)
                continue
            now = time.time()
            with self.lock:
                for cpu, (mhz, util) in values.items():
                    self.samples.setdefault(cpu, deque(maxlen=self.maxlen)).append((now, mhz, util))

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        if self.reader:
            self.reader.close()

    def window(self, cpu, start, end):
        """dict(mhz_mean, mhz_min, mhz_max, util_mean, samples) over [start, end]."""
        empty = dict(mhz_mean=None, mhz_min=None, mhz_max=None, util_mean=None, clock_samples=0)
        if cpu is None:
            return empty
        with self.lock:
            series = list(self.samples.get(cpu, ()))
        if not series:
            return empty
        times = [s[0] for s in series]
        # A sample at time t describes the interval ending at t.
        lo = bisect.bisect_left(times, start)
        hi = bisect.bisect_right(times, end + self.interval)
        chosen = series[lo:hi] or [series[min(lo, len(series) - 1)]]
        mhz = [s[1] for s in chosen if s[1] is not None]
        util = [s[2] for s in chosen if s[2] is not None]
        return dict(mhz_mean=round(sum(mhz) / len(mhz)) if mhz else None,
                    mhz_min=round(min(mhz)) if mhz else None,
                    mhz_max=round(max(mhz)) if mhz else None,
                    util_mean=round(sum(util) / len(util), 1) if util else None,
                    clock_samples=len(chosen))

    def current(self):
        """Latest (MHz, utilization) per CPU."""
        with self.lock:
            return {cpu: (s[-1][1], s[-1][2]) for cpu, s in self.samples.items() if s}
