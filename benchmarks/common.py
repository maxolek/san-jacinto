"""Shared helpers for the Tomahawk benchmark scripts.

Python 3.8+, standard library only. Engines are driven through UCI on stdin/stdout
and must support Tomahawk's `dumpstats` command and the `go ... eval` argument
(both present in every build configuration).
"""
import ctypes
import os
import re
import struct
import subprocess
import sys

IS_WINDOWS = sys.platform == "win32"

# (name, FEN, depth). Depths were chosen so each search takes about 0.1-1.5 s on a
# 28 W laptop P-core with the optimized engine; use --depth-delta to scale.
POSITIONS = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 10),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 8),
    ("sf_mid1", "r3k2r/2pb1ppp/2pp1q2/p7/1nP1B3/1P2P3/P2N1PPP/R2QK2R w KQkq a6 0 14", 10),
    ("sf_mid2", "4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24", 9),
    ("sf_mid3", "r3qbrk/6p1/2b2pPp/p3pP1Q/PpPpP2P/3P1B2/2PB3K/R5R1 w - - 16 42", 10),
    ("sf_mid4", "6k1/1R3p2/6p1/2Bp3p/3P2q1/P7/1P2rQ1K/5R2 b - - 4 44", 11),
    ("sf_mid5", "8/8/1p2k1p1/3p3p/1p1P1P1P/1P2PK2/8/8 w - - 3 54", 15),
    ("sf_mid6", "7r/2p3k1/1p1p1qp1/1P1Bp3/p1P2r1P/P7/4R3/Q4RK1 w - - 0 36", 10),
    ("sf_mid7", "r1bq1rk1/pp2b1pp/n1pp1n2/3P1p2/2P1p3/2N1P2N/PP2BPPP/R1BQ1RK1 b - - 2 10", 9),
    ("sf_mid8", "3r3k/2r4p/1p1b3q/p4P2/P2Pp3/1B2P3/3BQ1RP/6K1 w - - 3 87", 10),
    ("sf_mid9", "2r4r/1p4k1/1Pnp4/3Qb1pq/8/4BpPp/5P2/2RR1BK1 w - - 0 42", 8),
    ("sf_end1", "4q1bk/6b1/7p/p1p4p/PNPpP2P/KN4P1/3Q4/4R3 b - - 0 37", 11),
    ("sf_end2", "2q3r1/1r2pk2/pp3pp1/2pP3p/P1Pb1BbP/1P4Q1/R3NPP1/4R1K1 w - - 2 34", 10),
    ("sf_end3", "1r2r2k/1b4q1/pp5p/2pPp1p1/P3Pn2/1P1B1Q1P/2R3P1/4BR1K b - - 1 37", 9),
    ("sf_end4", "r3kbbr/pp1n1p1P/3ppnp1/q5N1/1P1pP3/P1N1B3/2P1QP2/R3KB1R b KQkq b3 0 17", 9),
    ("rook_end", "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1", 16),
    ("kpk_promo", "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1", 20),
    ("promo_race", "8/1P6/8/8/8/8/5kp1/K7 w - - 0 1", 13),
    ("ep_check", "8/8/3p4/KPp4r/1R2Pp1k/8/6P1/8 w - c6 0 2", 12),
    ("sf_mid10", "r2q1rk1/pp2ppbp/2p2np1/6B1/3PP1b1/Q1P2N2/P4PPP/3RKB1R b K - 0 13", 10),
]


def select_positions(only=""):
    """POSITIONS filtered by a comma-separated list of names (empty = all)."""
    if not only:
        return list(POSITIONS)
    names = set(only.split(","))
    unknown = names - {p[0] for p in POSITIONS}
    if unknown:
        sys.exit(f"unknown position name(s) in --only: {', '.join(sorted(unknown))}; "
                 f"choose from: {', '.join(p[0] for p in POSITIONS)}")
    return [p for p in POSITIONS if p[0] in names]


# ----------------------------------------------------------------------------
# CPU selection
# ----------------------------------------------------------------------------

def performance_cpus():
    """Logical CPUs to benchmark on: one per physical core of the fastest core type.

    On hybrid Intel CPUs these are the P-cores (E-cores are 30-50% slower and
    would add noise); with SMT only the first logical CPU of each core is used.
    """
    if IS_WINDOWS:
        try:
            return _windows_fastest_cores()
        except OSError:
            pass
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def cpus_for_threads(threads):
    """CPUs to pin a search with `threads` search threads to (None = no pinning).

    While the threads fit, one logical CPU per P-core, as for single-threaded runs.
    Beyond that the process is left unrestricted and the engine places its threads
    itself (Affinity auto: P-cores, then E-cores, then SMT siblings): Tomahawk never
    runs more search threads than the CPUs it may use, so pinning 8 threads to
    4 CPUs would silently measure 4 threads.
    """
    fast = performance_cpus()
    return fast if threads <= len(fast) else None


def _windows_fastest_cores():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    relation_processor_core = 0
    length = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(length))
    buf = ctypes.create_string_buffer(length.value)
    if not k32.GetLogicalProcessorInformationEx(relation_processor_core, buf, ctypes.byref(length)):
        raise OSError(ctypes.get_last_error(), "GetLogicalProcessorInformationEx")
    raw = buf.raw
    cores = []  # (efficiency class, first logical cpu)
    offset = 0
    while offset < length.value:
        relationship, size = struct.unpack_from("<II", raw, offset)
        if relationship == relation_processor_core:
            efficiency = raw[offset + 9]
            mask, group = struct.unpack_from("<QH", raw, offset + 32)
            if group == 0 and mask:
                cores.append((efficiency, (mask & -mask).bit_length() - 1))
        offset += size
    if not cores:
        raise OSError(0, "no processor cores reported")
    best = max(e for e, _ in cores)
    return sorted(cpu for e, cpu in cores if e == best)


def parse_cpus(text):
    """'auto' (None) or a comma-separated list of logical CPU numbers."""
    if not text or text == "auto":
        return None
    return [int(x) for x in text.split(",")]


def pin(proc, cpus):
    """Restrict a started process to `cpus` (list of logical CPUs) at high priority."""
    if not cpus:
        return
    if IS_WINDOWS:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        mask = 0
        for cpu in cpus:
            mask |= 1 << cpu
        handle = ctypes.c_void_p(int(proc._handle))
        if not k32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask")
        k32.SetPriorityClass(handle, 0x00000080)  # HIGH_PRIORITY_CLASS
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(proc.pid, set(cpus))


# ----------------------------------------------------------------------------
# Running the engine
# ----------------------------------------------------------------------------

def run_engine(engine, commands, cpus=None, timeout=3600):
    """Run one engine process on a list of UCI commands (quit is appended); returns stdout."""
    proc = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    pin(proc, cpus)
    head = repin_when_ready(proc, cpus)
    out, _ = proc.communicate("\n".join(list(commands) + ["quit"]) + "\n", timeout=timeout)
    return head + out


def repin_when_ready(proc, cpus):
    """Wait for `readyok`, then pin again; returns the output read so far.

    Older Tomahawk builds (before the Affinity option, e.g. dd04600) set their own
    process affinity at startup (pin_to_pcores() in main.cpp, a hard-coded mask for
    one particular CPU). Where that mask is valid it would replace the benchmark's
    choice of cores, so the pin is repeated once the engine has finished starting
    up. Current builds keep their search threads inside the process affinity, so
    for them this is a no-op.
    """
    if not cpus:
        return ""
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    head = []
    for line in proc.stdout:
        head.append(line)
        if line.strip() == "readyok":
            pin(proc, cpus)
            break
    return "".join(head)


def search(engine, fen, depth=None, movetime=None, threads=1, cpus=None, hash_mb=None, extra=()):
    """One search in a fresh process (fresh TT, history and killers).

    Returns dict(nodes, time_ms, bestmove, eval, pv, threads, output). `nodes` and
    `time_ms` come from `dumpstats` (time measured inside the engine, excluding
    startup). `threads` is the thread count the engine confirmed, which is lower
    than requested when it has fewer CPUs available.
    Benchmark Release/PROD builds: DEV builds count quiescence nodes separately.
    """
    cmds = []
    if threads and threads > 1:
        cmds.append(f"setoption name Threads value {threads}")
    if hash_mb:
        cmds.append(f"setoption name Hash value {hash_mb}")
    cmds += list(extra)
    cmds += ["isready", f"position fen {fen}"]
    cmds.append(f"go depth {depth} eval" if depth else f"go movetime {movetime} eval")
    cmds.append("dumpstats")
    out = run_engine(engine, cmds, cpus)
    m_best = re.search(r"^eval (-?\d+) bestmove (\S+)", out, re.M)
    m_nodes = re.search(r"^\s+Nodes\s+(\d+)", out, re.M)
    m_time = re.search(r"^\s+Time \(ms\)\s+(\d+)", out, re.M)
    m_pv = re.search(r"^\s+PV\s+(.*)$", out, re.M)
    m_threads = re.findall(r"^info string set Threads = (\d+)", out, re.M)
    if not (m_best and m_nodes and m_time):
        raise RuntimeError(f"could not parse output of {engine}:\n{out[-2000:]}")
    return {
        "nodes": int(m_nodes.group(1)),
        "time_ms": int(m_time.group(1)),
        "bestmove": m_best.group(2),
        "eval": int(m_best.group(1)),
        "pv": m_pv.group(1).strip() if m_pv else "",
        "threads": int(m_threads[-1]) if m_threads else (threads or 1),
        "output": out,
    }
