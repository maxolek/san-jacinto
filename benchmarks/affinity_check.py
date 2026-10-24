"""Check where the engine's search threads really run (Windows).

For each thread count, starts a timed search and compares the placement the
engine announces ("info string affinity ...") with the affinity mask Windows
reports for each busy thread (GetThreadGroupAffinity). Run it on a new machine
to see the automatic order, or after changing the Affinity code.

usage: python affinity_check.py <engine.exe> [--threads 1,2,4,8] [--affinity auto] [--cpus 0-7 [--cpu-sets]]
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import subprocess
import sys
import threading
import time

from sample_profile import k32, threads_of

THREAD_QUERY_INFORMATION = 0x0040


class GROUP_AFFINITY(ctypes.Structure):
    _fields_ = [("Mask", ctypes.c_size_t), ("Group", wt.WORD), ("Reserved", wt.WORD * 3)]


k32.OpenThread.restype = ctypes.c_void_p
k32.GetThreadGroupAffinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(GROUP_AFFINITY)]
k32.GetThreadTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(wt.FILETIME)] * 4
k32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                       ctypes.POINTER(ctypes.c_size_t)]
k32.CloseHandle.argtypes = [ctypes.c_void_p]


def expand(text):
    """'0,2,4-6' -> [0, 2, 4, 5, 6]"""
    out = []
    for part in text.replace(" ", "").split(","):
        if part:
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return out


def bits(mask):
    return [i for i in range(64) if mask >> i & 1]


def cpu_time(handle):
    ft = [wt.FILETIME() for _ in range(4)]
    k32.GetThreadTimes(handle, *[ctypes.byref(x) for x in ft])
    return sum((f.dwHighDateTime << 32) | f.dwLowDateTime for f in ft[2:])


def expected_masks(summary, count, process_cpus):
    """Sorted per-thread CPU lists implied by an 'info string affinity ...' line."""
    detail = summary.split(": ", 1)[1]
    for prefix, per_thread in (("search threads on cpus ", True), ("search thread on cpu ", True),
                               ("search thread on cpus ", False)):
        if detail.startswith(prefix):
            cpus = expand(detail[len(prefix):].split(" (")[0])
            return sorted([c] for c in cpus) if per_thread else [cpus]
    return [process_cpus] * count  # not pinned: every CPU the process was given (single processor group)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engine")
    ap.add_argument("--threads", default="1,2,4,8", help="thread counts to check")
    ap.add_argument("--affinity", default="auto", help="Affinity option value")
    ap.add_argument("--cpus", default="", help="restrict the engine process first, as a match runner would")
    ap.add_argument("--cpu-sets", action="store_true",
                    help="restrict with a process default CPU set (fastchess on Windows 11) instead of the affinity mask")
    ap.add_argument("--movetime", type=int, default=2000)
    args = ap.parse_args()

    proc = subprocess.Popen([args.engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    handle = ctypes.c_void_p(int(proc._handle))
    pmask, smask = ctypes.c_size_t(), ctypes.c_size_t()
    k32.GetProcessAffinityMask(handle, ctypes.byref(pmask), ctypes.byref(smask))
    process_cpus = bits(pmask.value)
    if args.cpus:
        process_cpus = expand(args.cpus)
        mask = sum(1 << c for c in process_cpus)
        if args.cpu_sets:
            ga = (GROUP_AFFINITY * 1)()
            ga[0].Mask, ga[0].Group = mask, 0
            k32.SetProcessDefaultCpuSetMasks.argtypes = [ctypes.c_void_p, ctypes.POINTER(GROUP_AFFINITY), wt.USHORT]
            ok = k32.SetProcessDefaultCpuSetMasks(handle, ga, 1)
        else:
            ok = k32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask))
        if not ok:
            sys.exit(f"could not restrict the process ({ctypes.get_last_error()})")

    lines = []
    threading.Thread(target=lambda: [lines.append(l.rstrip("\n")) for l in proc.stdout], daemon=True).start()

    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def wait_for(prefix, start):
        while True:
            for line in lines[start:]:
                if line.startswith(prefix):
                    return line
            if proc.poll() is not None:
                sys.exit("engine exited")
            time.sleep(0.01)

    send("uci")
    wait_for("uciok", 0)
    for line in lines:
        if line.startswith("info string cpu topology"):
            print(line[len("info string "):])
    print(f"process cpus: {','.join(map(str, process_cpus))}")
    n = len(lines)
    send(f"setoption name Affinity value {args.affinity}")
    print(wait_for("info string", n)[len("info string "):])

    failures = 0
    summary = None
    for requested in [int(t) for t in args.threads.split(",")]:
        n = len(lines)
        send(f"setoption name Threads value {requested}")
        count = int(wait_for("info string set Threads = ", n).split("=")[1].split()[0])
        n = len(lines)
        send("position startpos")
        send(f"go movetime {args.movetime}")
        time.sleep(min(0.6, args.movetime / 3000))
        handles = {tid: k32.OpenThread(THREAD_QUERY_INFORMATION, False, tid) for tid in threads_of(proc.pid)}
        before = {tid: cpu_time(h) for tid, h in handles.items() if h}
        time.sleep(min(0.5, args.movetime / 4000))
        busy = sorted(before, key=lambda t: cpu_time(handles[t]) - before[t], reverse=True)[:count]
        actual = []
        for tid in busy:
            ga = GROUP_AFFINITY()
            k32.GetThreadGroupAffinity(handles[tid], ctypes.byref(ga))
            actual.append(bits(ga.Mask))
        for h in handles.values():
            if h:
                k32.CloseHandle(h)
        wait_for("bestmove", n)
        announced = [l for l in lines[n:] if l.startswith("info string affinity")]
        summary = announced[0] if announced else summary  # printed only when the placement changes
        ok = summary is not None and sorted(actual) == expected_masks(summary, count, process_cpus)
        failures += not ok
        capped = f" (requested {requested}; the engine caps Threads at the cpus available)" if count != requested else ""
        print(f"\nthreads={count}: {'OK' if ok else 'MISMATCH'}{capped}")
        print(f"  engine: {summary[len('info string '):] if summary else '(no placement line)'}")
        print("  actual: " + "  ".join("{" + ",".join(map(str, m)) + "}" for m in sorted(actual)))

    send("quit")
    proc.wait(10)
    print(f"\n{failures} mismatch(es)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
