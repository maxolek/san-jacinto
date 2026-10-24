"""Sampling profiler for the engine on Windows x64 (no admin rights needed).

Starts the engine, runs a search, and samples the search thread's instruction
pointer ~500-1000 times per second (SuspendThread + GetThreadContext), then
attributes samples to functions, including functions inlined into them.

Symbols:
  * PDB next to the exe (clang / zig c++ -g builds): read with the system dbghelp.dll.
  * otherwise DWARF debug info (MinGW GCC builds with -g): resolved with binutils
    addr2line (on PATH with MSYS2, or pass --addr2line).
Profile an optimized build with debug info, e.g. CMake Release plus
-DCMAKE_CXX_FLAGS=-g (not the Profile config, which enables DEBUG bookkeeping).

usage: python sample_profile.py <engine.exe> [--fen FEN] [--depth 12] [--top 40] [--func NAME]
"""
import argparse
import bisect
import collections
import ctypes
import ctypes.wintypes as wt
import os
import struct
import subprocess
import sys
import threading
import time

import common

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
winmm = ctypes.WinDLL("winmm")
k32.OpenThread.restype = ctypes.c_void_p      # HANDLE, not a 32-bit int
k32.SuspendThread.restype = wt.DWORD          # (DWORD)-1 on failure
k32.SuspendThread.argtypes = [ctypes.c_void_p]

TH32CS_SNAPTHREAD = 0x4
TH32CS_SNAPMODULE = 0x8
THREAD_ACCESS = 0x0002 | 0x0008 | 0x0010 | 0x0040  # suspend/resume, get context, query
CONTEXT_CONTROL = 0x00100001                       # x64


class THREADENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ThreadID", wt.DWORD),
                ("th32OwnerProcessID", wt.DWORD), ("tpBasePri", ctypes.c_long),
                ("tpDeltaPri", ctypes.c_long), ("dwFlags", wt.DWORD)]


class MODULEENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("th32ModuleID", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("GlblcntUsage", wt.DWORD), ("ProccntUsage", wt.DWORD),
                ("modBaseAddr", ctypes.c_void_p), ("modBaseSize", wt.DWORD), ("hModule", wt.HMODULE),
                ("szModule", ctypes.c_wchar * 256), ("szExePath", ctypes.c_wchar * 260)]


class SYMBOL_INFO(ctypes.Structure):
    _fields_ = [("SizeOfStruct", ctypes.c_ulong), ("TypeIndex", ctypes.c_ulong),
                ("Reserved", ctypes.c_uint64 * 2), ("Index", ctypes.c_ulong), ("Size", ctypes.c_ulong),
                ("ModBase", ctypes.c_uint64), ("Flags", ctypes.c_ulong), ("Value", ctypes.c_uint64),
                ("Address", ctypes.c_uint64), ("Register", ctypes.c_ulong), ("Scope", ctypes.c_ulong),
                ("Tag", ctypes.c_ulong), ("NameLen", ctypes.c_ulong), ("MaxNameLen", ctypes.c_ulong),
                ("Name", ctypes.c_char * 1)]


def threads_of(pid):
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    snap = ctypes.c_void_p(k32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0))
    te = THREADENTRY32()
    te.dwSize = ctypes.sizeof(te)
    out = []
    ok = k32.Thread32First(snap, ctypes.byref(te))
    while ok:
        if te.th32OwnerProcessID == pid:
            out.append(te.th32ThreadID)
        ok = k32.Thread32Next(snap, ctypes.byref(te))
    k32.CloseHandle(snap)
    return out


def module_range(pid, exe_name):
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    snap = ctypes.c_void_p(k32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, pid))
    me = MODULEENTRY32W()
    me.dwSize = ctypes.sizeof(me)
    ok = k32.Module32FirstW(snap, ctypes.byref(me))
    found = None
    while ok:
        if me.szModule.lower() == exe_name.lower():
            found = (me.modBaseAddr, me.modBaseSize)
            break
        ok = k32.Module32NextW(snap, ctypes.byref(me))
    k32.CloseHandle(snap)
    return found


def image_base(exe):
    with open(exe, "rb") as f:
        data = f.read(4096)
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    return struct.unpack_from("<Q", data, opt + 24)[0] if magic == 0x20B else struct.unpack_from("<I", data, opt + 28)[0]


# ----------------------------------------------------------------------------
# Symbolizers: frames(rva) -> [outermost function, ..., innermost inlined function]
# ----------------------------------------------------------------------------

class PdbSymbolizer:
    FAKE_BASE = 0x10000000

    def __init__(self, exe):
        self.dbg = ctypes.WinDLL("dbghelp", use_last_error=True)
        self.h = ctypes.c_void_p(0x7777)
        self.dbg.SymSetOptions(0x2 | 0x4 | 0x40)  # UNDNAME | DEFERRED_LOADS | LOAD_ANYTHING
        if not self.dbg.SymInitialize(self.h, os.path.dirname(os.path.abspath(exe)).encode(), False):
            raise OSError(ctypes.get_last_error(), "SymInitialize")
        self.dbg.SymLoadModuleEx.restype = ctypes.c_uint64
        self.dbg.SymLoadModuleEx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                                             ctypes.c_uint64, wt.DWORD, ctypes.c_void_p, wt.DWORD]
        if not self.dbg.SymLoadModuleEx(self.h, None, exe.encode(), None, self.FAKE_BASE, 0, None, 0):
            raise OSError(ctypes.get_last_error(), "SymLoadModuleEx")
        self.buf = ctypes.create_string_buffer(ctypes.sizeof(SYMBOL_INFO) + 2048)
        self.si = SYMBOL_INFO.from_buffer(self.buf)
        self.dbg.SymAddrIncludeInlineTrace.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.dbg.SymAddrIncludeInlineTrace.restype = wt.DWORD
        self.dbg.SymQueryInlineTrace.argtypes = [ctypes.c_void_p, ctypes.c_uint64, wt.DWORD, ctypes.c_uint64,
                                                 ctypes.c_uint64, ctypes.POINTER(wt.DWORD), ctypes.POINTER(wt.DWORD)]
        self.dbg.SymFromInlineContext.argtypes = [ctypes.c_void_p, ctypes.c_uint64, wt.DWORD,
                                                  ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p]
        self._build_table()
        self.cache = {}

    def _build_table(self):
        # SymFromAddr is unreliable for some lld-produced PDBs; enumerate functions instead.
        CB = ctypes.WINFUNCTYPE(wt.BOOL, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p)
        rows = []

        def cb(pinfo, size, ctx):
            si = SYMBOL_INFO.from_address(pinfo)
            if si.Tag == 5:  # SymTagFunction
                rows.append((si.Address, si.Size,
                             ctypes.string_at(pinfo + SYMBOL_INFO.Name.offset, si.NameLen).decode(errors="replace")))
            return True

        self.dbg.SymEnumSymbols.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p, CB, ctypes.c_void_p]
        keep = CB(cb)
        self.dbg.SymEnumSymbols(self.h, self.FAKE_BASE, b"*", keep, None)
        rows.sort()
        self.table = rows
        self.starts = [r[0] for r in rows]

    def _name(self):
        return self.buf[SYMBOL_INFO.Name.offset:SYMBOL_INFO.Name.offset + self.si.NameLen].decode(errors="replace")

    def frames(self, rva):
        if rva in self.cache:
            return self.cache[rva]
        addr = self.FAKE_BASE + rva
        i = bisect.bisect_right(self.starts, addr) - 1
        out = [self.table[i][2] if i >= 0 and addr < self.table[i][0] + max(self.table[i][1], 1) else f"?{rva:x}"]
        n = self.dbg.SymAddrIncludeInlineTrace(self.h, addr)
        if n:
            ctx, idx, disp = wt.DWORD(0), wt.DWORD(0), ctypes.c_uint64(0)
            if self.dbg.SymQueryInlineTrace(self.h, addr, 0, addr, addr, ctypes.byref(ctx), ctypes.byref(idx)):
                inl = []
                for k in range(n):
                    self.si.SizeOfStruct = ctypes.sizeof(SYMBOL_INFO)
                    self.si.MaxNameLen = 2000
                    if self.dbg.SymFromInlineContext(self.h, addr, ctx.value + k, ctypes.byref(disp),
                                                     ctypes.addressof(self.si)):
                        inl.append(self._name())
                out.extend(reversed(inl))  # inline contexts come innermost first
        self.cache[rva] = out
        return out

    def prepare(self, rvas):
        pass


class Addr2LineSymbolizer:
    """DWARF symbols via binutils addr2line (MinGW GCC builds compiled with -g)."""

    def __init__(self, exe, tool):
        self.exe, self.tool = exe, tool
        self.base = image_base(exe)
        self.cache = {}

    @staticmethod
    def _short(name):
        cut = name.find("(")
        return name[:cut] if cut > 0 else name

    def prepare(self, rvas):
        todo = [r for r in rvas if r not in self.cache]
        if not todo:
            return
        stdin = "\n".join(hex(self.base + r) for r in todo) + "\n"
        out = subprocess.run([self.tool, "-e", self.exe, "-f", "-C", "-i", "-p", "-a"],
                             input=stdin, capture_output=True, text=True).stdout
        frames, current = {}, None
        for line in out.splitlines():
            if line.startswith("0x"):
                addr, _, rest = line.partition(": ")
                current = int(addr, 16) - self.base
                frames[current] = [self._short(rest.split(" at ")[0])]
            elif line.strip().startswith("(inlined by)") and current is not None:
                frames[current].append(self._short(line.strip()[len("(inlined by) "):].split(" at ")[0]))
        for rva in todo:
            innermost_first = frames.get(rva, [f"?{rva:x}"])
            self.cache[rva] = list(reversed(innermost_first))

    def frames(self, rva):
        if rva not in self.cache:
            self.prepare([rva])
        return self.cache[rva]


def make_symbolizer(exe, addr2line):
    if os.path.exists(os.path.splitext(exe)[0] + ".pdb"):
        return PdbSymbolizer(exe)
    return Addr2LineSymbolizer(exe, addr2line)


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exe")
    ap.add_argument("--fen", default="r1bq1rk1/pp2b1pp/n1pp1n2/3P1p2/2P1p3/2N1P2N/PP2BPPP/R1BQ1RK1 b - - 2 10")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--stacks", type=int, default=0, help="also print the N hottest full inline stacks")
    ap.add_argument("--func", default="", help="print the hottest addresses inside this function")
    ap.add_argument("--addr2line", default="addr2line", help="addr2line executable for DWARF builds")
    args = ap.parse_args()
    exe = os.path.abspath(args.exe)

    sym = make_symbolizer(exe, args.addr2line)
    fast = common.performance_cpus()
    engine_cpu = fast[1] if len(fast) > 1 else fast[0]
    sampler_cpus = [c for c in fast if c != engine_cpu][:1]

    proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    common.pin(proc, [engine_cpu])
    proc.stdin.write("uci\nisready\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            sys.exit("engine exited during startup")
        if "readyok" in line:
            break
    common.pin(proc, [engine_cpu])  # again: older builds set their own affinity at startup
    base, size = module_range(proc.pid, os.path.basename(exe))
    done = threading.Event()

    def reader():
        for line in proc.stdout:
            if line.startswith("bestmove"):
                done.set()
        done.set()

    threading.Thread(target=reader, daemon=True).start()
    proc.stdin.write(f"position fen {args.fen}\ngo depth {args.depth}\n")
    proc.stdin.flush()

    if sampler_cpus:
        mask = 1 << sampler_cpus[0]
        k32.SetProcessAffinityMask(ctypes.c_void_p(-1), ctypes.c_size_t(mask))
    winmm.timeBeginPeriod(1)
    handles = {tid: k32.OpenThread(THREAD_ACCESS, False, tid) for tid in threads_of(proc.pid)}
    ft = [wt.FILETIME() for _ in range(4)]

    def user_time(h):
        if h and k32.GetThreadTimes(ctypes.c_void_p(h), *[ctypes.byref(x) for x in ft]):
            return (ft[3].dwHighDateTime << 32) | ft[3].dwLowDateTime
        return 0

    before = {tid: user_time(h) for tid, h in handles.items()}
    time.sleep(0.25)
    search_tid = max(handles, key=lambda tid: user_time(handles[tid]) - before[tid])  # the busy thread
    h = handles[search_tid]

    buf = ctypes.create_string_buffer(1232 + 16)
    ctx = (ctypes.addressof(buf) + 15) & ~15  # CONTEXT must be 16-byte aligned
    ctx_flags = ctypes.c_uint32.from_address(ctx + 0x30)
    rip = ctypes.c_uint64.from_address(ctx + 0xF8)
    samples = collections.Counter()
    outside = 0
    t0 = time.time()
    while not done.is_set():
        if k32.SuspendThread(ctypes.c_void_p(h)) == 0xFFFFFFFF:
            break
        ctx_flags.value = CONTEXT_CONTROL
        if k32.GetThreadContext(ctypes.c_void_p(h), ctypes.c_void_p(ctx)):
            r = rip.value
            if base <= r < base + size:
                samples[r - base] += 1
            else:
                outside += 1
        k32.ResumeThread(ctypes.c_void_p(h))
        time.sleep(0.0005)
    winmm.timeEndPeriod(1)
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait()

    sym.prepare(list(samples))
    outer, level1, inner, stacks = (collections.Counter() for _ in range(4))
    for rva, c in samples.items():
        fr = sym.frames(rva)
        outer[fr[0]] += c
        level1[" > ".join(fr[:2])] += c
        inner[fr[-1]] += c
        stacks[" > ".join(fr)] += c
    total = max(sum(samples.values()), 1)
    print(f"samples in exe={total} outside exe={outside} wall={time.time() - t0:.1f}s")
    for title, counter, limit in (("containing (non-inlined) function", outer, 25),
                                  ("containing function > first inlined callee", level1, args.top),
                                  ("innermost (inlined) function", inner, 30)):
        print(f"---- by {title} ----")
        for name, c in counter.most_common(limit):
            print(f"{100.0 * c / total:6.2f}%  {name[:160]}")
    if args.stacks:
        print("---- full inline stacks ----")
        for name, c in stacks.most_common(args.stacks):
            print(f"{100.0 * c / total:6.2f}%  {name[:220]}")
    if args.func:
        print(f"---- hottest addresses in {args.func} (rva) ----")
        hot = [(rva, c) for rva, c in samples.items() if args.func in sym.frames(rva)[0]]
        fsum = max(sum(c for _, c in hot), 1)
        for rva, c in sorted(hot, key=lambda x: -x[1])[:40]:
            print(f"{100.0 * c / fsum:6.2f}%  {rva:8x}  {' > '.join(sym.frames(rva)[1:])[:120]}")


if __name__ == "__main__":
    main()
