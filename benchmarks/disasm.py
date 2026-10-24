"""Disassemble one function of a PDB-symbolized build, annotated with source lines
and inlined-function boundaries (useful next to sample_profile.py --func).

Requires `pip install capstone` and a .pdb next to the exe (clang / zig c++ -g).
For MinGW GCC builds use binutils instead: objdump -d -S -C tomahawk.exe

usage: python disasm.py <engine.exe> <function-name-substring>
"""
import ctypes
import ctypes.wintypes as wt
import struct
import sys

import capstone

from sample_profile import PdbSymbolizer


class IMAGEHLP_LINE64(ctypes.Structure):
    _fields_ = [("SizeOfStruct", wt.DWORD), ("Key", ctypes.c_void_p), ("LineNumber", wt.DWORD),
                ("FileName", ctypes.c_char_p), ("Address", ctypes.c_uint64)]


def rva_to_offset(data, rva):
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    sections = struct.unpack_from("<H", data, pe + 6)[0]
    table = pe + 24 + struct.unpack_from("<H", data, pe + 20)[0]
    for i in range(sections):
        vsize, va, rawsize, rawptr = struct.unpack_from("<IIII", data, table + 40 * i + 8)
        if va <= rva < va + max(vsize, rawsize):
            return rawptr + (rva - va)
    raise ValueError(hex(rva))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    exe, name = sys.argv[1], sys.argv[2]
    sym = PdbSymbolizer(exe)
    matches = [r for r in sym.table if name in r[2]]
    if not matches:
        sys.exit(f"no function matching {name!r}")
    for addr, size, fname in matches:
        print(f"{addr - sym.FAKE_BASE:#x} {size:6d} {fname}")
    addr, size, _ = max(matches, key=lambda r: r[1])
    rva = addr - sym.FAKE_BASE
    data = open(exe, "rb").read()
    code = data[rva_to_offset(data, rva):][:size]

    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.syntax = capstone.CS_OPT_SYNTAX_INTEL
    md.skipdata = True  # keep going over instructions capstone cannot decode (e.g. AVX-VNNI)
    line = IMAGEHLP_LINE64()
    line.SizeOfStruct = ctypes.sizeof(line)
    disp = wt.DWORD(0)
    sym.dbg.SymGetLineFromAddr64.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
    last = None
    for ins in md.disasm(code, rva):
        where = ""
        if sym.dbg.SymGetLineFromAddr64(sym.h, sym.FAKE_BASE + ins.address, ctypes.byref(disp), ctypes.byref(line)):
            where = f"{line.FileName.decode().split(chr(92))[-1]}:{line.LineNumber}"
        frames = sym.frames(ins.address)
        tag = frames[-1] if len(frames) > 1 else ""
        if (where, tag) != last:
            print(f"      ; {where} {tag}")
            last = (where, tag)
        print(f"{ins.address:8x}: {ins.mnemonic} {ins.op_str}")


if __name__ == "__main__":
    main()
