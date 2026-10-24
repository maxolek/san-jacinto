# Benchmarks

Tools for measuring Tomahawk's speed (NPS, time-to-depth) and for proving that an optimization did not change the search. They were written during the engine's speed work; the [reference results](#reference-results) at the end show what they measured.

Guiding rule: **an implementation-only optimization must leave the search bit-identical** (same nodes, best move, eval and PV at fixed depth, single thread). Then the time difference is a pure speed difference on identical work. If the search changed, NPS says little about strength; use game testing (`python -m tests.sprt`) instead. See Tomahawk's `docs/validation.md` for the wider picture.

## Requirements

- **Windows x64** is the tested platform. P-core detection, CPU pinning and the profiler use Win32 APIs. On Linux the drivers pin with `sched_setaffinity` and use every allowed CPU (untested).
- **Python 3.8+**, standard library only. `disasm.py` also needs `pip install capstone`.
- **The Tomahawk sources** for the build helpers: `build_ref.ps1 -Repo` / `build_unity.ps1 -Src`, else the `TOMAHAWK_ROOT` environment variable, else `..\..\tomahawk` (Tomahawk checked out next to san-jacinto, as the main README describes). The Python tools only need engine executables.
- **A compiler** for the build helpers: your normal CMake toolchain (MinGW-w64 GCC), or zig (`pip install ziglang`, a bundled clang) for builds with a `.pdb`.
- **Benchmark Release/PROD builds.** `DEV` counts quiescence nodes differently and `Debug` is unoptimized.

Scripts can be run from any directory. Builds and results go to `benchmarks/out/`, which is gitignored.

## Quick start: is my change faster, and still the same search?

```powershell
cd benchmarks
.\build_ref.ps1 -Name base -Ref HEAD    # last engine commit (any commit, branch or tag), built in a git worktree in out\src-base
.\build_ref.ps1 -Name new               # your working tree
python compare.py out\base.exe out\new.exe --reps 3
```

Output (end of a real run, GCC builds of `dd04600` vs. the multithreading branch):

```text
OK   promo_race   d=13 nodes    182939/   182939 best b7b8q/b7b8q eval 0/0 ms     428 ->     138  x3.094 [3.06-3.12]
OK   ep_check     d=12 nodes    110632/   110632 best b4b2/b4b2 eval 655/655 ms     287 ->     100  x2.879 [2.80-2.95]
OK   sf_mid10     d=10 nodes    189078/   189078 best f6e4/f6e4 eval 367/367 ms     600 ->     200  x3.022 [2.65-3.40]

nodes base=5206419 new=5206419  time base=15998ms new=5487ms
NPS base=325432 new=948864  total-time speedup x2.916
per-position geomean speedup x2.938  (95% bootstrap CI over positions x2.845..x3.052)
search mismatches: 0/20
```

- `search mismatches: 0/20` is required for an implementation-only change. `DIFF` rows show which position diverged first. Stop and find the cause before trusting any timing.
- The speedup is the per-position geometric mean, with a bootstrap confidence interval over the positions. Total time is also shown; long positions dominate it.
- **Learn your machine's noise floor** with an A/A run: `python compare.py out\new.exe out\new.exe`. On the laptop used here, A/A results stayed within about ±1–2%. Treat smaller differences as noise unless they repeat and the interval excludes 1.

## Methodology

Why the numbers are trustworthy, and what to keep when you change the scripts:

1. **Fixed depth, one thread, fresh process per search.** A fresh process has an empty TT, history and killers, so node counts are deterministic. They act as a signature of the search. Timings come from the engine's own `dumpstats`, which excludes startup.
2. **Paired concurrent runs** (`compare.py`). Base and new search the same position *at the same time* on two different physical P-cores, and swap cores on every repetition. Both see the same clock and thermal state. A laptop's clock can move by 2× with background load, so separate before/after runs are unreliable.
3. **P-cores only, one logical CPU per core.** On hybrid Intel CPUs, E-cores are 30–50% slower. SMT siblings share a core, so only the first logical CPU of each P-core is used. The scripts detect this automatically (`common.performance_cpus()`); override it with `compare.py --cpus` or `bench.py --cpu`.
4. **Pinning survives the engine's own affinity.** Older builds (such as `dd04600`) set their process affinity at startup with a hard-coded mask (`pin_to_pcores()` in `main.cpp`), so the scripts pin again once the engine answers `readyok`. Current builds never leave the process affinity they are given: with `Threads > 1` they pin each search thread to its own core *within* the scripts' CPUs (see "thread placement" in Tomahawk's `docs/engine.md`).
5. **Summaries over positions:** geometric mean with a bootstrap interval, so one long position cannot dominate.

Practical tips: plug in the laptop, use the high-performance power plan, and close heavy background programs (sync clients, browsers, IDE indexers). The positions take 0.1–1.5 s each on a ~3 GHz laptop P-core. On a faster desktop, add `--depth-delta 1` so the 1 ms timer resolution stays small relative to each search.

## Tools

| Tool | Purpose |
|---|---|
| [compare.py](compare.py) | A/B: search equivalence plus paired timing of two builds |
| [bench.py](bench.py) | One build: nodes, time and NPS per position (`--out` writes JSON) |
| [progress.py](progress.py) | Several configurations in one table (for example original vs. current vs. 2/4 threads) |
| [smp.py](smp.py) | Lazy SMP: stability of multi-threaded searches, then NPS and time-to-depth scaling |
| [affinity_check.py](affinity_check.py) | Where the search threads really run: the engine's announced placement vs. each thread's OS affinity mask |
| [perft.py](perft.py) | Perft suite on 6 positions with known counts (move generation, make/unmake) |
| [verify.py](verify.py) | Runs a `TOMAHAWK_VERIFY_NNUE` build: every accumulator update and evaluation is checked |
| [sample_profile.py](sample_profile.py) | Sampling profiler without admin rights: hot functions, including inlined callees |
| [disasm.py](disasm.py) | Annotated disassembly of one function (PDB builds) |
| [kernelbench.cpp](kernelbench.cpp) | NNUE inference microbenchmark (`eval_simd`, L1, L2 kernels) plus a core-clock estimate |
| [build_ref.ps1](build_ref.ps1) | CMake build of the working tree or any git ref, copied to `out\<Name>.exe` |
| [build_unity.ps1](build_unity.ps1) | Single-translation-unit build without CMake (g++, clang++ or zig): symbols, verify, kernelbench |
| [common.py](common.py) | Shared: position set, P-core detection, pinning, UCI driver |

Every Python tool has `--help`. Most accept `--only name1,name2` (position names are in [common.py](common.py)) and `--depth-delta N`.

### Builds

```powershell
.\build_ref.ps1 -Name new                                   # working tree, CMake Release
.\build_ref.ps1 -Name base -Ref dd04600                     # any commit, branch or tag
.\build_ref.ps1 -Name new-clang -CMakeArgs '-DCMAKE_CXX_COMPILER=clang++'
.\build_ref.ps1 -Name verify -CMakeArgs '-DCMAKE_CXX_FLAGS=-DTOMAHAWK_VERIFY_NNUE'
```

- The engine loads its NNUE weights from the source tree it was built from (`PROJECT_ROOT`). Keep `out\src-<Name>` while you use a `-Ref` build, and remove it later with `git -C <tomahawk> worktree remove <san-jacinto>\benchmarks\out\src-<Name>` (it is a worktree of the Tomahawk repository).
- CMake writes to `<source>\engines\prod`. The script passes `-DVERSION=bench-<Name>`, so the build lands there as `bench-<Name>.exe` next to your normal executable (Debug builds still write `engines\dev\tomahawk.exe`), and that exact file is copied to `out\<Name>.exe`. The engine then reports `id version bench-<Name>` over UCI.

`build_unity.ps1` compiles all sources as one translation unit. That gives the same cross-file inlining as LTO. Use it when LTO is unavailable, for profiling builds, or for a source folder outside git:

```powershell
.\build_unity.ps1 -Out out\new.exe                          # g++ -O3 -ffast-math -march=native
.\build_unity.ps1 -Out out\prof.exe -Symbols                # + -g (DWARF with GCC, PDB with zig)
.\build_unity.ps1 -Out out\prof.exe -Compiler zig -Symbols  # zig must be on PATH or give its full path
.\build_unity.ps1 -Out out\verify.exe -Extra '-DTOMAHAWK_VERIFY_NNUE'
.\build_unity.ps1 -Out out\kernelbench.exe -Kernelbench
```

With GCC on Windows, both build paths add `-Wa,-muse-unaligned-vector-move` when the assembler supports it (GCC bug 54412; see the build section of Tomahawk's README).

**Compare builds that use the same C++ standard library.** GCC (libstdc++) and clang/zig (libc++) break `std::sort` ties differently, so their node counts differ. For a cross-toolchain speed comparison, use `compare.py --allow-diff`, which reports NPS only.

### Correctness

```powershell
python perft.py out\new.exe            # after board / move generator changes (--quick: one ply less)
python verify.py out\verify.exe        # after NNUE / accumulator changes (slow; depth reduced by 2)
python verify.py out\verify.exe --threads 4 --only sf_mid5
```

`TOMAHAWK_VERIFY_NNUE` compiles checks into the search. After every move, the incrementally updated accumulators are compared with a full rebuild. Every evaluation is compared with the scalar reference `NNUE::evaluate()`. The first mismatch prints `info string NNUE VERIFY FAILED ...` and exits with code 3. Never benchmark this build.

### Speed

```powershell
python compare.py out\base.exe out\new.exe --reps 3                  # A/B with equivalence check
python compare.py out\base.exe out\new.exe --allow-diff              # intentional search change: NPS only
python bench.py out\new.exe --reps 3 --out out\new.json              # one build
python progress.py original=out\base.exe current=out\new.exe current-2T=out\new.exe@2 current-4T=out\new.exe@4
python smp.py out\new.exe --threads 1,2,4,8
```

- `progress.py` runs every configuration back to back for each position, alternating the order between repetitions. Speedups are relative to the first configuration.
- For Lazy SMP, time-to-depth is more meaningful than NPS: helper threads duplicate work, so NPS overstates the benefit. Neither replaces games at a fixed time control.
- Multi-threaded runs pin the process to the detected P-cores (one logical CPU each) while the thread count fits. Larger counts run unpinned, and the engine places the threads itself (P-cores first). The engine never runs more search threads than the CPUs it may use; `bench.py`, `progress.py` and `smp.py` flag any run where it used fewer threads than requested.

### Thread placement

```powershell
python affinity_check.py out\new.exe                                 # auto placement for 1, 2, 4, 8 threads
python affinity_check.py out\new.exe --threads 4 --affinity 0,6,8,1  # an explicit list
python affinity_check.py out\new.exe --threads 4 --cpus 2-5          # inside a restricted process, as a match runner would
python affinity_check.py out\new.exe --threads 4 --cpus 2-5 --cpu-sets   # restricted with a CPU set (fastchess on Windows 11)
```

It prints the detected topology and the automatic order, then for each thread count runs a short search and checks that the busy threads' OS affinity masks match the placement the engine announced. Any `MISMATCH` means the engine did not place its threads as it said. The check is Windows-only.

### Finding hot spots

```powershell
.\build_unity.ps1 -Out out\prof.exe -Symbols
python sample_profile.py out\prof.exe --depth 12 --top 30
python sample_profile.py out\prof.exe --func eval_simd               # hottest addresses in one function
python sample_profile.py out\prof.exe --stacks 20                    # hottest inline stacks
```

- The profiler suspends the search thread about 500–1000 times per second and records its instruction pointer. Samples are attributed to the containing function and to the first inlined callee.
- It reads symbols from a `.pdb` next to the exe (zig/clang builds, via Windows' dbghelp). Otherwise it reads DWARF through binutils `addr2line`: it must be on `PATH` (MSYS2 has it), or pass `--addr2line C:\msys64\ucrt64\bin\addr2line.exe`.
- Profile optimized builds with symbols, not the `Profile` CMake config, which adds `DEBUG` bookkeeping.

`python disasm.py out\prof.exe eval_simd` prints annotated assembly for PDB builds. For GCC builds, use `objdump -d -S -C out\prof.exe`.

`kernelbench` times NNUE inference on the real network. It prints the SIMD and scalar evaluations side by side, so you can see that they agree. It uses engine internals, so update it when the NNUE code changes.

## Suggested workflow

1. **Implementation-only optimization:** perft (if board or move generation changed), then `verify.py` (if NNUE changed), then `compare.py` against the previous build. Keep a change only with 0 mismatches and a speedup above your A/A noise.
2. **Search change** (nodes change on purpose): `compare.py --allow-diff` for the cost per node, then SPRT games for strength.
3. **Where is the time going?** Run `sample_profile.py`, then `--func` or `disasm.py` on the hot function, and `kernelbench` for NNUE kernels.
4. **Threads:** `smp.py` for stability and scaling, `affinity_check.py` for thread placement, then games at a fixed time for strength.

## Reference results

Measured on an i7-1270P laptop (4 P-cores used, clock varying about 2.4–4 GHz under load). The code measured is Tomahawk's multithreading branch (then uncommitted) on top of `dd04600`. Numbers are machine-specific; compare ratios, not absolute NPS.

**Single thread, original (`dd04600`) vs. current.** Both builds are unity builds (LTO-equivalent), and all 20 positions give identical searches.

| Toolchain | Original NPS | Current NPS | Total-time speedup | Geomean speedup (95% CI) |
|---|---|---|---|---|
| GCC 16.2 (w64devkit), `compare.py --reps 2` | 325K | 949K | 2.92× | 2.94× (2.85–3.05) |
| clang 21 (zig 0.16), `progress.py --reps 2` | 264K | 781K | 2.95× | — |

**Steps, one at a time** (clang, paired `compare.py`, each 0/20 mismatches):

| Step | Speedup |
|---|---|
| Lazy static eval | 1.14× |
| Fast `givesCheck`, clock polling, fixed-size PV | 1.18× |
| Board: repetition history, piece lookup, castling hash | 1.07× |
| SEE and move generation | 1.12× |
| NNUE accumulator stack (one fused copy+update per ply; undo is a pop) | 1.24× |
| NNUE inference: sparse L1, register-blocked L2/L3 (AVX2) | 1.40× |
| TT prefetch after make-move | 1.06× |
| Lazy SMP refactor at `Threads=1` | 0.99× (noise) |

**Lazy SMP** (current code; scaling is relative to the same build at 1 thread):

| Threads | NPS, clang | Time-to-depth, clang | NPS, GCC | Time-to-depth, GCC |
|---|---|---|---|---|
| 1 | 781K | 1.00× | 1.00M | 1.00× |
| 2 | 1.45M | 1.52× | 1.78M | 1.39× |
| 4 | 2.78M | 2.09× | 2.98M | 1.82× |

The clang rows come from `progress.py` over 20 positions with 2 repetitions. Against the original single-threaded engine, 4 threads reach 6.18× time-to-depth and 10.5× NPS. The GCC rows come from `smp.py` with 1 repetition, so they are noisier. Time-to-depth with Lazy SMP varies from run to run, because threads change each other's TT contents.

## Known gaps

- Linux support is untested.
- `build_ref.ps1` was tested only with CMake's Debug configuration, with zig and with GCC 16.2. Neither toolchain available here can link the Release configuration's LTO: zig's LTO misses MinGW symbols, and the w64devkit GCC was built without LTO. The Release path runs the same `cmake` commands as the build in Tomahawk's README, which your normal GCC handles. The GCC measurements above use `build_unity.ps1` instead.
- `kernelbench.cpp` and the `TOMAHAWK_VERIFY_NNUE` checks depend on engine internals, so update them with the NNUE code.
