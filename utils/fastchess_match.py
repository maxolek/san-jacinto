"""Engine-vs-engine matches with fastchess: one runner for SPRT, SMP ladders and net matches.

A match is a Candidate and a Baseline playing opening pairs: games 2k+1 and 2k+2
share an opening with colours reversed. prepare_match() plans each concurrent
game's CPUs (utils.cpu_slots), creates the run folder and builds the fastchess
process(es); run_match() plays it, journals every finished game and feeds a
utils.pentanomial.PentanomialSPRT state. A sequential state stops the match at
the first SPRT decision; a fixed one (sequential=False) plays every pair.

Run folder: config.json, fastchess_stdout.log, games.jsonl, stats.jsonl,
summary.json, games.pgn, fastchess.json (fastchess's resume state).
"""
import contextlib
import json
import os
import queue
import random
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from utils import cpu_slots, uci_probe
from utils.pentanomial import FINISHED_GAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# fastchess (`make` in the libs/fastchess submodule, or a release binary placed there)
FASTCHESS = PROJECT_ROOT / "libs" / "fastchess" / ("fastchess.exe" if os.name == "nt" else "fastchess")


@dataclass
class Side:
    path: str
    options: list = field(default_factory=list)  # fastchess engine keys, e.g. "option.Hash=64"
    threads: int = 1


@dataclass
class Match:
    run_dir: Path
    command: list
    processes: list  # [(cmd, cpus or None)]
    slots: list      # cpu slots, or None when games are not pinned


def limit_args(depth=None, time=None, tc=None, nodes=None):
    """fastchess -each keys for exactly one per-move limit."""
    if sum(x is not None for x in (depth, time, tc, nodes)) != 1:
        raise ValueError("Specify exactly one of --depth, --time, --tc or --nodes")
    if depth is not None:
        return [f"depth={depth}", "tc=inf"]
    if time is not None:
        return [f"st={time}", "timemargin=30"]
    if nodes is not None:
        return [f"nodes={nodes}", "tc=inf"]
    return [f"tc={tc}"]


def limit_label(depth=None, time=None, tc=None, nodes=None):
    if depth is not None:
        return f"depth={depth}"
    if time is not None:
        return f"st={time:g}"
    if nodes is not None:
        return f"nodes={nodes}"
    return str(tc)


def book_args(book, plies):
    if not book:
        return []
    return ["-openings", f"file={os.path.abspath(book)}", f"format={os.path.splitext(str(book))[1][1:]}",
            "order=random", f"plies={plies}"]


def tb_args(tb=None, tbpieces=None):
    """Syzygy adjudication; tb is one or more folders (';'-separated on Windows, ':' elsewhere).

    fastchess needs only the WDL tables (*.rtbw); a folder without any is refused here,
    before a run starts, rather than by fastchess at startup.
    """
    if not tb:
        return []
    empty = [d for d in str(tb).split(os.pathsep) if d and not any(Path(d).glob("*.rtbw"))]
    if empty:
        raise ValueError(f"no Syzygy WDL tables (*.rtbw) in {', '.join(empty)}. Adjudication needs only the WDL "
                         "files: the 3-4-5 piece set is about 380 MB, e.g. https://tablebase.lichess.ovh/tables/"
                         "standard/3-4-5-wdl/")
    return ["-tb", str(tb)] + (["-tbpieces", str(tbpieces)] if tbpieces else [])


def stop_fastchess(proc, force=False):
    """Stop only this run's process tree, including the engine children.

    fastchess handles SIGINT on POSIX. Windows has no equivalent targeted
    SIGINT delivery; taskkill /T stops the dedicated tree there. Our result
    journal is flushed before shutdown, independent of PGN finalization.
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode and proc.poll() is None:
            raise RuntimeError(f"Could not stop fastchess process tree: {result.stderr}")
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGINT)
        except ProcessLookupError:
            pass


def run_fastchess(processes, state, run_dir, plotter=None):
    """Journal raw results and ordered pair snapshots, with or without a GUI.

    `processes` is [(cmd, cpus)]: one fastchess process (cpus None), or one per cpu
    slot, started on the slot's cpus; slot games are renumbered into one sequence of
    opening pairs (cpu_slots.global_game). Stops at state.decided.
    """
    run_dir = Path(run_dir)
    process_options = (dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                       if os.name == "nt" else dict(start_new_session=True))
    slots = len(processes)
    stopping_at = None
    failure = None
    with contextlib.ExitStack() as stack:
        transcript = stack.enter_context(open(run_dir / "fastchess_stdout.log", "w", encoding="utf-8"))
        games = stack.enter_context(open(run_dir / "games.jsonl", "w", encoding="utf-8"))
        history = stack.enter_context(open(run_dir / "stats.jsonl", "w", encoding="utf-8"))
        line_queue = queue.Queue()
        procs, threads = [], []

        def reader(slot, proc):
            try:
                for line in proc.stdout:
                    line_queue.put((slot, line))
            finally:
                line_queue.put((slot, None))

        try:
            for slot, (cmd, cpus) in enumerate(processes):
                options = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                               **process_options)
                proc = cpu_slots.popen_pinned(cmd, cpus, **options) if cpus else subprocess.Popen(cmd, **options)
                procs.append(stack.enter_context(proc))
                threads.append(threading.Thread(target=reader, args=(slot, proc), daemon=True))
                threads[-1].start()
            running = slots
            while running:
                # Bound work per pass so high concurrency cannot starve the GUI.
                for _ in range(128):
                    try:
                        slot, line = line_queue.get(timeout=0.05)
                    except queue.Empty:
                        break
                    if line is None:
                        running -= 1
                        ret = procs[slot].wait(timeout=10)
                        if ret != 0 and stopping_at is None:
                            raise subprocess.CalledProcessError(ret, processes[slot][0])
                        if not running:
                            break
                        continue
                    match = FINISHED_GAME.search(line)
                    if match:
                        local, white, black, outcome = match.groups()
                        number = cpu_slots.global_game(int(local), slot, slots)
                        line = line.replace(f"Finished game {local} ", f"Finished game {number} ", 1)
                    if slots > 1:
                        line = f"[slot {slot}] {line}"
                    print(line, end="")
                    transcript.write(line)
                    transcript.flush()
                    if not match:
                        continue
                    record = dict(game=number, white=white, black=black, result=outcome)
                    if slots > 1:
                        record["slot"] = slot
                    games.write(json.dumps(record) + "\n")
                    games.flush()
                    updates = state.add_game(number, white, black, outcome)
                    for snapshot in updates:
                        history.write(json.dumps(snapshot, allow_nan=False) + "\n")
                        history.flush()
                    if updates:
                        snapshot = updates[-1]
                        print(f"[PENTA] pairs={snapshot['pairs']} bins={snapshot['pentanomial']} "
                              f"LLR={snapshot['llr']:+.4f} result={snapshot['result']}")
                    if state.decided and stopping_at is None:
                        stopping_at = time.monotonic()
                        for proc in procs:
                            stop_fastchess(proc)
                    if plotter:
                        for snapshot in updates:
                            plotter.update(snapshot)
                if plotter:
                    plotter._pump_events()
                if stopping_at is not None and time.monotonic() - stopping_at > 10:
                    for proc in procs:
                        stop_fastchess(proc, force=True)
        except BaseException as exc:
            failure = str(exc) or type(exc).__name__
            for proc in procs:
                stop_fastchess(proc, force=True)
            for proc in procs:
                proc.wait(timeout=10)
            raise
        finally:
            summary = state.snapshot()
            summary["error"] = failure
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
            for thread in threads:
                thread.join(timeout=2)
    return state.snapshot()


def slot_processes(cmd, slots, pairs, run_dir, book):
    """One fastchess process per cpu slot; together they play `pairs` opening pairs.

    Each process plays its games one at a time (-concurrency 1) and gets its own PGN and
    state file. With a book, all shuffle it with one seed and play consecutive segments,
    so no opening repeats, as in a single process. Pairs are split evenly up front and
    tested in order, so the slowest slot sets the pace (matters for --depth on slower cores).
    """
    seed = random.SystemRandom().randrange(1, 2**31)
    processes, start = [], 1
    for i, (rounds, cpus) in enumerate(zip(cpu_slots.split_rounds(pairs, len(slots)), slots)):
        extra = ["-concurrency", "1", "-rounds", str(rounds),
                 "-pgnout", f"file={run_dir / f'games.slot{i}.pgn'}",
                 "-config", f"outname={run_dir / f'fastchess.slot{i}.json'}"]
        if book:
            extra += ["-srand", str(seed), "-openings", f"start={start}"]
        processes.append((cmd + extra, cpus))
        start += rounds
    return processes


def merge_slot_pgns(run_dir, count):
    """Concatenate the slot processes' PGNs into games.pgn."""
    with open(Path(run_dir) / "games.pgn", "a", encoding="utf-8") as merged:
        for i in range(count):
            part = Path(run_dir) / f"games.slot{i}.pgn"
            if part.exists():
                merged.write(part.read_text(encoding="utf-8"))
                part.unlink()


def prepare_match(candidate, baseline, *, limits, pairs, concurrency, run_dir, fastchess=FASTCHESS,
                  book=None, book_depth=16, pin=True, maxmoves=100, extra=(), meta=None):
    """Plan cpus, create `run_dir`, build the fastchess process(es) and write config.json.

    Cpus are planned before the folder is created, so a capacity error leaves nothing
    behind. A nodes limit is refused for engines that ignore `go nodes`.
    """
    if any(arg.startswith("nodes=") for arg in limits):
        uci_probe.require_node_limit([candidate.path, baseline.path])
    for side in (candidate, baseline):
        if side.threads > 1:
            uci_probe.require_threads(os.path.abspath(side.path), side.threads)
    threads = max(candidate.threads, baseline.threads)
    slots = cpu_slots.plan([candidate.path, baseline.path], threads, concurrency, enabled=pin)
    try:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True)
        sides = []
        for name, side in (("Candidate", candidate), ("Baseline", baseline)):
            path = os.path.abspath(side.path)
            options = list(side.options) + [f"option.Threads={side.threads}"]
            if slots is None and threads > 1 and concurrency > 1:
                options.append("option.Affinity=none")  # auto would pin every engine's threads to the same cpus
            sides += ["-engine", f"name={name}", f"cmd={path}", f"dir={os.path.dirname(path)}", *options]
        if slots is None and threads > 1 and concurrency > 1:
            print("[AFFINITY] warning: multi-threaded games are not pinned; engines use Affinity=none")
        book_block = book_args(book, book_depth)
        cmd = [
            str(fastchess),
            # Keep fastchess's own state file (saved on exit) with this run, not in the cwd
            "-config", f"outname={run_dir / 'fastchess.json'}",
            *sides,
            "-each", "proto=uci", *limits,
            # each round plays one opening twice with colors reversed (games 2k+1, 2k+2)
            "-maxmoves", str(maxmoves),
            "-games", "2",
            "-rounds", str(pairs),
            *book_block,
            *extra,
            # -force-concurrency: like cutechess, allow more games than logical CPUs
            "-concurrency", str(concurrency), "-force-concurrency",
            "-pgnout", f"file={run_dir / 'games.pgn'}",
        ]
        if slots and threads == 1:
            cmd += cpu_slots.fastchess_args(slots)  # fastchess pins each game to its cpu
            processes = [(cmd, None)]
        elif slots:
            processes = slot_processes(cmd, slots, pairs, run_dir, bool(book_block))
        else:
            processes = [(cmd, None)]
        (run_dir / "config.json").write_text(json.dumps(dict(
            meta or {}, command=cmd, slots=slots,
            slot_commands=[c for c, _ in processes] if len(processes) > 1 else None), indent=2, default=str),
            encoding="utf-8")
    except BaseException:
        cpu_slots.release()
        raise
    return Match(run_dir, cmd, processes, slots)


def run_match(match, state, plotter=None):
    """Play a prepared match; returns the final state snapshot."""
    print(" ".join(match.command))
    if len(match.processes) > 1:
        print(f"[MATCH] as {len(match.processes)} processes, one per game slot (-concurrency 1 each)")
    try:
        return run_fastchess(match.processes, state, match.run_dir, plotter)
    finally:
        cpu_slots.release()
        if len(match.processes) > 1:
            merge_slot_pgns(match.run_dir, len(match.processes))
