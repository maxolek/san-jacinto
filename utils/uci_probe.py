"""Small UCI checks run before tests that depend on an engine feature.

honors_node_limit(exe): does `go nodes N` actually bound the search? An engine
that ignores the node limit makes a "fixed nodes" test silently test something
else; Tomahawk, for example, parses `go nodes` but its SearchLimits only know
time and depth, so with no clock it searches for its minimum time slice. The
check: `go nodes 1000000000` must NOT return within a second (no engine searches
a billion nodes that fast) and `go nodes 1000` must return within a few seconds.

uci_options(exe): the options an engine declares on plain `uci` with their type and
bounds, i.e. the only ones fastchess will set (it skips undeclared options and
out-of-range values with a warning); require_threads uses it.
"""
import os
import queue
import subprocess
import threading
import time

_cache = {}


class _Session:
    def __init__(self, exe):
        self.proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                     cwd=os.path.dirname(exe) or None)
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def send(self, command):
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def wait_for(self, prefix, timeout):
        """Lines up to and including the first starting with `prefix`; None on timeout or exit."""
        seen, deadline = [], time.monotonic() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return None
            if line is None:
                return None
            seen.append(line)
            if line.startswith(prefix):
                return seen

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if self.proc.poll() is None:
            self.proc.kill()


def _key(exe):
    exe = os.path.abspath(exe)
    return exe, os.path.getmtime(exe)


def uci_options(exe, timeout=15.0):
    """Options declared on `uci` (what fastchess can set): {name: {"type", "min", "max"}}."""
    key = ("options",) + _key(exe)
    if key not in _cache:
        session = _Session(os.path.abspath(exe))
        try:
            session.send("uci")
            lines = session.wait_for("uciok", timeout) or []
        finally:
            session.close()
        options = {}
        for line in lines:
            if line.startswith("option name "):
                name, _, rest = line[len("option name "):].partition(" type ")
                fields = rest.split()
                spec = {"type": fields[0] if fields else None}
                for bound in ("min", "max"):
                    if bound in fields[:-1]:
                        try:
                            spec[bound] = int(fields[fields.index(bound) + 1])
                        except ValueError:
                            pass
                options[name.strip()] = spec
        _cache[key] = options
    return _cache[key]


def require_threads(exe, threads):
    """Raise ValueError unless the engine accepts Threads=threads (fastchess would only warn and skip it)."""
    spec = uci_options(exe).get("Threads")
    if spec is None:
        raise ValueError(f"{exe} declares no Threads option, so it cannot search with {threads} threads")
    if spec.get("max") is not None and spec["max"] < threads:
        raise ValueError(f"{exe} accepts at most Threads={spec['max']}, not {threads}; the games would silently "
                         "be played with fewer threads")


def honors_node_limit(exe, timeout=15.0):
    """True / False as described above; None if the engine could not be checked."""
    key = ("nodes",) + _key(exe)
    if key in _cache:
        return _cache[key]
    session = _Session(os.path.abspath(exe))
    result = None
    try:
        session.send("uci")
        if session.wait_for("uciok", timeout) is None:
            return None
        session.send("isready")
        if session.wait_for("readyok", timeout) is None:
            return None
        session.send("position startpos")
        session.send("go nodes 1000000000")
        if session.wait_for("bestmove", 1.0) is not None:
            result = False  # returned at once: the limit is not what ends the search
        else:
            session.send("stop")
            if session.wait_for("bestmove", timeout) is None:
                return None
            session.send("isready")  # drain anything the stop produced (some engines print a 2nd bestmove)
            if session.wait_for("readyok", timeout) is None:
                return None
            session.send("go nodes 1000")
            if session.wait_for("bestmove", 5.0) is not None:
                result = True
            else:
                session.send("stop")
                session.wait_for("bestmove", timeout)
                result = False  # searches on regardless of the limit
    finally:
        session.close()
    _cache[key] = result
    return result


def require_node_limit(engines):
    """Raise ValueError if any engine ignores `go nodes`; warn if it could not be checked."""
    for exe in dict.fromkeys(os.path.abspath(e) for e in engines):
        honored = honors_node_limit(exe)
        if honored is False:
            raise ValueError(
                f"{exe} ignores `go nodes`, so a fixed-nodes test would not limit its search by nodes. "
                "The engine needs a node limit (for Tomahawk: a node budget in SearchLimits, set from "
                "`go nodes` in computeSearchTime and checked where the clock is polled).")
        if honored is None:
            print(f"[NODES] warning: could not check that {exe} honors `go nodes`")
