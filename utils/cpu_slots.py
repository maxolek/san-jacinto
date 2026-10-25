"""CPU claims of concurrent runs: one lock file per logical CPU.

utils/match_runner.py plans every tool's game slots (plan_slots, from the OS's core
layout in utils/cpu_topology.py) and claims each slot's CPUs here, so two runs at once
never pin games to the same CPU: CPUs another run holds are skipped (held_elsewhere).
A claim is an exclusive lock on cpu<N>.lock in the per-user temp folder
(san-jacinto-cpus in %TEMP% on Windows, /tmp/san-jacinto-cpus-<uid> elsewhere), held
by this process: it dies with the process, even a killed one, and unclaim() (the
runner's release_slots) gives it back earlier.

Slots no longer follow the CPU order the engine reports on `uci` ("info string cpu
topology ... auto order ..."): the runner reads the topology from the OS and tells
engines with an Affinity option (Tomahawk) their slot through that option.
"""
import os
import sys
import tempfile

_claims = {}  # cpu -> lock file descriptor held by this process


def claim(cpu):
    """Hold `cpu` for this process; False if another run holds it (True if this process does)."""
    return _claim(cpu)


def unclaim(cpu):
    """Give back one cpu this process claimed (no-op if it holds none)."""
    if cpu in _claims:
        _unclaim(cpu)


def claimed():
    """The cpus this process holds."""
    return set(_claims)


def held_elsewhere(cpu):
    """True if another process holds `cpu` (checked by briefly taking its lock)."""
    if cpu in _claims:
        return False
    if _claim(cpu):
        _unclaim(cpu)
        return False
    return True


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
