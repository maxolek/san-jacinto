"""fastchess helpers shared by the match tools: the binary, per-move limits and tablebases.

Every tool plays its games on utils/match_runner.py: tests/sprt.py, tuning/spsa_tuner.py
and tests/strength_estimator.py directly; tests/smp.py, nnue/registry.py `match`,
tests/tournament.py and tuning/cmaes_tuner.py through utils/fixed_match.py. This module
holds only what they share around the runner:

  FASTCHESS, default_fastchess()  the fastchess binary, found in the order of
                                  utils.match_runner.find_fastchess ($FASTCHESS, the
                                  libs/fastchess build, fastchess on PATH), so every tool's
                                  --fastchess defaults to the binary the runner would use
  limit_args(), limit_label()     fastchess -each keys for exactly one per-move limit, and the
                                  label stored with the games ("depth=8", "st=0.1", "10+0.1")
  tb_args()                       Syzygy adjudication arguments, checked before a run starts
"""
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# fastchess (`make` in the libs/fastchess submodule, or a release binary placed there)
LIBS_FASTCHESS = PROJECT_ROOT / "libs" / "fastchess" / ("fastchess.exe" if os.name == "nt" else "fastchess")


def default_fastchess():
    """$FASTCHESS, the libs/fastchess build or fastchess on PATH, the first that exists (else the libs path).

    The order of utils.match_runner.find_fastchess, so every tool plays with the same binary.
    """
    for candidate in (os.environ.get("FASTCHESS"), str(LIBS_FASTCHESS), shutil.which("fastchess")):
        if candidate and os.path.isfile(candidate):
            return Path(os.path.abspath(candidate))
    return LIBS_FASTCHESS


FASTCHESS = default_fastchess()


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
