"""Build provenance: which exact engine build played, and how it was built.

An engine is identified by its build, not its version string (several builds,
DEV and Prod, rebuilds, local patches, share a version):

    engine_key = sha256(binary)[:24] + "+" + (sha256(net)[:24] | "nonet")

Tomahawk loads its NNUE net from a file at runtime (the `nnue_weight_file`
default it reports on `uci` for DEV engines only), so the same binary with another net is another
engine. Stockfish's `EvalFile` is handled the same way.

How the build was made is recorded in a sidecar written next to the binary when
it is built (`<exe>.build.json`: git commit, branch, dirty flag and diff hash,
build type, compiler and flags from CMakeCache.txt). release.py writes it; for
binaries built by hand:

    python -m data.etl.provenance stamp <exe> --source ../tomahawk --build-type DEV [--build-dir ../tomahawk/build_dev]
    python -m data.etl.provenance show <exe>
    python -m data.etl.provenance register <exe> [--description TEXT]

A sidecar whose recorded binary hash no longer matches the binary (rebuilt
without stamping) is ignored, so stale git information is never attached.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .db import probe_engine_metadata, register_engine
from .schema import ensure_schema

NET_OPTIONS = ("nnue_weight_file", "EvalFile")  # Tomahawk, Stockfish
CMAKE_KEYS = ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_GENERATOR")


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def engine_key(binary_sha256, net_sha256=None):
    return f"{binary_sha256[:24]}+{net_sha256[:24] if net_sha256 else 'nonet'}"


def build_info_path(exe):
    return Path(f"{os.path.abspath(exe)}.build.json")


def _git(source_dir, *args):
    try:
        out = subprocess.run(["git", "-C", str(source_dir), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(source_dir):
    """Commit, branch and dirty state of the source tree (None outside a git checkout)."""
    commit = _git(source_dir, "rev-parse", "HEAD")
    if not commit:
        return None
    status = _git(source_dir, "status", "--porcelain", "--untracked-files=no") or ""
    info = dict(commit=commit, branch=_git(source_dir, "rev-parse", "--abbrev-ref", "HEAD"),
                describe=_git(source_dir, "describe", "--always", "--dirty", "--tags"), dirty=bool(status))
    if status:
        diff = _git(source_dir, "diff", "HEAD") or ""
        info["dirty_files"] = [line[3:] for line in status.splitlines()]
        info["dirty_diff_sha256"] = hashlib.sha256(diff.encode("utf-8", "replace")).hexdigest()
    return info


def cmake_info(build_dir):
    """Compiler, flags and build type from <build_dir>/CMakeCache.txt."""
    cache = Path(build_dir) / "CMakeCache.txt" if build_dir else None
    if not cache or not cache.is_file():
        return None
    entries = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        name, sep, value = line.partition("=")
        if sep and not line.startswith(("#", "//")):
            entries[name.split(":", 1)[0]] = value
    info = {key: entries.get(key) for key in CMAKE_KEYS}
    build_type = (entries.get("CMAKE_BUILD_TYPE") or "").upper()
    if build_type:
        info[f"CMAKE_CXX_FLAGS_{build_type}"] = entries.get(f"CMAKE_CXX_FLAGS_{build_type}")
    compiler = entries.get("CMAKE_CXX_COMPILER")
    if compiler:
        try:
            version = subprocess.run([compiler, "--version"], capture_output=True, text=True, timeout=30)
            info["compiler_version"] = version.stdout.splitlines()[0] if version.stdout else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    return info


def write_build_info(exe, source_dir, build_type=None, build_dir=None, version=None):
    """Record how `exe` was built in <exe>.build.json; returns the record."""
    record = dict(
        binary_sha256=sha256_file(exe),
        built_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        version=version,
        build_type=build_type,
        source_dir=str(Path(source_dir).resolve()),
        git=git_info(source_dir),
        cmake=cmake_info(build_dir),
    )
    build_info_path(exe).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def read_build_info(exe, binary_sha256=None):
    """The sidecar, if it belongs to this exact binary (else None)."""
    path = build_info_path(exe)
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("binary_sha256") != (binary_sha256 or sha256_file(exe)):
        print(f"[PROVENANCE] ignoring {path.name}: it describes an earlier build of this file")
        return None
    return record


def _net_file(options, exe):
    for name in NET_OPTIONS:
        value = (options.get(name) or {}).get("default")
        if value and value not in ("<empty>", "<default>"):
            path = Path(value) if os.path.isabs(value) else Path(os.path.dirname(os.path.abspath(exe))) / value
            return path if path.is_file() else None
    return None


def engine_provenance(exe, meta=None):
    """Everything known about the build at `exe`; `meta` is a probe_engine_metadata result to reuse."""
    exe = os.path.abspath(exe)
    meta = meta or probe_engine_metadata(exe)
    binary_sha256 = sha256_file(exe)
    net = _net_file(meta.get("options", {}), exe)
    net_sha256 = sha256_file(net) if net else None
    build = read_build_info(exe, binary_sha256)
    git = (build or {}).get("git") or {}
    cmake = (build or {}).get("cmake") or {}
    return dict(
        engine_key=engine_key(binary_sha256, net_sha256),
        binary_sha256=binary_sha256,
        binary_path=exe,
        binary_size=os.path.getsize(exe),
        binary_mtime_utc=datetime.fromtimestamp(os.path.getmtime(exe), timezone.utc).isoformat(timespec="seconds"),
        net_sha256=net_sha256,
        net_path=str(net) if net else None,
        uci=dict(name=meta.get("name"), version=meta.get("version"), author=meta.get("author"),
                 info_strings=meta.get("info_strings", [])),
        git_commit=git.get("commit"),
        git_branch=git.get("branch"),
        git_dirty=git.get("dirty"),
        build_type=(build or {}).get("build_type") or cmake.get("CMAKE_BUILD_TYPE"),
        compile_flags=" ".join(filter(None, [cmake.get("CMAKE_CXX_FLAGS"), cmake.get(
            f"CMAKE_CXX_FLAGS_{(cmake.get('CMAKE_BUILD_TYPE') or '').upper()}")])) or None,
        build=build,
    )


def _store(cnxn, engine_id, prov):
    cnxn.execute(
        """UPDATE engines SET engine_key=?, binary_sha256=?, binary_path=?, net_sha256=?, net_path=?,
               git_commit=?, git_branch=?, git_dirty=?, build_type=?, provenance=?,
               compile_flags=COALESCE(?, compile_flags)
           WHERE id=?""",
        (prov["engine_key"], prov["binary_sha256"], prov["binary_path"], prov["net_sha256"], prov["net_path"],
         prov["git_commit"], prov["git_branch"],
         None if prov["git_dirty"] is None else int(prov["git_dirty"]), prov["build_type"],
         json.dumps(prov), prov["compile_flags"], engine_id))
    cnxn.commit()


def resolve_engine(cnxn, exe, name=None, description=None):
    """The engines.id of the build at `exe`, registering it if new.

    Lookup is by engine_key. A row registered before provenance existed (no key)
    is adopted by the canonical binary of its version, <version>[.exe] as release.py
    names builds, when it is the only such row, so its history stays attached. A
    differently named build reporting that version (a patch, a test build) is
    registered as a new engine instead of taking over that history.
    """
    ensure_schema(cnxn)
    meta = probe_engine_metadata(exe)
    prov = engine_provenance(exe, meta)
    row = cnxn.execute("SELECT id, binary_path FROM engines WHERE engine_key=?", (prov["engine_key"],)).fetchone()
    if row:
        if row[1] != prov["binary_path"]:
            cnxn.execute("UPDATE engines SET binary_path=? WHERE id=?", (prov["binary_path"], row[0]))
            cnxn.commit()
        return row[0]
    version = meta.get("version")
    legacy = cnxn.execute("SELECT id FROM engines WHERE engine_key IS NULL AND version=?", (version,)).fetchall()
    if len(legacy) == 1 and version in (Path(exe).name, Path(exe).stem):  # Linux/macOS builds have no .exe
        print(f"[PROVENANCE] attaching build {prov['engine_key']} to engine id {legacy[0][0]} "
              f"(registered as version {version} before builds were tracked)")
        _store(cnxn, legacy[0][0], prov)
        return legacy[0][0]
    if description is None and not sys.stdin.isatty():
        description = f"Auto-registered build {prov['engine_key']}"
    engine_id = register_engine(cnxn, dict(engine_path=exe, name=name, description=description,
                                           engine_key=prov["engine_key"]))
    _store(cnxn, engine_id, prov)
    return engine_id


def _field(row, key):
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def locate_engine(row, candidates=()):
    """Path of the registered build `row` (a mapping with binary_path/binary_sha256), or None.

    The recorded path is used if the file there is still that build; otherwise the
    first of `candidates` (e.g. engines/prod/<version>.exe) that is. A build that
    loads its net from a file (net_path/net_sha256 recorded) is only found while
    that file is unchanged: the same binary with another net is another engine.
    Rows without a recorded hash (registered before builds were tracked) accept
    the first existing candidate, as before.
    """
    paths = [p for p in [_field(row, "binary_path"), *candidates] if p]
    if not _field(row, "binary_sha256"):
        return next((str(p) for p in paths if os.path.isfile(p)), None)
    for path in paths:
        if os.path.isfile(path) and sha256_file(path) == row["binary_sha256"]:
            net_sha256, net_path = _field(row, "net_sha256"), _field(row, "net_path")
            if net_sha256 and not (net_path and os.path.isfile(net_path) and sha256_file(net_path) == net_sha256):
                print(f"[PROVENANCE] {path}: the net it loads ({net_path}) is not the registered one")
                return None
            return str(path)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Engine build provenance")
    sub = parser.add_subparsers(dest="command", required=True)
    stamp = sub.add_parser("stamp", help="record how a binary was built (<exe>.build.json)")
    stamp.add_argument("exe")
    stamp.add_argument("--source", required=True, help="engine source checkout it was built from")
    stamp.add_argument("--build-type")
    stamp.add_argument("--build-dir", help="CMake build dir (for compiler and flags)")
    stamp.add_argument("--version")
    show = sub.add_parser("show", help="print what is known about a binary")
    show.add_argument("exe")
    register = sub.add_parser("register", help="register a binary in the raw database")
    register.add_argument("exe")
    register.add_argument("--name")
    register.add_argument("--description")
    register.add_argument("--db", help="raw database (default: CHESS_RAW_DB / data.etl.paths.RAW_DB)")
    args = parser.parse_args(argv)

    if args.command == "stamp":
        print(json.dumps(write_build_info(args.exe, args.source, args.build_type, args.build_dir, args.version),
                         indent=2))
    elif args.command == "show":
        print(json.dumps(engine_provenance(args.exe), indent=2))
    else:
        import sqlite3
        from .paths import RAW_DB, connect_raw
        cnxn = connect_raw(args.db or RAW_DB)
        print(f"engine id {resolve_engine(cnxn, args.exe, args.name, args.description)}")
        cnxn.close()


if __name__ == "__main__":
    main()
