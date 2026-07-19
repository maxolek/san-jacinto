"""Database connection helpers and engine probing."""
import sqlite3
import os
import time
import subprocess
import platform
import shutil
from pathlib import Path
import re
import queue 
import threading
from .paths import RAW_DB


# -----------------------
# ----- helpers ---------
# -----------------------

def _int_option(options, key):
    def _get_option(options, key):
        # try variants to handle different naming conventions
        candidates = [
            key,
            key.lower(),
            key.upper(),
            key.replace("_", " "),
            key.replace("_", " ").lower(),
            key.replace("_", " ").title(),
        ]
        for c in candidates:
            if c in options:
                return options[c]
        return None

    value = _get_option(options, key)
    if value is None:
        return None
    raw = value.get("default")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(raw))
        except Exception:
            return None


def _float_option(options, key):
    def _get_option(options, key):
        candidates = [
            key,
            key.lower(),
            key.upper(),
            key.replace("_", " "),
            key.replace("_", " ").lower(),
            key.replace("_", " ").title(),
        ]
        for c in candidates:
            if c in options:
                return options[c]
        return None

    value = _get_option(options, key)
    if value is None:
        return None
    raw = value.get("default")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _float_option_scaled(options, key, scale=100):
    v = _float_option(options, key)
    if v is None:
        return None
    try:
        return float(v) / scale
    except Exception:
        return None


def _bool_option(options, key):
    def _get_option(options, key):
        candidates = [
            key,
            key.lower(),
            key.upper(),
            key.replace("_", " "),
            key.replace("_", " ").lower(),
            key.replace("_", " ").title(),
        ]
        for c in candidates:
            if c in options:
                return options[c]
        return None

    value = _get_option(options, key)
    if value is None:
        return None
    raw = value.get("default")
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw.lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def clear_log_dir(log_dir):
    """Remove all files and subdirectories in a log directory."""
    if not os.path.isdir(log_dir):
        return

    for entry in os.listdir(log_dir):
        path = os.path.join(log_dir, entry)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            else:
                shutil.rmtree(path)
        except Exception as e:
            print(f"[WARN] Failed to delete {path}: {e}")


def get_db(path=None):
    """Get a SQLite connection with Row factory."""
    if path is None:
        path = str(RAW_DB)
    cnxn = sqlite3.connect(path)
    cnxn.row_factory = sqlite3.Row
    return cnxn



def register_engine(cnxn, engine):
    """Register an engine and return its row ID.
    
    If an engine with the same version is already registered, returns its ID instead.
    Prompts the user for a description if one is not provided
    """
    # First, extract version to check for duplicates
    if engine.get("engine_path"):
        engine_path = engine["engine_path"]
        print(f"[ETL] Probing engine UCI at: {engine_path}")
        engine_meta = probe_engine_metadata(engine_path)
        options = engine_meta.get("options", {})
        print(f"[ETL] Probed option keys: {list(options.keys())}")
        version = engine.get("version") or engine_meta.get("version")
        name = engine.get("name") or Path(engine_path).name.removesuffix('.exe')
        #description = engine.get("description") or f"Auto-registered from {engine_path}"

        # prompt user for description instead of auto-filling
        if engine.get("description"): 
            description = engine['description']
        else:
            print(f"\n[ETL] Registering new engine: {name} ({version})")
            name        = input("  Enter a name for this engine version (or press Enter to skip): ").strip()
            description = input("  Enter a description for this engine version (or press Enter to skip): ").strip()
            if not description: 
                description = f"Auto-registered from {engine_path}"

        move_overhead_ms = _int_option(options, "Move Overhead")
        max_threads = _int_option(options, "Threads")
        hash_size_mb = _int_option(options, "Hash")
        pondering = _bool_option(options, "Ponder")

        delta_prune_threshold = _int_option(options, "DELTA_PRUNE_THRESHOLD")
        see_prune_threshold = _int_option(options, "SEE_PRUNE_THRESHOLD")
        aspiration_window = _int_option(options, "ASPIRATION_WINDOW")
        aspiration_start_depth = _int_option(options, "ASPIRATION_START_DEPTH")
        aspiration_depth_scale = _int_option(options, "ASPIRATION_DEPTH_SCALE")
        aspiration_research_scale = _float_option(options, "ASPIRATION_RESEARCH_SCALE")
        draw_eval = _int_option(options, "DRAW_EVAL")
        contempt = _int_option(options, "CONTEMPT")
        r_nmp = _int_option(options, "R_NMP")
        r_lmr_const = _float_option_scaled(options, "R_LMR_CONST", 100)
        r_lmr_denom = _float_option_scaled(options, "R_LMR_DENOM", 100)
        lmr_move_order_threshold = _int_option(options, "LMR_MOVE_ORDER_THRESHOLD")
        lmr_depth_threshold = _int_option(options, "LMR_DEPTH_THRESHOLD")

        # diagnostics: report any missing mapped values
        mapped = {
            'delta_prune_threshold': delta_prune_threshold,
            'see_prune_threshold': see_prune_threshold,
            'aspiration_window': aspiration_window,
            'aspiration_start_depth': aspiration_start_depth,
            'aspiration_depth_scale': aspiration_depth_scale,
            'aspiration_research_scale': aspiration_research_scale,
            'draw_eval': draw_eval,
            'contempt': contempt,
            'r_nmp': r_nmp,
            'r_lmr_const': r_lmr_const,
            'r_lmr_denom': r_lmr_denom,
            'lmr_move_order_threshold': lmr_move_order_threshold,
            'lmr_depth_threshold': lmr_depth_threshold,
        }
        missing = [k for k, v in mapped.items() if v is None]
        if missing:
            print(f"[ETL][WARN] The following mapped engine params were not found or parsed: {missing}")

        if version:
            print(f"[ETL] Saving engine config as: {version}")
            save_engine_config(engine_path, version)
        else:
            print(f"[WARN] Engine version not found; skipping save_config for {engine_path}")
    else:
        version = engine.get("version")
        name = engine.get("name")
        description = engine.get("description")
        move_overhead_ms = engine.get("move_overhead_ms")
        max_threads = engine.get("max_threads")
        hash_size_mb = engine.get("hash_size_mb")
        pondering = engine.get("pondering")
        delta_prune_threshold = engine.get("delta_prune_threshold")
        see_prune_threshold = engine.get("see_prune_threshold")
        aspiration_window = engine.get("aspiration_window")
        aspiration_start_depth = engine.get("aspiration_start_depth")
        aspiration_depth_scale = engine.get("aspiration_depth_scale")
        aspiration_research_scale = engine.get("aspiration_research_scale")
        draw_eval = engine.get("draw_eval")
        contempt = engine.get("contempt")
        r_nmp = engine.get("r_nmp")
        r_lmr_const = engine.get("r_lmr_const")
        r_lmr_denom = engine.get("r_lmr_denom")
        lmr_move_order_threshold = engine.get("lmr_move_order_threshold")
        lmr_depth_threshold = engine.get("lmr_depth_threshold")

    # Check if engine with this version already exists
    # Check if engine with this version already exists
    if version:
        existing = cnxn.execute(
            "SELECT id FROM engines WHERE version=?", (version,)
        ).fetchone()
        if existing is not None:
            print(f"[ETL] Engine {name} ({version}) already registered with ID {existing[0]}")
            return existing[0]

    cur = cnxn.execute(
        """
        INSERT INTO engines (
            name, version, description, compile_flags,
            move_overhead_ms, max_threads, hash_size_mb, pondering,
            delta_prune_threshold, see_prune_threshold,
            aspiration_window, aspiration_start_depth, aspiration_depth_scale,
            aspiration_research_scale, draw_eval, contempt,
            r_nmp, r_lmr_const, r_lmr_denom,
            lmr_move_order_threshold, lmr_depth_threshold
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            version,
            description,
            engine.get("compile_flags"),
            # uci engine options
            move_overhead_ms,
            max_threads,
            hash_size_mb,
            pondering,
            # search params
            delta_prune_threshold,
            see_prune_threshold,
            aspiration_window,
            aspiration_start_depth,
            aspiration_depth_scale,
            aspiration_research_scale,
            draw_eval,
            contempt,
            r_nmp,
            r_lmr_const,
            r_lmr_denom,
            lmr_move_order_threshold,
            lmr_depth_threshold
        )
    )
    cnxn.commit()
    print(f"[ETL] Registered engine {name} ({version})")
    return cur.lastrowid


def get_engine_id(cnxn, version=None):
    """Look up engine ID by version string."""
    row = cnxn.execute(
        "SELECT id FROM engines WHERE version=?",
        (version,)
    ).fetchone()

    # if no engine found - register and return
    if row is None:
        register_engine(
            cnxn,
            {"engine_path": f"engines/dev/{version}.exe"}
        )

        row = cnxn.execute(
            "SELECT id FROM engines WHERE version=?",
            (version,)
        ).fetchone()

        if row is None:
            raise RuntimeError(f"Failed to register engine {version}")

    return row[0]


def probe_engine_metadata(engine_path, timeout=10.0):
    system = platform.system()
    engine_path = os.path.abspath(engine_path)

    if system == "Windows" and not engine_path.lower().endswith(".exe"):
        if os.path.exists(engine_path + ".exe"):
            engine_path += ".exe"

    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine not found at path: {engine_path}")

    def _run_probe(command):
        p = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )

        q = queue.Queue()

        def _reader():
            try:
                for line in p.stdout:
                    q.put(line)
            except Exception:
                pass
            finally:
                q.put(None)  # sentinel: stdout closed

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

        meta = {}
        options = {}
        start = time.time()
        try:
            p.stdin.write(f"{command}\n")
            p.stdin.flush()

            while True:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    raise RuntimeError(f"Timeout waiting for UCI response from {engine_path}")

                try:
                    line = q.get(timeout=min(remaining, 0.5))
                except queue.Empty:
                    continue

                if line is None:
                    raise RuntimeError(f"Engine closed stdout before uciok: {engine_path}")

                line = line.strip()
                if line.startswith("id name"):
                    meta["name"] = line[len("id name"):].strip()
                elif line.startswith("id version"):
                    meta["version"] = line[len("id version"):].strip()
                elif line.startswith("id author"):
                    meta["author"] = line[len("id author"):].strip()
                elif line.startswith("option name "):
                    m = re.match(r'^option name (.+?) type (\w+)(?: default (".*?"|[^ ]+))?(?: min ([^ ]+))?(?: max ([^ ]+))?', line)
                    if m:
                        opt_name = m.group(1).strip()
                        opt_type = m.group(2).strip()
                        default_val = m.group(3)
                        min_val = m.group(4)
                        max_val = m.group(5)

                        if isinstance(default_val, str):
                            default_val = default_val.strip()
                            if default_val.startswith('"') and default_val.endswith('"'):
                                default_val = default_val[1:-1]

                        entry = {"type": opt_type, "default": default_val}
                        if min_val is not None:
                            entry["min"] = min_val
                        if max_val is not None:
                            entry["max"] = max_val

                        options[opt_name] = entry
                elif line == "uciok":
                    break

            try:
                p.stdin.write("quit\n")
                p.stdin.flush()
                p.wait(timeout=1)
            except subprocess.TimeoutExpired:
                p.kill()

        finally:
            if p.poll() is None:
                p.kill()

        if "version" not in meta:
            raise RuntimeError(f"Failed to probe engine metadata: {engine_path}")

        meta["options"] = options
        return meta

    try:
        return _run_probe("uci_dev")
    except RuntimeError:
        return _run_probe("uci")


def save_engine_config(engine_path, config_name, timeout=10.0):
    """Run the engine via UCI and issue save_config <name>."""
    system = platform.system()
    engine_path = os.path.abspath(engine_path)

    if system == "Windows" and not engine_path.lower().endswith(".exe"):
        if os.path.exists(engine_path + ".exe"):
            engine_path += ".exe"

    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine not found at path: {engine_path}")

    p = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    start = time.time()
    try:
        p.stdin.write("uci\n")
        p.stdin.flush()

        while True:
            if time.time() - start > timeout:
                raise RuntimeError(f"Timeout waiting for UCI response from {engine_path}")
            line = p.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            if line.strip() == "uciok":
                break

        p.stdin.write(f"save_config {config_name}\n")
        p.stdin.flush()
        p.stdin.write("quit\n")
        p.stdin.flush()
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        raise RuntimeError(f"Timeout while saving engine config for {engine_path}")
    finally:
        if p.poll() is None:
            p.kill()

    return True


def extract_engine_id_from_search(search_path):
    """Extract engine_id (version string) from the first search object in a search log."""
    import json

    with open(search_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                print(f"[WARN] skipping malformed JSON line {line_no} in {search_path}")
                continue

            if not isinstance(data, dict):
                print(f"[WARN] skipping non-object JSON line {line_no} in {search_path}: {type(data).__name__}")
                continue

            if "engine_id" in data:
                return data["engine_id"]

    return None
