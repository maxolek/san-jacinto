"""Synthetic Tomahawk-like telemetry and a raw SQLite database built from it.

write_logs() generates game/search/timing/root_moves JSONL logs in the engine's
format; build_raw_db() creates the raw schema, ingests those logs with the real
ETL (data.etl) and then adds what engine logs don't carry: match-runner
environment columns, SPRT rows (measured Elo as REAL) and an STS run.
analytics_fixture() runs the real pipeline CLI on it once per test process.
Everything is deterministic for a given seed.
"""
import atexit
import contextlib
import importlib
import io
import json
import os
import random
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import chess

VERSIONS = ("1.0.0", "1.1.0", "1.2.0")
STRENGTH = {"1.0.0": 0.0, "1.1.0": 30.0, "1.2.0": 60.0}
FUNCTIONS = ("ROOT", "MAKEMOVE", "UNMAKE_MOVE", "MOVEGEN", "SCORE_ORDER", "NNUE", "EVAL", "SEE",
             "TT_PROBE", "TT_STORE")
MATE_SCORE = 100_000
# The first game opens with a named ECO variation (A00 Polish, Tuebingen variation).
FIRST_GAME_OPENING = ("b2b4", "g8h6")
# A match game whose moves can't be replayed: its recorded "Name, Variation" label is used.
FALLBACK_OPENING = ("Sicilian defence", "Najdorf variation")
# A match game from an EPD book: moves relative to the book position (Closed Ruy Lopez after 5...Be7).
# Only a prefix replays from the start (1.c3 b5 = "Saragossa"), so the recorded label must win.
EPD_BOOK_MOVES = ("c2c3", "b7b5", "b3c2", "d7d6")
EPD_BOOK_OPENING = ("Ruy Lopez", "Closed")
# A local path in an engine description: must never reach the public export.
PRIVATE_DESCRIPTION = r"Auto-registered from C:\Users\someone\engines\dev\{version}.exe"
# Builds that log three kinds of PVS re-search (share of the nodes) instead of one count, as newer
# Tomahawk builds do (include/search_stats.h); the others log pvs_researches.
PVS_KIND_VERSIONS = ("1.2.0",)
PVS_KINDS = {"depth": 0.0006, "full": 0.0003, "root": 0.0001}


def write_logs(directory, n_games=12, seed=7):
    """Write the four JSONL logs to `directory`; returns (games, searches)."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    files = {name: open(out / f"{name}.jsonl", "w", encoding="utf-8")
             for name in ("game", "search", "timing", "root_moves")}
    searches = 0
    try:
        for game_index in range(n_games):
            white, black = rng.sample(VERSIONS, 2)
            game_uuid = str(uuid.UUID(int=rng.getrandbits(128)))
            board = chess.Board()
            moves = []
            advantage = rng.gauss((STRENGTH[white] - STRENGTH[black]) / 2, 20)  # White's view
            for ply in range(rng.randint(24, 60)):
                if board.is_game_over():
                    break
                advantage += rng.gauss((STRENGTH[white] - STRENGTH[black]) / 60, 25)
                version = white if board.turn == chess.WHITE else black
                stm_eval = int(advantage if board.turn == chess.WHITE else -advantage)
                if searches % 23 == 5:
                    stm_eval = MATE_SCORE - 7          # side to move mates in 7 plies
                elif searches % 29 == 9:
                    stm_eval = -(MATE_SCORE - 6)       # side to move is mated in 6 plies
                legal = list(board.legal_moves)
                move = rng.choice(legal)
                if game_index == 0 and ply < len(FIRST_GAME_OPENING):
                    move = chess.Move.from_uci(FIRST_GAME_OPENING[ply])
                depth = rng.randint(5, 11)
                nodes = [int(rng.uniform(0.5, 1.5) * 3.0 ** d) for d in range(1, depth + 1)]
                qnodes = [int(x * rng.uniform(0.3, 1.2)) for x in nodes]
                times = [max(0.01, (a + b) / 2e3) for a, b in zip(nodes, qnodes)]
                evals = [stm_eval + (0 if abs(stm_eval) >= 99_000 else int(rng.gauss(0, 60 / (d + 1))))
                         for d in range(depth)]
                fh = [[int(x * 0.02 * p) for p in (0.8, 0.08, 0.04, 0.03, 0.03, 0.02)] for x in nodes]
                iteration_moves = [move.uci() if d > depth // 2 else rng.choice(legal).uci() for d in range(depth)]
                search_uuid = str(uuid.UUID(int=rng.getrandbits(128)))
                record = dict(
                    search_uuid=search_uuid, game_uuid=game_uuid, engine_id=version, fen=board.fen(), ply=ply + 1,
                    time_ms=sum(times), root_eval=evals[-1], completed_depth=depth - 1, max_depth=depth,
                    max_qdepth=depth + rng.randint(2, 10), best_move=move.uci(), principal_variation=[move.uci()],
                    nodes=sum(nodes), qnodes=sum(qnodes), tt_stores=int(sum(nodes) * 0.6),
                    tt_hits=int(sum(nodes) * rng.uniform(0.2, 0.4)), tt_fill=rng.random(),
                    fail_highs=int(sum(nodes) * 0.02), fail_lows=int(sum(nodes) * 0.01),
                    aspiration_fail_high_researches=rng.randint(0, 3), aspiration_fail_low_researches=rng.randint(0, 3),
                    fail_high_index=[sum(f[i] for f in fh) for i in range(6)],
                    see_prunes=int(sum(qnodes) * 0.05), delta_prunes=int(sum(qnodes) * 0.03),
                    nmp=int(sum(nodes) * 0.01), nmp_failhigh=int(sum(nodes) * 0.006),
                    tt_overwritten=int(sum(nodes) * 0.01),
                    itdepth_nodes=nodes, itdepth_qnodes=qnodes, itdepth_time_ms=times, itdepth_eval=evals,
                    itdepth_move=iteration_moves, itdepth_qdepth=[d + 8 for d in range(depth)],
                    itdepth_ttstores=[int(x * 0.6) for x in nodes], itdepth_tthits=[int(x * 0.3) for x in nodes],
                    itdepth_ttfill=[rng.random() for _ in nodes], itdepth_fail_highs=[int(x * 0.02) for x in nodes],
                    itdepth_fail_lows=[int(x * 0.01) for x in nodes],
                    itdepth_aspiration_failhigh_researches=[0] * depth,
                    itdepth_aspiration_faillow_researches=[0] * depth,
                    itdepth_fh_index=fh, itdepth_see_prunes=[int(x * 0.05) for x in qnodes],
                    itdepth_delta_prunes=[int(x * 0.03) for x in qnodes],
                    itdepth_nmp=[int(x * 0.01) for x in nodes], itdepth_nmp_failhigh=[int(x * 0.006) for x in nodes],
                    treedepth_nodes=nodes[::-1], treedepth_qnodes=qnodes[::-1], treedepth_tt_stores=[1] * depth,
                    treedepth_tt_hits=[1] * depth, treedepth_fail_highs=[1] * depth, treedepth_fail_lows=[1] * depth,
                    treedepth_fh_index=fh[::-1], treedepth_see_prunes=[1] * depth,
                    treedepth_delta_prunes=[1] * depth,
                    treedepth_nmp=[1] * depth, treedepth_nmp_failhigh=[1] * depth,
                )
                if version in PVS_KIND_VERSIONS:
                    for kind, share in PVS_KINDS.items():
                        values = [int(x * share) for x in nodes]
                        record.update({f"pvs_researches_{kind}": sum(values), f"itdepth_pvs_researches_{kind}": values,
                                       f"treedepth_pvs_researches_{kind}": [1] * depth})
                else:
                    values = [int(x * 0.001) for x in nodes]
                    record.update(pvs_researches=sum(values), itdepth_pvs_researches=values,
                                  treedepth_pvs_researches=[1] * depth)
                files["search"].write(json.dumps(record) + "\n")
                timing = {"type": "timing", "search_uuid": search_uuid, "game_uuid": game_uuid,
                          "engine_id": version, "total_search_time_ms": sum(times)}
                for function in FUNCTIONS:
                    share = 1.0 if function == "ROOT" else rng.uniform(0.01, 0.2)
                    timing[function] = {"total_ms": sum(times) * share, "calls": rng.randint(100, 100000)}
                files["timing"].write(json.dumps(timing) + "\n")
                root_moves = [{"move": m.uci(), "eval": stm_eval + rng.randint(-150, 50),
                               "time_ms": rng.randint(0, 50), "nodes": rng.randint(10, 5000)} for m in legal[:8]]
                files["root_moves"].write(json.dumps({"search_uuid": search_uuid, "depth": depth,
                                                      "moves": root_moves}) + "\n")
                if searches % 2 == 0:
                    # a last iteration cut off by the clock after two moves (no rng draws: the data stays the same)
                    files["root_moves"].write(json.dumps({"search_uuid": search_uuid, "depth": depth + 1,
                                                          "moves": root_moves[:2]}) + "\n")
                board.push(move)
                moves.append(move.uci())
                searches += 1
            p_white = 1 / (1 + 10 ** (-advantage / 400))
            draw = rng.random() < 0.3
            result = 3 if draw else (1 if rng.random() < p_white else 2)
            for side, version in (("white", white), ("black", black)):
                files["game"].write(json.dumps(dict(
                    game_uuid=game_uuid, engine_id=version, side=side, wtime=10000, btime=10000, winc=100,
                    binc=100, movestogo=None, depth=None, nodes=None, movetime=None, result=result,
                    reason=rng.randint(1, 6), opening=None, start_fen=chess.STARTING_FEN, moves=moves,
                    time_s=rng.uniform(5, 60))) + "\n")
    finally:
        for f in files.values():
            f.close()
    return n_games, searches


def _quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def build_raw_db(directory, n_games=12, seed=7):
    """Create `directory`/chess.db from synthetic logs; returns its path."""
    from data import etl
    from data.databases.init_raw_db import init_raw_db

    directory = Path(directory)
    logs = directory / "logs"
    write_logs(logs, n_games=n_games, seed=seed)
    raw = _quiet(init_raw_db, directory / "chess.db")
    cnxn = sqlite3.connect(raw)
    try:
        for version in VERSIONS:
            _quiet(etl.register_engine, cnxn, {"name": "Tomahawk", "version": version,
                                               "description": PRIVATE_DESCRIPTION.format(version=version),
                                               "compile_flags": "-O3 -I/home/someone/include"})
        _quiet(etl.ingest_log_dir, cnxn, logs, clear=False)
        enrich(cnxn)
        cnxn.commit()
    finally:
        cnxn.close()
    return raw


def enrich(cnxn):
    """Match-runner environment on most games, termination codes 7..10, SPRT rows and an STS run."""
    game_ids = [row[0] for row in cnxn.execute("SELECT id FROM games ORDER BY id")]
    for i, game_id in enumerate(game_ids):
        if i % 4 == 3:
            continue  # engine-log games without environment data
        p_core = i % 2 == 0
        cnxn.execute("""
            UPDATE games SET source = 'match', run_id = '20260928_101129_354256', run_game = ?, pair_index = ?,
                env_host = 'bench-host', env_cpu_model = 'Synthetic CPU 9000', env_os = 'Windows-11',
                env_cpu = ?, env_core_type = ?, env_cluster = ?, env_concurrency = 4,
                env_mhz_mean = ?, env_mhz_min = ?, env_mhz_max = ?, env_util_mean = 97.5,
                env_power_source = 'ac', env_core_mixed = ?
            WHERE id = ?
        """, (i + 1, i // 2 + 1, i % 8, "P" if p_core else "E", i % 3,
              4700.0 + 37 * i if p_core else 3100.0 + 23 * i, 4500.0 if p_core else 2900.0,
              5000.0 if p_core else 3300.0, 1 if i == 5 else 0, game_id))
    for code, game_id in zip((7, 8, 9, 10), game_ids):
        cnxn.execute("UPDATE games SET termination = ? WHERE id = ?", (code, game_id))
    cnxn.execute("UPDATE games SET moves = '[]', opening = ? WHERE id = ?",
                 (", ".join(FALLBACK_OPENING), game_ids[2]))
    cnxn.execute("UPDATE games SET moves = ?, opening = ? WHERE id = ?",
                 (json.dumps(EPD_BOOK_MOVES), ", ".join(EPD_BOOK_OPENING), game_ids[4]))

    engines = [row[0] for row in cnxn.execute("SELECT id FROM engines ORDER BY id")]
    for i, (candidate, baseline, book, elo) in enumerate((
            (engines[1], engines[0], r"C:\Users\someone\books\UHO_4060_v2.epd", 12.37),
            (engines[2], engines[1], "/home/someone/books/8moves_v3.pgn", -3.5))):
        experiment = cnxn.execute("INSERT INTO experiments (engine_id, comparison_engine_id, type, metadata) "
                                  "VALUES (?, ?, 'SPRT', ?)",
                                  (candidate, baseline, json.dumps({"host": "bench-host", "cwd": "C:\\work"}))).lastrowid
        cnxn.execute("""
            INSERT INTO sprt (experiment_id, baseline_engine_id, candidate_engine_id, opening_book, book_depth,
                time_control, elo0, elo1, alpha, beta, result, elo_diff, llr, los, candidate_wins,
                candidate_losses, candidate_draws, games_played, run_time_s, pairs, penta_ll, penta_ld,
                penta_dd, penta_wl, penta_wd, penta_ww, score, elo_low, elo_high, nelo, nelo_low, nelo_high,
                draw_ratio, variance_ratio, concurrency, core_summary, excluded_pairs, run_id, book_seed,
                env_host, env_cpu_model)
            VALUES (?, ?, ?, ?, 8, '10+0.1', 0.0, 2.5, 0.05, 0.05, ?, ?, 2.95, 0.93, 120, 100, 180, 400, 3600.5,
                200, 10, 40, 60, 20, 45, 25, 0.517, 1.2, 23.5, 20.1, 1.0, 39.0, 0.45, 0.8, 4,
                '{"P": {"pairs": 120}, "E": {"pairs": 80}}', 2, '20260928_101129_354256', 42,
                'bench-host', 'Synthetic CPU 9000')
        """, (experiment, baseline, candidate, book, "pass" if i == 0 else "inconclusive", elo))

    # STS: a decoy run first, so that sts row ids overlap the real run's experiment id
    # (searches.sts_id holds the experiment id, not an sts row id).
    decoy = cnxn.execute("INSERT INTO experiments (engine_id, type) VALUES (?, 'STS')", (engines[0],)).lastrowid
    for k in range(6):
        cnxn.execute("INSERT INTO sts (experiment_id, suite, position_name, fen, engine_move, move_is_correct) "
                     "VALUES (?, 'decoy.epd', ?, '8/8/8/8/8/8/8/K6k w - - 0 1', 'a1a2', 0)",
                     (decoy, f"STS(decoy) - Decoy {k}"))
    sts_run = cnxn.execute("INSERT INTO experiments (engine_id, type) VALUES (?, 'STS')", (engines[0],)).lastrowid
    rows = cnxn.execute("SELECT id, fen, move FROM searches WHERE engine_id = ? ORDER BY id LIMIT 5",
                        (engines[0],)).fetchall()
    for n, (search_id, fen, move) in enumerate(rows):
        epd = " ".join(fen.split()[:4]) + " 0 1"   # tests/sts.py rewrites the move counters
        cnxn.execute("INSERT INTO sts (experiment_id, suite, position_name, fen, engine_move, expected_move, "
                     "move_is_correct) VALUES (?, '/home/someone/sts/STS1.epd', ?, ?, ?, ?, ?)",
                     (sts_run, f"STS({n}) - Undermining", epd, move, move if n % 2 == 0 else "a2a3", n % 2 == 0))
        cnxn.execute("UPDATE searches SET sts_id = ? WHERE id = ?", (sts_run, search_id))
    return {"sts_experiment": sts_run, "sts_searches": [row[0] for row in rows]}

# ─────────────────────────────────────────────────────────────────────────────
# Running the pipeline in-process
# ─────────────────────────────────────────────────────────────────────────────
def run_module_in_process(module, extra_args=None):
    """`python -m module args` in this process: the module's main() with that command line.

    (Starting a Python process per pipeline step is slow on some machines.)
    """
    old_argv = sys.argv
    sys.argv = [module] + list(extra_args or [])
    try:
        status = importlib.import_module(module).main()
    except SystemExit as exit_:
        status = exit_.code
    finally:
        sys.argv = old_argv
    if status not in (0, None):
        raise RuntimeError(f"{module} exited with status {status}")


_ANALYTICS = {}


def analytics_fixture(n_games=6):
    """(root, raw_db, analytics_db, pipeline_output) built once per process.

    The analytics database comes from `run_analytics_pipeline --full` with a
    Stockfish path that doesn't exist, i.e. the no-Stockfish path. Tests must
    only read it (copy it to change it).
    """
    if not _ANALYTICS:
        temporary = tempfile.TemporaryDirectory(prefix="sj-data-tests-", ignore_cleanup_errors=True)
        atexit.register(temporary.cleanup)
        root = Path(temporary.name)
        raw = build_raw_db(root, n_games=n_games)
        analytics = root / "chess_analytics.duckdb"
        from data.databases import run_analytics_pipeline

        output = io.StringIO()
        missing = str(root / "no-such-stockfish" / "stockfish.exe")
        with mock.patch.dict(os.environ, {"STOCKFISH": missing}), contextlib.redirect_stdout(output):
            run_analytics_pipeline.main(["--full", "--raw-db", str(raw), "--analytics-db", str(analytics)],
                                        runner=run_module_in_process)
        _ANALYTICS.update(temporary=temporary, root=root, raw=raw, analytics=analytics, output=output.getvalue())
    return _ANALYTICS["root"], _ANALYTICS["raw"], _ANALYTICS["analytics"], _ANALYTICS["output"]
