"""Games recorded by the match runner, and how engine telemetry links to them.

The match runner (utils/match_runner.py, used by tests/sprt.py and friends)
writes one record per game to <run>/games.jsonl with the authoritative result,
termination, opening and the environment the game ran in (logical cpu, P/E core
type, measured clock, power source), and all moves to <run>/games.pgn.

Engines log their own telemetry per process (search_<pid>.jsonl, ...), with
instance_id = process id and a game_uuid per game, but a game record only when
the engine itself saw the game end, and the two engines of a game never share a
uuid. So instead of trusting engine game records, every runner game becomes one
`games` row (source 'match'), and each engine game_uuid is linked to the runner
game whose positions its searches were made in (a vote over the searched FENs;
the two games of an opening pair share only their first positions).
"""
from __future__ import annotations

import io
import json
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

RESULT_CODES = {"1-0": 1, "0-1": 2, "1/2-1/2": 3}
# utils/match_runner termination classes (tests/sprt.termination_class) -> raw termination codes
TERMINATION_CODES = {
    "mate": 1, "stalemate": 2, "repetition": 3, "50-move rule": 4, "time forfeit": 5, "resignation": 6,
    "adjudication": 7, "move limit": 7, "tablebase": 7, "insufficient material": 8, "illegal move": 9, 
    "crash": 10, "stall": 10,
}
_TAG = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


def epd(fen):
    """Position part of a FEN (placement, side, castling, en passant)."""
    return " ".join(fen.split()[:4])


def read_run(run_dir):
    """(games.jsonl records, config meta) of a match-runner run directory."""
    run_dir = Path(run_dir)
    games = []
    path = run_dir / "games.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            games = [json.loads(line) for line in f if line.strip()]
    meta = {}
    config = run_dir / "config.json"
    if config.exists():
        meta = json.loads(config.read_text(encoding="utf-8")).get("meta", {})
    return games, meta


def pgn_moves(run_dir):
    """{(job, job_game): (uci moves, [epd after each ply, starting position included])} from games.pgn."""
    import chess.pgn

    path = Path(run_dir) / "games.pgn"
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            try:
                key = (int(game.headers.get("Job", "0")), int(game.headers.get("JobGame", "0")))
            except ValueError:
                continue
            board = game.board()
            moves, positions = [], [epd(board.fen())]
            for move in game.mainline_moves():
                moves.append(move.uci())
                board.push(move)
                positions.append(epd(board.fen()))
            out[key] = (moves, positions, game.headers.get("FEN"))
    return out


def stored_games(cnxn, run_id, experiment_id, first="Candidate", second="Baseline"):
    """(games, moves) like read_run / pgn_moves, from a runs played games in the raw database
    
    for linking telemetry when the run folder is gone (e.g. a later retry-ingest):
    game 2k+1 of an opening pair has 'first' as white. withou the folder the engine
    processes of a game are unknown, so ties are not broken by job
    """
    import chess
    games, moves = [], []
    rows = cnxn.execute("SELECT run_game, moves, start_fen FROM games WHERE source = 'match' AND run_id = ? "
                        "AND experiment_id = ?", (run_id, experiment_id)).fetchall()
    
    for number, played, fen in rows:
        board = chess.Board(fen) if fen else chess.Board()
        ucis, positions = json.loads(played or "[]"), [epd(board.fen())]

        for move in ucis:
            board.push_uci(move)
            positions.append(epd(board.fen()))

        white, black = (first, second) if number % 2 else (second, first)
        games.append(dict(game=number, job=0, job_game=number, white=white, black=black))
        moves[(0, number)] = (ucis, positions, fen)

    return games, moves

def link_engine_games(logroot, run_dir, games=None, moves=None, names_by_version=None, min_support=0.5, engine_name=None):
    """{engine game_uuid: run game number} for every engine game in logroot's instance logs.

    Each search's FEN votes for the runner games that contain that position.
    A vote counts only if the searching engine had the move in that game (when
    `names_by_version` maps engine versions to the runner's engine names), and
    the winning game must contain at least `min_support` of the engine game's
    searched positions: games the runner killed (e.g. when an SPRT stopped) are
    not in games.pgn and share only their first position with a finished game,
    so they stay unlinked instead of polluting that game. Ties (the shared
    first positions of a pair) are broken by the engine process's own job.
    """
    games = games if games is not None else read_run(run_dir)[0]
    moves = moves if moves is not None else pgn_moves(run_dir)
    by_position = defaultdict(set)
    job_of_pid = defaultdict(set)
    record_of = {}
    for record in games:
        key = (record.get("job"), record.get("job_game"))
        record_of[record["game"]] = record
        for position in moves.get(key, ((), ()))[1]:
            by_position[position].add(record["game"])
        for pid in record.get("engine_pids") or ():
            job_of_pid[pid].add(record.get("job"))
    game_job = {record["game"]: record.get("job") for record in games}

    votes = defaultdict(Counter)   # game_uuid -> Counter(run game)
    searches = Counter()           # game_uuid -> searches with a FEN
    owner = {}                     # game_uuid -> instance_id
    for path in Path(logroot).glob("search_*.jsonl"):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                uuid, fen = record.get("game_uuid"), record.get("fen")
                if not uuid or not fen:
                    continue
                owner.setdefault(uuid, record.get("instance_id"))
                searches[uuid] += 1
                name = engine_name if engine_name is not None else \
                    (names_by_version or {}).get(record.get("engine_id"))
                white_to_move = fen.split()[1] == "w" if len(fen.split()) > 1 else True
                for number in by_position.get(epd(fen), ()):
                    if name is not None:
                        played = record_of[number]
                        if (played.get("white") if white_to_move else played.get("black")) != name:
                            continue
                    votes[uuid][number] += 1
    links = {}
    for uuid, counter in votes.items():
        best = max(counter.values())
        if best < max(1, min_support * searches[uuid]):
            continue
        tied = [n for n, c in counter.items() if c == best]
        if len(tied) > 1:
            jobs = job_of_pid.get(owner.get(uuid), set())
            tied = [n for n in tied if game_job.get(n) in jobs] or tied
        if len(tied) == 1:
            links[uuid] = tied[0]
    return links


def _tc_fields(time_control):
    """wtime, btime, winc, binc (ms), depth, movetime (ms) from a run's time control text."""
    empty = dict(wtime=None, btime=None, winc=None, binc=None, depth=None, movetime=None)
    out = dict(empty)
    
    if not time_control:
        return out
    text = str(time_control).strip().split("@")[0].strip()
    kind, value = None, text 
    match = re.fullmatch(r"(depth|nodes|st|tc)\s*[= ]\s*(\S+)", text)
    if match: kind, value = match.groups()
    try:
        if kind == "depth":
            out['depth'] = int(value)
        elif kind == "nodes":
            out['nodes'] = int(value)
        elif kind == "st":
            out['movetime'] = round(float(value) * 1000)
        elif value.endswith("s/move"):
            out['movetime'] = round(float(value[:-6]) * 1000)
        else:
            base, _, inc = value.partition("+")
            base = base.split("/")[-1] # moves/time
            if ":" in base:
                minutes, seconds = base.split(':', 1)
                base_s = 60 * float(minutes) + float(seconds)
            else:
                base_s = float(base)
            increment = float(inc or 0)
            if not (math.isfinite(base_s) and math.isfinite(increment)):
                return out 
            out['wtime'] = out['btime'] = round(base_s * 1000)
            out['winc'] = out['binc'] = round(increment * 1000)
    except (ValueError, OverflowError):
        return dict(empty)
    return out

def game_rows(run_dir, engine_ids, experiment_id=None):
    """Rows for the raw `games` table, one per finished runner game (see ingest.bulk_log_match_games)."""
    games, meta = read_run(run_dir)
    moves = pgn_moves(run_dir)
    machine = meta.get("machine", {})
    tc = _tc_fields(meta.get("time_control"))
    rows = []
    for record in games:
        played = moves.get((record.get("job"), record.get("job_game")), ((), (), None))
        opening = ", ".join(x for x in (record.get("opening"), record.get("variation")) if x)
        rows.append(dict(
            experiment_id=experiment_id,
            white_engine_id=engine_ids.get(record["white"]),
            black_engine_id=engine_ids.get(record["black"]),
            **tc,
            result=RESULT_CODES.get(record.get("result"), 0),
            termination=TERMINATION_CODES.get(record.get("termination")),
            opening=opening or None,
            start_fen=played[2] if len(played) > 2 else None,
            moves=json.dumps(list(played[0])),
            run_time_s=record.get("seconds"),
            source="match",
            run_id=record.get("run") or meta.get("run"),
            run_game=record.get("game"),
            pair_index=record.get("pair"),
            env_host=machine.get("host"),
            env_cpu_model=machine.get("cpu_model"),
            env_os=machine.get("os"),
            env_cpu=record.get("cpu"),
            env_core_type=record.get("core_type"),
            env_cluster=record.get("cluster"),
            env_concurrency=record.get("concurrency"),
            env_mhz_mean=record.get("mhz_mean"),
            env_mhz_min=record.get("mhz_min"),
            env_mhz_max=record.get("mhz_max"),
            env_util_mean=record.get("util_mean"),
            env_power_source=record.get("power_source") or machine.get("power_source"),
            env_core_mixed=int(bool(record.get("core_mixed"))),
        ))
    return rows


def count_lines(paths):
    total = 0
    for path in paths:
        with io.open(path, encoding="utf-8", errors="replace") as f:
            total += sum(1 for _ in f)
    return total
