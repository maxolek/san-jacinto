#!/usr/bin/env python3
"""Mine an engine's mistakes from past games into an EPD test suite.

Reads the PGNs that SPRT runs and tournaments leave in logs/, finds the moves
where a selected player threw away winning chances, confirms them with
Stockfish and appends those positions to one EPD suite that tests/sts.py
replays (a position is solved when the engine plays one of the good moves):

    python -m tests.mistakes --player Candidate
    python -m tests.sts --engine <engine> --sts logs/mistakes/mistakes.epd --depth 10

Why a script and not a dashboard view:
  * Mining needs Stockfish. Every position is screened at a cheap depth and
    every candidate is searched again at depth with MultiPV: minutes of
    engine time, not a page load.
  * It is reproducible. Fixed depths, Threads=1 and an empty hash per game
    give the same entries from the same PGNs, and every entry records its
    game, ply and verification depth.
  * It grows one suite over time. New positions are merged into the existing
    file, one entry per position, so re-running over the same logs adds
    nothing and the suite becomes a regression test of past mistakes.

How a move becomes an entry:
  1. Screen: every position of the game is analysed once (--screen-depth or
     --screen-nodes). The eval of position k+1, negated, is the eval after
     move k. The move is a candidate when the mover's win% (lichess formula)
     drops by at least --threshold points. Book moves are never mined.
  2. Verify: the candidate position is searched at --verify-depth with
     --multipv lines. The good moves (bm) are those within --good-margin cp of
     the best. The played move (am) is searched on its own at the same depth;
     the position is kept when that move is not good and still loses at least
     --threshold points.
  3. Select: one entry per position (largest loss), at most --max-per-game per
     game (largest losses first); positions already in the suite are skipped.
     A CSV with the details sits next to the suite (same name, .csv).
"""
import argparse
import csv
import glob
import hashlib
import itertools
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine
import chess.pgn

# paths
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
TOMAHAWK_ROOT = PROJECT_ROOT.parent / "tomahawk"
DEFAULT_STOCKFISH = TOMAHAWK_ROOT / "engines" / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
# games written by tests/sprt.py and tests/tournament.py
DEFAULT_PGN_GLOBS = ["logs/sprt_results/*/games.pgn", "logs/tournament_results/*/*.pgn"]
DEFAULT_OUT = "logs/mistakes/mistakes.epd"

# lichess win% model: evals are clipped to +-1000 cp; a mate counts as
# +-(MATE_SCORE - moves) before clipping
MATE_SCORE = 10000
CP_CLIP = 1000
WIN_SLOPE = 0.00368208

BOOK_COMMENT = "book"     # fastchess comment on opening-book moves
ENDGAME_MAX_PIECES = 6    # knights, bishops, rooks and queens of both sides
OPENING_PLIES = 20
PHASES = ("opening", "middlegame", "endgame")

CSV_FIELDS = ["id", "fen", "bm", "am", "loss", "cp_loss", "best", "played", "depth",
              "player", "game", "ply", "phase", "sf_best_cp", "played_cp",
              "source", "white", "black", "round"]


# --------------------------
#   evals and positions
# --------------------------

def clip_cp(cp):
    return max(-CP_CLIP, min(CP_CLIP, cp))


def win_percent(cp):
    """Lichess win% (0-100) for the side whose eval is `cp` centipawns."""
    return 50 + 50 * (2 / (1 + math.exp(-WIN_SLOPE * clip_cp(cp))) - 1)


def score_cp(score, color):
    """Centipawns of an engine PovScore from `color`'s view; mate in n is +-(MATE_SCORE - n)."""
    return score.pov(color).score(mate_score=MATE_SCORE)


def position_key(board):
    """The position without move counters: board.epd()."""
    return board.epd()


def entry_id(key):
    return "mistake." + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def classify_phase(board, ply):
    """endgame with <= 6 knights, bishops, rooks and queens left; else opening before ply 20."""
    pieces = chess.popcount(board.occupied & ~board.pawns & ~board.kings)
    if pieces <= ENDGAME_MAX_PIECES:
        return "endgame"
    return "opening" if ply < OPENING_PLIES else "middlegame"


def is_selected(name, players, exclude):
    """Whether moves by `name` are mined: in `players` (None: everyone) and not matching `exclude`."""
    if players and name not in players:
        return False
    return not (exclude and re.search(exclude, name))


# --------------------------
#   suite entries
# --------------------------

@dataclass
class Mistake:
    """A verified mistake: one suite line and one CSV row."""
    key: str           # position_key() of the position before the move
    fen: str
    bm: list           # SAN of the good moves, best first
    am: str            # SAN of the played move
    best: str          # SAN of Stockfish's best move
    loss: float        # win% points lost by the played move
    cp_loss: int       # best minus played, both clipped to +-1000 cp
    best_cp: int       # mover's view; mates as +-(MATE_SCORE - moves)
    played_cp: int
    depth: int
    player: str
    source: Path
    game_index: int    # 1-based game number in `source`
    ply: int
    phase: str
    white: str
    black: str
    game_round: str


def c0_value(text):
    """Keep c0 a list of key=value words: no spaces, ';' or quotes in a value."""
    return re.sub(r'[\s;"]+', "_", text) or "?"


def format_epd_line(m):
    """Suite line; bm comes right after the position because tests/sts.py only reads it there."""
    c0 = " ".join([
        f"loss={m.loss:.1f}",
        f"cp_loss={m.cp_loss}",
        f"best={m.best}",
        f"played={m.am}",
        f"depth={m.depth}",
        f"player={c0_value(m.player)}",
        f"game={c0_value(m.source.name)}#{m.game_index}",
        f"ply={m.ply}",
        f"phase={m.phase}",
    ])
    return f'{m.key} bm {" ".join(m.bm)}; am {m.am}; id "{entry_id(m.key)}"; c0 "{c0}";'


def csv_row(m):
    return {
        "id": entry_id(m.key), "fen": m.fen, "bm": " ".join(m.bm), "am": m.am,
        "loss": f"{m.loss:.1f}", "cp_loss": m.cp_loss, "best": m.best, "played": m.am,
        "depth": m.depth, "player": m.player, "game": f"{m.source.name}#{m.game_index}",
        "ply": m.ply, "phase": m.phase, "sf_best_cp": m.best_cp, "played_cp": m.played_cp,
        "source": str(m.source), "white": m.white, "black": m.black, "round": m.game_round,
    }


def dedupe_by_position(mistakes):
    """One mistake per position, the one with the largest loss; input order is kept."""
    best = {}
    for i, m in enumerate(mistakes):
        if m.key not in best or m.loss > mistakes[best[m.key]].loss:
            best[m.key] = i
    return [mistakes[i] for i in sorted(best.values())]


def cap_per_game(mistakes, limit):
    """At most `limit` mistakes per game (0: no cap), largest losses first; input order is kept."""
    if limit <= 0:
        return list(mistakes)
    by_game = defaultdict(list)
    for i, m in enumerate(mistakes):
        by_game[(m.source, m.game_index)].append(i)
    keep = set()
    for indices in by_game.values():
        keep.update(sorted(indices, key=lambda i: -mistakes[i].loss)[:limit])
    return [mistakes[i] for i in sorted(keep)]


def normalise_key(epd_fields):
    """position_key() of the four EPD fields, e.g. dropping an en passant square nobody can use."""
    try:
        return position_key(chess.Board(epd_fields))
    except ValueError:
        return epd_fields


def read_suite_keys(path):
    """Position keys of the entries already in an EPD file (none if it does not exist)."""
    if not path.exists():
        return set()
    keys = set()
    with open(path, encoding="utf-8-sig") as suite:
        for line in suite:
            fields = line.split(";", 1)[0].split()
            if len(fields) >= 4 and not line.lstrip().startswith("#"):
                keys.add(normalise_key(" ".join(fields[:4])))
    return keys


def ends_with_newline(path):
    """True for a missing or empty file, or one whose last byte is a newline."""
    if not path.exists() or path.stat().st_size == 0:
        return True
    with open(path, "rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def append_suite(path, mistakes):
    path.parent.mkdir(parents=True, exist_ok=True)
    separator = "" if ends_with_newline(path) else "\n"
    with open(path, "a", encoding="utf-8") as suite:
        suite.write(separator + "".join(format_epd_line(m) + "\n" for m in mistakes))


def append_csv(path, mistakes):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(csv_row(m) for m in mistakes)


def merge_into_suite(out, mistakes):
    """Append the mistakes whose position is not in the suite yet (EPD and CSV); returns them."""
    known = read_suite_keys(out)
    new = []
    for m in mistakes:
        if m.key not in known:
            known.add(m.key)
            new.append(m)
    if new:
        append_suite(out, new)
        append_csv(out.with_suffix(".csv"), new)
    return new


# --------------------------
#   games
# --------------------------

def warn(message):
    print(f"[MISTAKES] warning: {message}", flush=True)


def resolve_path(text):
    """Relative paths are relative to the repo root, like tests/sts.py --sts."""
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def collect_pgn_files(patterns):
    """PGN files named by paths or glob patterns, without duplicates, in a stable order."""
    files = []
    for pattern in patterns:
        path = resolve_path(pattern)
        if path.is_file():
            matches = [path]
        else:
            found = glob.glob(pattern, root_dir=PROJECT_ROOT, recursive=True)
            matches = sorted(p for p in (PROJECT_ROOT / f for f in found) if p.is_file())
        if not matches:
            warn(f"no PGN files match {pattern}")
        for match in matches:
            match = match.resolve()
            if match not in files:
                files.append(match)
    return files


def read_games(path):
    """Yield (1-based index, game) for every game of a PGN file."""
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        for index in itertools.count(1):
            game = chess.pgn.read_game(handle)
            if game is None:
                return
            yield index, game


@dataclass
class GameMove:
    board: chess.Board  # position before the move, without move history
    move: chess.Move
    ply: int            # board.ply() before the move; honours [FEN] move numbers
    player: str
    book: bool


def game_moves(game):
    """The mainline as GameMoves, plus the final position. Raises ValueError for a bad start."""
    board = game.board()  # honours [FEN] / [SetUp]
    if not board.is_valid():
        raise ValueError(f"invalid start position {board.fen()}")
    names = {chess.WHITE: game.headers.get("White", "?"), chess.BLACK: game.headers.get("Black", "?")}
    moves = []
    for node in game.mainline():
        moves.append(GameMove(board.copy(stack=False), node.move, board.ply(), names[board.turn],
                              node.comment.strip() == BOOK_COMMENT))
        board.push(node.move)
    return moves, board.copy(stack=False)


# --------------------------
#   Stockfish passes
# --------------------------

def evaluate(engine, board, limit, game_key):
    """Eval in cp for the side to move."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_game_over():
        return 0  # stalemate, insufficient material, 75-move rule
    info = engine.analyse(board, limit, game=game_key, info=chess.engine.INFO_SCORE)
    return score_cp(info["score"], board.turn)


def screen_game(engine, moves, final_board, indices, limit, threshold, game_key):
    """Screening pass: [(GameMove, loss)] for the moves at `indices` losing >= threshold win%.

    Every position is analysed once: the eval of position k+1 (the opponent to
    move), negated, is the mover's eval after move k.
    """
    positions = [m.board for m in moves] + [final_board]
    evals = {}

    def eval_at(k):
        if k not in evals:
            evals[k] = evaluate(engine, positions[k], limit, game_key)
        return evals[k]

    candidates = []
    for k in indices:
        loss = win_percent(eval_at(k)) - win_percent(-eval_at(k + 1))
        if loss >= threshold:
            candidates.append((moves[k], loss))
    return candidates


@dataclass
class Verdict:
    best: chess.Move
    best_cp: int        # mover's view
    good: list          # good moves, best first
    played_cp: int
    loss: float         # win% points


def verify(engine, board, played, depth, multipv, good_margin, threshold, game_key):
    """Verification pass: a Verdict when `played` is a real mistake at `depth`, else None.

    Good moves are the MultiPV moves within `good_margin` cp of the best. Evals
    are clipped to +-1000 cp like the win% formula, so every clearly winning
    move counts as good, not only the fastest mate. While even the last line is
    still good, the search is repeated with twice the lines (up to every legal
    move), so bm lists every good move and a replay never fails a good one.
    """
    limit = chess.engine.Limit(depth=depth)
    legal = board.legal_moves.count()
    lines_wanted = min(multipv, legal)
    while True:
        lines = engine.analyse(board, limit, multipv=lines_wanted, game=game_key)
        ranked = [(info["pv"][0], score_cp(info["score"], board.turn)) for info in lines if info.get("pv")]
        if not ranked:
            return None
        best, best_cp = ranked[0]
        good = [move for move, cp in ranked if clip_cp(best_cp) - clip_cp(cp) <= good_margin]
        if len(good) < len(ranked) or lines_wanted >= legal:
            break
        lines_wanted = min(2 * lines_wanted, legal)
    if played in good:
        return None
    info = engine.analyse(board, limit, root_moves=[played], game=game_key)
    played_cp = score_cp(info["score"], board.turn)
    loss = win_percent(best_cp) - win_percent(played_cp)
    if clip_cp(best_cp) - clip_cp(played_cp) <= good_margin or loss < threshold:
        return None  # good after all (outside the MultiPV lines), or too small a loss
    return Verdict(best, best_cp, good, played_cp, loss)


def make_mistake(gm, verdict, depth, source, index, headers):
    board = gm.board
    return Mistake(
        key=position_key(board),
        fen=board.fen(),
        bm=[board.san(move) for move in verdict.good],
        am=board.san(gm.move),
        best=board.san(verdict.best),
        loss=verdict.loss,
        cp_loss=clip_cp(verdict.best_cp) - clip_cp(verdict.played_cp),
        best_cp=verdict.best_cp,
        played_cp=verdict.played_cp,
        depth=depth,
        player=gm.player,
        source=source,
        game_index=index,
        ply=gm.ply,
        phase=classify_phase(board, gm.ply),
        white=headers.get("White", "?"),
        black=headers.get("Black", "?"),
        game_round=headers.get("Round", "?"),
    )


# --------------------------
#   mining
# --------------------------

@dataclass
class Stats:
    games_read: int = 0
    games_scanned: int = 0
    screened: Counter = field(default_factory=Counter)    # moves screened per player
    candidates: Counter = field(default_factory=Counter)  # screening candidates per player


def screen_limit(args):
    if args.screen_nodes:
        return chess.engine.Limit(nodes=args.screen_nodes)
    return chess.engine.Limit(depth=args.screen_depth)


def is_minable(gm, args):
    return (not gm.book and gm.ply >= args.min_ply
            and is_selected(gm.player, args.player, args.exclude_player))


def mine_game(engine, game, source, index, args, stats):
    """Screen one game and verify its candidates; returns the game's verified mistakes."""
    label = f"{source.name}#{index}"
    for error in game.errors:
        warn(f"{label}: {error} (mining the moves before it)")
    try:
        moves, final_board = game_moves(game)
    except ValueError as exc:
        warn(f"{label}: {exc}; game skipped")
        return []
    indices = [k for k, gm in enumerate(moves) if is_minable(gm, args)]
    if not indices:
        return []
    stats.games_scanned += 1
    game_key = (str(source), index)  # a new key sends ucinewgame: each game starts with an empty hash
    candidates = screen_game(engine, moves, final_board, indices, screen_limit(args),
                             args.threshold, game_key)
    mistakes = []
    for gm, _ in candidates:
        verdict = verify(engine, gm.board, gm.move, args.verify_depth, args.multipv,
                         args.good_margin, args.threshold, game_key)
        if verdict is not None:
            mistakes.append(make_mistake(gm, verdict, args.verify_depth, source, index, game.headers))
    stats.screened.update(moves[k].player for k in indices)
    stats.candidates.update(gm.player for gm, _ in candidates)
    white, black = game.headers.get("White", "?"), game.headers.get("Black", "?")
    print(f"[MISTAKES] {label} {white} - {black}: {len(indices)} moves screened, "
          f"{len(candidates)} candidates, {len(mistakes)} mistakes", flush=True)
    return mistakes


def scan(engine, pgn_files, args, stats):
    """Verified mistakes of every game, in file / game / ply order."""
    mistakes = []
    for source in pgn_files:
        print(f"[MISTAKES] Reading {source}", flush=True)
        for index, game in read_games(source):
            if args.max_games is not None and stats.games_scanned >= args.max_games:
                return mistakes
            stats.games_read += 1
            mistakes += mine_game(engine, game, source, index, args, stats)
    return mistakes


# --------------------------
#   summary
# --------------------------

def replay_command(out):
    try:
        shown = out.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        shown = str(out)
    if " " in shown:
        shown = f'"{shown}"'
    return f"python -m tests.sts --engine <engine> --sts {shown} --depth 10"


def print_summary(args, stats, n_files, verified, unique, selected, new, out):
    limit = f"nodes {args.screen_nodes}" if args.screen_nodes else f"depth {args.screen_depth}"
    print("\n=== Mistake mining summary ===")
    print(f"PGN files:          {n_files}")
    print(f"Games scanned:      {stats.games_scanned} (of {stats.games_read} read)")
    print(f"Moves screened:     {sum(stats.screened.values())} (screen {limit})")
    print(f"Candidates:         {sum(stats.candidates.values())} (win% drop >= {args.threshold:g})")
    print(f"Verified mistakes:  {len(verified)} (verify depth {args.verify_depth}, "
          f"multipv {args.multipv}, good margin {args.good_margin} cp)")
    print(f"New suite entries:  {len(new)} (not added: {len(verified) - len(unique)} repeated positions, "
          f"{len(unique) - len(selected)} over --max-per-game, "
          f"{len(selected) - len(new)} already in the suite)")

    verified_by_player = Counter(m.player for m in verified)
    new_by_player = Counter(m.player for m in new)
    players = sorted(set(stats.screened) | set(verified_by_player))
    width = max(len(name) for name in [*players, *PHASES]) + 2
    print("Per player:")
    for name in players:
        print(f"  {name:<{width}}screened {stats.screened[name]:>6}  "
              f"candidates {stats.candidates[name]:>5}  "
              f"verified {verified_by_player[name]:>5}  new {new_by_player[name]:>5}")
    verified_by_phase = Counter(m.phase for m in verified)
    new_by_phase = Counter(m.phase for m in new)
    print("Per phase:")
    for phase in PHASES:
        print(f"  {phase:<{width}}verified {verified_by_phase[phase]:>5}  new {new_by_phase[phase]:>5}")

    if not out.exists():
        print(f"Suite: {out} (not created: no entries yet)")
        return
    print(f"Suite: {out} ({len(read_suite_keys(out))} positions)")
    print(f"CSV:   {out.with_suffix('.csv')}")
    print("Replay the suite:")
    print(f"  {replay_command(out)}")


# --------------------------
#   main
# --------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tests.mistakes", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    games = parser.add_argument_group("games")
    games.add_argument("--pgn", nargs="+", action="extend", metavar="FILE_OR_GLOB",
                       help="PGN files or glob patterns, relative to the repo root "
                            f"(default: {' '.join(DEFAULT_PGN_GLOBS)})")
    games.add_argument("--player", action="append", metavar="NAME",
                       help="mine only this player's moves; repeatable (default: every player)")
    games.add_argument("--exclude-player", default="(?i)stockfish", metavar="REGEX",
                       help="never mine players whose name matches; '' mines everyone "
                            "(default: %(default)s)")
    games.add_argument("--max-games", type=int, metavar="N", help="stop after scanning N games")
    games.add_argument("--min-ply", type=int, default=0, metavar="PLY",
                       help="skip moves played before this ply (default: %(default)s)")

    screening = parser.add_argument_group("screening")
    limit = screening.add_mutually_exclusive_group()
    limit.add_argument("--screen-depth", type=int, default=10, metavar="D",
                       help="(default: %(default)s)")
    limit.add_argument("--screen-nodes", type=int, metavar="N",
                       help="screen with a node limit instead of a depth")
    screening.add_argument("--threshold", type=float, default=15.0, metavar="POINTS",
                           help="smallest win%% drop that counts as a mistake (default: %(default)s)")

    verification = parser.add_argument_group("verification")
    verification.add_argument("--verify-depth", type=int, default=18, metavar="D",
                              help="(default: %(default)s)")
    verification.add_argument("--multipv", type=int, default=4, metavar="N",
                              help="(default: %(default)s)")
    verification.add_argument("--good-margin", type=int, default=25, metavar="CP",
                              help="moves this close to the best are good moves, bm "
                                   "(default: %(default)s)")

    output = parser.add_argument_group("output")
    output.add_argument("--out", default=DEFAULT_OUT, metavar="EPD",
                        help="suite to merge into, relative to the repo root; the CSV goes "
                             "next to it (default: %(default)s)")
    output.add_argument("--max-per-game", type=int, default=3, metavar="N",
                        help="most entries per game, largest losses first; 0: no cap "
                             "(default: %(default)s)")

    stockfish = parser.add_argument_group("stockfish")
    stockfish.add_argument("--stockfish", metavar="PATH",
                           default=os.environ.get("STOCKFISH_PATH") or str(DEFAULT_STOCKFISH),
                           help="Stockfish binary: $STOCKFISH_PATH, else "
                                "../tomahawk/engines/stockfish/stockfish-windows-x86-64-avx2.exe "
                                "(now: %(default)s)")
    stockfish.add_argument("--threads", type=int, default=1, metavar="N",
                           help="more than 1 makes runs irreproducible (default: %(default)s)")
    stockfish.add_argument("--hash", type=int, default=64, metavar="MB",
                           help="(default: %(default)s)")
    args = parser.parse_args(argv)
    check_args(parser, args)
    return args


def check_args(parser, args):
    minimums = [("--screen-depth", args.screen_depth, 1), ("--screen-nodes", args.screen_nodes, 1),
                ("--verify-depth", args.verify_depth, 1), ("--multipv", args.multipv, 1),
                ("--threads", args.threads, 1), ("--hash", args.hash, 1),
                ("--max-games", args.max_games, 1), ("--good-margin", args.good_margin, 0),
                ("--min-ply", args.min_ply, 0), ("--max-per-game", args.max_per_game, 0)]
    for flag, value, minimum in minimums:
        if value is not None and value < minimum:
            parser.error(f"{flag} must be >= {minimum}")
    if args.exclude_player:
        try:
            re.compile(args.exclude_player)
        except re.error as exc:
            parser.error(f"--exclude-player is not a valid regex: {exc}")
    if not Path(args.stockfish).is_file():
        parser.error(f"Stockfish not found at {args.stockfish}; "
                     "pass --stockfish PATH or set STOCKFISH_PATH")
    for name in args.player or []:
        if not is_selected(name, None, args.exclude_player):
            warn(f"--player {name} matches --exclude-player {args.exclude_player!r}; "
                 "its moves are skipped unless you pass --exclude-player ''")


def main(argv=None):
    args = parse_args(argv)
    pgn_files = collect_pgn_files(args.pgn or DEFAULT_PGN_GLOBS)
    if not pgn_files:
        print("[MISTAKES] No PGN files to mine; pass --pgn FILE_OR_GLOB")
        return 1
    out = resolve_path(args.out).resolve()

    stats = Stats()
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine:  # quits Stockfish on errors too
        engine.configure({"Threads": args.threads, "Hash": args.hash})
        print(f"[MISTAKES] {engine.id.get('name', 'Stockfish')}: {args.stockfish}", flush=True)
        verified = scan(engine, pgn_files, args, stats)

    unique = dedupe_by_position(verified)
    selected = cap_per_game(unique, args.max_per_game)
    new = merge_into_suite(out, selected)
    print_summary(args, stats, len(pgn_files), verified, unique, selected, new, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
