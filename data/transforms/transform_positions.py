"""Position characteristics (position_features) and optional Stockfish ground truth.

Usage (from the repository root):
  python -m data.transforms.transform_positions                    # Stockfish if found
  python -m data.transforms.transform_positions --stockfish PATH   # explicit binary
  python -m data.transforms.transform_positions --depth 12 --multipv 5 --workers 6
  python -m data.transforms.transform_positions --skip-stockfish

Stockfish is looked up via --stockfish, $STOCKFISH (or $STOCKFISH_PATH), then
`stockfish` on PATH. Every searched position gets Stockfish's top --multipv lines
at --depth, stored once per FEN in the stockfish_moves table; an engine move
outside those lines is scored alone, so its centipawn loss is always known.
Only positions without an analysis of that depth and width are analysed, by
--workers single-threaded Stockfish processes. The sf_* columns of search_stats
are then filled from stockfish_moves, with or without Stockfish, so a full
reload of the analytics tables never needs Stockfish again. Evaluations are
stored from the side to move's point of view, like the engine's own eval.
"""

import argparse
import chess
import chess.engine
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import duckdb
import pandas as pd

from ..etl.paths import ANALYTICS_DB, PROJECT_ROOT

# pawn value of pieces (approx)
piece_values = {
    'p': 1,
    'n': 3,
    'b': 3,
    'r': 5,
    'q': 9,
    'k': 10000
}

# piece contributions to game phasing
phase_values = {
    'p': 0,
    'n': 1,
    'b': 1,
    'r': 2,
    'q': 4,
    'k': None
}

# HELPERS

def is_passed(board: chess.Board, square: chess.Square, color: bool) -> bool:
    """
    Determines if the pawn on `square` is a passed pawn.
    
    A pawn is passed if there are **no enemy pawns** in the same file or adjacent files
    ahead of it (toward promotion).
    
    Parameters:
        board : chess.Board
        square : chess.Square
        color : True for White, False for Black
    
    Returns:
        bool : True if passed pawn, False otherwise
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)

    # Ranks ahead depending on color
    if color == chess.WHITE:
        ranks_ahead = range(rank+1, 8)
    else:
        ranks_ahead = range(rank-1, -1, -1)

    # Files to check: same file + adjacent
    files_to_check = [f for f in [file-1, file, file+1] if 0 <= f <= 7]

    # Check each square ahead for enemy pawns
    for r in ranks_ahead:
        for f in files_to_check:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN and piece.color != color:
                return False
    return True


# GAME CHARACTERISTICS

def get_game_phase(fen):
    """
    Returns 'opening', 'midgame', or 'endgame' based on the position.
    uses material based game phasing
    """
    board = chess.Board(fen)
    ply = board.fullmove_number * 2
    if not board.turn:
        ply -= 1  # subtract 1 if black to move, since fullmove_number counts full moves

    # Count pieces
    piece_counts = Counter()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_counts[piece.symbol().lower()] += 1

    # Compute material-based phase
    # TotalPhase = phase at start of game
    starting_phase = (phase_values['p']*16 + phase_values['n']*4 + 
                      phase_values['b']*4 + phase_values['r']*4 + 
                      phase_values['q']*2)

    phase = starting_phase
    phase -= piece_counts.get('p',0)*phase_values['p']
    phase -= piece_counts.get('n',0)*phase_values['n']
    phase -= piece_counts.get('b',0)*phase_values['b']
    phase -= piece_counts.get('r',0)*phase_values['r']
    phase -= piece_counts.get('q',0)*phase_values['q']

    # Thresholds for 3-phase model
    opening_phase_threshold = 0.25 * starting_phase
    middle_phase_threshold  = 2/3 * starting_phase
    opening_ply_max = 30  # first 15 moves

    if phase < opening_phase_threshold and ply <= opening_ply_max:
        return "opening"
    elif phase < middle_phase_threshold:
        return "midgame"
    else:
        return "endgame"

# Classifies positions purely from the FEN string — no engine metrics required.
# Returns a dict of sub-scores and a final label: Tactical / Positional / Endgame
 
import re
 
# Piece values for material counting (centipawns)
_PIECE_VALUES = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 0}
 
def _parse_fen_board(fen: str) -> dict:
    """Extract structured info from a FEN string."""
    parts = fen.strip().split()
    board_str = parts[0]
    side_to_move = parts[1] if len(parts) > 1 else "w"
    castling = parts[2] if len(parts) > 2 else "-"
    ep_square = parts[3] if len(parts) > 3 else "-"
 
    # Build piece lists
    pieces = {"w": {}, "b": {}}
    file_, rank_ = 0, 7
    for ch in board_str:
        if ch == "/":
            file_ = 0
            rank_ -= 1
        elif ch.isdigit():
            file_ += int(ch)
        else:
            color = "w" if ch.isupper() else "b"
            ptype = ch.lower()
            sq = rank_ * 8 + file_
            pieces[color].setdefault(ptype, []).append(sq)
            file_ += 1
 
    return {
        "pieces": pieces,
        "side": side_to_move,
        "castling": castling,
        "ep": ep_square,
    }
 
 
def _material_score(pieces: dict) -> dict:
    """Total material for each side and imbalance."""
    totals = {}
    for color in ("w", "b"):
        totals[color] = sum(
            _PIECE_VALUES.get(p, 0) * len(sqs)
            for p, sqs in pieces[color].items()
        )
    imbalance = abs(totals["w"] - totals["b"])
    total = totals["w"] + totals["b"]
    return {"white": totals["w"], "black": totals["b"],
            "imbalance": imbalance, "total": total}
 
 
def _pawn_structure(pieces: dict) -> dict:
    """Doubled, isolated, and passed pawn counts (approximate from squares)."""
    result = {}
    for color in ("w", "b"):
        pawns = pieces[color].get("p", [])
        files = [sq % 8 for sq in pawns]
        file_counts = {}
        for f in files:
            file_counts[f] = file_counts.get(f, 0) + 1
        doubled  = sum(1 for c in file_counts.values() if c > 1)
        isolated = sum(1 for f in file_counts
                       if (f - 1) not in file_counts and (f + 1) not in file_counts)
        result[color] = {"count": len(pawns), "doubled": doubled, "isolated": isolated,
                          "files": set(file_counts.keys())}
    return result
 
 
def _piece_mobility_proxy(pieces: dict) -> dict:
    """
    Very rough mobility proxy: count of non-pawn, non-king pieces.
    More pieces = more potential moves = richer positional play.
    """
    active = {}
    for color in ("w", "b"):
        active[color] = sum(
            len(sqs) for p, sqs in pieces[color].items() if p not in ("p", "k")
        )
    return active
 
 
def _king_safety(pieces: dict) -> dict:
    """
    Rough king safety: is the king still on its starting square / near a corner?
    Kings on e1/e8 (squares 4 / 60) are likely uncasled = slightly exposed.
    """
    safety = {}
    start_squares = {"w": 4, "b": 60}
    for color in ("w", "b"):
        king_sqs = pieces[color].get("k", [])
        if not king_sqs:
            safety[color] = "unknown"
            continue
        ksq = king_sqs[0]
        rank = ksq // 8
        file_ = ksq % 8
        # Cornered = safer; central = exposed
        corner_dist = min(file_, 7 - file_)   # 0 at a/h file, 3.5 at centre
        safety[color] = "exposed" if corner_dist >= 2 and rank in (0, 7) else "sheltered"
    return safety
 
 
def classify_position(fen: str) -> dict:
    """
    Classify a position purely from its FEN.
 
    Returns
    -------
    dict with keys:
        label       : "Tactical" | "Positional" | "Endgame"
        tactical_score   : 0–1 float
        positional_score : 0–1 float
        endgame_score    : 0–1 float
        features    : sub-feature breakdown dict
    """
    if not fen or not isinstance(fen, str):
        return {"label": "Unknown", "tactical_score": 0, "positional_score": 0,
                "endgame_score": 0, "features": {}}
 
    try:
        info    = _parse_fen_board(fen)
        board   = chess.Board(fen)
        pieces  = info["pieces"]
        mat     = _material_score(pieces)
        pawns   = _pawn_structure(pieces)
        mob     = _piece_mobility_proxy(pieces)
        king_s  = _king_safety(pieces)
 
        # ── Endgame signals ──────────────────────────────────────────────────
        total_mat   = mat["total"]
        # Endgame threshold: less than ~26cp worth of non-pawn material each
        # (roughly rook + minor piece or less)
        minor_mat = {c: sum(_PIECE_VALUES.get(p, 0) * len(sqs)
                            for p, sqs in pieces[c].items() if p not in ("p", "k"))
                     for c in ("w", "b")}
        endgame_score = max(0.0, 1.0 - (minor_mat["w"] + minor_mat["b"]) / 3000)
 
        # ── Tactical signals ─────────────────────────────────────────────────
        # Use immediate tactical indicators (captures/checks/hanging pieces/pawn tension)
        # rather than material imbalance which is often a consequence of tactics.
        # 1) captures & checks available for side-to-move
        captures_side = 0
        checks_side = 0
        try:
            for mv in board.legal_moves:
                if board.is_capture(mv):
                    captures_side += 1
                if board.gives_check(mv):
                    checks_side += 1
        except Exception:
            captures_side = 0
            checks_side = 0
        cap_score = min(captures_side / 6.0, 1.0) * 0.4
        check_score = min(checks_side / 2.0, 1.0) * 0.25

        # 2) hanging pieces: pieces attacked and undefended
        hanging = 0
        try:
            for color_key, items in pieces.items():
                color_bool = chess.WHITE if color_key == 'w' else chess.BLACK
                for ptype, sqs in items.items():
                    for sq in sqs:
                        attackers = board.attackers(not color_bool, sq)
                        defenders = board.attackers(color_bool, sq)
                        if attackers and not defenders:
                            hanging += 1
        except Exception:
            hanging = 0
        hang_score = min(hanging / 4.0, 1.0) * 0.25

        # 3) En passant / castling / pawn-structure (retain small signals)
        ep_score = 0.15 if info["ep"] != "-" else 0.0
        castling = info["castling"]
        castling_lost = sum(1 for c in "KQkq" if c not in castling)
        castling_score = castling_lost / 4 * 0.1
        pawn_damage = (
            pawns["w"]["doubled"] + pawns["w"]["isolated"] +
            pawns["b"]["doubled"] + pawns["b"]["isolated"]
        )
        pawn_score = min(pawn_damage / 8, 1.0) * 0.1

        tactical_score = cap_score + check_score + hang_score + ep_score + castling_score + pawn_score
        tactical_score = min(max(tactical_score, 0.0), 1.0)
        # Discount tactics somewhat in endgames
        tactical_score *= (1.0 - endgame_score * 0.4)
 
        # ── Positional signals ───────────────────────────────────────────────
        # High piece count, balanced material, intact pawn structure = positional
        piece_richness  = min((mob["w"] + mob["b"]) / 10, 1.0)
        # use a normalized material-imbalance for positional weighting only
        max_imbalance = 900.0
        imbalance_score = min(mat.get("imbalance", 0) / max_imbalance, 1.0)
        balance_score = 1.0 - imbalance_score
        structure_score = 1.0 - min(pawn_damage / 8, 1.0)
        positional_score = (piece_richness * 0.4 + balance_score * 0.35 + structure_score * 0.25)
        positional_score *= (1.0 - endgame_score * 0.6)
 
        # ── Normalise and label ──────────────────────────────────────────────
        scores = {"Tactical": tactical_score, "Positional": positional_score, "Endgame": endgame_score}
        label  = max(scores, key=scores.get)
 
        return {
            "label":            label,
            "tactical_score":   round(tactical_score,   3),
            "positional_score": round(positional_score, 3),
            "endgame_score":    round(endgame_score,    3),
            "features": {
                "total_material":   total_mat,
                "material_imbalance": mat["imbalance"],
                "ep_available":     info["ep"] != "-",
                "castling_rights_lost": castling_lost,
                "white_doubled_pawns":  pawns["w"]["doubled"],
                "white_isolated_pawns": pawns["w"]["isolated"],
                "black_doubled_pawns":  pawns["b"]["doubled"],
                "black_isolated_pawns": pawns["b"]["isolated"],
                "active_pieces_white":  mob["w"],
                "active_pieces_black":  mob["b"],
                "king_safety_white":    king_s["w"],
                "king_safety_black":    king_s["b"],
            }
        }
    except Exception as e:
        return {"label": "Unknown", "tactical_score": 0, "positional_score": 0,
                "endgame_score": 0, "features": {"error": str(e)}}
 
 
def classify_df(df: pd.DataFrame, fen_col: str = "fen") -> pd.DataFrame:
    """Vectorised classification: adds pos_label / pos_tactical / pos_positional / pos_endgame columns."""
    if fen_col not in df.columns or df.empty:
        return df
    results = df[fen_col].map(classify_position)
    df = df.copy()
    df["pos_label"]      = results.map(lambda r: r["label"])
    df["pos_tactical"]   = results.map(lambda r: r["tactical_score"])
    df["pos_positional"] = results.map(lambda r: r["positional_score"])
    df["pos_endgame"]    = results.map(lambda r: r["endgame_score"])
    return df

def get_position_balance(fen):
    board = chess.Board(fen)
    balance = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.symbol().lower()]
            if piece.color == chess.WHITE:
                balance += value
            else:
                balance -= value
    return balance


def get_pawn_characteristics(fen):
    board = chess.Board(fen)
    pawns = {chess.WHITE: [], chess.BLACK: []}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            pawns[piece.color].append(square)

    
    
    def analyze(color):
        backward = 0
        doubled = 0
        passed = 0
        files = [square % 8 for square in pawns[color]]
        counts = Counter(files)
        doubled = sum(v-1 for v in counts.values() if v > 1)
        # passed/backward detection simplified
        # can improve with pawn masks
        passed = sum(1 for sq in pawns[color] if is_passed(board, sq, color))
        backward = len(pawns[color]) - passed - doubled
        return backward, doubled, passed
    
    return analyze(chess.WHITE), analyze(chess.BLACK)


def get_king_safety(fen):
    board = chess.Board(fen)
    
    def king_metrics(color):
        king_square = board.king(color)
        rank, file = divmod(king_square, 8)
        
        # Pawn shield
        shield_squares = []
        if color == chess.WHITE:
            if rank+1 < 8:
                shield_squares = [chess.square(f, rank+1) for f in range(max(file-1,0), min(file+2,8))]
        else:
            if rank-1 >= 0:
                shield_squares = [chess.square(f, rank-1) for f in range(max(file-1,0), min(file+2,8))]
        shield_pawns = sum(1 for sq in shield_squares if board.piece_at(sq) and 
                           board.piece_at(sq).piece_type == chess.PAWN and board.piece_at(sq).color==color)
        
        # Open files near king
        open_files = sum(1 for f in range(max(file-1,0), min(file+2,8)) 
                         if not any(board.piece_at(chess.square(f, r)) for r in range(8)))
        
        # Tropism: sum of weighted inverse-distance of enemy pieces to king
        tropism = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color != color:
                pr, pf = divmod(sq, 8)
                distance = abs(rank-pr) + abs(file-pf)  # Manhattan distance
                distance = max(distance,1)  # avoid div by zero
                tropism += piece_values[piece.symbol().lower()] / distance
        
        return {'shield_pawns': shield_pawns, 'open_files': open_files, 'tropism': tropism}
    
    return king_metrics(chess.WHITE), king_metrics(chess.BLACK)



def get_mobility_characteristics(fen):
    board = chess.Board(fen)
    mobility = {}
    
    for color in [chess.WHITE, chess.BLACK]:
        if board.turn != color:
            board.push(chess.Move.null())  # switch turn to color for legal moves
        legal_moves = list(board.legal_moves)
        capture_moves = sum(board.is_capture(m) for m in legal_moves)
        check_moves = sum(board.gives_check(m) for m in legal_moves)

        # Enemy territory squares
        if color == chess.WHITE:
            enemy_squares = set(range(32,64))  # ranks 4-7
        else:
            enemy_squares = set(range(0,32))   # ranks 0-3
        
        # Legal moves into enemy territory
        legal_enemy = sum(1 for m in legal_moves if m.to_square in enemy_squares)
        
        # Controlled squares in enemy territory
        controlled_squares = set()
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == color:
                controlled_squares.update(board.attacks(sq) & enemy_squares)
        
        mobility[color] = {
            'num_moves': len(legal_moves),
            'capture_ratio': capture_moves / max(len(legal_moves),1),
            'check_ratio': check_moves / max(len(legal_moves),1),
            'legal_enemy': legal_enemy,
            'controlled_enemy': len(controlled_squares)
        }
    
    return mobility[chess.WHITE], mobility[chess.BLACK]

def _table_exists(cnxn, name):
    row = cnxn.execute("""
        SELECT 1 FROM information_schema.tables
        WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = ?
    """, [name]).fetchone()
    return row is not None


POSITION_FEATURES_DDL = """
    CREATE TABLE IF NOT EXISTS position_features (
        search_id   INTEGER,
        game_id     INTEGER,
        fen         TEXT,
        game_phase  TEXT,
        position_type TEXT,
        pos_tactical FLOAT,
        pos_positional FLOAT,
        pos_endgame FLOAT,
        balance     INTEGER,
        white_backwards INTEGER,
        white_doubled INTEGER,
        white_passed INTEGER,
        black_backwards INTEGER,
        black_doubled INTEGER,
        black_passed INTEGER,
        white_shield_pawns INTEGER,
        white_open_files INTEGER,
        white_tropism FLOAT,
        black_shield_pawns INTEGER,
        black_open_files INTEGER,
        black_tropism FLOAT,
        white_num_moves INTEGER,
        white_capture_ratio FLOAT,
        white_check_ratio FLOAT,
        white_legal_enemy INTEGER,
        white_controlled_enemy INTEGER,
        black_num_moves INTEGER,
        black_capture_ratio FLOAT,
        black_check_ratio FLOAT,
        black_legal_enemy INTEGER,
        black_controlled_enemy INTEGER
    )
"""


def position_feature_row(search_id, game_id, fen):
    """position_features values for one position; features are NULL when the FEN can't be analysed."""
    try:
        game_phase = get_game_phase(fen)
        classification = classify_position(fen)
        balance = get_position_balance(fen)
        # Pawns
        (w_backward, w_doubled, w_passed), (b_backward, b_doubled, b_passed) = get_pawn_characteristics(fen)
        # King safety + tropism
        ws, bs = get_king_safety(fen)
        # Mobility + space
        wm, bm = get_mobility_characteristics(fen)
    except Exception:
        return (search_id, game_id, fen, None, "Unknown") + (None,) * 26
    return (
        search_id, game_id, fen, game_phase, classification["label"],
        classification["tactical_score"], classification["positional_score"], classification["endgame_score"],
        balance,
        w_backward, w_doubled, w_passed,
        b_backward, b_doubled, b_passed,
        ws['shield_pawns'], ws['open_files'], ws['tropism'],
        bs['shield_pawns'], bs['open_files'], bs['tropism'],
        wm['num_moves'], wm['capture_ratio'], wm['check_ratio'],
        wm['legal_enemy'], wm['controlled_enemy'],
        bm['num_moves'], bm['capture_ratio'], bm['check_ratio'],
        bm['legal_enemy'], bm['controlled_enemy'],
    )


def build_position_features(cnxn):
    """Analyse every searched position without a position_features row (incremental).

    The table is always created, even when there is nothing to analyse, so the
    views that join it can be installed.
    """
    cnxn.execute(POSITION_FEATURES_DDL)
    if not _table_exists(cnxn, "search_stats"):
        print("  position_features: search_stats missing -- nothing to process")
        return 0

    # Only fetch positions not already processed (incremental)
    rows = cnxn.execute("""
        SELECT s.id AS search_id, s.fen, s.game_id
        FROM search_stats s
        LEFT JOIN position_features pf ON s.id = pf.search_id
        WHERE pf.search_id IS NULL AND s.fen IS NOT NULL
        ORDER BY s.id
    """).fetchall()

    if not rows:
        print(f"  position_features: 0 new positions to process")
        return 0

    print(f"  position_features: processing {len(rows)} new positions...")

    BATCH_SIZE = 1000
    batch = []

    for i, (search_id, fen, game_id) in enumerate(rows):
        batch.append(position_feature_row(search_id, game_id, fen))

        if len(batch) >= BATCH_SIZE:
            _insert_batch(cnxn, batch)
            batch = []
            if (i + 1) % 5000 == 0:
                print(f"    ...processed {i + 1}/{len(rows)}")

    # flush remaining
    if batch:
        _insert_batch(cnxn, batch)

    print(f"  position_features: done ({len(rows)} rows inserted)")
    return len(rows)


def _insert_batch(cnxn, batch):
    """Insert a batch of position feature tuples."""
    cnxn.executemany(
        """
        INSERT INTO position_features (
            search_id, game_id, fen, game_phase, position_type,
            pos_tactical, pos_positional, pos_endgame, balance,
            white_backwards, white_doubled, white_passed,
            black_backwards, black_doubled, black_passed,
            white_shield_pawns, white_open_files, white_tropism,
            black_shield_pawns, black_open_files, black_tropism,
            white_num_moves, white_capture_ratio, white_check_ratio,
            white_legal_enemy, white_controlled_enemy,
            black_num_moves, black_capture_ratio, black_check_ratio,
            black_legal_enemy, black_controlled_enemy
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stockfish ground truth (optional)
# ─────────────────────────────────────────────────────────────────────────────
MATE_SCORE = 100_000  # the engine's MATE_SCORE: mate is +/-(MATE_SCORE - plies to mate)
# Stockfish's top lines per position. On balanced positions, 5 lines at depth 10
# take ~0.1 s on one core; against a depth-18 reference, depth 10's best move
# loses 7 cp on average (depth 8: 15 cp; depth 12 costs ~3x more for 6 cp).
DEFAULT_DEPTH = 10
DEFAULT_MULTIPV = 5
BATCH_SIZE = 200      # positions per write, so an interrupted run keeps its progress
ENGINE_OPTIONS = {"Threads": 1, "Hash": 16}   # per Stockfish process (options it doesn't have are skipped)
HANDSHAKE_TIMEOUT = 30.0  # seconds for a new engine to answer `uci` and `isready`

STOCKFISH_COLUMNS = (
    ("sf_eval", "INTEGER"),        # Stockfish's first line: side to move's point of view, engine scale
    ("sf_best_move", "VARCHAR"),   # first move of Stockfish's first line
    ("sf_time_ms", "DOUBLE"),      # time of the position's analysis
    ("sf_computed_at", "TIMESTAMP"),
    ("sf_pv", "VARCHAR"),          # JSON list of UCI moves of the first line
    ("sf_eval_pov", "VARCHAR"),    # 'stm'; NULL marks evaluations stored from White's view by older versions
    ("sf_depth", "INTEGER"),       # Stockfish's search depth
    ("sf_multipv", "INTEGER"),     # lines analysed: the N of "Stockfish's top N"
    ("sf_move_rank", "INTEGER"),   # line of the engine's move: 1..sf_multipv, 0 = not among the lines
    ("sf_move_eval", "INTEGER"),   # Stockfish's eval of the engine's move (side to move)
)

# Stockfish's analysis per position (FEN), shared by every search of that
# position. It outlives the per-search tables: after load_analytics --full the
# sf_* columns are restored from here instead of running Stockfish again.
STOCKFISH_MOVES_DDL = """
    CREATE TABLE IF NOT EXISTS stockfish_moves (
        fen          VARCHAR NOT NULL,
        move         VARCHAR NOT NULL,   -- UCI
        rank         INTEGER,            -- Stockfish's line (1 = best); NULL: an engine move outside the lines, scored alone
        eval         INTEGER,            -- side to move's point of view, engine scale
        pv           VARCHAR,            -- JSON list of UCI moves, starting with `move`
        depth        INTEGER,            -- depth searched
        multipv      INTEGER,            -- lines requested for the position (NULL on single-move rows)
        time_ms      DOUBLE,             -- the analysis' time (the whole MultiPV search on line rows)
        stockfish    VARCHAR,            -- the engine's UCI id name
        computed_at  TIMESTAMP
    )
"""


def _executable(value):
    path = Path(value).expanduser()
    if path.is_file():
        return str(path)
    return shutil.which(str(value))


def _bundled_stockfish():
    """A binary in the repository's engines/stockfish/ folder (the old default location)."""
    folder = PROJECT_ROOT / "engines" / "stockfish"
    if not folder.is_dir():
        return None
    for candidate in sorted(folder.glob("stockfish*")):
        if not candidate.is_file():
            continue
        if os.name == "nt" and candidate.suffix.lower() != ".exe":
            continue
        if os.name != "nt" and not os.access(candidate, os.X_OK):
            continue
        return str(candidate)
    return None


def find_stockfish(path=None):
    """Stockfish from `path` (--stockfish), else $STOCKFISH / $STOCKFISH_PATH, else PATH,
    else the repository's engines/stockfish/ folder. Returns None when there is none.

    A configured path that doesn't exist is reported and not replaced by another binary.
    """
    for source, value in (("--stockfish", path),
                          ("STOCKFISH", os.environ.get("STOCKFISH")),
                          ("STOCKFISH_PATH", os.environ.get("STOCKFISH_PATH"))):
        if value:
            found = _executable(value)
            if not found:
                print(f"  stockfish: {source}={value} is not an executable file")
            return found
    return shutil.which("stockfish") or _bundled_stockfish()


def stockfish_score(score):
    """Engine-scale integer of a python-chess Score: centipawns, or +/-(MATE_SCORE - plies to mate)."""
    if score is None:
        return None
    mate = score.mate()
    if mate is None:
        cp = score.score()
        return None if cp is None else int(cp)
    if mate > 0:
        return MATE_SCORE - (2 * mate - 1)   # the side to move mates in `mate` moves
    return -(MATE_SCORE + 2 * mate)          # the side to move is mated in -mate moves


class EngineTerminated(Exception):
    """The UCI engine exited (or closed its output) in the middle of a command."""


class UciEngine:
    """A minimal UCI client for depth-limited analysis.

    python-chess parses every intermediate `info` line and replays each PV on a board;
    with MultiPV that costs more Python time than Stockfish's own search takes, so one
    Python process could keep only ~3 engines busy. This client keeps the last scored
    line per MultiPV index as plain strings. Its blocking pipe reads release the GIL,
    so many engines run side by side from one process.
    """

    def __init__(self, command, handshake_timeout=None):
        args = list(command) if isinstance(command, (list, tuple)) else [str(command)]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                     errors="replace", bufsize=1)
        self.name, self.options, self.multipv = None, set(), None
        # A program that never answers `uci` would block the reads below forever: kill it instead.
        watchdog = threading.Timer(handshake_timeout or HANDSHAKE_TIMEOUT, self.proc.kill)
        watchdog.daemon = True
        watchdog.start()
        try:
            self._send("uci")
            while (line := self._readline()) != "uciok":
                if line.startswith("id name "):
                    self.name = line[len("id name "):].strip()
                elif line.startswith("option name ") and " type " in line:
                    self.options.add(line[len("option name "):line.index(" type ")])
            for name, value in ENGINE_OPTIONS.items():
                if name in self.options:
                    self._send(f"setoption name {name} value {value}")
            self._ready()
        except BaseException:
            self.quit()
            raise
        finally:
            watchdog.cancel()

    def _send(self, command):
        try:
            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()
        except OSError as exc:
            raise EngineTerminated(str(exc)) from exc

    def _readline(self):
        line = self.proc.stdout.readline()
        if not line:
            raise EngineTerminated(f"the engine exited (code {self.proc.poll()})")
        return line.strip()

    def _ready(self):
        self._send("isready")
        while self._readline() != "readyok":
            pass

    def search(self, fen, depth, multipv=1, moves=()):
        """[(eval, pv)] of the final line per MultiPV index, best first; evals from the side to move.

        Every search starts with a cleared hash (ucinewgame), so its result depends only on the
        position and the settings, never on what was analysed before or by which process.
        """
        if multipv != self.multipv and "MultiPV" in self.options:
            self._send(f"setoption name MultiPV value {multipv}")
            self.multipv = multipv
        self._send("ucinewgame")
        self._ready()
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}" + (" searchmoves " + " ".join(moves) if moves else ""))
        lines = {}   # MultiPV index -> (depth, exact, eval, pv) of the latest line, deepest first
        while not (line := self._readline()).startswith("bestmove"):
            if not line.startswith("info ") or " pv " not in line or " score " not in line:
                continue
            tokens = line.split()
            try:
                at = tokens.index("score")
                kind, value = tokens[at + 1], int(tokens[at + 2])
                line_depth = int(tokens[tokens.index("depth") + 1]) if "depth" in tokens else 0
                index = int(tokens[tokens.index("multipv") + 1]) if "multipv" in tokens else 1
            except (ValueError, IndexError):
                continue
            exact = "lowerbound" not in tokens and "upperbound" not in tokens
            pv = tokens[tokens.index("pv") + 1:]
            score = chess.engine.Cp(value) if kind == "cp" else chess.engine.Mate(value)
            previous = lines.get(index)
            if pv and (previous is None or (line_depth, exact) >= previous[:2]):
                lines[index] = (line_depth, exact, stockfish_score(score), pv)
        return [(value, pv) for _, _, value, pv in (lines[i] for i in sorted(lines))]

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except (EngineTerminated, subprocess.TimeoutExpired):
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        finally:
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    stream.close()
                except OSError:
                    pass


def analyse_lines(engine, fen, depth=DEFAULT_DEPTH, multipv=DEFAULT_MULTIPV):
    """Stockfish's top `multipv` lines of a position: ([(move, eval, pv), ...] best first, milliseconds)."""
    started = time.perf_counter()
    lines = engine.search(fen, depth, multipv)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return [(pv[0], value, pv) for value, pv in lines], elapsed_ms


def score_move(engine, fen, move, depth=DEFAULT_DEPTH):
    """Stockfish's (eval, pv, milliseconds) for one move of a position, searched alone (searchmoves)."""
    started = time.perf_counter()
    lines = engine.search(fen, depth, 1, moves=[move])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not lines:
        return None, [move], elapsed_ms
    value, pv = lines[0]
    if pv[0] != move:   # the engine searched another move: its score says nothing about this one
        return None, [move], elapsed_ms
    return value, pv, elapsed_ms


def _legal(board, uci):
    try:
        return board.is_legal(chess.Move.from_uci(uci))
    except ValueError:
        return False


def _legal_moves(fen, moves):
    """The moves (UCI) that are legal in the position; none when the FEN is not a valid position."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return []
    return [move for move in moves if _legal(board, move)] if board.is_valid() else []


def default_workers():
    """Stockfish processes to run side by side: half the logical CPUs (about one per physical core)."""
    return max(1, (os.cpu_count() or 2) // 2)


class _EnginePool:
    """One single-threaded Stockfish process per worker thread, started on first use."""

    def __init__(self, command):
        self.command = command
        self.local = threading.local()
        self.lock = threading.Lock()
        self.engines = []
        self.name = None

    def get(self):
        engine = getattr(self.local, "engine", None)
        if engine is None:
            engine = UciEngine(self.command)
            with self.lock:
                self.engines.append(engine)
                self.name = self.name or engine.name
            self.local.engine = engine
        return engine

    def restart(self):
        engine = getattr(self.local, "engine", None)
        self.local.engine = None
        if engine is not None:
            with self.lock:
                if engine in self.engines:
                    self.engines.remove(engine)
            _quit(engine)
        return self.get()

    def close(self):
        with self.lock:
            engines, self.engines = self.engines, []
        for engine in engines:
            _quit(engine)


def _quit(engine):
    try:
        engine.quit()
    except Exception:
        pass


def _analyse_task(pool, fen, moves, depth, multipv, need_lines):
    """stockfish_moves rows (without stockfish/computed_at) for one position, or None when the
    position can't be analysed: its lines (when `need_lines`), then a single-move score for
    every engine move that is legal and not among the lines."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    if not board.is_valid() or not any(board.legal_moves):
        return None
    moves = [move for move in moves if _legal(board, move)]
    if not need_lines and not moves:
        return []
    for attempt in (1, 2):
        engine = pool.get() if attempt == 1 else pool.restart()
        try:
            rows, ranked = [], set()
            if need_lines:
                lines, elapsed_ms = analyse_lines(engine, fen, depth, multipv)
                for rank, (move, value, pv) in enumerate(lines, 1):
                    rows.append((fen, move, rank, value, json.dumps(pv), depth, multipv, elapsed_ms))
                ranked = {move for move, _, _ in lines}
            for move in moves:
                if move not in ranked:
                    value, pv, elapsed_ms = score_move(engine, fen, move, depth)
                    rows.append((fen, move, None, value, json.dumps(pv), depth, None, elapsed_ms))
            return rows
        except EngineTerminated:
            if attempt == 2:
                raise
    return None


def ensure_stockfish_columns(conn):
    for name, dtype in STOCKFISH_COLUMNS:
        conn.execute(f'ALTER TABLE search_stats ADD COLUMN IF NOT EXISTS "{name}" {dtype}')


def ensure_stockfish_tables(conn):
    """The stockfish_moves cache, and the sf_* columns when search_stats exists."""
    conn.execute(STOCKFISH_MOVES_DDL)
    if _table_exists(conn, "search_stats"):
        ensure_stockfish_columns(conn)


def convert_legacy_stockfish_evals(conn):
    """Older versions stored sf_eval from White's point of view: flip black-to-move rows once."""
    count = conn.execute(
        "SELECT count(*) FROM search_stats WHERE sf_eval IS NOT NULL AND sf_eval_pov IS NULL"
    ).fetchone()[0]
    if count:
        conn.execute("""
            UPDATE search_stats
            SET sf_eval = CASE WHEN split_part(fen, ' ', 2) = 'b' THEN -sf_eval ELSE sf_eval END,
                sf_eval_pov = 'stm'
            WHERE sf_eval IS NOT NULL AND sf_eval_pov IS NULL
        """)
        print(f"  stockfish: converted {count} stored evaluations from White's view to the side to move's")
    return count


def _search_columns(conn):
    return {row[0] for row in conn.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_catalog = current_database() AND table_schema = 'main' AND table_name = 'search_stats'
    """).fetchall()}


def _pending_tasks(conn, depth, multipv, limit=None):
    """[(fen, engine moves, need_lines, depth, multipv)] of the Stockfish work to do.

    First the positions without lines of at least `depth` and `multipv` (at most `limit`, in
    a fixed pseudo-random order). They are analysed at the larger of the requested and the
    stored depth, and likewise for the number of lines, so an analysis never gets shallower
    or narrower. Then the analysed positions with engine moves that are neither among the
    lines nor scored alone: those moves are scored at the depth of the position's lines, so a
    move's loss always compares two evaluations of the same depth.
    """
    has_move = "move" in _search_columns(conn)
    moves = ("coalesce(list(DISTINCT s.move ORDER BY s.move) FILTER (WHERE s.move IS NOT NULL AND s.move <> ''), [])"
             if has_move else "[]")
    analysed = """
        SELECT fen, min(depth) AS depth, min(multipv) AS multipv
        FROM stockfish_moves WHERE rank IS NOT NULL GROUP BY fen
    """
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE _sf_todo AS
        WITH analysed AS ({analysed})
        SELECT p.fen, greatest({int(depth)}, coalesce(a.depth, 0)) AS depth,
               greatest({int(multipv)}, coalesce(a.multipv, 0)) AS multipv
        FROM (SELECT DISTINCT fen FROM search_stats WHERE fen IS NOT NULL AND fen <> '') p
        LEFT JOIN analysed a ON a.fen = p.fen
        WHERE a.fen IS NULL OR a.depth < {int(depth)} OR a.multipv < {int(multipv)}
        ORDER BY hash(p.fen), p.fen
        {f'LIMIT {int(limit)}' if limit and limit > 0 else ''}
    """)
    tasks = [(fen, list(moves_), True, task_depth, task_multipv)
             for fen, task_depth, task_multipv, moves_ in conn.execute(f"""
        SELECT t.fen, t.depth, t.multipv, {moves} FROM _sf_todo t LEFT JOIN search_stats s ON s.fen = t.fen
        GROUP BY t.fen, t.depth, t.multipv ORDER BY hash(t.fen), t.fen
    """).fetchall()]
    if has_move:
        tasks += [(fen, list(moves_), False, lines_depth, None) for fen, lines_depth, moves_ in conn.execute(f"""
            WITH analysed AS ({analysed})
            SELECT s.fen, a.depth, list(DISTINCT s.move ORDER BY s.move)
            FROM search_stats s
            JOIN analysed a ON a.fen = s.fen
            LEFT JOIN stockfish_moves m
                   ON m.fen = s.fen AND m.move = s.move AND (m.rank IS NOT NULL OR m.depth = a.depth)
            WHERE s.move IS NOT NULL AND s.move <> '' AND m.fen IS NULL
              AND s.fen NOT IN (SELECT fen FROM _sf_todo)
            GROUP BY s.fen, a.depth ORDER BY hash(s.fen), s.fen
        """).fetchall()]
    conn.execute("DROP TABLE _sf_todo")
    return tasks


def _write_stockfish_rows(conn, replace, rows, stockfish, computed_at, chunk=400):
    """Store analysed rows: positions in `replace` lose their previous rows first, and a
    single-move score replaces an older one of the same move. One transaction."""
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE _sf_rows (
                fen VARCHAR, move VARCHAR, rank INTEGER, eval INTEGER, pv VARCHAR,
                depth INTEGER, multipv INTEGER, time_ms DOUBLE
            )
        """)
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            conn.execute("INSERT INTO _sf_rows VALUES " + ", ".join(["(?, ?, ?, ?, ?, ?, ?, ?)"] * len(part)),
                         [value for row in part for value in row])
        if replace:
            conn.execute("CREATE OR REPLACE TEMP TABLE _sf_replace (fen VARCHAR)")
            for start in range(0, len(replace), chunk):
                part = replace[start:start + chunk]
                conn.execute("INSERT INTO _sf_replace VALUES " + ", ".join(["(?)"] * len(part)), part)
            conn.execute("DELETE FROM stockfish_moves WHERE fen IN (SELECT fen FROM _sf_replace)")
            conn.execute("DROP TABLE _sf_replace")
        conn.execute("""
            DELETE FROM stockfish_moves AS t USING _sf_rows AS r
            WHERE t.rank IS NULL AND r.rank IS NULL AND t.fen = r.fen AND t.move = r.move
        """)
        conn.execute("""
            INSERT INTO stockfish_moves
            SELECT fen, move, rank, eval, pv, depth, multipv, time_ms, ?, ? FROM _sf_rows
        """, [stockfish, computed_at])
        conn.execute("DROP TABLE _sf_rows")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def analyse_positions(conn, engine_path, depth=DEFAULT_DEPTH, multipv=DEFAULT_MULTIPV, limit=None,
                      workers=None, batch_size=BATCH_SIZE):
    """Run Stockfish on the searched positions that need it and store the results in
    stockfish_moves. Returns (positions analysed, single moves scored).

    `engine_path` is a binary path or a command list. `workers` Stockfish processes
    (default: default_workers()) analyse positions side by side; the results are written
    every `batch_size` positions, so an interrupted run keeps its progress.
    """
    ensure_stockfish_tables(conn)
    tasks = []
    for fen, moves, need_lines, task_depth, task_multipv in _pending_tasks(conn, depth, multipv, limit):
        if not need_lines:
            moves = _legal_moves(fen, moves)   # an illegal engine move ('0000') is never scored
            if not moves:
                continue
        tasks.append((fen, moves, need_lines, task_depth, task_multipv))
    if not tasks:
        print("  stockfish: 0 positions to analyse")
        return 0, 0
    positions_todo = sum(1 for task in tasks if task[2])
    deeper = sum(1 for task in tasks if task[2] and (task[3] > depth or task[4] > multipv))
    workers = max(1, min(workers or default_workers(), len(tasks)))
    try:
        probe = UciEngine(engine_path)
    except (OSError, EngineTerminated) as exc:
        raise RuntimeError(f"cannot start {engine_path}: {exc}") from exc
    probe.quit()
    if multipv > 1 and "MultiPV" not in probe.options:
        raise RuntimeError(f"{probe.name or engine_path} has no MultiPV option, so it cannot give its top {multipv} lines")
    print(f"  stockfish: {positions_todo} positions to analyse (depth {depth}, {multipv} lines"
          + (f"; {deeper} kept deeper or wider, as analysed before" if deeper else "")
          + f") and {len(tasks) - positions_todo} more to score engine moves in; "
            f"{workers} {probe.name or 'engine'} process(es)")
    pool = _EnginePool(engine_path)
    replace, rows = [], []
    analysed = moves_scored = skipped = failed = done = 0
    started = last_report = time.perf_counter()

    def flush():
        # the batch is taken first: a write that fails (or is interrupted) is not retried
        nonlocal replace, rows
        batch_replace, batch_rows, replace, rows = replace, rows, [], []
        if batch_rows or batch_replace:
            computed_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
            _write_stockfish_rows(conn, batch_replace, batch_rows, pool.name, computed_at)

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stockfish")
    queue = iter(tasks)
    running = {}   # future -> (fen, need_lines); a few tasks per process, so memory stays flat

    def submit_next():
        task = next(queue, None)
        if task is not None:
            fen, moves, need_lines, task_depth, task_multipv = task
            running[executor.submit(_analyse_task, pool, fen, moves, task_depth, task_multipv,
                                    need_lines)] = (fen, need_lines)

    try:
        for _ in range(workers * 4):
            submit_next()
        pending = 0
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                fen, need_lines = running.pop(future)
                submit_next()
                done += 1
                try:
                    result = future.result()
                except Exception as e:
                    # nothing stored: retried on the next run
                    failed += 1
                    if failed <= 5:
                        print(f"[WARN] Stockfish failed on {fen}: {e}")
                    if failed >= 20 and failed > done // 2:
                        raise RuntimeError(f"Stockfish failed on {failed} of {done} positions; stopping")
                    continue
                if result is None:
                    skipped += 1
                    continue
                if need_lines:
                    replace.append(fen)
                    analysed += 1
                moves_scored += sum(1 for row in result if row[2] is None)
                rows.extend(result)
                pending += 1
                if pending >= batch_size:
                    flush()
                    pending = 0
            now = time.perf_counter()
            if now - last_report >= 30 and done < len(tasks):
                last_report = now
                rate = done / (now - started)
                print(f"  ...{done}/{len(tasks)} ({rate:.1f}/s, about {(len(tasks) - done) / rate / 60:.0f} min left)")
    finally:
        try:
            flush()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            pool.close()
    extra = "".join(f", {n} {what}" for n, what in ((skipped, "skipped (invalid or no legal moves)"),
                                                    (failed, "failed")) if n)
    print(f"  stockfish: analysed {analysed} positions and scored {moves_scored} engine moves outside the lines "
          f"in {time.perf_counter() - started:.0f} s{extra}")
    return analysed, moves_scored


# The engine's move among Stockfish's lines: its line, 0 when it was scored alone
# (outside the lines), NULL when unknown (no engine move, or not scored yet).
SYNC_SQL = """
CREATE OR REPLACE TEMP TABLE _sf_sync AS
WITH best AS (
    SELECT * FROM stockfish_moves WHERE rank = 1
    QUALIFY row_number() OVER (PARTITION BY fen ORDER BY computed_at DESC) = 1
), scored AS (
    SELECT fen, move, rank, eval FROM stockfish_moves
    QUALIFY row_number() OVER (PARTITION BY fen, move ORDER BY rank NULLS LAST, computed_at DESC) = 1
), fresh AS (
    SELECT s.id,
           b.eval AS sf_eval, b.move AS sf_best_move, b.time_ms AS sf_time_ms, b.computed_at AS sf_computed_at,
           b.pv AS sf_pv, 'stm' AS sf_eval_pov, b.depth AS sf_depth, b.multipv AS sf_multipv,
           CASE WHEN m.rank IS NOT NULL THEN m.rank WHEN m.move IS NOT NULL THEN 0 END AS sf_move_rank,
           m.eval AS sf_move_eval,
           s.sf_eval AS old_eval, s.sf_best_move AS old_best_move, s.sf_computed_at AS old_computed_at,
           s.sf_pv AS old_pv, s.sf_eval_pov AS old_eval_pov, s.sf_depth AS old_depth, s.sf_multipv AS old_multipv,
           s.sf_move_rank AS old_move_rank, s.sf_move_eval AS old_move_eval
    FROM search_stats s
    JOIN best b ON b.fen = s.fen
    LEFT JOIN scored m ON m.fen = s.fen AND m.move = {move}
)
SELECT id, sf_eval, sf_best_move, sf_time_ms, sf_computed_at, sf_pv, sf_eval_pov, sf_depth, sf_multipv,
       sf_move_rank, sf_move_eval
FROM fresh
WHERE sf_eval IS DISTINCT FROM old_eval OR sf_best_move IS DISTINCT FROM old_best_move
   OR sf_computed_at IS DISTINCT FROM old_computed_at OR sf_pv IS DISTINCT FROM old_pv
   OR sf_eval_pov IS DISTINCT FROM old_eval_pov OR sf_depth IS DISTINCT FROM old_depth
   OR sf_multipv IS DISTINCT FROM old_multipv OR sf_move_rank IS DISTINCT FROM old_move_rank
   OR sf_move_eval IS DISTINCT FROM old_move_eval
"""


def sync_search_stats(conn):
    """Copy Stockfish's analysis from stockfish_moves to the sf_* columns of every search of
    an analysed position; returns the number of search_stats rows changed.

    Needs no Stockfish, so it also restores the columns after load_analytics --full.
    Searches of positions without an analysis keep their values.
    """
    if not _table_exists(conn, "search_stats"):
        return 0
    ensure_stockfish_tables(conn)
    move = "s.move" if "move" in _search_columns(conn) else "NULL"
    conn.execute(SYNC_SQL.format(move=move))
    changed = conn.execute("SELECT count(*) FROM _sf_sync").fetchone()[0]
    if changed:
        conn.execute("""
            UPDATE search_stats AS s
            SET sf_eval = u.sf_eval, sf_best_move = u.sf_best_move, sf_time_ms = u.sf_time_ms,
                sf_computed_at = u.sf_computed_at, sf_pv = u.sf_pv, sf_eval_pov = u.sf_eval_pov,
                sf_depth = u.sf_depth, sf_multipv = u.sf_multipv, sf_move_rank = u.sf_move_rank,
                sf_move_eval = u.sf_move_eval
            FROM _sf_sync AS u
            WHERE s.id = u.id
        """)
    conn.execute("DROP TABLE _sf_sync")
    return changed


def update_search_stats_with_stockfish(conn, engine_path, depth=DEFAULT_DEPTH, limit=None,
                                       multipv=DEFAULT_MULTIPV, workers=1, batch_size=BATCH_SIZE):
    """Analyse the positions that need it, then update search_stats; returns the rows changed."""
    analyse_positions(conn, engine_path, depth=depth, multipv=multipv, limit=limit, workers=workers,
                      batch_size=batch_size)
    return sync_search_stats(conn)


def run(cnxn, stockfish=None, skip_stockfish=False, depth=DEFAULT_DEPTH, limit=None, multipv=DEFAULT_MULTIPV,
        workers=None):
    """Pipeline step: Stockfish's lines for new positions when a binary is available, the sf_*
    columns of search_stats from the stored analysis (always), then position features."""
    ensure_stockfish_tables(cnxn)
    if _table_exists(cnxn, "search_stats"):
        convert_legacy_stockfish_evals(cnxn)
        if skip_stockfish:
            print("  stockfish: skipped (--skip-stockfish)")
        else:
            engine_path = find_stockfish(stockfish)
            if engine_path is None:
                print("  stockfish: not found -- skipping new analysis "
                      "(use --stockfish PATH, set STOCKFISH, or put stockfish on PATH)")
            else:
                try:
                    analyse_positions(cnxn, engine_path, depth=depth, multipv=multipv, limit=limit, workers=workers)
                except Exception as e:
                    print(f"[WARN] Ground-truth computation failed or stopped: {e}")
        changed = sync_search_stats(cnxn)
        stored = cnxn.execute("SELECT count(DISTINCT fen) FROM stockfish_moves WHERE rank IS NOT NULL").fetchone()[0]
        print(f"  stockfish: {stored} positions analysed in total; {changed} search_stats rows updated from them")
    build_position_features(cnxn)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Position features and optional Stockfish ground truth")
    parser.add_argument("--db", type=Path, help=f"Analytics database (default: CHESS_ANALYTICS_DB or {ANALYTICS_DB})")
    parser.add_argument("--stockfish", help="Stockfish binary (default: $STOCKFISH, then stockfish on PATH)")
    parser.add_argument("--skip-stockfish", action="store_true",
                        help="No new Stockfish analysis (stored results are still applied)")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help=f"Stockfish search depth (default: {DEFAULT_DEPTH})")
    parser.add_argument("--multipv", type=int, default=DEFAULT_MULTIPV,
                        help=f"Stockfish lines per position, the N of 'top N' (default: {DEFAULT_MULTIPV})")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Stockfish processes side by side (default: {default_workers()}, half the logical CPUs)")
    parser.add_argument("--limit", type=int, help="Analyse at most this many new positions with Stockfish")
    args = parser.parse_args(argv)
    if args.depth < 1 or args.multipv < 1 or (args.workers is not None and args.workers < 1):
        parser.error("--depth, --multipv and --workers must be positive")

    db = args.db or os.environ.get('CHESS_ANALYTICS_DB') or ANALYTICS_DB
    with duckdb.connect(str(db)) as cnxn:
        run(cnxn, stockfish=args.stockfish, skip_stockfish=args.skip_stockfish, depth=args.depth,
            limit=args.limit, multipv=args.multipv, workers=args.workers)


if __name__ == "__main__":
    main()
