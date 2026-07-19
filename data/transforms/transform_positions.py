"""
get characteristics of a position
"""

import chess
from collections import Counter
import pandas as pd
import datetime
import time

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

def build_position_features(cnxn):
    # Create table if it doesn't exist
    cnxn.execute("""
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
    """)

    # Only fetch positions not already processed (incremental)
    rows = cnxn.execute("""
        SELECT d.search_id, d.fen, d.game_id, d.sts_id
        FROM dim_positions d
        LEFT JOIN position_features pf ON d.search_id = pf.search_id
        WHERE pf.search_id IS NULL AND d.fen IS NOT NULL
    """).fetchall()

    if not rows:
        print(f"  position_features: 0 new positions to process")
        return

    print(f"  position_features: processing {len(rows)} new positions...")

    BATCH_SIZE = 1000
    batch = []

    for i, row in enumerate(rows):
        search_id, fen, game_id, sts_id = row

        # Extract features
        game_phase = get_game_phase(fen)
        classification = classify_position(fen)
        balance = get_position_balance(fen)

        tactical_score   = classification["tactical_score"]
        positional_score = classification["positional_score"]
        endgame_score    = classification["endgame_score"]
        position_type    = classification["label"]

        # Pawns
        wp, bp = get_pawn_characteristics(fen)
        w_backward, w_doubled, w_passed = wp
        b_backward, b_doubled, b_passed = bp

        # King safety + tropism
        ws, bs = get_king_safety(fen)

        # Mobility + space
        wm, bm = get_mobility_characteristics(fen)

        # ---- batch insert ----
        batch.append((
            search_id, game_id, fen, game_phase, position_type,
            tactical_score, positional_score, endgame_score, balance,
            w_backward, w_doubled, w_passed,
            b_backward, b_doubled, b_passed,
            ws['shield_pawns'], ws['open_files'], ws['tropism'],
            bs['shield_pawns'], bs['open_files'], bs['tropism'],
            wm['num_moves'], wm['capture_ratio'], wm['check_ratio'],
            wm['legal_enemy'], wm['controlled_enemy'],
            bm['num_moves'], bm['capture_ratio'], bm['check_ratio'],
            bm['legal_enemy'], bm['controlled_enemy'],
        ))

        if len(batch) >= BATCH_SIZE:
            _insert_batch(cnxn, batch)
            batch = []
            if (i + 1) % 5000 == 0:
                print(f"    ...processed {i + 1}/{len(rows)}")

    # flush remaining
    if batch:
        _insert_batch(cnxn, batch)

    print(f"  position_features: done ({len(rows)} rows inserted)")


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


import duckdb
import shutil
import os
from pathlib import Path
from ..etl.paths import ANALYTICS_DB

if __name__ == "__main__":
    DB = os.environ.get('CHESS_ANALYTICS_DB') or str(ANALYTICS_DB)
    cwd_dir = Path(__file__).resolve().parent

    cnxn = duckdb.connect(DB)

    # Inline: compute Stockfish eval + best move and store in `search_stats` columns
    def find_engine(path_arg: str | None) -> str:
        if path_arg:
            p = Path(path_arg)
            if p.exists():
                return str(p)
            raise SystemExit(f"Engine not found at {path_arg}")
        for name in ("stockfish", "stockfish_15", "stockfish_14"):
            p = shutil.which(name)
            if p:
                return p
        raise SystemExit("Stockfish binary not found; pass engine path or install stockfish on PATH")

    def cp_from_score(score, board):
        try:
            sc = score.white()
            val = sc.score(mate_score=100000)
            return int(val) if val is not None else None
        except Exception:
            try:
                val = score.score(mate_score=100000)
                return int(val) if val is not None else None
            except Exception:
                return None
    """
    def ensure_search_stats_columns(conn, N=3):
        cur = conn.execute("PRAGMA table_info('search_stats')").fetchall()
        cols = {r[1] for r in cur}
        if 'sf_eval' not in cols:
            conn.execute("ALTER TABLE search_stats ADD COLUMN sf_eval INTEGER")
        if 'sf_best_move' not in cols:
            conn.execute("ALTER TABLE search_stats ADD COLUMN sf_best_move TEXT")
        if 'sf_time_ms' not in cols:
            conn.execute("ALTER TABLE search_stats ADD COLUMN sf_time_ms DOUBLE")
        if 'sf_computed_at' not in cols:
            conn.execute("ALTER TABLE search_stats ADD COLUMN sf_computed_at TIMESTAMP")
        if 'sf_pv' not in cols:
            conn.execute("ALTER TABLE search_stats ADD COLUMN sf_pv TEXT")
        # per-ply `sf_pv_#` columns deprecated: we only store full PV in `sf_pv` JSON
    """
    def update_search_stats_with_stockfish(conn, engine_path=None, depth=12, mpv=3, limit=None, skip_existing=True):
        engine_path = find_engine(engine_path)
        #ensure_search_stats_columns(conn, mpv)
        q = "SELECT id, fen FROM search_stats WHERE fen IS NOT NULL"
        if skip_existing:
            q += " AND sf_eval IS NULL"
        if limit and limit > 0:
            q += f" LIMIT {limit}"
        rows = conn.execute(q).fetchall()

        import chess
        import chess.engine

        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        # request MultiPV from engine so `info` contains multiple principal variations 
        #   which # of stockfish's favorite moves did we play?
        N = int(mpv or 1)
        try:
            engine.configure({"MultiPV": N})
        except Exception:
            # some engine builds may not expose configure or MultiPV; continue
            pass

        updates = []
        cnt = 0
        print(f"  ...processing {len(rows)} search_stats rows with Stockfish depth={depth}, MultiPV={N}...")
        try:
            for sid, fen in rows:
                if (cnt > 0 and ((len(rows) < 10000 and cnt % 1000 == 0) or (len(rows) >= 10000 and cnt % 10000 == 0))): print(f"  ...processed {cnt}/{len(rows)} search_stats rows")

                board = chess.Board(fen)
                t0 = time.time()
                
                try:
                    info = engine.analyse(board, chess.engine.Limit(depth=depth))
                    t1 = time.time()
                    sf_time_ms = (t1 - t0) * 1000.0
                    sf_eval = cp_from_score(info.get("score"), board)
                    # extract PV moves (if present) and best move
                    pv = info.get("pv") or []
                    pv_uci = [m.uci() for m in pv]
                    mv = engine.play(board, chess.engine.Limit(depth=depth))
                    sf_best = mv.move.uci() if mv and mv.move else (pv_uci[0] if pv_uci else None)
                    # we store the full PV as JSON in `sf_pv`; per-ply columns are deprecated
                    pv_cols = []
                except Exception as e:
                    print(f"[WARN] Engine failed for search_id={sid}: {e}")
                    sf_time_ms = None
                    sf_eval = None
                    sf_best = None
                    pv_cols = [None] * N

                import json
                # build tuple: sf_eval, sf_best, sf_time_ms, sf_computed_at, sf_pv(JSON), id
                rowvals = [sf_eval, sf_best, sf_time_ms, datetime.datetime.now(datetime.UTC), json.dumps(pv_uci), sid]
                updates.append(tuple(rowvals))
                cnt += 1
        finally:
            engine.quit()

        # Apply updates in bulk (sf_eval, sf_best_move, sf_time_ms, sf_computed_at, sf_pv_1, sf_pv_2, sf_pv_3, sf_pv)
        if updates:
            sql = "UPDATE search_stats SET sf_eval = ?, sf_best_move = ?, sf_time_ms = ?, sf_computed_at = ?, sf_pv = ? WHERE id = ?"
            conn.executemany(sql, updates)
            print(f"Updated {len(updates)} search_stats rows with Stockfish ground truth (sf_pv JSON)")
        else:
            print("No updates performed")

    try:
        update_search_stats_with_stockfish(
            cnxn, 
            engine_path=cwd_dir.parent.parent / "engines" / "stockfish" / "stockfish-windows-x86-64-avx2.exe", 
            depth=8, 
            mpv=1, 
            limit=None, 
            skip_existing=True
        )
    except Exception as e:
        print(f"[WARN] Ground-truth computation failed or skipped: {e}")

    build_position_features(cnxn)

    cnxn.close()