"""Opening/ECO classification from move lists.

Approach: build a trie from the ECO database once at import time, keyed
by UCI move sequence. For a given game, walk its moves one at a time
down the trie and remember the deepest node that corresponds to a
named opening. That gives the most specific opening reached — e.g.
c4 Nf6 = English, but c4 Nf6 d4 e6 g3 d5 = Catalan (a deeper, more
specific match under the same opening prefix).
"""
import json
from pathlib import Path
from functools import lru_cache
import chess

_ECO_FILE = Path(__file__).resolve().parent.parent / "openings" / "eco_db.json"


def _build_eco_trie():
    """
    Build a trie: each node is a dict of {move: child_node}, plus an
    optional '_entry' key holding (eco, name) if an opening ends here.
    Built once per process.
    """
    root = {}
    if not _ECO_FILE.exists():
        return root

    with open(_ECO_FILE, 'r') as f:
        eco_list = json.load(f)

    for entry in eco_list:
        moves_seq = entry.get("moves") or []
        if not moves_seq:
            continue
        eco = entry.get("eco", "")
        name = entry.get("name", "")

        node = root
        for mv in moves_seq:
            node = node.setdefault(mv, {})
        # Deeper/duplicate entries at the same exact position: last one
        # wins on load (or you could prefer the first — doesn't matter
        # much since exact-duplicate move sequences are rare).
        node["_entry"] = (eco, name)

    return root


_ECO_TRIE = _build_eco_trie()


def _normalize_tokens(moves):
    if moves is None:
        return []
    if isinstance(moves, str):
        s = moves.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return s.split()
    try:
        return list(moves)
    except Exception:
        return []


def _to_uci_moves(tokens):
    """Convert SAN or UCI tokens into a clean list of UCI moves,
    stopping at the first illegal/unparsable move."""
    board = chess.Board()
    uci_moves = []
    for tok in tokens:
        try:
            mv = chess.Move.from_uci(tok)
            if mv in board.legal_moves:
                board.push(mv)
                uci_moves.append(mv.uci())
                continue
        except Exception:
            pass
        try:
            mv = board.parse_san(tok)
            board.push(mv)
            uci_moves.append(mv.uci())
            continue
        except Exception:
            break
    return uci_moves


@lru_cache(maxsize=200_000)
def get_opening_from_moves(moves):
    """
    Returns (eco, name) for the deepest/most specific opening position
    reached by this game's move list. Single linear walk down the trie —
    O(number of moves), independent of ECO database size.
    """
    tokens = _normalize_tokens(moves)
    uci_moves = _to_uci_moves(tuple(tokens) if not isinstance(tokens, tuple) else tokens)

    node = _ECO_TRIE
    last_match = ("", "")

    for mv in uci_moves:
        if mv not in node:
            break
        node = node[mv]
        if "_entry" in node:
            last_match = node["_entry"]

    return last_match


def get_opening_name(moves):
    return get_opening_from_moves(moves)[1]


def get_opening_code(moves):
    return get_opening_from_moves(moves)[0]