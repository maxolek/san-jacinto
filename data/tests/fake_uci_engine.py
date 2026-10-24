"""A tiny UCI engine for tests: fixed scores from the side to move, a PV from the legal moves.

Moves are ranked by their UCI text, last first: the move at position p of that order
(p = 0 is the best) scores `cp 57 - 10 * p`, except that the best move scores `mate 2`
when the position's fullmove number is 50 and `mate -1` at 51 (UCI scores are from the
side to move). MultiPV (a declared option) prints that many lines; `go ... searchmoves m`
searches only the given moves and scores them by their place in the full order. Every
received command is appended to the file named by $FAKE_UCI_LOG. Switches for failure
tests: $FAKE_UCI_CRASH_ON_GO = k exits without answering the k-th `go`,
$FAKE_UCI_NO_MULTIPV hides the MultiPV option, $FAKE_UCI_IGNORE_SEARCHMOVES searches
every move even when told to search only some.
"""
import os
import sys

import chess


def score(board, position):
    if position == 0 and board.fullmove_number == 50:
        return "mate 2"
    if position == 0 and board.fullmove_number == 51:
        return "mate -1"
    return f"cp {57 - 10 * position}"


def main():
    log = os.environ.get("FAKE_UCI_LOG")
    crash_on = int(os.environ.get("FAKE_UCI_CRASH_ON_GO", "0"))   # exit silently on this `go` (1-based)
    no_multipv = bool(os.environ.get("FAKE_UCI_NO_MULTIPV"))      # declare no MultiPV option
    ignore_searchmoves = bool(os.environ.get("FAKE_UCI_IGNORE_SEARCHMOVES"))
    searches = 0
    board = chess.Board()
    multipv = 1
    for line in sys.stdin:
        command = line.strip()
        if log:
            with open(log, "a", encoding="utf-8") as f:
                f.write(command + "\n")
        if command == "uci":
            options = "" if no_multipv else "option name MultiPV type spin default 1 min 1 max 500\n"
            print(f"id name FakeFish\nid author san-jacinto tests\n{options}uciok", flush=True)
        elif command == "isready":
            print("readyok", flush=True)
        elif command.startswith("setoption name MultiPV value"):
            multipv = int(command.split()[-1])
        elif command.startswith("position"):
            tokens = command.split()
            if tokens[1] == "startpos":
                board = chess.Board()
                rest = tokens[2:]
            else:
                board = chess.Board(" ".join(tokens[2:8]))
                rest = tokens[8:]
            if rest and rest[0] == "moves":
                for move in rest[1:]:
                    board.push_uci(move)
        elif command.startswith("go"):
            searches += 1
            if searches == crash_on:
                sys.exit(3)
            tokens = command.split()
            order = sorted(board.legal_moves, key=lambda m: m.uci(), reverse=True)
            candidates = order
            if "searchmoves" in tokens and not ignore_searchmoves:
                wanted = set(tokens[tokens.index("searchmoves") + 1:])
                candidates = [m for m in order if m.uci() in wanted]
            for rank, move in enumerate(candidates[:multipv], 1):
                board.push(move)
                replies = sorted(board.legal_moves, key=lambda m: m.uci())
                board.pop()
                pv = " ".join(m.uci() for m in [move] + replies[:1])
                print(f"info depth 1 seldepth 1 multipv {rank} score {score(board, order.index(move))} "
                      f"nodes 20 pv {pv}", flush=True)
            print(f"bestmove {candidates[0].uci() if candidates else '0000'}", flush=True)
        elif command == "quit":
            break


if __name__ == "__main__":
    main()
