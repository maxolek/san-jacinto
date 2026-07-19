"""convert a UCI position string to PGN and final FEN"""

import chess 
import chess.pgn

def parse_uci_position(uci_string: str) -> chess.Board:
    parts = uci_string.strip().split()

    if parts[0] == "position":
        parts = parts[1:]
    if parts[0] == "startpos":
        board = chess.Board()
        parts = parts[1:]
    elif parts[0] == "fen":
        fen_parts = []
        parts = parts[1:]
        while parts and parts[0] != "moves":
            fen_parts.append(parts.pop(0))
        board = chess.Board(" ".join(fen_parts))
    else:
        board = chess.Board()

    if parts and parts[0] == "moves":
        for uci_move in parts[1:]:
            board.push_uci(uci_move)

    return board

def board_to_pgn(board: chess.Board) -> str:
    game = chess.pgn.Game()
    node = game
    moves = list(board.move_stack)
    board.reset()

    for move in moves:
        node = node.add_variation(move)
        board.push(move)

    game.headers["Result"] = board.result() if board.is_game_over() else "*"
    return str(game)

if __name__ == "__main__":
    uci_input = input("UCI position string: ").strip()
    board = parse_uci_position(uci_input)

    print("\n--- Final FEN ---")
    print(board.fen())
    print("\n--- PGN ---")
    print(board_to_pgn(board))