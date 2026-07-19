import argparse
import json
from pathlib import Path
import chess.pgn


def build(pgn_path: Path, out_path: Path):
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN not found: {pgn_path}")
    out = []
    with open(pgn_path, 'r', encoding='utf-8') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            headers = game.headers
            eco = headers.get('ECO') or headers.get('Eco') or ''
            name = headers.get('Opening') or headers.get('Event') or headers.get('Site') or ''
            # collect moves as UCI
            b = game.board()
            moves = []
            for mv in game.mainline_moves():
                moves.append(mv.uci())
                b.push(mv)
            out.append({"eco": eco, "name": name, "moves": moves})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("Wrote:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn', type=Path, help='Path to ECO PGN file. If omitted, looks for eco.pgn next to this script.')
    parser.add_argument('--out', type=Path, help='Output JSON file. Defaults to eco_db.json next to this script.')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    pgn = args.pgn or (script_dir / 'eco.pgn')
    out = args.out or (script_dir / 'eco_db.json')

    build(pgn, out)


if __name__ == '__main__':
    main()

