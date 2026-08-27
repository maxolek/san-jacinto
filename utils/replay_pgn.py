
#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path

import chess
import chess.pgn


CLOCK_RE = re.compile(
    r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]"
)


def parse_clock(comment: str) -> int | None:
    """
    Parse a Lichess clock annotation.

    [%clk ...] is the player's remaining time AFTER the move.

    Returns milliseconds.
    """
    match = CLOCK_RE.search(comment or "")

    if not match:
        return None

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))

    return int(
        (hours * 3600 + minutes * 60 + seconds) * 1000
    )


def parse_time_control(time_control: str):
    """
    Parse a time control such as:

        30+2

    Returns:
        initial_time_ms, increment_ms
    """

    if not time_control:
        raise ValueError(
            "PGN has no TimeControl header."
        )

    if "+" not in time_control:
        raise ValueError(
            f"Unsupported TimeControl: {time_control!r}. "
            "Expected something like '30+2'."
        )

    base, increment = time_control.split("+", 1)

    try:
        initial_time_ms = int(float(base) * 1000)
        increment_ms = int(float(increment) * 1000)
    except ValueError:
        raise ValueError(
            f"Could not parse TimeControl: {time_control!r}"
        )

    return initial_time_ms, increment_ms


def format_clock(ms: int) -> str:
    """Format milliseconds for display."""

    if ms < 60000:
        return f"{ms / 1000:.2f}s"

    minutes = ms // 60000
    seconds = (ms % 60000) / 1000

    return f"{minutes}:{seconds:05.2f}"


def wait_for(engine, expected: str):
    """
    Read engine output until a line starts with `expected`.
    """

    while True:
        line = engine.stdout.readline()

        if not line:
            raise RuntimeError(
                "Engine terminated unexpectedly."
            )

        line = line.strip()

        if line.startswith(expected):
            return line


def uci_handshake(engine):
    """Perform the UCI handshake."""

    engine.stdin.write("uci\n")
    engine.stdin.flush()

    wait_for(engine, "uciok")

    engine.stdin.write("isready\n")
    engine.stdin.flush()

    wait_for(engine, "readyok")


def send_position(engine, moves):
    """
    Set the engine to the exact historical position.

    The complete move list is deliberately supplied so that
    repetition detection and other history-dependent logic
    sees the same game history.
    """

    command = "position startpos"

    if moves:
        command += " moves " + " ".join(moves)

    engine.stdin.write(command + "\n")


def parse_eval_line(line: str):
    """
    Parse:

        eval 57 bestmove e2e4

    Returns:

        eval_cp
        bestmove
    """

    parts = line.split()

    if len(parts) < 2:
        return None, None

    try:
        eval_cp = int(parts[1])
    except ValueError:
        eval_cp = None

    bestmove = None

    if "bestmove" in parts:
        index = parts.index("bestmove")

        if index + 1 < len(parts):
            bestmove = parts[index + 1]

    return eval_cp, bestmove


def get_search_result(
    engine,
    moves,
    wtime,
    btime,
    winc,
    binc,
):
    """
    Run the exact time-managed search used for replay.

    The engine is given:

        go eval wtime ... btime ... winc ... binc ...

    `eval` is an extension to the normal search command. It
    returns the final search evaluation in addition to the
    normal bestmove.

    After bestmove, dumpstats is requested to obtain the
    completed search's depth and NPS.
    """

    # ------------------------------------------------------------
    # Set exact historical position.
    # ------------------------------------------------------------

    send_position(engine, moves)

    # ------------------------------------------------------------
    # Run time-managed search.
    # ------------------------------------------------------------

    engine.stdin.write(
        f"go eval "
        f"wtime {wtime} "
        f"btime {btime} "
        f"winc {winc} "
        f"binc {binc}\n"
    )

    engine.stdin.flush()

    eval_cp = None
    bestmove = None

    # ------------------------------------------------------------
    # Wait for the engine's final eval/bestmove line.
    # ------------------------------------------------------------

    while True:

        line = engine.stdout.readline()

        if not line:
            raise RuntimeError(
                "Engine terminated during search."
            )

        line = line.strip()

        if line.startswith("eval"):

            eval_cp, bestmove = parse_eval_line(line)

            break

    # ------------------------------------------------------------
    # Request statistics from the completed search.
    # ------------------------------------------------------------

    engine.stdin.write("dumpstats\n")
    engine.stdin.flush()

    depth = None
    nps = None

    in_search_stats = False

    while True:

        line = engine.stdout.readline()

        if not line:
            raise RuntimeError(
                "Engine terminated during dumpstats."
            )

        line = line.strip()

        # The stats output ends with a line of '=' characters.
        if (
            in_search_stats
            and line.startswith("=")
        ):
            break

        if line == "SEARCH STATS":
            in_search_stats = True
            continue

        if not in_search_stats:
            continue

        # --------------------------------------------------------
        # Depth
        #
        # Example:
        #
        # Depth                           11
        # Completed Depth                 11
        # --------------------------------------------------------

        match = re.match(
            r"Depth\s+(\d+)",
            line,
        )

        if match:
            depth = int(match.group(1))
            continue

        # --------------------------------------------------------
        # NPS
        #
        # Example:
        #
        # NPS                         858189
        # --------------------------------------------------------

        match = re.match(
            r"NPS\s+(\d+)",
            line,
        )

        if match:
            nps = int(match.group(1))
            continue

    return {
        "eval_cp": eval_cp,
        "bestmove": bestmove,
        "depth": depth,
        "nps": nps,
    }


def determine_engine_color(game):
    """
    Determine Tomahawk's side from the PGN.

    We look for 'tomahawk' in exactly one player name.
    """

    white = game.headers.get("White", "")
    black = game.headers.get("Black", "")

    white_is_engine = "tomahawk" in white.lower()
    black_is_engine = "tomahawk" in black.lower()

    if white_is_engine and not black_is_engine:
        return chess.WHITE

    if black_is_engine and not white_is_engine:
        return chess.BLACK

    raise ValueError(
        f"Could not uniquely identify Tomahawk.\n"
        f"White: {white}\n"
        f"Black: {black}\n"
        f"Expected 'tomahawk' in exactly one player name."
    )


def move_label(board):
    """Return PGN-style move number."""

    if board.turn == chess.WHITE:
        return f"{board.fullmove_number}."

    return f"{board.fullmove_number}..."


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Replay a PGN and search only Tomahawk's moves."
        )
    )

    parser.add_argument(
        "--pgn",
        required=True,
        type=Path,
        help="PGN file",
    )

    parser.add_argument(
        "--engine",
        required=True,
        type=Path,
        help="UCI engine executable",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Print every Tomahawk position.",
    )

    parser.add_argument(
        "--fen",
        action="store_true",
        help="Print FEN for differing positions.",
    )

    args = parser.parse_args()

    # ============================================================
    # READ PGN
    # ============================================================

    with args.pgn.open(
        "r",
        encoding="utf-8",
    ) as f:

        game = chess.pgn.read_game(f)

    if game is None:
        print(
            "ERROR: Could not read PGN.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ============================================================
    # DETERMINE ENGINE SIDE
    # ============================================================

    try:
        engine_color = determine_engine_color(game)

    except ValueError as e:

        print(
            f"ERROR: {e}",
            file=sys.stderr,
        )

        sys.exit(1)

    engine_side_name = (
        "White"
        if engine_color == chess.WHITE
        else "Black"
    )

    # ============================================================
    # TIME CONTROL
    # ============================================================

    time_control = game.headers.get("TimeControl")

    try:

        initial_time, increment = parse_time_control(
            time_control
        )

    except ValueError as e:

        print(
            f"ERROR: {e}",
            file=sys.stderr,
        )

        sys.exit(1)

    # ============================================================
    # SUMMARY
    # ============================================================

    print(
        f"Game: "
        f"{game.headers.get('White', '?')} - "
        f"{game.headers.get('Black', '?')}"
    )

    print(
        f"Tomahawk side: {engine_side_name}"
    )

    print(
        f"TimeControl: {time_control} "
        f"(initial={format_clock(initial_time)}, "
        f"increment={increment / 1000:g}s)"
    )

    print(
        f"Engine: {args.engine}"
    )

    print()

    # ============================================================
    # START ENGINE
    # ============================================================

    engine = subprocess.Popen(
        [str(args.engine)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    try:

        uci_handshake(engine)

        engine.stdin.write("ucinewgame\n")
        engine.stdin.write("isready\n")
        engine.stdin.flush()

        wait_for(engine, "readyok")

        # ========================================================
        # INITIAL CLOCK STATE
        # ========================================================

        white_clock = initial_time
        black_clock = initial_time

        # Complete ACTUAL game history in UCI format.
        moves = []

        board = game.board()

        positions_searched = 0
        differences = 0

        # ========================================================
        # OUTPUT HEADER
        # ========================================================

        print(
            f"{'MOVE':<8}"
            f"{'GAME':<8}"
            f"{'NEW':<10}"
            f"{'CLOCK':<10}"
            f"{'EVAL':<10}"
            f"{'DEPTH':<8}"
            f"{'NPS':<10}"
        )

        print("-" * 90)

        # ========================================================
        # REPLAY GAME
        # ========================================================

        for node in game.mainline():

            move = node.move

            current_color = board.turn

            label = move_label(board)

            # UCI representation of the actual game move.
            game_move = board.uci(move)

            # ----------------------------------------------------
            # ONLY SEARCH TOMahawk'S MOVES.
            # ----------------------------------------------------

            if current_color == engine_color:

                # Clocks BEFORE the move.
                search_wtime = white_clock
                search_btime = black_clock

                fen_before = board.fen()

                # ------------------------------------------------
                # Run the new engine's exact search.
                # ------------------------------------------------

                result = get_search_result(
                    engine=engine,
                    moves=moves,
                    wtime=search_wtime,
                    btime=search_btime,
                    winc=increment,
                    binc=increment,
                )

                engine_move = result["bestmove"]

                different = (
                    engine_move is not None
                    and engine_move != game_move
                )

                positions_searched += 1

                if different:
                    differences += 1

                # ------------------------------------------------
                # Display.
                # ------------------------------------------------

                if args.all or different:

                    if result["eval_cp"] is None:

                        eval_string = "?"

                    else:

                        eval_string = (
                            f"{result['eval_cp'] / 100:+.2f}"
                        )

                    engine_clock = (
                        search_wtime
                        if engine_color == chess.WHITE
                        else search_btime
                    )

                    print(
                        f"{label:<8}"
                        f"{game_move:<8}"
                        f"{str(engine_move):<10}"
                        f"{format_clock(engine_clock):<10}"
                        f"{eval_string:<10}"
                        f"{str(result['depth'] or '?'):<8}"
                        f"{str(result['nps'] or '?'):<10}"
                    )

                    if different:

                        print(
                            f"    DIFFERENCE: "
                            f"game={game_move}, "
                            f"engine={engine_move}"
                        )

                        if args.fen:

                            print(
                                f"    FEN: {fen_before}"
                            )

            # ----------------------------------------------------
            # ALWAYS ADVANCE WITH THE ACTUAL GAME MOVE.
            # ----------------------------------------------------

            board.push(move)
            moves.append(game_move)

            # ----------------------------------------------------
            # UPDATE POST-MOVE CLOCK.
            #
            # [%clk] is the clock AFTER the move.
            # ----------------------------------------------------

            post_move_clock = parse_clock(
                node.comment
            )

            if post_move_clock is None:

                print(
                    f"WARNING: No clock found for "
                    f"{label} {game_move}",
                    file=sys.stderr,
                )

            elif current_color == chess.WHITE:

                white_clock = post_move_clock

            else:

                black_clock = post_move_clock

        # ========================================================
        # SUMMARY
        # ========================================================

        print()
        print("=" * 70)

        print(
            f"Tomahawk positions searched: {positions_searched}"
        )

        print(
            f"Move-choice differences:      {differences}"
        )

        print("=" * 70)

    finally:

        try:

            engine.stdin.write("quit\n")
            engine.stdin.flush()

        except Exception:
            pass

        try:

            engine.terminate()
            engine.wait(timeout=2)

        except Exception:

            try:
                engine.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
