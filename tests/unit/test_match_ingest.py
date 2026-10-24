"""Match-runner game ingestion: every played game becomes a `games` row with its
environment, and engine telemetry is linked to the right game by position."""
import json
import random
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import chess
    import chess.pgn
except ImportError:  # pragma: no cover
    chess = None

GAMES_COLUMNS = """
    id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id INTEGER, white_engine_id INTEGER,
    black_engine_id INTEGER, wtime INTEGER, btime INTEGER, winc INTEGER, binc INTEGER,
    movestogo INTEGER, depth INTEGER, nodes INTEGER, movetime INTEGER, result TEXT,
    termination TEXT, opening TEXT, start_fen TEXT, moves TEXT, run_time_s REAL"""
ENV_COLUMNS = """, source TEXT, run_id TEXT, run_game INTEGER, pair_index INTEGER, env_host TEXT,
    env_cpu_model TEXT, env_os TEXT, env_cpu INTEGER, env_core_type TEXT, env_cluster INTEGER,
    env_concurrency INTEGER, env_mhz_mean REAL, env_mhz_min REAL, env_mhz_max REAL,
    env_util_mean REAL, env_power_source TEXT, env_core_mixed INTEGER"""


def play(rng, plies=30):
    """A random game from a shared 4-ply opening, so both games of a pair start alike."""
    board = chess.Board()
    for move in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(move)
    while len(board.move_stack) < plies and not board.is_game_over():
        board.push(rng.choice(list(board.legal_moves)))
    return board


@unittest.skipIf(chess is None, "python-chess is not installed")
class MatchIngest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.run_dir, self.logroot = base / "run_x", base / "logs"
        self.run_dir.mkdir()
        self.logroot.mkdir()
        rng = random.Random(3)
        records, pgn = [], []
        searches = {101: [], 102: []}  # pid -> search records (candidate, baseline processes)
        for number in (1, 2, 3, 4):
            board = play(rng)
            pair = (number + 1) // 2
            white = "Candidate" if number % 2 else "Baseline"
            black = "Baseline" if number % 2 else "Candidate"
            records.append(dict(run="run_x", game=number, pair=pair, job=1, job_game=number, white=white,
                                black=black, result="1/2-1/2", termination="move limit", seconds=1.0,
                                opening="Ruy Lopez", variation="", cpu=9, core_type="E", cluster=4,
                                concurrency=2, mhz_mean=2600, mhz_min=2500, mhz_max=2700, util_mean=99.0,
                                power_source="ac", core_mixed=False, engine_pids=[101, 102]))
            game = chess.pgn.Game.from_board(board)
            game.headers["Job"], game.headers["JobGame"] = "1", str(number)
            pgn.append(str(game))
            replay = chess.Board()
            for ply, move in enumerate(board.move_stack):
                if ply >= 4:  # engines only search after the book
                    pid = 101 if (replay.turn == chess.WHITE) == (white == "Candidate") else 102
                    searches[pid].append(dict(game_uuid=f"{pid}-g{number}", instance_id=pid,
                                              engine_id="cand" if pid == 101 else "base",
                                              fen=replay.fen(), ply=ply))
                replay.push(move)
        # A game the runner killed (not in games.pgn): it shares only the opening's first
        # searched position with the finished games, then goes its own way.
        killed = play(random.Random(99), plies=16)
        replay = chess.Board()
        for ply, move in enumerate(killed.move_stack):
            if ply >= 4 and replay.turn == chess.WHITE:
                searches[101].append(dict(game_uuid="101-killed", instance_id=101, engine_id="cand",
                                          fen=replay.fen(), ply=ply))
            replay.push(move)
        (self.run_dir / "games.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        (self.run_dir / "games.pgn").write_text("\n\n".join(pgn) + "\n")
        (self.run_dir / "config.json").write_text(json.dumps(dict(meta=dict(
            run="run_x", time_control="10+0.1", machine=dict(host="h", cpu_model="cpu", os="os")))))
        for pid, rows in searches.items():
            (self.logroot / f"search_{pid}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        self.searches = searches

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_engine_game_links_to_its_played_game(self):
        from data.etl.match_games import link_engine_games
        expected = {f"{pid}-g{n}": n for pid in (101, 102) for n in (1, 2, 3, 4)}
        self.assertEqual(link_engine_games(self.logroot, self.run_dir), expected)
        names = {"cand": "Candidate", "base": "Baseline"}
        self.assertEqual(link_engine_games(self.logroot, self.run_dir, names_by_version=names), expected)

    def test_killed_games_stay_unlinked(self):
        from data.etl.match_games import link_engine_games
        self.assertNotIn("101-killed", link_engine_games(self.logroot, self.run_dir))

    def test_games_rows_carry_result_moves_and_environment(self):
        from data.etl.ingest import bulk_log_match_games
        cnxn = sqlite3.connect(":memory:")
        cnxn.execute(f"CREATE TABLE games ({GAMES_COLUMNS}{ENV_COLUMNS})")
        mapping = bulk_log_match_games(cnxn, self.run_dir, 7, {"Candidate": 1, "Baseline": 2})
        self.assertEqual(sorted(mapping), [("run_x", n) for n in (1, 2, 3, 4)])
        row = cnxn.execute("SELECT experiment_id, white_engine_id, black_engine_id, result, termination, "
                           "wtime, winc, env_core_type, env_cpu, env_mhz_mean, env_host, source, pair_index, "
                           "moves FROM games WHERE run_game = 2").fetchone()
        self.assertEqual(row[:13], (7, 2, 1, "3", "7", 10000, 100, "E", 9, 2600.0, "h", "match", 1))
        self.assertGreater(len(json.loads(row[13])), 4)
        cnxn.close()

    def test_old_schema_still_ingests_games(self):
        from data.etl.ingest import bulk_log_match_games
        cnxn = sqlite3.connect(":memory:")
        cnxn.execute(f"CREATE TABLE games ({GAMES_COLUMNS})")
        mapping = bulk_log_match_games(cnxn, self.run_dir, 7, {"Candidate": 1, "Baseline": 2})
        self.assertEqual(len(mapping), 4)
        self.assertEqual(cnxn.execute("SELECT COUNT(*) FROM games").fetchone()[0], 4)
        cnxn.close()


if __name__ == "__main__":
    unittest.main()
