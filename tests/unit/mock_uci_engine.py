#!/usr/bin/env python3
"""A tiny UCI engine for testing the match tools without a real engine.

Plays legal moves with python-chess: mates when it can, captures the most
valuable piece it can, otherwise a random move. The `Strength` option (0-100)
is the probability of using those rules instead of a random move, so two
instances with different strengths give a measurable Elo difference.

It declares the `Affinity` option like Tomahawk. If MOCK_ENGINE_LOG names a
directory, every `go` appends one JSON line to go_<pid>.jsonl there with the
process id, the Affinity value it received, the CPUs its process may use and
the CPU it is running on, so tests can verify pinning. With `uci_logging` true
and a `log_dir`, it writes Tomahawk-style game_<pid>.jsonl / search_<pid>.jsonl
records (instance_id = process id).

To test the runner's core guard it can misbehave like an old build:
MOCK_ENGINE_ESCAPE_STARTUP=0-7 moves the process to those CPUs at start-up
(like pin_to_pcores()), and the option `Escape` = midgame moves it to every CPU
on its sixth search.
"""
import ctypes
import json
import os
import random
import sys
import time
import uuid

import chess

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def current_cpu():
    if sys.platform == "win32":
        return ctypes.windll.kernel32.GetCurrentProcessorNumber()
    try:
        return os.sched_getcpu()
    except AttributeError:
        return None


def allowed():
    try:
        import psutil
        return sorted(psutil.Process().cpu_affinity())
    except Exception:
        return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def move_self(cpus):
    try:
        import psutil
        psutil.Process().cpu_affinity(list(cpus))
    except Exception:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cpus))


def cpu_list(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


class Engine:
    def __init__(self):
        self.board = chess.Board()
        self.options = {"Affinity": "auto", "Strength": "50", "Threads": "1", "Hash": "16",
                        "uci_logging": "false", "log_dir": "", "Version": "mock-1", "Escape": "none"}
        self.game_uuid = None
        self.side = None
        self.searches = 0
        self.rng = random.Random(os.getpid() ^ int(time.time() * 1000))
        if os.environ.get("MOCK_ENGINE_ESCAPE_STARTUP"):
            move_self(cpu_list(os.environ["MOCK_ENGINE_ESCAPE_STARTUP"]))

    def send(self, text):
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    def log_path(self, stem):
        directory = self.options["log_dir"]
        if self.options["uci_logging"].lower() != "true" or not directory:
            return None
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{stem}_{os.getpid()}.jsonl")

    def write(self, stem, record):
        path = self.log_path(stem)
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def finish_game(self):
        if self.game_uuid:
            outcome = self.board.outcome(claim_draw=True)
            self.write("game", dict(game_uuid=self.game_uuid, instance_id=os.getpid(),
                                    engine_id=self.options["Version"], side=self.side,
                                    result=outcome.result() if outcome else "*",
                                    reason=outcome.termination.name.lower() if outcome else None,
                                    start_fen=chess.STARTING_FEN,
                                    moves=[m.uci() for m in self.board.move_stack]))
        self.game_uuid, self.side = None, None

    def choose(self):
        moves = list(self.board.legal_moves)
        if self.rng.random() * 100 < float(self.options["Strength"]):
            for move in moves:
                self.board.push(move)
                mate = self.board.is_checkmate()
                self.board.pop()
                if mate:
                    return move
            captures = [m for m in moves if self.board.is_capture(m)]
            if captures:
                return max(captures, key=lambda m: VALUES.get(
                    (self.board.piece_at(m.to_square) or chess.Piece(chess.PAWN, True)).piece_type, 0))
        return self.rng.choice(moves)

    def go(self):
        started = time.time()
        self.searches += 1
        if self.options["Escape"] == "midgame" and self.searches == 6:
            move_self(range(os.cpu_count() or 1))
        if os.environ.get("MOCK_ENGINE_LOG"):
            os.makedirs(os.environ["MOCK_ENGINE_LOG"], exist_ok=True)
            path = os.path.join(os.environ["MOCK_ENGINE_LOG"], f"go_{os.getpid()}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(pid=os.getpid(), affinity_option=self.options["Affinity"],
                                        allowed=allowed(), cpu=current_cpu())) + "\n")
        if self.game_uuid is None:
            self.game_uuid = str(uuid.uuid4())
            self.side = "white" if self.board.turn == chess.WHITE else "black"
        move = self.choose()
        busy = float(os.environ.get("MOCK_ENGINE_BUSY_MS", "5")) / 1000
        while time.time() - started < busy:
            pass
        elapsed = max(1, int((time.time() - started) * 1000))
        stm_eval = self.material()
        search_uuid = str(uuid.uuid4())
        record = self.telemetry(search_uuid, elapsed, stm_eval, move)
        self.write("search", record)
        self.write("timing", {"type": "timing", "search_uuid": search_uuid, "game_uuid": self.game_uuid,
                              "instance_id": os.getpid(), "engine_id": self.options["Version"],
                              "total_search_time_ms": elapsed,
                              "ROOT": {"total_ms": elapsed, "calls": 1},
                              "evaluate": {"total_ms": elapsed * 0.4, "calls": record["nodes"]},
                              "movegen": {"total_ms": elapsed * 0.2, "calls": record["nodes"] // 2}})
        self.write("root_moves", {"search_uuid": search_uuid, "depth": record["max_depth"],
                                  "moves": [{"move": m.uci(), "eval": stm_eval, "time_ms": 0, "nodes": 10}
                                            for m in list(self.board.legal_moves)[:6]]})
        self.send(f"info depth {record['max_depth']} score cp {stm_eval} nodes {record['nodes']} "
                  f"time {elapsed} pv {move.uci()}")
        self.send(f"bestmove {move.uci()}")

    def material(self):
        """Material balance in centipawns from the side to move."""
        values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900}
        score = 0
        for piece_type, value in values.items():
            score += value * (len(self.board.pieces(piece_type, self.board.turn))
                              - len(self.board.pieces(piece_type, not self.board.turn)))
        return score

    def telemetry(self, search_uuid, elapsed, stm_eval, move):
        """A search record with the fields Tomahawk logs (values are synthetic)."""
        depth = 3 + len(self.board.move_stack) % 5
        nodes = [int(40 * 3 ** d * (0.5 + self.rng.random())) for d in range(1, depth + 1)]
        qnodes = [int(x * (0.3 + 0.9 * self.rng.random())) for x in nodes]
        times = [elapsed * x / sum(nodes) for x in nodes]
        fh = [[int(x * 0.02 * p) for p in (0.8, 0.08, 0.04, 0.03, 0.03, 0.02)] for x in nodes]
        total = sum(nodes)
        return dict(
            search_uuid=search_uuid, game_uuid=self.game_uuid, instance_id=os.getpid(),
            engine_id=self.options["Version"], fen=self.board.fen(), ply=len(self.board.move_stack) + 1,
            time_ms=elapsed, root_eval=stm_eval, completed_depth=depth - 1, max_depth=depth,
            max_qdepth=depth + 6, best_move=move.uci(), principal_variation=[move.uci()],
            nodes=total, qnodes=sum(qnodes), tt_stores=int(total * 0.6), tt_hits=int(total * 0.3),
            tt_fill=self.rng.random(), fail_highs=int(total * 0.02), fail_lows=int(total * 0.01),
            aspiration_fail_high_researches=0, aspiration_fail_low_researches=0,
            fail_high_index=[sum(f[i] for f in fh) for i in range(6)],
            see_prunes=int(sum(qnodes) * 0.05), delta_prunes=int(sum(qnodes) * 0.03),
            nmp=int(total * 0.01), nmp_failhigh=int(total * 0.006), tt_overwritten=int(total * 0.01),
            itdepth_nodes=nodes, itdepth_qnodes=qnodes, itdepth_time_ms=times, itdepth_eval=[stm_eval] * depth,
            itdepth_move=[move.uci()] * depth, itdepth_qdepth=[d + 6 for d in range(depth)],
            itdepth_ttstores=[int(x * 0.6) for x in nodes], itdepth_tthits=[int(x * 0.3) for x in nodes],
            itdepth_ttfill=[self.rng.random() for _ in nodes], itdepth_fail_highs=[int(x * 0.02) for x in nodes],
            itdepth_fail_lows=[int(x * 0.01) for x in nodes], itdepth_aspiration_failhigh_researches=[0] * depth,
            itdepth_aspiration_faillow_researches=[0] * depth, itdepth_fh_index=fh,
            itdepth_see_prunes=[int(x * 0.05) for x in qnodes], itdepth_delta_prunes=[int(x * 0.03) for x in qnodes],
            itdepth_pvs_researches=[int(x * 0.001) for x in nodes], itdepth_nmp=[int(x * 0.01) for x in nodes],
            itdepth_nmp_failhigh=[int(x * 0.006) for x in nodes],
            treedepth_nodes=nodes[::-1], treedepth_qnodes=qnodes[::-1], treedepth_tt_stores=[1] * depth,
            treedepth_tt_hits=[1] * depth, treedepth_fail_highs=[1] * depth, treedepth_fail_lows=[1] * depth,
            treedepth_fh_index=fh[::-1], treedepth_see_prunes=[1] * depth, treedepth_delta_prunes=[1] * depth,
            treedepth_pvs_researches=[1] * depth, treedepth_nmp=[1] * depth, treedepth_nmp_failhigh=[1] * depth,
        )

    def run(self):
        for line in sys.stdin:
            tokens = line.split()
            if not tokens:
                continue
            cmd = tokens[0]
            if cmd == "uci":
                self.send("id name MockEngine")
                self.send(f"id version {self.options['Version']}")
                self.send("id author san-jacinto tests")
                self.send("option name Affinity type string default auto")
                self.send("option name Strength type spin default 50 min 0 max 100")
                self.send("option name Threads type spin default 1 min 1 max 64")
                self.send("option name Hash type spin default 16 min 1 max 1024")
                self.send("option name uci_logging type check default false")
                self.send("option name log_dir type string default <empty>")
                self.send("option name Version type string default mock-1")
                self.send("option name Escape type string default none")
                self.send("uciok")
            elif cmd == "isready":
                self.send("readyok")
            elif cmd == "setoption" and " name " in line:
                rest = line.strip().split(" name ", 1)[1]
                name, _, value = rest.partition(" value ")
                self.options[name.strip()] = value.strip()
            elif cmd == "ucinewgame":
                self.finish_game()
                self.board = chess.Board()
            elif cmd == "position":
                if "startpos" in tokens:
                    self.board = chess.Board()
                elif "fen" in tokens:
                    end = tokens.index("moves") if "moves" in tokens else len(tokens)
                    self.board = chess.Board(" ".join(tokens[tokens.index("fen") + 1:end]))
                if "moves" in tokens:
                    for mv in tokens[tokens.index("moves") + 1:]:
                        self.board.push_uci(mv)
            elif cmd == "go":
                self.go()
            elif cmd == "quit":
                self.finish_game()
                break


if __name__ == "__main__":
    Engine().run()
