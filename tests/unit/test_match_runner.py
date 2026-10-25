"""The fastchess match runner (utils/match_runner.py) and the SPRT built on it (tests/sprt.py).

Fast tests cover slot planning (core types, SMT, multi-cpu slots, the cpu lock files other
runs hold), the fastchess command line, PGN game numbering, termination classes, time
control labels and how engine telemetry joins the played games. The integration tests run
fastchess with tests/unit/mock_uci_engine.py (skipped without fastchess): games.jsonl,
game numbering and pairs, the environment of every game, PGN tags, the database rows, and
the off-core guard.
"""
import contextlib
import gc
import io
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils import cpu_slots, cpu_topology, match_runner  # noqa: E402
from utils.cpu_topology import Core, Topology  # noqa: E402
from utils.match_runner import EngineSpec, MatchRunner, Slot, plan_slots, release_slots, tc_arguments  # noqa: E402

try:
    import chess
    import chess.pgn
except ImportError:  # pragma: no cover
    chess = None
try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

MOCK = ROOT / "tests" / "unit" / "mock_uci_engine.py"
try:
    FASTCHESS = match_runner.find_fastchess()
except FileNotFoundError:
    FASTCHESS = None
PINNING = cpu_topology.topology().supported
BOOK = """[Event "?"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *

[Event "?"]
[Result "*"]

1. d4 d5 2. c4 e6 *

[Event "?"]
[Result "*"]

1. c4 e5 2. Nc3 Nf6 *

[Event "?"]
[Result "*"]

1. e4 c5 2. Nf3 d6 *
"""
# tests/unit/mock_uci_engine.py, but every attempt to change its own affinity (MOCK_ENGINE_ESCAPE_STARTUP,
# option Escape=midgame) is recorded in MOCK_ENGINE_LOG/escape_<pid>.jsonl with the cpus it has after it.
ESCAPING = '''import json, os, sys
sys.path.insert(0, {tests_unit!r})
import psutil
import mock_uci_engine as mock


def move_self(cpus):
    process = psutil.Process()
    try:
        process.cpu_affinity(list(cpus))
        error = None
    except Exception as exc:
        error = repr(exc)
    with open(os.path.join(os.environ["MOCK_ENGINE_LOG"], "escape_%d.jsonl" % os.getpid()), "a") as f:
        f.write(json.dumps(dict(pid=os.getpid(), wanted=sorted(cpus), error=error,
                                allowed=sorted(process.cpu_affinity()))) + "\\n")


os.makedirs(os.environ.get("MOCK_ENGINE_LOG") or ".", exist_ok=True)
mock.move_self = move_self
mock.Engine().run()
'''


def hybrid():
    """4 P-cores with SMT (each its own L2) + 8 E-cores in two L2 clusters: an i7-1270P."""
    cores = [Core((2 * i, 2 * i + 1), speed=1, cluster=i) for i in range(4)]
    cores += [Core((8 + i,), speed=0, cluster=4) for i in range(4)]
    cores += [Core((12 + i,), speed=0, cluster=5) for i in range(4)]
    return Topology(cores=tuple(cores), supported=True, allowed=tuple(range(16)))


def quietly(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class SlotPlanning(unittest.TestCase):
    def test_one_cpu_per_core_fastest_first_all_but_one_core(self):
        slots = plan_slots(topo=hybrid(), claim=False)
        self.assertEqual([s.cpu for s in slots], [1, 2, 4, 6, 8, 12, 9, 13, 10, 14, 11])  # cpu 0 left idle
        self.assertEqual([s.core_type for s in slots], ["P"] * 4 + ["E"] * 7)
        self.assertEqual(slots[0].cpus, (1,))
        self.assertEqual(slots[0].label, "P/cpu01")

    def test_smt_siblings_only_beyond_the_physical_cores(self):
        slots = plan_slots(14, topo=hybrid(), claim=False)
        self.assertEqual([s.cpu for s in slots][-2:], [0, 3])
        with self.assertRaises(ValueError):
            plan_slots(17, topo=hybrid(), claim=False)

    def test_multi_threaded_slots_keep_one_core_type(self):
        slots = plan_slots(topo=hybrid(), threads=2, claim=False)
        self.assertEqual([s.cpus for s in slots], [(1, 2), (4, 6), (8, 12), (9, 13), (10, 14)])
        self.assertEqual([s.core_type for s in slots], ["P", "P", "E", "E", "E"])
        self.assertEqual(slots[0].label, "P/cpu1,2")
        self.assertEqual(len(plan_slots(8, topo=hybrid(), threads=2, claim=False)), 8)

    def test_unpinned(self):
        slots = plan_slots(3, topo=hybrid(), pin=False)
        self.assertEqual([(s.cpu, s.cpus, s.core_type) for s in slots], [(None, (), "-")] * 3)

    def test_cpus_another_run_holds_are_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(tempfile, "tempdir", tmp):
            env = dict(os.environ, TEMP=tmp, TMP=tmp, TMPDIR=tmp)
            other = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); from utils import cpu_slots; "
                 "print(all(cpu_slots.claim(c) for c in (1, 2)), flush=True); sys.stdin.read()", str(ROOT)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
            try:
                self.assertEqual(other.stdout.readline().strip(), "True")
                slots = plan_slots(topo=hybrid())
                try:
                    cpus = [s.cpu for s in slots]
                    # cores 0/1 and 2/3 have a cpu in use elsewhere: their free siblings come last
                    self.assertEqual(cpus, [4, 6, 8, 12, 9, 13, 10, 14, 11, 15])
                    self.assertEqual(cpu_slots.claimed(), set(cpus))
                finally:
                    release_slots(slots)
                self.assertEqual(cpu_slots.claimed(), set())
                with self.assertRaises(ValueError):  # 16 cpus, 2 held elsewhere
                    plan_slots(15, topo=hybrid())
                self.assertEqual(cpu_slots.claimed(), set())
            finally:
                other.stdin.close()
                other.wait(timeout=10)
                other.stdout.close()
            slots = plan_slots(topo=hybrid())  # the other run has ended: its cpus are free again
            release_slots(slots)
            self.assertEqual(slots[0].cpu, 1)


class FastchessCommand(unittest.TestCase):
    def test_engine_block(self):
        engine = EngineSpec("A", os.path.abspath("engine.exe"), ("script.py", "a b"), {"Hash": 16},
                            workdir="w", pin_option="Affinity")
        args = engine.fastchess_args(Slot(0, 3, "P"))
        self.assertEqual(args[:4], ["-engine", "name=A", f"cmd={engine.cmd}", "dir=w"])
        self.assertIn('args=script.py "a b"' if os.name == "nt" else "args=script.py 'a b'", args)
        self.assertIn("option.Affinity=3", args)
        self.assertIn("option.Hash=16", args)
        self.assertIn("option.Affinity=2,4", engine.fastchess_args(Slot(0, 2, "P", cpus=(2, 4))))
        threaded = EngineSpec("A", "engine.exe", (), {"Threads": 4}, pin_option="Affinity")
        self.assertIn("option.Affinity=none", threaded.fastchess_args(Slot(0, None), shared=True))
        self.assertFalse(any(a.startswith("option.Affinity") for a in engine.fastchess_args(Slot(0, None), True)))
        self.assertFalse(any(a.startswith("option.Affinity")
                             for a in EngineSpec("S", "sf.exe", pin_option="").fastchess_args(Slot(0, 3, "P"))))

    def test_job_command(self):
        runner = MatchRunner.__new__(MatchRunner)
        runner.fastchess, runner.slots, runner.recover = "fastchess.exe", [Slot(0, 3, "P")], True
        runner.tc_args, runner.plies, runner.extra_args = tc_arguments(nodes=5000), 8, ["-maxmoves", "100"]
        spec = SimpleNamespace(engines=(EngineSpec("A", "a.exe", pin_option=""), EngineSpec("B", "b.exe", pin_option="")),
                               openings=[None] * 3)
        cmd = runner.command(runner.slots[0], spec, Path("j/job1_book.epd"), Path("j/job1.pgn"), Path("j/job1.json"))
        text = " ".join(cmd)
        self.assertIn(f"-config outname={Path('j/job1.json')}", text)
        self.assertIn("-each proto=uci nodes=5000 tc=inf", text)
        self.assertIn("-games 2 -rounds 3 -concurrency 1", text)
        self.assertIn(f"-openings file={Path('j/job1_book.epd')} format=epd order=sequential plies=8", text)
        self.assertIn(f"-pgnout file={Path('j/job1.pgn')}", text)
        self.assertNotIn("-use-affinity", cmd)
        self.assertEqual(cmd[-3:], ["-recover", "-maxmoves", "100"])

    def test_time_controls(self):
        self.assertEqual(tc_arguments(depth=6), ["depth=6", "tc=inf"])
        self.assertEqual(tc_arguments(movetime=0.1), ["st=0.1", "timemargin=30"])
        self.assertEqual(tc_arguments(tc="10+0.1"), ["tc=10+0.1"])
        self.assertEqual(tc_arguments(nodes=20000), ["nodes=20000", "tc=inf"])
        with self.assertRaises(ValueError):
            tc_arguments(depth=6, nodes=100)

    def test_warnings(self):
        self.assertEqual(match_runner.WARNING.match("Warning; Illegal move a1a1 played by A").group(2),
                         "Illegal move a1a1 played by A")
        self.assertTrue(match_runner.WARNING.match("Warning: No opening book specified!"))
        self.assertIsNone(match_runner.WARNING.match("Started game 1 of 4 (A vs B)"))

    def test_one_warning_of_each_kind_reaches_the_controller_and_every_one_is_counted(self):
        runner = MatchRunner.__new__(MatchRunner)
        runner.warnings = {}
        job = SimpleNamespace(id=1, log=io.StringIO())
        runner.running = {0: job}
        controller = OneJob(None)
        lines = [f"Warning; Bestmove does not match beginning of last PV - move {move} from Stockfish"
                 for move in ("e2e4", "g8f6", "a7a8q", "e2e4", "h2h1n")]
        lines += ["Warning; Illegal move a1a1 played by A", "Warning; Engine B is not responsive",
                  "Warning; Illegal move h7h8 played by A"]
        for line in lines:
            runner._handle(controller, 1, 0.0, line)
        self.assertEqual(controller.warnings, ["Bestmove does not match beginning of last PV - move e2e4 from "
                                               "Stockfish", "Illegal move a1a1 played by A",
                                               "Engine B is not responsive"])
        self.assertEqual(runner.warning_count("Bestmove does not match beginning of last PV"), 5)
        self.assertEqual(runner.warning_count("Illegal move"), 2)
        self.assertEqual(runner.warning_count(), 8)
        self.assertEqual(match_runner.warning_key("Illegal move e7e8q played by Engine2 at ply 51"),
                         "Illegal move <move> played by Engine# at ply #")

class PgnNumbering(unittest.TestCase):
    def test_round_and_white_give_the_job_game(self):
        def game(round_no, white, black):
            return (f'[Event "Fastchess Tournament"]\n[Site "?"]\n[Round "{round_no}"]\n[White "{white}"]\n'
                    f'[Black "{black}"]\n[Result "1/2-1/2"]\n\n1. e4 e5 1/2-1/2\n')
        with tempfile.TemporaryDirectory() as tmp:
            pgn = Path(tmp) / "job7.pgn"
            # not in play order, to show the numbering does not rely on it
            pgn.write_text("\n".join([game(1, "A", "B"), game(2, "B", "A"), game(1, "B", "A"), game(2, "A", "B")]))
            runner = MatchRunner.__new__(MatchRunner)
            runner.games_pgn = Path(tmp) / "games.pgn"
            job = SimpleNamespace(id=7, pgn=pgn, slot=Slot(0, 5, "P", cpus=(5, 7)),
                                  spec=SimpleNamespace(engines=(SimpleNamespace(name="A"), SimpleNamespace(name="B"))))
            runner._append_pgn(job)
            text = runner.games_pgn.read_text()
        numbers = [int(line.split('"')[1]) for line in text.splitlines() if line.startswith("[JobGame ")]
        self.assertEqual(numbers, [1, 4, 2, 3])
        self.assertEqual(text.count('[Job "7"]'), 4)
        self.assertEqual(text.count('[Cpu "5,7"]'), 4)
        self.assertEqual(text.count('[Slot "P/cpu5,7"]'), 4)
        self.assertTrue(text.splitlines()[1].startswith("[Job "))  # right after [Event]


class Labels(unittest.TestCase):
    def test_fastchess_terminations(self):
        from tests import sprt
        from utils.match_runner import termination_class
        self.assertIs(sprt.termination_class, termination_class)  # one classification for every tool
        observed = {  # texts fastchess 1.8.2 printed for these endings
            "White mates": ("mate", None), "Black mates": ("mate", None),
            "Draw by stalemate": ("stalemate", None),
            "Draw by insufficient mating material": ("insufficient material", None),
            "Draw by fifty moves rule": ("50-move rule", None), "Draw by 3-fold repetition": ("repetition", None),
            "White disconnects": ("crash", "white"), "Black's connection stalls": ("stall", "black"),
            "White makes an illegal move": ("illegal move", "white"),
            "Black loses on time (114ms overrun)": ("time forfeit", "black"),
            "Draw by adjudication": ("adjudication", None), "Black wins by adjudication": ("adjudication", None),
            "White wins by adjudication: SyzygyTB": ("tablebase", None),
            "Draw by adjudication: SyzygyTB": ("tablebase", None), "Game interrupted": ("interrupted", None),
        }
        for text, expected in observed.items():
            self.assertEqual(termination_class(text), expected, text)
        from data.etl.match_games import TERMINATION_CODES
        for label, _ in observed.values():
            if label != "interrupted":
                self.assertIn(label, TERMINATION_CODES)

    def test_time_control_labels(self):
        from data.etl.match_games import _tc_fields
        none = dict(wtime=None, btime=None, winc=None, binc=None, depth=None, nodes=None, movetime=None)
        cases = {
            "depth=8": dict(depth=8), "depth 8": dict(depth=8), "nodes=50000": dict(nodes=50000),
            "st=0.1": dict(movetime=100), "0.29s/move": dict(movetime=290),
            "10+0.1": dict(wtime=10000, btime=10000, winc=100, binc=100),
            "40/60+0.6": dict(wtime=60000, btime=60000, winc=600, binc=600),
            "1:00+1": dict(wtime=60000, btime=60000, winc=1000, binc=1000),
            "blitz@T4": {}, "inf": {}, "tc=inf": {}, "depth=x": {}, None: {},
        }
        for label, fields in cases.items():
            self.assertEqual(_tc_fields(label), {**none, **fields}, label)


@unittest.skipIf(chess is None, "python-chess is not installed")
class TelemetryJoinsPlayedGames(unittest.TestCase):
    """Engine logs of a run link to its played games; only an unmatched engine game gets a row."""

    def test_engine_games_link_to_played_games_and_are_never_stored_twice(self):
        self.check()

    def test_a_retry_after_the_run_folder_was_removed_links_to_the_stored_games(self):
        self.check(remove_run_dir=True)

    def check(self, remove_run_dir=False):
        from data.databases.init_raw_db import init_raw_db
        from data.etl import bulk_log_match_games
        from tests import sprt
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        db, _ = quietly(init_raw_db, base / "db")
        cnxn = sqlite3.connect(db)
        self.addCleanup(cnxn.close)
        cnxn.execute("INSERT INTO engines (name, version) VALUES ('cand', 'c'), ('base', 'b')")
        cnxn.execute("INSERT INTO experiments (engine_id, type) VALUES (1, 'SPRT')")
        cnxn.commit()
        run_dir, logs = base / "20261025_101010_000001", base / "logs"
        run_dir.mkdir()
        rng = random.Random(5)
        records, pgn = [], []
        searches = {"candidate": [], "baseline": []}
        for number in (1, 2):
            board = chess.Board()
            for move in ("e2e4", "e7e5", "g1f3", "b8c6"):
                board.push_uci(move)
            while len(board.move_stack) < 30 and not board.is_game_over():
                board.push(rng.choice(list(board.legal_moves)))
            white = "Candidate" if number == 1 else "Baseline"
            records.append(dict(run=run_dir.name, game=number, pair=1, job=1, job_game=number, white=white,
                                black="Baseline" if number == 1 else "Candidate", result="1/2-1/2",
                                termination="adjudication", seconds=1.0, cpu=1, core_type="P",
                                engine_pids=[11, 22]))
            game = chess.pgn.Game.from_board(board)
            game.headers["Job"], game.headers["JobGame"] = "1", str(number)
            pgn.append(str(game))
            replay = chess.Board()
            for ply, move in enumerate(board.move_stack):
                if ply >= 4:
                    side = "candidate" if (replay.turn == chess.WHITE) == (white == "Candidate") else "baseline"
                    searches[side].append(dict(game_uuid=f"{side}-g{number}", search_uuid=f"{side}-{number}-{ply}",
                                               instance_id=11 if side == "candidate" else 22, engine_id="x",
                                               fen=replay.fen(), ply=ply, nodes=10))
                replay.push(move)
        (run_dir / "games.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        (run_dir / "games.pgn").write_text("\n\n".join(pgn) + "\n")
        (run_dir / "config.json").write_text(json.dumps(dict(meta=dict(run=run_dir.name, time_control="depth=2"))))
        played = bulk_log_match_games(cnxn, run_dir, 1, {"Candidate": 1, "Baseline": 2})
        self.assertEqual(len(played), 2)
        if remove_run_dir:
            shutil.rmtree(run_dir)
        # an engine game of a killed game: only the starting position of the book is known
        searches["candidate"].append(dict(game_uuid="candidate-killed", search_uuid="k1", instance_id=11,
                                          engine_id="x", fen="8/8/8/8/8/8/8/K6k w - - 0 1", ply=4, nodes=1))
        for side, rows in searches.items():
            (logs / side).mkdir(parents=True)
            (logs / side / "search_1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            uuids = sorted({r["game_uuid"] for r in rows})
            (logs / side / "game_1.jsonl").write_text("\n".join(json.dumps(
                dict(game_uuid=u, side="white", result=3, moves=[])) for u in uuids) + "\n")
        ids = dict(candidate=1, baseline=2, experiment=1, loaded=[], run_dir=str(run_dir), run_id=run_dir.name)
        ok, out = quietly(sprt.ingest_telemetry, cnxn, logs, ids)
        self.assertTrue(ok, out)
        rows = cnxn.execute("SELECT source, count(*) FROM games GROUP BY source ORDER BY source").fetchall()
        self.assertEqual(rows, [("engine", 1), ("match", 2)])  # the killed game only
        by_game = dict(cnxn.execute("SELECT g.run_game, count(*) FROM searches s JOIN games g ON g.id = s.game_id "
                                    "WHERE g.source = 'match' GROUP BY g.run_game").fetchall())
        self.assertEqual(sum(by_game.values()), sum(len(r) for r in searches.values()) - 1)
        engines = cnxn.execute("SELECT DISTINCT s.engine_id, g.run_game FROM searches s JOIN games g ON "
                               "g.id = s.game_id WHERE g.source = 'match' ORDER BY 2, 1").fetchall()
        self.assertEqual(engines, [(1, 1), (2, 1), (1, 2), (2, 2)])
        self.assertEqual(cnxn.execute("SELECT count(*) FROM telemetry_loads").fetchone()[0], 2)


@unittest.skipUnless(os.name == "nt", "job objects are Windows only")
class JobObjectLimit(unittest.TestCase):
    """The runner's job objects: a process (and its children) cannot leave the slot's cpus."""

    CHILD = ("import json, subprocess, sys, psutil\n"
             "p = psutil.Process()\n"
             "p.cpu_affinity(list(range(psutil.cpu_count())))  # an old build pinning itself elsewhere\n"
             "nested = subprocess.run([sys.executable, '-c', 'import psutil; print(psutil.Process().cpu_affinity())'],"
             " capture_output=True, text=True).stdout.strip()\n"
             "print(json.dumps([p.cpu_affinity(), nested]))\n")

    def test_processes_in_the_job_stay_on_its_cpus(self):
        cpu = max(cpu_topology.allowed_cpus())
        job = cpu_topology.KillOnCloseJob(cpus=[cpu])
        self.addCleanup(job.close)
        self.assertEqual(job.affinity, [cpu])
        proc = subprocess.Popen([sys.executable, "-c", self.CHILD], stdout=subprocess.PIPE, text=True,
                                creationflags=match_runner.CREATE_SUSPENDED)
        self.assertTrue(job.add(proc._handle))
        self.assertEqual(cpu_topology.resume_process(proc.pid), 1)
        out, _ = proc.communicate(timeout=60)
        own, nested = json.loads(out)
        self.assertEqual(own, [cpu])
        self.assertEqual(nested, str([cpu]))  # through the venv launcher and its own job, too

    def test_closing_the_job_kills_what_is_left(self):
        job = cpu_topology.KillOnCloseJob()
        self.assertIsNone(job.affinity)
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.assertTrue(job.add(proc._handle))
        job.close()
        self.assertIsNotNone(proc.wait(timeout=10))


class EnginePaths(unittest.TestCase):
    def test_paths_with_a_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "with space" / "engine.exe"
            engine.parent.mkdir()
            engine.write_bytes(b"")
            self.assertEqual(match_runner.fastchess_path(Path(tmp) / "plain.exe"), str(Path(tmp) / "plain.exe"))
            if os.name != "nt":  # fastchess passes the path as argv[0]
                self.assertEqual(match_runner.fastchess_path(engine), str(engine))
                return
            try:
                short = match_runner.fastchess_path(engine)
            except ValueError as e:  # a volume without 8.3 names: refused with a clear message
                self.assertIn("space", str(e))
                return
            self.assertNotIn(" ", short)
            self.assertTrue(os.path.samefile(short, engine))
            spec = EngineSpec("A", str(engine), pin_option="")
            self.assertIn(f"cmd={short}", spec.fastchess_args(Slot(0, 1, "P")))
            self.assertIn(f"dir={engine.parent}", spec.fastchess_args(Slot(0, 1, "P")))  # cwd may keep spaces


class SelfPlay(unittest.TestCase):
    def upload(self, ids):
        from data.databases.init_raw_db import init_raw_db
        from tests import sprt
        from utils.pentanomial import PentanomialSPRT
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        db, _ = quietly(init_raw_db, Path(tmp.name) / "db")
        state = PentanomialSPRT(0, 10)
        state.add_game(1, "Candidate", "Baseline", "1-0")
        state.add_game(2, "Baseline", "Candidate", "1/2-1/2")
        args = sprt.resolve_args(["--engine-a", "a.exe", "--engine-b", "b.exe", "--depth", "3", "--db", str(db)])
        with mock.patch("data.etl.resolve_engine", side_effect=lambda cnxn, exe: ids[exe]):
            _, out = quietly(sprt.upload_logs, args, state.snapshot(), runtime=1.0)
        cnxn = sqlite3.connect(db)
        self.addCleanup(cnxn.close)
        return out, cnxn.execute("SELECT count(*) FROM sprt").fetchone()[0], \
            cnxn.execute("SELECT engine_a_id, engine_b_id FROM match_results").fetchall()

    def test_a_build_against_itself_gets_no_pairwise_result(self):
        out, sprt_rows, results = self.upload({"a.exe": 4, "b.exe": 4})
        self.assertEqual((sprt_rows, results), (1, []))
        self.assertIn("same build", out)

    def test_two_builds_get_one(self):
        out, sprt_rows, results = self.upload({"a.exe": 4, "b.exe": 5})
        self.assertEqual((sprt_rows, results), (1, [(4, 5)]))


@unittest.skipIf(FASTCHESS is None, "fastchess is not built (libs/fastchess)")
@unittest.skipIf(chess is None, "python-chess is not installed")
class FastchessRuns(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.book = self.dir / "book.pgn"
        self.book.write_text(BOOK)
        self.mock_log = self.dir / "mock_log"

    def test_sprt_run_records_every_game_with_its_core_and_links_telemetry(self):
        from data.databases.init_raw_db import init_raw_db
        from tests import sprt
        db, _ = quietly(init_raw_db, self.dir / "db")
        results, logroot = self.dir / "results", self.dir / "logs"
        baseline = self.dir / "mock_baseline.py"  # another build (another hash): a pairwise result is stored
        baseline.write_text(MOCK.read_text() + "\n# baseline build\n")
        argv = ["--engine-a", str(MOCK), "--engine-b", str(baseline), "--param", "Strength=70", "--depth", "1",
                "--max-games", "8", "--concurrency", "2", "--pairs-per-job", "2", "--maxmoves", "40",
                "--book", str(self.book), "--book-depth", "4", "--seed", "3", "--elo1", "10", "--log", "--quiet",
                "--db", str(db), "--results-dir", str(results), "--logroot", str(logroot)]
        with mock.patch.dict(os.environ, MOCK_ENGINE_LOG=str(self.mock_log)), \
                mock.patch.object(sys, "stdin", io.StringIO("")):
            result, out = quietly(sprt.main, argv)
        run_dir = Path(result["run_dir"])
        self.assertEqual(result["games"], 8, out)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual((summary["mixed_core_games"], summary["voided_pairs"]), (0, 0), out)

        games = [json.loads(line) for line in (run_dir / "games.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(g["game"] for g in games), list(range(1, 9)))
        by_number = {g["game"]: g for g in games}
        for pair in range(1, 5):
            first, second = by_number[2 * pair - 1], by_number[2 * pair]
            self.assertEqual((first["pair"], second["pair"]), (pair, pair))
            self.assertEqual((first["white"], second["white"]), ("Candidate", "Baseline"))  # colours reversed
            self.assertEqual((first["job"], first["job_game"] + 1), (second["job"], second["job_game"]))
            self.assertEqual(first["opening"], second["opening"])
        jobs = {}
        for g in games:
            jobs.setdefault(g["job"], []).append(g["job_game"])
        for job, numbers in jobs.items():  # each job plays its pairs as games 1, 2, 3, ...
            self.assertEqual(sorted(numbers), list(range(1, len(numbers) + 1)), job)
            self.assertEqual(len(numbers) % 2, 0)
        for g in games:
            self.assertFalse(g["core_mixed"])
            if PINNING:
                self.assertTrue(g["pinned"])
                self.assertIn(g["core_type"], ("P", "E"))
                self.assertIsInstance(g["cpu"], int)
                self.assertGreater(g["clock_samples"], 0)
                self.assertTrue(g["engine_pids"])
                self.assertEqual(g["affinity_limited"], os.name == "nt")
            for key in ("mhz_mean", "slot_label", "concurrency", "power_source", "seconds", "termination"):
                self.assertIn(key, g)

        tags = {}
        with open(run_dir / "games.pgn", encoding="utf-8") as f:
            while (game := chess.pgn.read_game(f)) is not None:
                tags[(int(game.headers["Job"]), int(game.headers["JobGame"]))] = game.headers
        self.assertEqual(set(tags), {(g["job"], g["job_game"]) for g in games})
        for g in games:
            headers = tags[(g["job"], g["job_game"])]
            self.assertEqual((headers["White"], headers["Result"]), (g["white"], g["result"]))
            self.assertEqual((headers["CoreType"], headers["Slot"]), (g["core_type"], g["slot_label"]))
            self.assertEqual(headers["Cpu"], str(g["cpu"]) if g["cpu"] is not None else "-")

        if PINNING:  # every search ran on its game's cpu, and Tomahawk-style engines were told it
            cpus = {g["cpu"] for g in games}
            for path in self.mock_log.glob("go_*.jsonl"):
                for line in path.read_text().splitlines():
                    record = json.loads(line)
                    self.assertEqual(len(record["allowed"]), 1, record)
                    self.assertIn(record["allowed"][0], cpus)
                    self.assertEqual(record["affinity_option"], str(record["allowed"][0]))
        names = {p.name.split("_")[0].split(".")[0] for p in (run_dir / "jobs").iterdir()}
        self.assertEqual(names, {f"job{job:05d}" for job in jobs})  # fastchess wrote nothing else

        cnxn = sqlite3.connect(db)
        self.addCleanup(cnxn.close)
        q = lambda sql: cnxn.execute(sql).fetchall()  # noqa: E731
        self.assertEqual(q("SELECT source, count(*), count(DISTINCT run_game) FROM games GROUP BY source"),
                         [("match", 8, 8)])
        self.assertEqual(q("SELECT result, games_played FROM sprt"), [(summary["result"], 8)])
        self.assertEqual(q("SELECT source, engine_a_id != engine_b_id, wins + draws + losses, time_control "
                           "FROM match_results"), [("sprt", 1, 8, "depth=1")])
        self.assertEqual(q("SELECT count(*) FROM searches WHERE game_id IS NULL"), [(0,)])
        self.assertGreater(q("SELECT count(*) FROM searches")[0][0], 0)
        self.assertEqual(q("SELECT count(*) FROM telemetry_loads"), [(2,)])
        self.assertFalse((logroot / run_dir.name).exists())  # loaded, then removed

    def run_one_job(self, first, second, candidate_options=None, pairs=2, env=None):
        """One job of `pairs` opening pairs on one slot; env applies to the job's engines only."""
        from utils.openings import OpeningBook
        slots = plan_slots(concurrency=1)
        self.addCleanup(release_slots, slots)
        book = OpeningBook(self.book, plies=4, seed=1)
        engines = (EngineSpec.from_path("A", first, candidate_options).resolve(),
                   EngineSpec.from_path("B", second).resolve())
        controller = OneJob(match_runner.JobSpec(engines, book.take(pairs), book))
        runner = MatchRunner(None, slots, self.dir / "run", tc_arguments(depth=1), plies=4,
                             extra_args=["-maxmoves", "40"])
        with mock.patch.dict(os.environ, env or {}):
            quietly(runner.run, controller)
        return runner, controller

    def escaping_engine(self):
        """The mock engine, recording each attempt to change its affinity and the cpus it has after it."""
        path = self.dir / "escaping_engine.py"
        path.write_text(ESCAPING.format(tests_unit=str(MOCK.parent)))
        return path

    def escapes(self, games):
        pids = {pid for g in games for pid in g.env["engine_pids"]}
        records = [json.loads(line) for path in self.mock_log.glob("escape_*.jsonl")
                   for line in path.read_text().splitlines()]
        return [r for r in records if r["pid"] in pids]  # not the probe run before the job

    @unittest.skipUnless(PINNING, "no cpu affinity on this platform")
    def test_an_engine_leaving_its_core_during_a_game(self):
        engine = self.escaping_engine()  # Escape=midgame: tries to move to every cpu on its 6th search
        runner, controller = self.run_one_job(engine, MOCK, {"Escape": "midgame"},
                                              env=dict(MOCK_ENGINE_LOG=str(self.mock_log)))
        games = controller.games
        self.assertEqual(len(games), 4)
        attempts = self.escapes(games)
        self.assertTrue(attempts)
        cpu = games[0].env["cpu"]
        if os.name == "nt":  # the job object's affinity limit: the attempt has no effect, every game counts
            self.assertEqual([r["allowed"] for r in attempts], [[cpu]] * len(attempts))
            self.assertTrue(all(g.env["affinity_limited"] for g in games))
            self.assertEqual((runner.mixed_games, sum(g.env["core_mixed"] for g in games)), (0, 0))
        else:  # the engine got away: the guard excludes the game and moves it back
            self.assertGreaterEqual(runner.mixed_games, 1)
            self.assertTrue(any(g.env["core_mixed"] for g in games))
            self.assertTrue(any("left slot" in w for w in controller.warnings))

    @unittest.skipUnless(PINNING, "no cpu affinity on this platform")
    def test_engines_pinning_themselves_elsewhere_at_startup(self):
        engine = self.escaping_engine()  # like old Tomahawk builds (pin_to_pcores in main)
        everything = f"0-{(os.cpu_count() or 2) - 1}"
        runner, controller = self.run_one_job(engine, engine, pairs=1, env=dict(
            MOCK_ENGINE_ESCAPE_STARTUP=everything, MOCK_ENGINE_LOG=str(self.mock_log)))
        games = sorted(controller.games, key=lambda g: g.game_no)
        self.assertEqual(len(games), 2)
        attempts = self.escapes(games)
        self.assertGreaterEqual(len({r["pid"] for r in attempts}), 2)  # both engines tried
        cpu = games[0].env["cpu"]
        if os.name == "nt":  # they stay on the slot: nothing to exclude
            self.assertEqual([r["allowed"] for r in attempts], [[cpu]] * len(attempts))
            self.assertEqual([g.env["core_mixed"] for g in games], [False, False])
            self.assertEqual(runner.jobs[0].escapes, 0)
            for path in self.mock_log.glob("go_*.jsonl"):  # every search ran on the slot
                for line in path.read_text().splitlines():
                    self.assertEqual(json.loads(line)["allowed"], [cpu])
        else:
            # they started inside game 1 (fastchess starts engines with the job's first game) and may have
            # searched off their core: excluded. Moved back since, so game 2 counts.
            self.assertGreater(runner.jobs[0].escapes, 0)
            self.assertEqual([g.env["core_mixed"] for g in games], [True, False])

    @unittest.skipUnless(os.name == "nt", "Windows command lines")
    def test_an_engine_in_a_folder_with_a_space(self):
        folder = self.dir / "with space"
        folder.mkdir()
        shim = folder / "mock.bat"
        shim.write_text(f'@"{sys.executable}" "{MOCK}" %*\n')
        # fastchess puts the engine path unquoted on the command line: with the long path Windows
        # would start <tmp>\with.exe (which is no engine) instead of the shim
        shutil.copy(sys.executable, self.dir / "with.exe")
        try:
            short = match_runner.fastchess_path(shim)
        except ValueError:
            self.skipTest("no 8.3 short names on this volume")
        self.assertNotIn(" ", short)
        self.assertTrue(os.path.samefile(short, shim))
        runner, controller = self.run_one_job(shim, shim, pairs=1)
        self.assertEqual([g.result != "*" for g in controller.games], [True, True])
        self.assertIn("MockEngine", (self.dir / "run" / "games.pgn").read_text())


# Stand-ins for fastchess, started by the runner like a job's fastchess (suspended in a job object on
# Windows, in a process group of its own elsewhere).
FAKE_GAME = 'print("Started game 1 of 2 (A vs B)", flush=True)\n'
# fastchess dies in the middle of a game, and an engine it started lives on: the engine inherited
# fastchess's output (and ignores the end of its input), so the job's output stays open
FAKE_LEFTOVER = '''import subprocess, sys
engine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL,
                          stdout=sys.stdout)
print("leftover", engine.pid, flush=True)
print("Started game 1 of 2 (A vs B)", flush=True)
sys.exit(3)
'''


def open_handles():
    """Handles (Windows) or file descriptors this process holds."""
    process = psutil.Process()
    return process.num_handles() if os.name == "nt" else process.num_fds()


def gone(pid, timeout=10.0):
    """True once process `pid` has ended (within `timeout` seconds)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.1)
    return False


class JobLifecycle(unittest.TestCase):
    """What a job leaves behind when it ends, with a Python stand-in for fastchess."""

    def run_fake(self, script, jobs=1, slots=1, timeout=60.0):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        engines = (EngineSpec("A", "a.exe", pin_option=""), EngineSpec("B", "b.exe", pin_option=""))
        controller = Jobs(match_runner.JobSpec(engines, [SimpleNamespace(text="")]) for _ in range(jobs))
        runner = MatchRunner(sys.executable, [Slot(i, None) for i in range(slots)], tmp.name, [],
                             sample_clocks=False)
        runner.command = lambda *args, **kwargs: [sys.executable, "-c", script]
        errors = []

        def play():
            try:
                runner.run(controller)
            except BaseException as e:  # pragma: no cover - reported below
                errors.append(e)

        started = time.monotonic()
        thread = threading.Thread(target=play, daemon=True)
        thread.start()
        thread.join(timeout)
        self.assertFalse(thread.is_alive(), f"the runner still waits for its jobs after {timeout:.0f}s")
        self.assertEqual(errors, [])
        return runner, controller, time.monotonic() - started

    @unittest.skipIf(psutil is None, "psutil is not installed")
    def test_a_finished_job_keeps_no_process_or_pipe(self):
        self.run_fake(FAKE_GAME, jobs=4, slots=2)  # whatever the first jobs set up once
        gc.collect()
        before = open_handles()
        runner, controller, _ = self.run_fake(FAKE_GAME, jobs=20, slots=2)
        self.assertEqual(len(controller.ended), 20)
        self.assertTrue(all(job.proc is None and job.output_ended for job in runner.jobs))
        gc.collect()
        # each job's pipe and process handle (Windows) or pipe (POSIX) used to stay open until the run ended
        self.assertLess(open_handles() - before, 10)

    @unittest.skipIf(psutil is None, "psutil is not installed")
    def test_a_job_ends_when_fastchess_dies_and_an_engine_keeps_its_output_open(self):
        runner, controller, seconds = self.run_fake(FAKE_LEFTOVER, timeout=45)
        self.assertLess(seconds, 30)  # the engine would hold the output for 60 s
        job, = controller.ended
        self.assertEqual((job.returncode, job.killed, job.pairs_started), (3, False, 1))
        pid = next(int(line.split()[1]) for line in job.log_path.read_text().splitlines()
                   if line.startswith("leftover "))
        self.assertTrue(gone(pid), "the engine fastchess left behind is still running")

    def test_stopping_a_job_whose_fastchess_has_exited_kills_what_it_left(self):
        runner = MatchRunner.__new__(MatchRunner)
        container = mock.Mock()
        job = SimpleNamespace(proc=mock.Mock(pid=4321, **{"poll.return_value": 1}), container=container,
                              output_ended=False, killed=False)
        runner.kill(job, "stopped")
        container.close.assert_called_once_with()  # the job object: kills every process still in it
        self.assertFalse(job.killed)  # fastchess ended by itself


class OptionChecks(unittest.TestCase):
    """utils.uci_probe.option_problems: what fastchess would skip with only a warning."""

    DECLARED = {
        "Hash": "option name Hash type spin default 16 min 1 max 1024",
        "Scale": "option name Scale type spin default 0.5 min 0.1 max 2.5",
        "Ponder": "option name Ponder type check default false",
        "Style": "option name Style type combo default Normal var Solid var Normal var Risky Play",
        "Clear Hash": "option name Clear Hash type button",
        "log_dir": "option name log_dir type string default <empty>",
    }

    def problems(self, values):
        from utils.uci_probe import option_problems
        return option_problems(self.DECLARED, values)

    def test_values_fastchess_sends(self):
        self.assertEqual(self.problems({"Hash": 64, "Scale": "2.5", "Ponder": "true", "Style": "Risky Play",
                                        "Clear Hash": "true", "log_dir": "any thing"}), [])
        self.assertEqual(self.problems({"Hash": (1, 1024), "Scale": (0.1, 2.0), "log_dir": (0, 9)}), [])

    def test_values_fastchess_skips(self):
        cases = {"hash": "not declared (the engine calls it Hash)", "Threads": "Threads: not declared",
                 "Hash": "value 2048 exceeds the engine's [1, 1024]", "Scale": "value 3 exceeds",
                 "Ponder": "true or false", "Style": "not one of Solid, Normal, Risky Play",
                 "Clear Hash": "a button option takes true"}
        values = {"hash": 64, "Threads": 2, "Hash": 2048, "Scale": 3, "Ponder": "True", "Style": "Wild",
                  "Clear Hash": "false"}
        problems = self.problems(values)
        self.assertEqual(len(problems), len(cases))
        for problem, (name, message) in zip(problems, cases.items()):
            self.assertIn(message, problem, name)
        self.assertEqual(self.problems({"Hash": "lots"}), ["Hash=lots: not a number (spin option)"])
        self.assertEqual(self.problems({"Hash": (0, 64)}), ["Hash: range [0, 64] exceeds the engine's [1, 1024]"])


@unittest.skipIf(chess is None, "python-chess is not installed (the mock engine needs it)")
class SprtRefusals(unittest.TestCase):
    """An SPRT that cannot test what it was asked to is refused or ends with an error, never 'not accepted'."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "book.pgn").write_text(BOOK)
        self.results = self.dir / "results"

    def argv(self, *extra):
        return ["--engine-a", str(MOCK), "--engine-b", str(MOCK), "--depth", "1", "--max-games", "8",
                "--concurrency", "2", "--no-pin", "--fastchess", sys.executable, "--book", str(self.dir / "book.pgn"),
                "--book-depth", "4", "--quiet", "--results-dir", str(self.results), "--logroot",
                str(self.dir / "logs"), *extra]

    def test_candidate_options_the_engine_would_not_take_are_refused_before_the_run(self):
        from tests import sprt
        for param, message in (("strength=70", "the engine calls it Strength"),
                               ("Strength=150", r"value 150 exceeds the engine's \[0, 100\]"),
                               ("Strength=strong", "not a number"),
                               ("uci_logging=yes", "true or false")):
            with self.subTest(param), self.assertRaisesRegex(ValueError, message):
                quietly(sprt.main, self.argv("--param", param, "--no-db"))
        self.assertFalse(self.results.exists())  # refused before anything was created

    def test_jobs_failing_in_a_row_end_the_run_with_an_error_and_no_upload(self):
        from data.databases.init_raw_db import init_raw_db
        from tests import sprt
        db, _ = quietly(init_raw_db, self.dir / "db")
        if os.name == "nt":  # fastchess that cannot start a game: every job fails at once
            fake = self.dir / "fastchess.bat"
            fake.write_text("@exit /b 1\n")
        else:
            fake = self.dir / "fastchess"
            fake.write_text("#!/bin/sh\nexit 1\n")
            fake.chmod(0o755)
        with self.assertRaisesRegex(RuntimeError, "jobs in a row failed"):
            quietly(sprt.main, self.argv("--fastchess", str(fake), "--db", str(db)))
        run_dir, = self.results.iterdir()
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertIn("jobs in a row failed", summary["error"])
        self.assertEqual(summary["games_played"], 0)
        cnxn = sqlite3.connect(db)
        self.addCleanup(cnxn.close)
        self.assertEqual(cnxn.execute("SELECT count(*) FROM sprt").fetchone()[0], 0)  # not an inconclusive SPRT

    def test_the_controller_stops_after_failing_jobs(self):
        from tests import sprt
        args = sprt.resolve_args(self.argv())
        book = SimpleNamespace(take=lambda n: [SimpleNamespace(text="")] * n)
        controller = sprt.SprtController(args, (None, None), book, [Slot(0, None), Slot(1, None)], self.dir)
        self.addCleanup(controller.close)
        log = self.dir / "job.log"
        log.write_text("fastchess output\n")
        with contextlib.redirect_stdout(io.StringIO()):
            for number in range(1, 4):
                spec = controller.next_job(Slot(0, None))
                self.assertIsNotNone(spec)
                controller.on_job_end(SimpleNamespace(id=number, spec=spec, pairs=len(spec.openings), pairs_started=0,
                                                      killed=False, returncode=1, games=[], log_path=log,
                                                      slot=Slot(0, None)))
        self.assertIn("3 jobs in a row failed", controller.stop_reason)
        self.assertTrue(controller.finished())
        self.assertIsNone(controller.next_job(Slot(0, None)))


class RetryIngest(unittest.TestCase):
    def test_a_retry_without_the_database_fails_and_creates_none(self):
        from tests import sprt
        with tempfile.TemporaryDirectory() as tmp:
            run, missing = Path(tmp) / "run", Path(tmp) / "missing.db"
            (run / "candidate").mkdir(parents=True)
            (run / "engines.json").write_text(json.dumps(dict(candidate=1, baseline=2, experiment=1, loaded=[])))
            with self.assertRaisesRegex(FileNotFoundError, "missing.db not found"):
                sprt.main(["--retry-ingest", str(run), "--db", str(missing)])
            self.assertFalse(missing.exists())
            self.assertTrue((run / "candidate").is_dir())  # the logs are kept
            with self.assertRaisesRegex(FileNotFoundError, "engines.json not found"):
                sprt.retry_ingest(Path(tmp) / "elsewhere", db=missing)


class OneJob:
    """A controller that plays one job and records what the runner reports."""

    def __init__(self, spec):
        self.spec, self.games, self.ended, self.warnings = spec, [], [], []

    def next_job(self, slot):
        spec, self.spec = self.spec, None
        return spec

    def on_pair_start(self, job, pair_no):
        return True

    def on_game(self, game):
        self.games.append(game)

    def on_job_end(self, job):
        self.ended.append(job)

    def on_warning(self, job, text):
        self.warnings.append(text)

    def finished(self):
        return False

    def tick(self):
        pass


class Jobs(OneJob):
    """A controller that plays the given jobs, each on the next free slot."""

    def __init__(self, specs):
        super().__init__(None)
        self.specs = list(specs)

    def next_job(self, slot):
        return self.specs.pop(0) if self.specs else None


if __name__ == "__main__":
    unittest.main()
