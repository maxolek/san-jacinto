"""tests/tournament.py on the match runner (utils/fixed_match.py), and data/etl/telemetry.py.

Without fastchess (run_fixed_match replaced by FixedMatchController fed with games): the
pairings (gauntlet, round robin, whole pairs), what release.py's Namespace gives the runner,
how a time control's run is stored (one experiment: every played game with its core, a
match_results row per pairing with DD and WL apart, a rating fit, its PGN), that a time
control that stops, is interrupted or cannot be stored completely stores none of its
results, that a stored run's telemetry stays loadable when the fit fails, and the telemetry
of a run of three engines (one log folder each) linked to its games outside the write
transaction, loaded once, retried with and without the run folder. With fastchess: a
gauntlet of three mock engines
(tests/unit/mock_uci_engine.py copies registered as versions) stored in a temporary database
with its telemetry.
"""
import argparse
import contextlib
import io
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.databases.init_raw_db import init_raw_db  # noqa: E402
from data.etl import paths  # noqa: E402
from data.etl.provenance import sha256_file  # noqa: E402
from tests import release, tournament  # noqa: E402
from utils import cpu_topology  # noqa: E402
from utils.fixed_match import FixedMatchController, MatchInterrupted, MatchStopped  # noqa: E402
from utils.match_runner import EngineSpec, GameResult, find_fastchess  # noqa: E402

try:
    import chess
    import chess.pgn
except ImportError:  # pragma: no cover
    chess = None
try:
    FASTCHESS = find_fastchess()
except FileNotFoundError:
    FASTCHESS = None
MOCK = ROOT / "tests" / "unit" / "mock_uci_engine.py"
PINNING = cpu_topology.topology().supported
BOOK = "".join(f'[Event "?"]\n[Result "*"]\n\n{line} *\n\n' for line in (
    "1. e4 e5 2. Nf3 Nc6", "1. d4 d5 2. c4 e6", "1. c4 e5 2. Nc3 Nf6", "1. e4 c5 2. Nf3 d6"))
SLOT = SimpleNamespace(index=0, cpu=3, core_type="P", label="P/cpu03")


def quietly(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


class FakeJob:
    def __init__(self, job_id, spec, slot):
        self.id, self.spec, self.slot, self.log_path = job_id, spec, slot, None
        self.games, self.pairs_started, self.killed, self.returncode = [], 0, False, 0


class FakeMatch:
    """run_fixed_match without fastchess: FixedMatchController plays every pairing's pairs on one slot.

    Pair k of a job: the first engine wins as White and loses as Black (WL) when k is odd, two draws
    (DD) when k is even. `stop`: {call index: "stopped" | "interrupted"} raised after that call's games.
    """

    def __init__(self, stop=None):
        self.calls, self.stop = [], stop or {}

    def __call__(self, pairings, **kwargs):
        self.calls.append(dict(kwargs, pairings=pairings))
        run_dir = Path(kwargs["results_dir"]) / kwargs["run_name"]
        run_dir.mkdir(parents=True)
        meta = dict(run=run_dir.name, time_control=kwargs["time_control"], machine=dict(host="h", cpu_model="cpu"))
        (run_dir / "config.json").write_text(json.dumps(dict(meta=meta)))
        ctl = FixedMatchController(pairings, kwargs["book"], [SLOT], run_dir, quiet=True, status_every=0,
                                   pairs_per_job=kwargs["pairs_per_job"])
        pgn, number = [], 0
        while (spec := ctl.next_job(SLOT)) is not None:
            number += 1
            job = FakeJob(number, spec, SLOT)
            first, second = (e.name for e in spec.engines)
            for pair_no in range(1, len(spec.openings) + 1):
                job.pairs_started = pair_no
                ctl.on_pair_start(job, pair_no)
                for half, result in enumerate(("1-0", "1-0") if pair_no % 2 else ("1/2-1/2", "1/2-1/2"), 1):
                    white, black = (first, second) if half == 1 else (second, first)
                    game_no = 2 * pair_no - 2 + half
                    env = dict(slot=0, slot_label=SLOT.label, cpu=SLOT.cpu, core_type=SLOT.core_type,
                               core_mixed=False, mhz_mean=3000.0)
                    reason = "White mates" if result == "1-0" else "Draw by adjudication"
                    game = GameResult(job=job, pair_no=pair_no, game_no=game_no, white=white, black=black,
                                      result=result, reason=reason, started=1000.0, finished=1001.0, env=env)
                    job.games.append(game)
                    ctl.on_game(game)
                    pgn.append(f'[Event "?"]\n[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n'
                               f'[Job "{number}"]\n[JobGame "{game_no}"]\n\n1. e4 e5 {result}\n')
            ctl.on_job_end(job)
        ctl.close()
        (run_dir / "games.pgn").write_text("\n".join(pgn))
        result = ctl.result()
        stop = self.stop.get(len(self.calls) - 1)
        if stop == "stopped":
            raise MatchStopped("TOURNAMENT stopped after 0 of 2 pairs: 3 jobs in a row failed", result)
        if stop == "interrupted":
            raise MatchInterrupted(result)
        return result


class TempDb(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.db, _ = quietly(init_raw_db, self.dir / "db")
        self.cnxn = sqlite3.connect(self.db)
        self.addCleanup(self.cnxn.close)
        self.book = self.dir / "book.pgn"
        self.book.write_text(BOOK)

    def register(self, version):
        """A Prod build of `version` (an exe with its hash registered); (engine id, path)."""
        exe = self.dir / f"{version}.exe"
        exe.write_bytes(f"build {version}".encode())
        cur = self.cnxn.execute("INSERT INTO engines (name, version, binary_path, binary_sha256, build_type) "
                                "VALUES ('Tomahawk', ?, ?, ?, 'Prod')", (version, str(exe), sha256_file(exe)))
        self.cnxn.commit()
        return cur.lastrowid, str(exe)

    def q(self, sql, params=()):
        return self.cnxn.execute(sql, params).fetchall()


class Pairings(unittest.TestCase):
    def test_gauntlet_round_robin_and_whole_pairs(self):
        engines = [EngineSpec(name, f"{name}.exe") for name in ("Candidate", "v1.0", "v0.9")]
        gauntlet = tournament.make_pairings(engines, 3)
        self.assertEqual([(p.first.name, p.second.name, p.pairs) for p in gauntlet],
                         [("Candidate", "v1.0", 3), ("Candidate", "v0.9", 3)])
        robin = tournament.make_pairings(engines, 2, round_robin=True)
        self.assertEqual([(p.first.name, p.second.name) for p in robin],
                         [("Candidate", "v1.0"), ("Candidate", "v0.9"), ("v1.0", "v0.9")])
        self.assertEqual(tournament.opponents_in(gauntlet), {"Candidate": None, "v1.0": "Candidate",
                                                             "v0.9": "Candidate"})
        self.assertEqual(set(tournament.opponents_in(robin).values()), {None})
        pairs, out = quietly(tournament.pairs_per_pairing, 5)
        self.assertEqual(pairs, 3)
        self.assertIn("Playing 6 games per pairing (complete opening pairs), not 5", out)
        self.assertEqual(quietly(tournament.pairs_per_pairing, 4), (2, ""))
        self.assertEqual(quietly(tournament.pairs_per_pairing, 0)[0], 1)


@unittest.skipIf(chess is None, "python-chess is not installed")
class Storage(TempDb):
    def args(self, **kwargs):
        return argparse.Namespace(**{**dict(
            engine=str(self.dir / "cand.exe"), fastchess="fastchess.exe", opening_book=str(self.book),
            tournament_games=4, tournament_tc=["0:10+0.1", "1:00+0.6"], tournament_engines=2, concurrency=None,
            results_dir=self.dir / "results", logroot=self.dir / "logs"), **kwargs})

    def test_a_gauntlet_is_stored_per_time_control(self):
        cand, _ = self.register("2.0")
        old, _ = self.register("1.0")
        older, _ = self.register("0.9")
        fake = FakeMatch()
        with mock.patch.object(tournament, "run_fixed_match", fake):
            played, out = quietly(tournament.run_tournament, self.args(), self.cnxn, cand)
        self.assertEqual([(p["category"], p["time_control"]) for p in played],
                         [("ultra_fast", "0:10+0.1"), ("bullet", "1:00+0.6")])
        for call, tc in zip(fake.calls, ("0:10+0.1", "1:00+0.6")):
            self.assertEqual([(p.first.name, p.second.name, p.pairs) for p in call["pairings"]],
                             [("Candidate", "v0.9", 2), ("Candidate", "v1.0", 2)])  # the newest registered first
            self.assertEqual((call["tc_args"], call["time_control"], call["plies"], call["book"].plies),
                             ([f"tc={tc}"], tc, 16, 16))
            self.assertEqual((call["extra_args"], call["concurrency"], call["pin"], call["label"]),
                             (["-maxmoves", "200"], None, True, "TOURNAMENT"))
            self.assertIs(call["book"], fake.calls[0]["book"])  # one shuffle: every run its own openings
            for engine in (call["pairings"][0].first, call["pairings"][0].second):
                self.assertEqual(engine.options["uci_logging"], "false")
                self.assertEqual(Path(engine.options["log_dir"]),
                                 (self.dir / "logs").resolve() / call["run_name"] / engine.name)
        self.assertEqual(len({c["run_name"] for c in fake.calls}), 2)

        experiments = self.q("SELECT id, type, engine_id FROM experiments ORDER BY id")
        self.assertEqual([(t, e) for _, t, e in experiments], [("tournament", cand)] * 2)
        self.assertEqual([p["experiment_id"] for p in played], [i for i, _, _ in experiments])
        # every played game once, with its core; one row per pairing with DD and WL apart
        self.assertEqual(self.q("SELECT source, count(*), count(DISTINCT run_id), min(env_core_type), "
                                "min(env_cpu) FROM games GROUP BY source"), [("match", 16, 2, "P", 3)])
        engines = self.q("SELECT DISTINCT white_engine_id, black_engine_id FROM games")
        self.assertEqual(sorted(engines), sorted([(cand, older), (older, cand), (cand, old), (old, cand)]))
        rows = self.q("SELECT tc_category, time_control, engine_a_id, engine_b_id, wins, draws, losses, "
                      "penta_ll, penta_ld, penta_dd, penta_wl, penta_wd, penta_ww FROM match_results "
                      "WHERE source = 'tournament' ORDER BY id")
        self.assertEqual(rows, [(cat, tc, cand, b, 1, 2, 1, 0, 0, 1, 1, 0, 0)
                                for cat, tc in (("ultra_fast", "0:10+0.1"), ("bullet", "1:00+0.6"))
                                for b in (older, old)])
        self.assertEqual(self.q("SELECT tc_category FROM rating_fits ORDER BY id"), [("ultra_fast",), ("bullet",)])
        results_dir, = (self.dir / "results").iterdir()
        for cat in ("ultra_fast", "bullet"):
            self.assertEqual((results_dir / f"{cat}.pgn").read_text().count("[Event "), 8)
        self.assertFalse((self.dir / "logs").exists())  # no logging: nothing kept
        self.assertIn("Candidate vs v0.9", out)
        self.assertIn("Engine logging was disabled", out)

    def test_round_robin(self):
        cand, _ = self.register("2.0")
        old, _ = self.register("1.0")
        older, _ = self.register("0.9")
        fake = FakeMatch()
        with mock.patch.object(tournament, "run_fixed_match", fake):
            quietly(tournament.run_tournament, self.args(tournament_tc=["blitz"], round_robin=True, tournament_games=2),
                    self.cnxn, cand)
        self.assertEqual([(p.first.name, p.second.name, p.pairs) for p in fake.calls[0]["pairings"]],
                         [("Candidate", "v0.9", 1), ("Candidate", "v1.0", 1), ("v0.9", "v1.0", 1)])
        self.assertEqual(self.q("SELECT engine_a_id, engine_b_id, wins + draws + losses, time_control "
                                "FROM match_results ORDER BY id"),
                         [(cand, older, 2, "3:00+1.8"), (cand, old, 2, "3:00+1.8"), (older, old, 2, "3:00+1.8")])
        self.assertEqual(self.q("SELECT count(*) FROM games WHERE source = 'match'"), [(6,)])

    def test_a_time_control_that_stops_stores_none_of_its_results_and_raises(self):
        cand, _ = self.register("2.0")
        self.register("1.0")
        fake = FakeMatch(stop={1: "stopped"})
        out = io.StringIO()
        with mock.patch.object(tournament, "run_fixed_match", fake), contextlib.redirect_stdout(out), \
                self.assertRaisesRegex(RuntimeError, "jobs in a row failed"):
            tournament.run_tournament(self.args(log=True), self.cnxn, cand)
        self.assertIn("bullet (tc=1:00+0.6) was not completed: none of its results are stored", out.getvalue())
        # the first time control stays stored, the second is not
        self.assertEqual(self.q("SELECT tc_category, count(*) FROM match_results GROUP BY 1"), [("ultra_fast", 1)])
        self.assertEqual(self.q("SELECT count(*) FROM experiments"), [(1,)])
        self.assertEqual(self.q("SELECT count(*) FROM games"), [(4,)])
        results_dir, = (self.dir / "results").iterdir()
        self.assertTrue((results_dir / "bullet.pgn").exists())  # its games are kept as played

    def test_a_time_control_that_cannot_be_stored_completely_stores_nothing(self):
        """The 2nd pairing's row is refused (database locked): its experiment, games and 1st row go too."""
        from data import ratings
        cand, _ = self.register("2.0")
        self.register("1.0")
        self.register("0.9")
        record, calls = ratings.record_match, []

        def locked_second(cnxn, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise sqlite3.OperationalError("database is locked")
            return record(cnxn, **kwargs)

        with mock.patch.object(tournament, "run_fixed_match", FakeMatch()), \
                mock.patch.object(ratings, "record_match", locked_second), \
                self.assertRaisesRegex(sqlite3.OperationalError, "database is locked"):
            quietly(tournament.run_tournament, self.args(tournament_tc=["blitz"]), self.cnxn, cand)
        self.assertEqual(len(calls), 2)
        self.assertFalse(self.cnxn.in_transaction)
        for table in ("experiments", "games", "match_results", "rating_fits"):
            self.assertEqual(self.q(f"SELECT count(*) FROM {table}"), [(0,)], table)
        # the connection is left usable: the next time control is stored whole
        with mock.patch.object(tournament, "run_fixed_match", FakeMatch()):
            played, _ = quietly(tournament.run_tournament, self.args(tournament_tc=["blitz"]), self.cnxn, cand)
        self.assertEqual(self.q("SELECT id, end_time_utc IS NOT NULL FROM experiments"),
                         [(played[0]["experiment_id"], 1)])
        self.assertEqual(self.q("SELECT count(*) FROM games"), [(8,)])
        self.assertEqual(self.q("SELECT count(*) FROM match_results"), [(2,)])

    def test_a_failing_rating_fit_leaves_the_telemetry_ready_for_a_retry(self):
        cand, _ = self.register("2.0")
        old, _ = self.register("1.0")
        fake = FakeMatch()

        def fit_fails(cnxn, category):
            # the engines logged into their folders; the stored run's ids are already beside them
            run_logs, = (self.dir / "logs").resolve().iterdir()
            self.assertTrue((run_logs / "engines.json").exists())
            raise RuntimeError("the fit failed")

        def logging_match(pairings, **kwargs):
            for engine in {e.name: e for p in pairings for e in (p.first, p.second)}.values():
                Path(engine.options["log_dir"]).mkdir(parents=True)
            return fake(pairings, **kwargs)

        out = io.StringIO()
        with mock.patch.object(tournament, "run_fixed_match", logging_match), \
                mock.patch.object(tournament.ratings, "fit", fit_fails), contextlib.redirect_stdout(out), \
                self.assertRaisesRegex(RuntimeError, "the fit failed"):
            tournament.run_tournament(self.args(tournament_tc=["blitz"], log=True), self.cnxn, cand)
        run_logs, = (self.dir / "logs").resolve().iterdir()
        ids = json.loads((run_logs / "engines.json").read_text())
        experiment, = self.q("SELECT id FROM experiments")
        self.assertEqual((ids["experiment"], ids["engines"], ids["opponents"], ids["loaded"]),
                         (experiment[0], {"Candidate": cand, "v1.0": old}, {"Candidate": old, "v1.0": cand}, []))
        self.assertEqual(Path(ids["run_dir"]), Path(fake.calls[0]["results_dir"]).resolve() / run_logs.name)
        self.assertIn(f"--retry-ingest {run_logs}", out.getvalue())

    def test_ctrl_c_stores_nothing_of_the_time_control(self):
        cand, _ = self.register("2.0")
        self.register("1.0")
        with mock.patch.object(tournament, "run_fixed_match", FakeMatch(stop={0: "interrupted"})), \
                self.assertRaises(KeyboardInterrupt):
            quietly(tournament.run_tournament, self.args(), self.cnxn, cand)
        for table in ("experiments", "games", "match_results", "rating_fits"):
            self.assertEqual(self.q(f"SELECT count(*) FROM {table}"), [(0,)], table)

    def test_release_namespace(self):
        """release.run_tournament's Namespace: its engine, book, games, time controls, cpus and tablebases."""
        cand, cand_exe = self.register("2.0")
        old, _ = self.register("1.0")
        tb = self.dir / "tb"
        tb.mkdir()
        (tb / "KQvK.rtbw").write_bytes(b"")
        args = release.build_parser().parse_args([
            "--version", "9.9.9", "--name", "n", "--description", "d", "--tournament", "--tournament_tc", "bullet",
            "--tournament_games", "4", "--concurrency", "3", "--no-pin", "--tb", str(tb), "--tbpieces", "5",
            "--opening_book", str(self.book), "--fastchess", "fc.exe"])
        args.dev_engine = cand_exe
        fake = FakeMatch()
        with mock.patch.object(tournament, "run_fixed_match", fake), \
                mock.patch.object(tournament, "RESULTS_DIR", self.dir / "results"), \
                mock.patch.object(tournament, "TOURNAMENT_LOG_DIR", self.dir / "logs"):
            quietly(release.run_tournament, args, self.cnxn, cand)
        call, = fake.calls
        pairing, = call["pairings"]
        self.assertEqual((pairing.first.name, pairing.first.cmd, pairing.second.name, pairing.pairs),
                         ("Candidate", cand_exe, "v1.0", 2))
        # telemetry on (as in the release's SPRT): each engine logs into <logroot>/<run>/<engine name>/
        for engine in (pairing.first, pairing.second):
            self.assertEqual(engine.options["uci_logging"], "true")
            self.assertEqual(Path(engine.options["log_dir"]).parent.parent, (self.dir / "logs").resolve())
            self.assertEqual(Path(engine.options["log_dir"]).name, engine.name)
        self.assertEqual((call["tc_args"], call["concurrency"], call["pin"], call["fastchess"]),
                         (["tc=1:00+0.6"], 3, False, "fc.exe"))
        self.assertEqual(call["extra_args"], ["-maxmoves", "200", "-tb", str(tb), "-tbpieces", "5"])
        self.assertEqual(Path(call["book"].path), self.book)
        self.assertEqual(self.q("SELECT engine_a_id, engine_b_id, tc_category FROM match_results"),
                         [(cand, old, "bullet")])
        self.assertTrue((self.dir / "results").is_dir())


@unittest.skipIf(chess is None, "python-chess is not installed")
class Telemetry(TempDb):
    """A run of three engines (A vs B, A vs C), each logging into its own folder."""

    def setUp(self):
        super().setUp()
        from data.etl import bulk_log_match_games
        self.cnxn.execute("INSERT INTO engines (name, version) VALUES ('a', 'a'), ('b', 'b'), ('c', 'c')")
        self.cnxn.execute("INSERT INTO experiments (engine_id, type) VALUES (1, 'tournament')")
        self.cnxn.commit()
        self.run_dir, self.logs = self.dir / "20261025_101010_000001_blitz", self.dir / "logs"
        self.run_dir.mkdir()
        rng = random.Random(7)
        records, pgn, self.searches = [], [], {"A": [], "B": [], "C": []}
        pids = {1: {"A": 11, "B": 22}, 2: {"A": 12, "C": 33}}
        for number, (job, job_game, white, black) in enumerate(
                [(1, 1, "A", "B"), (1, 2, "B", "A"), (2, 1, "A", "C"), (2, 2, "C", "A")], 1):
            board = chess.Board()
            for move in ("e2e4", "e7e5", "g1f3", "b8c6"):
                board.push_uci(move)
            while len(board.move_stack) < 30 and not board.is_game_over():
                board.push(rng.choice(list(board.legal_moves)))
            records.append(dict(run=self.run_dir.name, game=number, pair=(number + 1) // 2, job=job,
                                job_game=job_game, white=white, black=black, result="1/2-1/2",
                                termination="adjudication", seconds=1.0, cpu=1, core_type="P",
                                engine_pids=list(pids[job].values()), pairing=f"A vs {'B' if job == 1 else 'C'}"))
            game = chess.pgn.Game.from_board(board)
            game.headers["Job"], game.headers["JobGame"] = str(job), str(job_game)
            pgn.append(str(game))
            replay = chess.Board()
            for ply, move in enumerate(board.move_stack):
                if ply >= 4:
                    name = white if replay.turn == chess.WHITE else black
                    self.searches[name].append(dict(game_uuid=f"{name}-g{number}", search_uuid=f"{name}-{number}-{ply}",
                                                    instance_id=pids[job][name], engine_id="x", fen=replay.fen(),
                                                    ply=ply, nodes=10))
                replay.push(move)
        (self.run_dir / "games.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        (self.run_dir / "games.pgn").write_text("\n\n".join(pgn) + "\n")
        (self.run_dir / "config.json").write_text(json.dumps(dict(meta=dict(run=self.run_dir.name,
                                                                            time_control="3:00+1.8"))))
        played, _ = quietly(bulk_log_match_games, self.cnxn, self.run_dir, 1, {"A": 1, "B": 2, "C": 3})
        self.assertEqual(len(played), 4)
        # engine games of killed games (only a position no played game has): B's opponent is known, A's is not
        for name, pid in (("A", 11), ("B", 22)):
            self.searches[name].append(dict(game_uuid=f"{name}-killed", search_uuid=f"{name}-k", instance_id=pid,
                                            engine_id="x", fen="8/8/8/8/8/8/8/K6k w - - 0 1", ply=4, nodes=1))

    def write_logs(self):
        for name, rows in self.searches.items():
            (self.logs / name).mkdir(parents=True, exist_ok=True)
            (self.logs / name / "search_1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            uuids = sorted({r["game_uuid"] for r in rows})
            (self.logs / name / "game_1.jsonl").write_text("\n".join(json.dumps(
                dict(game_uuid=u, side="white", result=3, moves=[])) for u in uuids) + "\n")

    def ids(self):
        from data import etl
        return etl.telemetry_ids(1, {"A": 1, "B": 2, "C": 3}, {"A": None, "B": 1, "C": 1}, self.run_dir)

    def check(self):
        self.assertEqual(self.q("SELECT source, count(*) FROM games GROUP BY source ORDER BY source"),
                         [("engine", 2), ("match", 4)])
        self.assertEqual(self.q("SELECT white_engine_id, black_engine_id FROM games WHERE source = 'engine' "
                                "ORDER BY white_engine_id"), [(1, None), (2, 1)])
        linked = self.q("SELECT g.run_game, s.engine_id, count(*) FROM searches s JOIN games g ON g.id = s.game_id "
                        "WHERE g.source = 'match' GROUP BY 1, 2 ORDER BY 1, 2")
        self.assertEqual([(game, engine) for game, engine, _ in linked],
                         [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 3), (4, 1), (4, 3)])
        self.assertEqual(sum(n for _, _, n in linked), sum(len(r) for r in self.searches.values()) - 2)
        self.assertEqual(self.q("SELECT side FROM telemetry_loads WHERE experiment_id = 1 ORDER BY side"),
                         [("A",), ("B",), ("C",)])

    def test_each_engine_links_to_the_games_it_played_and_is_loaded_once(self):
        from data import etl
        self.write_logs()
        ids = self.ids()
        ok, out = quietly(etl.ingest_telemetry, self.cnxn, self.logs, ids)
        self.assertTrue(ok, out)
        self.check()
        self.assertEqual(ids["loaded"], ["A", "B", "C"])
        self.assertFalse(any((self.logs / name).exists() for name in "ABC"))
        # the process stopped after the commits, before it noted them: the same logs come again
        self.write_logs()
        ok, out = quietly(etl.ingest_telemetry, self.cnxn, self.logs, self.ids())
        self.assertTrue(ok)
        self.assertEqual(out.count("loaded already"), 3)
        self.check()

    def test_a_retry_after_the_run_folder_was_removed_links_by_the_stored_engines(self):
        self.write_logs()
        (self.logs / "engines.json").write_text(json.dumps(self.ids()))
        shutil.rmtree(self.run_dir)
        self.cnxn.commit()
        with mock.patch.object(paths, "RAW_DB", Path(self.db)):
            ok, out = quietly(tournament.main, ["--retry-ingest", str(self.logs)])
        self.assertTrue(ok, out)
        self.assertFalse(self.logs.exists())
        self.check()

    def test_the_games_are_read_once_and_linked_outside_the_write_transaction(self):
        from data import etl
        from data.etl import match_games, telemetry
        self.write_logs()
        reads, links = [], []
        pgn_moves, link = match_games.pgn_moves, match_games.link_engine_games

        def counted_read(run_dir):
            reads.append(run_dir)
            return pgn_moves(run_dir)

        def checked_link(*args, **kwargs):
            links.append(self.cnxn.in_transaction)
            return link(*args, **kwargs)

        with mock.patch.object(telemetry, "pgn_moves", counted_read), \
                mock.patch.object(match_games, "pgn_moves", counted_read), \
                mock.patch.object(telemetry, "link_engine_games", checked_link):
            ok, out = quietly(etl.ingest_telemetry, self.cnxn, self.logs, self.ids())
            self.assertTrue(ok, out)
            self.assertEqual((len(reads), links), (1, [False, False, False]))  # one parse for three engines
            self.check()
            # a side loaded before is skipped before anything is read
            self.write_logs()
            ok, out = quietly(etl.ingest_telemetry, self.cnxn, self.logs, self.ids())
        self.assertTrue(ok, out)
        self.assertEqual(out.count("loaded already"), 3)
        self.assertEqual((len(reads), len(links)), (1, 3))
        self.check()

    def test_a_loader_that_loses_the_race_for_a_side_takes_it_as_loaded(self):
        from data import etl
        from data.etl import telemetry
        self.write_logs()
        link = telemetry.link_engine_games
        other = sqlite3.connect(self.db, timeout=0)  # would fail at once if the side's transaction were open
        self.addCleanup(other.close)

        def another_loader_first(*args, engine_name=None, **kwargs):
            if engine_name == "B":  # another process loads B while this one links it
                other.execute("INSERT INTO telemetry_loads (experiment_id, side) VALUES (1, 'B')")
                other.commit()
            return link(*args, engine_name=engine_name, **kwargs)

        ids = self.ids()
        with mock.patch.object(telemetry, "link_engine_games", another_loader_first):
            ok, out = quietly(etl.ingest_telemetry, self.cnxn, self.logs, ids)
        self.assertTrue(ok, out)
        self.assertIn("B telemetry of experiment 1 was loaded already", out)
        self.assertEqual(ids["loaded"], ["A", "B", "C"])
        self.assertFalse((self.logs / "B").exists())
        self.assertEqual(self.q("SELECT count(*) FROM searches WHERE engine_id = 2"), [(0,)])  # not loaded twice
        self.assertEqual(self.q("SELECT side FROM telemetry_loads ORDER BY side"), [("A",), ("B",), ("C",)])


@unittest.skipIf(FASTCHESS is None, "fastchess is not built (libs/fastchess)")
@unittest.skipIf(chess is None, "python-chess is not installed")
class FastchessGauntlet(TempDb):
    def mock_build(self, version, strength):
        """A copy of the mock engine reporting `version` (its own hash: its own build)."""
        path = self.dir / f"mock_{version}.py"
        path.write_text(MOCK.read_text().replace("mock-1", version).replace('"Strength": "50"',
                                                                             f'"Strength": "{strength}"'))
        return path

    def test_a_gauntlet_with_telemetry(self):
        from data import etl
        candidate = self.mock_build("2.0", 70)
        for version, strength in (("1.0", 50), ("0.9", 30)):
            quietly(etl.resolve_engine, self.cnxn, str(self.mock_build(version, strength)), description="mock")
        results, logroot = self.dir / "results", self.dir / "logs"
        argv = ["--engine", str(candidate), "--n_engines", "2", "--tc", "2+0.05", "--games", "2", "--concurrency", "2",
                "--pairs-per-job", "1", "--book", str(self.book), "--fastchess", str(FASTCHESS), "--log", "--quiet",
                "--results-dir", str(results), "--logroot", str(logroot)]
        with mock.patch.object(paths, "RAW_DB", Path(self.db)), mock.patch.object(sys, "stdin", io.StringIO("")):
            played, out = quietly(tournament.main, argv)
        run, = played
        self.assertEqual(run["category"], "ultra_fast", out)
        self.assertEqual({k: (s["pairs"], s["complete"]) for k, s in run["pairings"].items()},
                         {"Candidate vs v0.9": (1, True), "Candidate vs v1.0": (1, True)}, out)
        ids = dict(self.q("SELECT version, id FROM engines"))
        self.assertEqual(self.q("SELECT count(*), count(DISTINCT run_game), min(run_id), max(run_id) FROM games "
                                "WHERE source = 'match'"),
                         [(4, 4, Path(run["run_dir"]).name, Path(run["run_dir"]).name)])
        if PINNING:
            self.assertEqual(self.q("SELECT count(*) FROM games WHERE source = 'match' AND env_core_type IN "
                                    "('P', 'E') AND env_cpu IS NOT NULL"), [(4,)])
        self.assertEqual(sorted(self.q("SELECT engine_a_id, engine_b_id, wins + draws + losses, penta_wl IS NOT NULL "
                                       "FROM match_results WHERE source = 'tournament'")),
                         sorted([(ids["2.0"], ids["1.0"], 2, 1), (ids["2.0"], ids["0.9"], 2, 1)]))
        self.assertEqual(self.q("SELECT count(*) FROM rating_fits WHERE tc_category = 'ultra_fast'"), [(1,)])
        # telemetry: every search linked to a played game, by the engine that played it
        self.assertEqual(sorted(s for s, in self.q("SELECT side FROM telemetry_loads")), ["Candidate", "v0.9", "v1.0"])
        self.assertGreater(self.q("SELECT count(*) FROM searches")[0][0], 0)
        self.assertEqual(self.q("SELECT count(*) FROM searches WHERE game_id IS NULL"), [(0,)])
        self.assertEqual(self.q("SELECT count(*) FROM searches s JOIN games g ON g.id = s.game_id "
                                "WHERE s.engine_id NOT IN (g.white_engine_id, g.black_engine_id)"), [(0,)])
        self.assertEqual(self.q("SELECT count(*) FROM games WHERE source = 'engine'"), [(0,)])
        self.assertFalse((logroot / Path(run["run_dir"]).name).exists())  # loaded, then removed
        tournament_dir = Path(run["run_dir"]).parent
        with open(tournament_dir / "ultra_fast.pgn", encoding="utf-8") as f:
            games = 0
            while chess.pgn.read_game(f) is not None:
                games += 1
        self.assertEqual(games, 4)
        commands = [p.read_text().splitlines()[0] for p in (Path(run["run_dir"]) / "jobs").glob("job*.log")]
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertIn("-recover", command)
            self.assertIn("-maxmoves 200", command)
            self.assertIn("option.uci_logging=true", command)


if __name__ == "__main__":
    unittest.main()
