"""Public snapshots: exact relations and columns, privacy, validation, retention and archive."""
import contextlib
import io
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
import time
import unittest
from pathlib import Path

import duckdb

from data.databases.export_public import export_public, prune_snapshots, sha256
from data.databases.public_schema import (FORBIDDEN_COLUMNS, FORBIDDEN_RELATIONS, FULL_VIEWS, public_relations,
                                          public_tables)
from data.databases.test_schema import validate_schema
from data.transforms.aggregates import AGG_TABLES, STOCKFISH_TABLES
from data.transforms.validate import run_validation

from .fixtures import analytics_fixture

SNAPSHOT_FILE = re.compile(r"[0-9a-f]{64}\.duckdb")


def _relations(cnxn):
    return {r[0]: r[1] for r in cnxn.execute(
        "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema = 'main'").fetchall()}


def _columns(cnxn, relation):
    return tuple(r[0] for r in cnxn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position",
        [relation]).fetchall())


class PublicExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.source, _ = analytics_fixture()
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(cls.tmp.name)
        cls.output = root / "public" / "data"
        cls.archive = root / "release" / "analytics-public.tar.gz"
        cls.manifests = []
        with contextlib.redirect_stdout(io.StringIO()):
            # two lite exports into the same folder, keeping one
            for _ in range(2):
                cls.manifests.append(export_public(cls.source, cls.output, cls.archive, keep=1))
            cls.full_output = root / "public-full" / "data"
            cls.full_manifest = export_public(cls.source, cls.full_output, None, full=True)
        cls.lite_file = cls.output / cls.manifests[-1]["snapshot"]
        cls.full_file = cls.full_output / cls.full_manifest["snapshot"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_snapshot(self, path, full):
        with duckdb.connect(str(path), read_only=True) as cnxn:
            relations = _relations(cnxn)
            self.assertEqual(set(relations), public_relations(full))
            for table, columns in public_tables(full).items():
                with self.subTest(table=table):
                    self.assertEqual(relations[table], "BASE TABLE")
                    self.assertEqual(_columns(cnxn, table), tuple(columns))
            for view in (FULL_VIEWS if full else ()):
                self.assertEqual(relations[view], "VIEW")
                cnxn.execute(f"SELECT count(*) FROM {view}").fetchone()
            for relation in FORBIDDEN_RELATIONS:
                self.assertNotIn(relation, relations)
            for table, columns in FORBIDDEN_COLUMNS.items():
                self.assertFalse(set(columns) & set(_columns(cnxn, table)), table)
            # no local path or private text in any published string
            for table, kind in relations.items():
                if kind != "BASE TABLE":
                    continue
                for (column,) in cnxn.execute("SELECT column_name FROM information_schema.columns "
                                              "WHERE table_name = ? AND data_type = 'VARCHAR'", [table]).fetchall():
                    leaked = cnxn.execute(f'SELECT count(*) FROM "{table}" WHERE "{column}" ILIKE ? '
                                          f'OR "{column}" ILIKE ? OR "{column}" ILIKE ?',
                                          ["%someone%", "%Auto-registered%", '%"cwd"%']).fetchone()[0]
                    self.assertEqual(leaked, 0, f"{table}.{column} publishes private text")
            profile = "full" if full else "lite"
            info = dict(cnxn.execute("SELECT key, value FROM snapshot_info").fetchall())
            self.assertEqual(info["profile"], profile)
            self.assertEqual(run_validation(cnxn, profile=f"public-{profile}").errors, [])
            return {t: cnxn.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in AGG_TABLES}

    def test_lite_snapshot_has_exactly_the_allowed_relations(self):
        counts = self.check_snapshot(self.lite_file, full=False)
        # the fixture has no Stockfish analysis, so only the Stockfish comparison tables are empty
        self.assertEqual({t for t, n in counts.items() if not n}, set(STOCKFISH_TABLES), counts)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(validate_schema(self.lite_file))                      # detects public-lite
            self.assertFalse(validate_schema(self.lite_file, profile="public-full"))

    def test_full_snapshot_has_exactly_the_allowed_relations(self):
        self.check_snapshot(self.full_file, full=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(validate_schema(self.full_file))
        with duckdb.connect(str(self.full_file), read_only=True) as cnxn:
            searches = cnxn.execute("SELECT count(*) FROM search_stats").fetchone()[0]
            self.assertEqual(cnxn.execute("SELECT count(*) FROM search_context").fetchone()[0], searches)
            # Stockfish's lines are published only for positions that were searched
            self.assertEqual(cnxn.execute("SELECT count(*) FROM stockfish_moves WHERE fen NOT IN "
                                          "(SELECT fen FROM search_stats)").fetchone()[0], 0)

    def test_full_snapshot_publishes_pvs_researches_of_every_build_but_not_its_kinds(self):
        """Newer builds log three kinds of PVS re-search instead of pvs_researches (fixtures.PVS_KIND_VERSIONS):
        the snapshot holds their sum as pvs_researches, the same value the source's views derive."""
        derived = {
            "search_stats": "SELECT search_id, total_pvs_researches FROM search_metrics",
            "iterative_deepening_stats": "SELECT search_id, depth, pvs_researches FROM search_iteration_metrics",
            "search_tree_stats": "SELECT search_id, depth, pvs_researches FROM search_tree_features",
        }
        keys = {"search_stats": "id", "iterative_deepening_stats": "search_id, depth",
                "search_tree_stats": "search_id, depth"}
        with duckdb.connect(str(self.source), read_only=True) as source, \
                duckdb.connect(str(self.full_file), read_only=True) as snapshot:
            for table, sql in derived.items():
                with self.subTest(table=table):
                    self.assertFalse({c for c in _columns(snapshot, table) if c.startswith("pvs_researches_")})
                    self.assertTrue({c for c in _columns(source, table) if c.startswith("pvs_researches_")})
                    published = snapshot.execute(
                        f"SELECT {keys[table]}, pvs_researches FROM {table} ORDER BY ALL").fetchall()
                    self.assertEqual(published, source.execute(f"{sql} ORDER BY ALL").fetchall())
                    self.assertNotIn(None, {row[-1] for row in published})  # every fixture build logs them
            # the snapshot's own views agree with the source's
            self.assertEqual(snapshot.execute("SELECT search_id, total_pvs_researches FROM search_metrics "
                                              "ORDER BY ALL").fetchall(),
                             source.execute(f"{derived['search_stats']} ORDER BY ALL").fetchall())

    def test_full_snapshot_keeps_only_stockfish_lines_of_published_positions(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            source = Path(tmp) / "source.duckdb"
            shutil.copyfile(self.source, source)
            with duckdb.connect(str(source)) as cnxn:
                fen = cnxn.execute("SELECT min(fen) FROM search_stats").fetchone()[0]
                cnxn.execute("INSERT INTO stockfish_moves (fen, move, rank, eval, pv, depth, multipv) VALUES "
                             "(?, 'a2a3', 1, 12, '[\"a2a3\"]', 10, 5), "
                             "('8/8/8/8/8/8/8/K6k w - - 0 1', 'a1a2', 1, 0, '[\"a1a2\"]', 10, 5)", [fen])
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = export_public(source, Path(tmp) / "out", None, full=True)
            with duckdb.connect(str(Path(tmp) / "out" / manifest["snapshot"]), read_only=True) as snapshot:
                self.assertEqual(snapshot.execute("SELECT fen FROM stockfish_moves").fetchall(), [(fen,)])

    def test_snapshots_agree_with_the_source_aggregates(self):
        def same(a, b):
            if isinstance(a, float) and isinstance(b, float):
                return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
            return a == b

        with duckdb.connect(str(self.source), read_only=True) as source, \
                duckdb.connect(str(self.lite_file), read_only=True) as lite:
            for table in AGG_TABLES:
                with self.subTest(table=table):
                    expected = source.execute(f'SELECT * FROM "{table}" ORDER BY ALL').fetchall()
                    actual = lite.execute(f'SELECT * FROM "{table}" ORDER BY ALL').fetchall()
                    self.assertEqual(len(actual), len(expected))
                    for got, want in zip(actual, expected):
                        self.assertTrue(all(same(a, b) for a, b in zip(got, want)), (got, want))

    def test_manifest(self):
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest, self.manifests[-1])
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["profile"], "lite")
        self.assertEqual(manifest["snapshot"], f"snapshots/{manifest['sha256']}.duckdb")
        self.assertEqual(sha256(self.lite_file), manifest["sha256"])
        self.assertEqual(self.lite_file.stat().st_size, manifest["size_bytes"])
        self.assertGreater(manifest["searches"], 0)
        self.assertEqual(manifest["row_counts"]["search_stats"], manifest["searches"])
        self.assertEqual(self.full_manifest["row_counts"]["search_stats"], manifest["searches"])
        self.assertLess(manifest["size_bytes"], self.full_manifest["size_bytes"])

    def test_keeps_only_the_newest_snapshots(self):
        kept = sorted(p.name for p in (self.output / "snapshots").iterdir())
        self.assertEqual(kept, [self.lite_file.name])
        self.assertTrue(SNAPSHOT_FILE.fullmatch(kept[0]))
        self.assertNotEqual(self.manifests[0]["sha256"], self.manifests[1]["sha256"])
        with tarfile.open(self.archive, "r:gz") as bundle:
            names = sorted(bundle.getnames())
        self.assertEqual(names, ["data/manifest.json"] + [f"data/snapshots/{name}" for name in kept])
        self.assertEqual([p.name for p in self.output.iterdir() if p.name.startswith(".")], [])

    def test_prune_keeps_the_current_and_newest_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            files = []
            for age, digit in enumerate("abcd"):
                path = folder / f"{digit * 64}.duckdb"
                path.write_bytes(b"x")
                stamp = time.time() - 1000 * (4 - age)          # a oldest ... d newest
                os.utime(path, (stamp, stamp))
                files.append(path)
            (folder / "notes.txt").write_text("not a snapshot")
            kept, removed = prune_snapshots(folder, 3, current=files[0])   # the current one is the oldest
            self.assertEqual({p.name for p in kept}, {files[0].name, files[3].name, files[2].name})
            self.assertEqual([p.name for p in removed], [files[1].name])
            self.assertEqual(sorted(p.name for p in folder.iterdir()),
                             sorted([files[0].name, files[2].name, files[3].name, "notes.txt"]))

    def test_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            export_public(self.source, self.output, keep=0)
        with self.assertRaises(ValueError):
            export_public(self.source, self.output, self.output / "bundle.tar.gz")


if __name__ == "__main__":
    unittest.main()
