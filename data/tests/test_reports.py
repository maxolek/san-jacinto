"""Reports must not fail on consoles or pipes that can't encode a character (Windows cp1252),
and public snapshots must not publish local paths."""
import contextlib
import io
import unittest

import duckdb

from data.databases import test_schema
from data.transforms.validate import ValidationResult, validate_publication


class PipeSafeOutputTest(unittest.TestCase):
    def strict_stdout(self, encoding):
        return io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")

    def test_validation_report_on_a_narrow_encoding(self):
        result = ValidationResult()
        result.warn("check", "missing \u2014 fen \u2713 \u26a0", 3)
        result.error("other", "\u2717 broken", 1)
        for encoding in ("cp1252", "ascii"):
            stream = self.strict_stdout(encoding)
            with contextlib.redirect_stdout(stream):
                self.assertFalse(result.report())
            stream.flush()
            text = stream.buffer.getvalue().decode(encoding)
            self.assertIn("WARN  [check]", text)
            self.assertIn("ERROR [other]", text)

    def test_schema_report_on_a_narrow_encoding(self):
        stream = self.strict_stdout("cp1252")
        with contextlib.redirect_stdout(stream):
            test_schema._report("SCHEMA \u2713 \u26a0 ok")
        stream.flush()
        self.assertIn("SCHEMA", stream.buffer.getvalue().decode("cp1252"))


class PublicationCheckTest(unittest.TestCase):
    def test_local_paths_fail_public_validation(self):
        with duckdb.connect() as cnxn:
            cnxn.execute("CREATE TABLE sprt_runs AS SELECT 'UHO.epd' AS opening_book "
                         "UNION ALL SELECT '/home/someone/books/8moves.pgn'")
            cnxn.execute("CREATE TABLE sts_runs AS SELECT 'C:\\sts\\STS1.epd' AS suite")
            result = ValidationResult()
            validate_publication(cnxn, result)
        self.assertEqual([(check, count) for check, _, count in result.errors], [("privacy", 1), ("privacy", 1)])


if __name__ == "__main__":
    unittest.main()
