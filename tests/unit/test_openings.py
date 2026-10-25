"""utils/openings.py: reading PGN and EPD opening books.

A book saved with a UTF-8 byte order mark (as some Windows editors write it) must give the
same openings as one without: read as plain UTF-8, the mark would be glued to the first
line, turning the first PGN header into junk "moves" and the first EPD position into an
invalid FEN that fastchess would be given.
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.openings import OpeningBook  # noqa: E402

PGN = '[Event "?"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 *\n\n[Event "?"]\n[Result "*"]\n\n1. d4 d5 *\n'
EPD = ('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - c0 "e4";\n'
       'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - c0 "d4";\n')
BOM = b"\xef\xbb\xbf"


class Books(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def book(self, name, text, bom):
        path = self.dir / name
        path.write_bytes((BOM if bom else b"") + text.encode("utf-8"))
        return OpeningBook(path, plies=4, shuffle=False)

    def test_a_byte_order_mark_is_not_part_of_the_first_opening(self):
        for bom in (False, True):
            with self.subTest(bom=bom):
                pgn = self.book("book.pgn", PGN, bom)
                self.assertEqual([o.moves for o in pgn.openings], [("e4", "e5", "Nf3", "Nc6"), ("d4", "d5")])
                self.assertTrue(pgn.openings[0].text.startswith('[Event "?"]'))
                epd = self.book("book.epd", EPD, bom)
                self.assertEqual([o.fen for o in epd.openings],
                                 ["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                                  "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"])
                # the job book files fastchess reads start with the opening itself
                for book, start in ((pgn, b'[Event "?"]'), (epd, b"rnbqkbnr/")):
                    out = self.dir / f"job.{book.format}"
                    book.write(book.take(2), out)
                    self.assertTrue(out.read_bytes().startswith(start), out.read_bytes()[:20])


if __name__ == "__main__":
    unittest.main()
