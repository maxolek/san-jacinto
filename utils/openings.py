"""Opening books for paired games, and a register of results per opening.

`OpeningBook` reads a PGN or EPD book, shuffles it once with a recorded seed
and hands out openings in that order, so every pair gets a distinct opening
until the book is exhausted, and a run can be reproduced from its seed.

`OpeningRegister` keeps the candidate's results per named opening (ECO code,
opening and variation, e.g. "C11 French, Steinitz variation": +3 =5 -1), with
pentanomial counts per opening. Names come from data/openings/eco_db.json via
the same trie as the analytics pipeline (needs python-chess to read SAN). Without
python-chess the book's [ECO] tag and the most common name for that code are used.
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from utils.pentanomial import PairTally

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ECO_DB = PROJECT_ROOT / "data" / "openings" / "eco_db.json"

_COMMENT = re.compile(r"\{[^}]*\}|;[^\n]*")
_NAG = re.compile(r"\$\d+")
_MOVE_NUMBER = re.compile(r"^\d+\.+$|^\d+\.+(?=\S)")
_RESULT = {"1-0", "0-1", "1/2-1/2", "*"}
_TAG = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


class Opening:
    """One book entry: the verbatim text written to a job's book file."""

    __slots__ = ("index", "text", "moves", "fen", "eco")

    def __init__(self, index, text, moves=(), fen=None, eco=""):
        self.index, self.text, self.moves, self.fen, self.eco = index, text, tuple(moves), fen, eco

    def __repr__(self):
        return f"Opening({self.index}, {' '.join(self.moves[:6]) or self.fen})"


def _strip_variations(text):
    out, depth = [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def san_moves(movetext, plies=None):
    """SAN tokens of a PGN movetext, without numbers, comments, NAGs or result."""
    text = _strip_variations(_COMMENT.sub(" ", movetext))
    text = _NAG.sub(" ", text)
    moves = []
    for token in text.split():
        token = _MOVE_NUMBER.sub("", token)
        if not token or token in _RESULT:
            continue
        moves.append(token)
        if plies is not None and len(moves) >= plies:
            break
    return moves


def parse_pgn_book(text, plies=None):
    games, headers, movetext, in_moves = [], [], [], False

    def flush():
        if headers or movetext:
            tags = {m.group(1).lower(): m.group(2) for m in map(_TAG.match, headers) if m}
            body = "\n".join(headers) + "\n\n" + "\n".join(movetext).strip() + "\n"
            games.append(Opening(len(games), body, san_moves(" ".join(movetext), plies),
                                 tags.get("fen"), tags.get("eco", "")))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_moves:
                flush()
                headers, movetext, in_moves = [], [], False
            headers.append(stripped)
        elif stripped:
            in_moves = True
            movetext.append(stripped)
    flush()
    return [g for g in games if g.moves or g.fen]


def parse_epd_book(text):
    out = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 4:
            fen = " ".join(fields[:4]) + " 0 1"
            out.append(Opening(len(out), line.strip(), (), fen, ""))
    return out


class OpeningBook:
    def __init__(self, path, plies=16, seed=None, shuffle=True):
        self.path = Path(path)
        self.format = "epd" if self.path.suffix.lower() == ".epd" else "pgn"
        if self.path.suffix.lower() not in (".pgn", ".epd"):
            raise ValueError(f"Opening book must be .pgn or .epd (got {self.path.name}); "
                             "polyglot .bin books are engine books, not opening suites")
        # utf-8-sig: a byte order mark (editors on Windows write one) is not part of the first opening
        text = self.path.read_text(encoding="utf-8-sig", errors="replace")
        self.plies = plies
        self.openings = parse_pgn_book(text, plies) if self.format == "pgn" else parse_epd_book(text)
        if not self.openings:
            raise ValueError(f"No openings found in {self.path}")
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2 ** 31)
        self.order = list(range(len(self.openings)))
        if shuffle:
            random.Random(self.seed).shuffle(self.order)
        self.cursor = 0
        self.wrapped = False

    def __len__(self):
        return len(self.openings)

    def take(self, count):
        out = []
        for _ in range(count):
            if self.cursor >= len(self.order):
                self.cursor, self.wrapped = 0, True
            out.append(self.openings[self.order[self.cursor]])
            self.cursor += 1
        return out

    def write(self, openings, path):
        """A book file with exactly these openings, in order (for fastchess order=sequential)."""
        sep = "\n" if self.format == "pgn" else ""
        Path(path).write_text(sep.join(o.text.rstrip("\n") + "\n" for o in openings), encoding="utf-8")


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _eco_names_by_code():
    try:
        entries = json.loads(ECO_DB.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    names = {}
    for entry in entries:
        if entry.get("eco"):
            names.setdefault(entry["eco"], Counter())[entry.get("name", "")] += 1
    return {code: counts.most_common(1)[0][0] for code, counts in names.items()}


@lru_cache(maxsize=1)
def _trie_classifier():
    """data.etl.openings.get_opening_details, or None without python-chess."""
    try:
        from data.etl.openings import get_opening_details
        return get_opening_details
    except Exception:
        return None


@lru_cache(maxsize=100_000)
def _classify_moves(moves):
    classifier = _trie_classifier()
    if classifier is None or not moves:
        return None
    try:
        return classifier(" ".join(moves))
    except Exception:
        return None


def classify(opening):
    """(eco, name, variation) of the most specific named opening the book line reaches."""
    details = _classify_moves(opening.moves)
    if details and details[1]:
        return details
    if opening.eco:
        return opening.eco, _eco_names_by_code().get(opening.eco, ""), ""
    if opening.fen:
        return "", "(from FEN)", ""
    return "", "Unclassified", ""


def display_name(details):
    eco, name, variation = details
    label = f"{name}, {variation}" if variation else name
    return f"{eco} {label}".strip()


# ----------------------------------------------------------------------------
# Register
# ----------------------------------------------------------------------------

class OpeningRegister:
    """Candidate results per named opening; each pair is counted once."""

    def __init__(self):
        self.rows = {}  # display name -> dict(eco, name, variation, tally, by_core)

    def record_pair(self, opening, candidate_scores, core_type=None):
        details = classify(opening)
        key = display_name(details)
        row = self.rows.get(key)
        if row is None:
            row = self.rows[key] = dict(eco=details[0], name=details[1], variation=details[2],
                                        tally=PairTally(), by_core={})
        row["tally"].add_pair(*candidate_scores)
        if core_type:
            row["by_core"].setdefault(core_type, PairTally()).add_pair(*candidate_scores)
        return key

    def summaries(self):
        out = []
        for key, row in self.rows.items():
            s = row["tally"].summary()
            out.append(dict(opening=key, eco=row["eco"], name=row["name"], variation=row["variation"],
                            pairs=s["pairs"], wins=s["wins"], draws=s["draws"], losses=s["losses"],
                            score=s["raw_score"], elo=s["raw_elo"], elo_low=s["elo_low"], elo_high=s["elo_high"],
                            pentanomial=s["pentanomial"],
                            by_core={core: t.summary()["pentanomial"] for core, t in row["by_core"].items()}))
        out.sort(key=lambda r: (-r["pairs"], r["opening"]))
        return out

    def families(self):
        """The same results grouped by opening family (name without variation)."""
        grouped = {}
        for row in self.rows.values():
            name = row["name"] or "Unclassified"
            entry = grouped.setdefault(name, dict(tally=PairTally(), variations=0))
            family = entry["tally"]
            family.counts = [a + b for a, b in zip(family.counts, row["tally"].counts)]
            family.wdl = [a + b for a, b in zip(family.wdl, row["tally"].wdl)]
            for k, v in row["tally"].detail.items():
                family.detail[k] += v
            entry["variations"] += 1
        out = []
        for name, entry in grouped.items():
            s = entry["tally"].summary()
            out.append(dict(opening=name, family=name, variations=entry["variations"], pairs=s["pairs"],
                            wins=s["wins"], draws=s["draws"], losses=s["losses"], score=s["raw_score"],
                            elo=s["raw_elo"], elo_low=s["elo_low"], elo_high=s["elo_high"],
                            pentanomial=s["pentanomial"]))
        out.sort(key=lambda r: (-r["pairs"], r["family"]))
        return out

    def format_table(self, limit=12, min_pairs=1, level="variation"):
        """Console table at 'variation' level (ECO + opening + variation) or 'family' level."""
        source = self.families() if level == "family" else self.summaries()
        rows = [r for r in source if r["pairs"] >= min_pairs]
        if not rows:
            return "  (no completed pairs yet)"
        title = "opening family" if level == "family" else "opening (ECO, name, variation)"
        lines = [f"  {title:52s} {'pairs':>5s}  {'+W =D -L':>12s}  {'score':>6s}  {'elo':>6s}"]
        for r in rows[:limit]:
            wdl = f"+{r['wins']} ={r['draws']} -{r['losses']}"
            elo = f"{r['elo']:>+6.0f}" if r["elo"] is not None else f"{'-':>6s}"
            lines.append(f"  {r['opening'][:52]:52s} {r['pairs']:>5d}  {wdl:>12s}  "
                         f"{r['score']:>6.1%}  {elo}")
        if len(rows) > limit:
            csv_name = "openings_by_family.csv" if level == "family" else "openings.csv"
            lines.append(f"  ... {len(rows) - limit} more in {csv_name}")
        return "\n".join(lines)

    def save(self, directory, run_meta=None, history_csv=None):
        """openings.json plus openings.csv (variation level) and openings_by_family.csv in the
        run directory, and optionally append to a cumulative CSV (one row per variation, with
        its family name) so results per opening can be followed across runs at either level."""
        directory = Path(directory)
        rows = self.summaries()
        families = self.families()
        (directory / "openings.json").write_text(
            json.dumps(dict(meta=run_meta or {}, openings=rows, families=families), indent=2),
            encoding="utf-8")
        fields = ["opening", "eco", "name", "variation", "pairs", "wins", "draws", "losses",
                  "score", "elo", "elo_low", "elo_high"]
        with open(directory / "openings.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        family_fields = ["family", "variations", "pairs", "wins", "draws", "losses", "score", "elo",
                         "elo_low", "elo_high"]
        with open(directory / "openings_by_family.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=family_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(families)
        if history_csv:
            history_csv = Path(history_csv)
            history_csv.parent.mkdir(parents=True, exist_ok=True)
            new = not history_csv.exists()
            meta = run_meta or {}
            with open(history_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["recorded_utc", "run", "candidate", "baseline",
                                                       "time_control"] + fields, extrasaction="ignore")
                if new:
                    writer.writeheader()
                stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                for row in rows:
                    writer.writerow(dict(row, recorded_utc=stamp, run=meta.get("run"),
                                         candidate=meta.get("candidate"), baseline=meta.get("baseline"),
                                         time_control=meta.get("time_control")))
