"""Global Elo ratings: one maximum-likelihood fit over every stored rating game.

Rating matches are stored pairwise in match_results (engine a vs engine b:
W/D/L and, when known, pentanomial counts) by tests/tournament.py and the other
runners. A fit reads every result of one time-control category, fits logistic
Elo with utils.rating_fit (with standard errors), stores it (rating_fits,
ratings) and mirrors the standard categories into engine_ratings, which the
dashboard reads. Unlike per-run K-factor updates, the result does not depend on
the order games were played in, and every engine's rating uses all games
involving it, directly or through common opponents.

Scale: a fit keeps the previous fit's anchor engine and Elo, so ratings do not
jump between fits. The first fit anchors the oldest engine of the largest
connected group at its current engine_ratings value (else 1500), continuing the
earlier scale. The mirror writes the anchor's group; engines that never played
it are set to NULL (another scale); engines without rating games keep their
earlier value.

SPRT results are recorded too but left out of fits unless asked for: a test that
stops when it crosses a bound overestimates the engines that pass.

    python -m data.ratings fit [--tc blitz ...] [--include-sprt] [--anchor ENGINE] [--anchor-elo 1500]
    python -m data.ratings show [--tc blitz]
    python -m data.ratings backfill-sprt
"""
import argparse
import math
import sqlite3

from data.etl.schema import ensure_schema

CATEGORIES = ("ultra_fast", "bullet", "blitz", "rapid", "classical")  # mirrored into engine_ratings
DEFAULT_SOURCES = ("tournament", "manual")
ANCHOR_ELO = 1500.0
Z95 = 1.959964


def classify_tc(tc_str):
    """Classify a time control string into ultra_fast/bullet/blitz/rapid/classical.

    Parses "seconds+increment", "min:sec+increment" and "seconds". Estimated game
    time = base + 40 * increment (about 40 moves per side).
    """
    tc_str = tc_str.strip()
    parts = tc_str.split("+")
    clock = parts[0].split("/")[-1]  # moves/time+inc: the clock part
    if ":" in clock:
        minutes, seconds = clock.split(":")
        base_sec = int(minutes) * 60 + float(seconds)
    else:
        base_sec = float(clock)
    inc = float(parts[1]) if len(parts) > 1 else 0
    game_time_min = (base_sec + 40 * inc) / 60.0
    if game_time_min < 0.25:
        return "ultra_fast"
    if game_time_min < 2:
        return "bullet"
    if game_time_min < 10:
        return "blitz"
    if game_time_min < 30:
        return "rapid"
    return "classical"


def category(tc=None, depth=None, time=None, nodes=None, threads=1):
    """Rating category of a per-move limit: a TC class, or the fixed limit itself (e.g. nodes=50000).

    Multi-threaded games get their own category (e.g. blitz@T4): threads change strength.
    """
    if tc:
        label = classify_tc(tc)
    elif depth is not None:
        label = f"depth={depth}"
    elif nodes is not None:
        label = f"nodes={nodes}"
    elif time is not None:
        label = f"st={time:g}"
    else:
        raise ValueError("no time control or per-move limit")
    return label if not threads or threads <= 1 else f"{label}@T{threads}"


def record_match(cnxn, *, source, engine_a, engine_b, wins, draws, losses, tc_category,
                 time_control=None, penta=None, experiment_id=None):
    """Store one pairwise result (engine a's wins/draws/losses); returns its id."""
    ensure_schema(cnxn)
    penta = list(penta) if penta else [None] * 5
    cur = cnxn.execute(
        """INSERT INTO match_results (experiment_id, source, engine_a_id, engine_b_id, time_control, tc_category,
               wins, draws, losses, penta_ll, penta_ld, penta_dd, penta_wd, penta_ww)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (experiment_id, source, engine_a, engine_b, time_control, tc_category, wins, draws, losses, *penta))
    cnxn.commit()
    return cur.lastrowid


def record_snapshot(cnxn, *, source, engine_a, engine_b, snapshot, tc_category, time_control=None,
                    experiment_id=None):
    """Store a PentanomialSPRT snapshot (candidate = engine a)."""
    if not snapshot.get("games_played"):
        return None
    return record_match(cnxn, source=source, engine_a=engine_a, engine_b=engine_b,
                        wins=snapshot["candidate_wins"], draws=snapshot["candidate_draws"],
                        losses=snapshot["candidate_losses"], penta=snapshot["pentanomial"],
                        tc_category=tc_category, time_control=time_control, experiment_id=experiment_id)


def load_results(cnxn, tc_category, sources=DEFAULT_SOURCES):
    ensure_schema(cnxn)
    marks = ",".join("?" for _ in sources)
    return cnxn.execute(
        f"""SELECT engine_a_id, engine_b_id, SUM(wins), SUM(draws), SUM(losses) FROM match_results
            WHERE tc_category = ? AND source IN ({marks}) GROUP BY engine_a_id, engine_b_id""",
        (tc_category, *sources)).fetchall()


def _current_elo(cnxn, tc_category, engine):
    """The engine's rating on the category's current scale, or None.

    Its latest fitted rating in the anchor's group (component 0), else its
    engine_ratings value (e.g. from before fits were stored).
    """
    row = cnxn.execute("""SELECT r.elo FROM ratings r JOIN rating_fits f ON f.id = r.fit_id
                          WHERE f.tc_category = ? AND r.engine_id = ? AND r.component = 0
                          ORDER BY f.id DESC LIMIT 1""", (tc_category, engine)).fetchone()
    if row is None and tc_category in CATEGORIES:
        row = cnxn.execute(f"SELECT elo_{tc_category} FROM engine_ratings WHERE engine_id = ? "
                           f"AND elo_{tc_category} IS NOT NULL ORDER BY id DESC LIMIT 1", (engine,)).fetchone()
    return row[0] if row else None


def _choose_anchor(cnxn, tc_category, results):
    """(engine, Elo) fixing the scale of a fit.

    The previous fit's anchor and Elo while that engine still has games, so the
    scale stays put from fit to fit. Otherwise the oldest engine of the largest
    connected group (a stray old engine cannot move the main pool to another
    component), starting at its current rating so ratings continue the earlier
    scale, else ANCHOR_ELO.
    """
    valid = [(a, b) for a, b, w, d, l in results if a != b and (w or 0) + (d or 0) + (l or 0) > 0]
    played = {engine for pair in valid for engine in pair}
    if not played:
        return None, ANCHOR_ELO
    previous = cnxn.execute("SELECT anchor_engine_id, anchor_elo FROM rating_fits WHERE tc_category = ? "
                            "ORDER BY id DESC LIMIT 1", (tc_category,)).fetchone()
    if previous and previous[0] in played:
        return previous[0], previous[1]
    parent = {engine: engine for engine in played}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in valid:
        parent[find(a)] = find(b)
    groups = {}
    for engine in played:
        groups.setdefault(find(engine), []).append(engine)
    anchor = min(max(groups.values(), key=lambda g: (len(g), -min(g))))
    elo = _current_elo(cnxn, tc_category, anchor)
    return anchor, elo if elo is not None else ANCHOR_ELO


def fit(cnxn, tc_category, sources=DEFAULT_SOURCES, anchor=None, anchor_elo=None, prior_draws=1.0,
        write=True):
    """Fit every result of `tc_category`; returns (fit_id or None, FitResult), or (None, None) without games.

    Anchor: see _choose_anchor. An explicit anchor starts at anchor_elo, default its
    current rating (so the scale does not jump), else ANCHOR_ELO.
    """
    from utils.rating_fit import fit_ratings

    results = load_results(cnxn, tc_category, sources)
    if anchor is None:
        anchor, default_elo = _choose_anchor(cnxn, tc_category, results)
        if anchor is None:
            return None, None
        anchor_elo = anchor_elo if anchor_elo is not None else default_elo
    elif anchor_elo is None:
        current = _current_elo(cnxn, tc_category, anchor)
        anchor_elo = current if current is not None else ANCHOR_ELO
    result = fit_ratings(results, anchor=anchor, anchor_elo=anchor_elo, prior_draws=prior_draws)
    if not write:
        return None, result
    cur = cnxn.execute(
        """INSERT INTO rating_fits (tc_category, sources, anchor_engine_id, anchor_elo, prior_draws, games, engines)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (tc_category, ",".join(sources), anchor, anchor_elo, prior_draws,
         sum(r[2] + r[3] + r[4] for r in results), len(result.ratings)))
    fit_id = cur.lastrowid
    previous = {}
    for engine in sorted(result.ratings):
        component = result.component[engine]
        prev = previous.get(component)
        delta = result.diff(engine, prev) if prev is not None else (None, None)
        cnxn.execute(
            """INSERT INTO ratings (fit_id, engine_id, elo, se, games, points, component,
                   prev_engine_id, delta_prev, delta_prev_se) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fit_id, engine, result.ratings[engine], result.se(engine), result.games[engine],
             result.points[engine], component, prev, *delta))
        previous[component] = engine
    if tc_category in CATEGORIES:
        _mirror(cnxn, tc_category, result)
    cnxn.commit()
    return fit_id, result


def _mirror(cnxn, tc_category, result):
    """engine_ratings.elo_<category>: the fit's ratings (component 0 only; others are on another scale)."""
    elo_col, games_col = f"elo_{tc_category}", f"games_{tc_category}"
    for engine, elo in result.ratings.items():
        comparable = result.component[engine] == 0
        values = (round(elo, 1) if comparable else None, result.games[engine] if comparable else None)
        if cnxn.execute("SELECT 1 FROM engine_ratings WHERE engine_id = ?", (engine,)).fetchone():
            cnxn.execute(f"UPDATE engine_ratings SET {elo_col} = ?, {games_col} = ? WHERE engine_id = ?",
                         (*values, engine))
        else:
            cnxn.execute(f"INSERT INTO engine_ratings (engine_id, {elo_col}, {games_col}) VALUES (?, ?, ?)",
                         (engine, *values))


def latest_fit(cnxn, tc_category):
    ensure_schema(cnxn)
    row = cnxn.execute("SELECT id FROM rating_fits WHERE tc_category = ? ORDER BY id DESC LIMIT 1",
                       (tc_category,)).fetchone()
    return row[0] if row else None


def history(cnxn, fit_id):
    """Rows of a stored fit in engine order, with names and versions."""
    return cnxn.execute(
        """SELECT r.engine_id, e.name, e.version, e.build_type, r.elo, r.se, r.games, r.points, r.component,
                  r.prev_engine_id, r.delta_prev, r.delta_prev_se
           FROM ratings r LEFT JOIN engines e ON e.id = r.engine_id
           WHERE r.fit_id = ? ORDER BY r.component, r.engine_id""", (fit_id,)).fetchall()


def format_history(cnxn, fit_id, highlight=()):
    """The version history of a fit as a printable table (95% intervals)."""
    meta = cnxn.execute("SELECT tc_category, anchor_engine_id, anchor_elo, games, sources, fitted_utc "
                        "FROM rating_fits WHERE id = ?", (fit_id,)).fetchone()
    lines = [f"Ratings, {meta[0]} (fit {fit_id}, {meta[5]}): {meta[3]} games from {meta[4]}, "
             f"anchor engine {meta[1]} = {meta[2]:.0f}; +- = 95% interval",
             f"{'':2}{'id':>4} {'engine':<28} {'Elo':>7} {'+-':>6} {'games':>6} {'score':>6} "
             f"{'vs previous':>12} {'+-':>6}"]
    component = None
    for (engine, name, version, build_type, elo, se, games, points, comp, prev, delta, delta_se) \
            in history(cnxn, fit_id):
        if comp != component:
            if comp:
                lines.append(f"  -- component {comp}: never played the engines above; its scale is unrelated --")
            component = comp
        label = f"{name or '?'} {version or ''}" + (f" [{build_type}]" if build_type else "")
        mark = "*" if engine in highlight else " "
        vs = f"{delta:+12.1f} {Z95 * delta_se:6.1f}" if delta is not None else f"{'':>12} {'':>6}"
        lines.append(f"{mark:2}{engine:>4} {label[:28]:<28} {elo:7.1f} {Z95 * se:6.1f} {games:6d} "
                     f"{100 * points / games if games else math.nan:5.1f}% {vs}")
    return "\n".join(lines)


def backfill_sprt(cnxn):
    """Copy SPRT results not yet in match_results (source 'sprt'); returns the number added.

    Rows whose candidate and baseline are the same engine id (version-based lookups
    filed both builds under one row) carry no rating information and are skipped.
    """
    ensure_schema(cnxn)
    rows = cnxn.execute(
        """SELECT s.experiment_id, s.candidate_engine_id, s.baseline_engine_id, s.time_control, s.time_per_move,
                  s.depth_per_move, s.nodes_per_move, s.threads, s.candidate_wins, s.candidate_draws, s.candidate_losses
           FROM sprt s
           WHERE s.games_played > 0 AND s.candidate_engine_id != s.baseline_engine_id AND NOT EXISTS (
               SELECT 1 FROM match_results m WHERE m.source = 'sprt' AND m.experiment_id = s.experiment_id)""").fetchall()
    added = 0
    for experiment, cand, base, tc, time_per_move, depth, nodes, threads, wins, draws, losses in rows:
        try:
            cat = category(tc=tc, depth=depth, time=time_per_move, nodes=nodes, threads=threads or 1)
        except ValueError:
            continue
        record_match(cnxn, source="sprt", engine_a=cand, engine_b=base, wins=wins or 0, draws=draws or 0,
                     losses=losses or 0, tc_category=cat, time_control=tc, experiment_id=experiment)
        added += 1
    return added


def main(argv=None):
    from data.etl.paths import RAW_DB, connect_raw

    parser = argparse.ArgumentParser(description="Global Elo ratings over all stored rating matches")
    parser.add_argument("--db", default=None, help="raw database (default: CHESS_RAW_DB / data.etl.paths.RAW_DB)")
    sub = parser.add_subparsers(dest="command", required=True)
    fit_p = sub.add_parser("fit", help="fit and store ratings")
    fit_p.add_argument("--tc", nargs="+", default=None, help="categories (default: every category with results)")
    fit_p.add_argument("--include-sprt", action="store_true", help="also use SPRT results (biased by early stopping)")
    fit_p.add_argument("--anchor", type=int, default=None,
                       help="engine id fixed at --anchor-elo (default: the previous fit's anchor, else the oldest "
                            "engine of the largest connected group)")
    fit_p.add_argument("--anchor-elo", type=float, default=None,
                       help="default: the anchor's current rating, else 1500")
    fit_p.add_argument("--prior", type=float, default=1.0, help="virtual draws added to every pairing")
    fit_p.add_argument("--dry-run", action="store_true", help="print without storing")
    show_p = sub.add_parser("show", help="print the latest stored fit")
    show_p.add_argument("--tc", nargs="+", default=None)
    sub.add_parser("backfill-sprt", help="copy existing SPRT results into match_results")
    args = parser.parse_args(argv)

    cnxn = connect_raw(args.db or RAW_DB)
    ensure_schema(cnxn)
    categories = args.tc if getattr(args, "tc", None) else [
        r[0] for r in cnxn.execute("SELECT DISTINCT tc_category FROM match_results ORDER BY 1")]
    if args.command == "backfill-sprt":
        print(f"{backfill_sprt(cnxn)} SPRT result(s) added (used by fits only with --include-sprt)")
    elif args.command == "fit":
        sources = DEFAULT_SOURCES + (("sprt",) if args.include_sprt else ())
        for cat in categories:
            fit_id, result = fit(cnxn, cat, sources, args.anchor, args.anchor_elo, args.prior, write=not args.dry_run)
            if result is None:
                print(f"{cat}: no results from {', '.join(sources)}")
            elif fit_id is None:
                for engine in sorted(result.ratings):
                    print(f"{cat} engine {engine}: {result.ratings[engine]:.1f} +- {Z95 * result.se(engine):.1f}")
            else:
                print(format_history(cnxn, fit_id))
    else:
        for cat in categories:
            fit_id = latest_fit(cnxn, cat)
            print(format_history(cnxn, fit_id) if fit_id else f"{cat}: no stored fit")
    cnxn.close()


if __name__ == "__main__":
    main()
