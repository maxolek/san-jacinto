#!/usr/bin/env python3
"""NNUE net registry: connect each trained net to its training run, validation loss and Elo.

    python -m nnue.registry add <checkpoint dir or .bin> [--run-id ID] [--superbatch N] [--notes TEXT]
    python -m nnue.registry scan [--root nnue/checkpoints]     register every checkpoint found
    python -m nnue.registry report [--tc blitz]               val loss vs Elo, with correlation
    python -m nnue.registry match --engine X --net-a A.bin --net-b B.bin (--nodes N | --tc TC) [--pairs 200]

A net is identified by the sha256 of the file the engine loads (bullet's
quantised.bin, copied to the engine's weights folder). Engines registered with
provenance record the hash of the net they load (engines.net_sha256), which links
every net to the ratings (data/ratings.py) and SPRT results of the engines that
used it. Losses come from the logs the bullet fork writes into every checkpoint
(<run>-<N>/log.txt and val_log.txt, stitched across curriculum stages by
nnue/plot.py's loader, which works without matplotlib), the training settings from
train_config.json (written by train.py, which also registers the run's
checkpoints when it finishes).

train.py's best-validation folders (<run>/best/, best_<val set>/ and, for an
earlier run with the same net id, best*_previous_<time>/) hold hard links to the
files of checkpoint <run>-<N>, described by their best.json. A scan registers
such a folder as that checkpoint (run and superbatch from best.json) once the
checkpoint itself has been pruned, and skips it while the checkpoint exists.

The report answers "does lower validation loss mean more Elo?": per net, its
loss and the Elo of engines using it, with Spearman and Pearson correlations.

Net-vs-net matches play one engine binary with two nets through a UCI option the
engine declares on plain `uci` (fastchess skips undeclared options). Tomahawk
declares nnue_weight_file only for DEV engines and resolves a name under
bin/nnue_wgts/, so `match` refuses with that explanation until it is declared on
`uci` (--net-option and --net-value choose the option and how the net is named).
The games run on the match runner (utils/fixed_match.py: one game per core, each
with its core type, cpu and clock); the run folder goes to --results-dir (default
logs/net_matches).
"""
import argparse
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

NNUE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = NNUE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data.etl.paths import LOGS_DIR, RAW_DB, connect_raw  # noqa: E402
from data.etl.provenance import sha256_file  # noqa: E402
from data.etl.schema import ensure_schema  # noqa: E402

CHECKPOINTS = NNUE_ROOT / "checkpoints"
DEFAULT_BOOK = PROJECT_ROOT / "bin" / "opening_books" / "8moves_v3.pgn"
CHECKPOINT_NAME = re.compile(r"^(?P<run>.+)-(?P<superbatch>\d+)$")


def find_net_file(path):
    """The net file of a checkpoint (bullet's quantised.bin) or the file itself."""
    path = Path(path)
    if path.is_file():
        return path
    if (path / "quantised.bin").is_file():
        return path / "quantised.bin"
    bins = sorted(path.glob("*.bin"))
    return bins[0] if len(bins) == 1 else None


def load_losses(run_dir):
    """{superbatch: {"train": mean training loss, "val": validation loss}} of a training run.

    Read from the loss logs bullet writes into each checkpoint (<run>-<N>/log.txt,
    val_log.txt) with nnue/plot.py's loader, which stitches curriculum stages
    together. "val" is the superbatch's last validation, measured on the weights
    saved at its end (as train.py's best tracking does).
    """
    from nnue.plot import load_checkpoint_logs

    # Without batches_per_superbatch, (superbatch, batch) is placed at superbatch + batch / (max batch + 1).
    series = load_checkpoint_logs(str(run_dir))
    sums = {}
    for x, loss in series.get("train", []):
        total = sums.setdefault(int(x), [0.0, 0])
        total[0] += loss
        total[1] += 1
    losses = {sb: {"train": s / n} for sb, (s, n) in sums.items()}
    for x, loss in series.get("val", []):  # sorted: the superbatch's last validation wins
        losses.setdefault(int(x), {})["val"] = loss
    return losses


def load_best_info(folder):
    """best.json of a best-validation folder train.py keeps (best/, best_<val set>/, best*_previous_<time>/)."""
    try:
        info = json.loads((Path(folder) / "best.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) and isinstance(info.get("superbatch"), int) else None


def _run_info(net_file, run_id=None, superbatch=None):
    """Run id, superbatch, run folder and best.json (None for a checkpoint) of a net file:
    <run dir>/<run>-<superbatch>/quantised.bin, or a best folder <run dir>/best*/quantised.bin."""
    folder = net_file.parent
    best = load_best_info(folder)  # first: a best_<val set> folder name can look like <run>-<N>
    if best is not None:
        run_id = run_id or best.get("net_id")
        superbatch = superbatch if superbatch is not None else best["superbatch"]
        return run_id, superbatch, folder.parent, best
    match = CHECKPOINT_NAME.match(folder.name)
    if match:
        run_id = run_id or match["run"]
        superbatch = superbatch if superbatch is not None else int(match["superbatch"])
        return run_id, superbatch, folder.parent, None
    return run_id, superbatch, None, None


def _same_loss(a, b):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)


def register_net(cnxn, path, run_id=None, superbatch=None, notes=None):
    """Register (or refresh) a net; returns its id."""
    ensure_schema(cnxn)
    net_file = find_net_file(path)
    if net_file is None:
        raise FileNotFoundError(f"no net file in {path}")
    digest = sha256_file(net_file)
    run_id, superbatch, run_dir, best = _run_info(net_file, run_id, superbatch)
    losses = load_losses(run_dir) if run_dir else {}
    config_path = run_dir / "train_config.json" if run_dir else None
    config = config_path.read_text(encoding="utf-8") if config_path and config_path.is_file() else None
    if best is not None and not _same_loss(losses.get(superbatch, {}).get("val"), best.get("val_loss")):
        # The run's logs do not show this state (an archived best of an earlier run with
        # this net id, whose logs and train_config.json the later run replaced): only its
        # best.json describes it, and its loss is by definition that run's best.
        losses = {superbatch: {"val": best["val_loss"]}} if best.get("val_loss") is not None else {}
        config = None
    at = losses.get(superbatch, {})
    val = {sb: v["val"] for sb, v in losses.items() if "val" in v}
    best_sb = min(val, key=val.get) if val else None
    values = (run_id, superbatch, str(net_file), at.get("val"), at.get("train"),
              val.get(best_sb), best_sb, config, notes)
    row = cnxn.execute("SELECT id FROM nets WHERE net_sha256 = ?", (digest,)).fetchone()
    if row:
        cnxn.execute("""UPDATE nets SET run_id=COALESCE(?, run_id), superbatch=COALESCE(?, superbatch),
                            file_path=?, val_loss=COALESCE(?, val_loss), train_loss=COALESCE(?, train_loss),
                            best_val_loss=COALESCE(?, best_val_loss), best_val_superbatch=COALESCE(?, best_val_superbatch),
                            train_config=COALESCE(?, train_config), notes=COALESCE(?, notes) WHERE id=?""",
                     (*values, row[0]))
        net_id = row[0]
    else:
        net_id = cnxn.execute("""INSERT INTO nets (net_sha256, run_id, superbatch, file_path, val_loss, train_loss,
                                     best_val_loss, best_val_superbatch, train_config, notes)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (digest, *values)).lastrowid
    cnxn.commit()
    return net_id


def _copy_of_existing_checkpoint(net_file):
    """True for the net of a best folder whose checkpoint (best.json's "checkpoint") still holds the same net."""
    best = load_best_info(net_file.parent)
    if best is None or not best.get("checkpoint"):
        return False
    source = net_file.parent.parent / Path(str(best["checkpoint"])).name / net_file.name
    if not source.is_file():
        return False
    try:
        if source.samefile(net_file):  # train.py hard-links best folders
            return True
    except OSError:
        pass
    return sha256_file(source) == sha256_file(net_file)


def scan(cnxn, root=CHECKPOINTS, run_id=None):
    """Register every checkpoint under root (one training run's folder, or all of them).

    A best folder is registered as the checkpoint it was copied from only once that
    checkpoint is gone; while it exists, the checkpoint's own entry is the same net.
    """
    pattern = f"{run_id}/*/quantised.bin" if run_id else "**/quantised.bin"
    ids = []
    for net_file in sorted(Path(root).glob(pattern)):
        if not _copy_of_existing_checkpoint(net_file):
            ids.append(register_net(cnxn, net_file))
    return ids


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else None


def spearman(x, y):
    return pearson(_ranks(x), _ranks(y)) if len(x) >= 3 else None


def report_rows(cnxn, tc_category):
    """Per net: losses, the engines using it and their latest rating in the category, SPRT and net-match results."""
    from data.ratings import latest_fit

    ensure_schema(cnxn)
    fit_id = latest_fit(cnxn, tc_category)
    nets = cnxn.execute("SELECT id, run_id, superbatch, val_loss, best_val_loss, net_sha256 FROM nets ORDER BY id")
    rows = []
    for net_id, run_id, superbatch, val_loss, best_val, digest in nets.fetchall():
        engines = cnxn.execute(
            """SELECT e.id, e.version, r.elo, r.se, r.games FROM engines e
               LEFT JOIN ratings r ON r.engine_id = e.id AND r.fit_id = ? AND r.component = 0
               WHERE e.net_sha256 = ? ORDER BY e.id""", (fit_id, digest)).fetchall()
        rated = [e for e in engines if e[2] is not None]
        best = max(rated, key=lambda e: e[4]) if rated else None  # the most-played engine with this net
        sprt = cnxn.execute(
            """SELECT COUNT(*), SUM(s.result = 'pass') FROM sprt s JOIN engines e ON e.id = s.candidate_engine_id
               WHERE e.net_sha256 = ?""", (digest,)).fetchone()
        rows.append(dict(net=net_id, run=run_id, superbatch=superbatch, val_loss=val_loss, best_val_loss=best_val,
                         engines=[e[0] for e in engines], elo=best[2] if best else None,
                         se=best[3] if best else None, games=best[4] if best else None,
                         sprts=sprt[0] or 0, sprt_passes=sprt[1] or 0))
    return rows, fit_id


def format_report(rows, tc_category, fit_id):
    lines = [f"NNUE nets vs Elo ({tc_category}, rating fit {fit_id or 'none'}; Elo of the most-played engine "
             f"using the net, +- 95%)",
             f"{'net':>4} {'run':<24} {'sb':>6} {'val loss':>10} {'best val':>10} {'engines':<12} "
             f"{'Elo':>8} {'+-':>6} {'SPRT pass':>10}"]
    for r in sorted(rows, key=lambda r: (r["val_loss"] is None, r["val_loss"] or 0)):
        loss = f"{r['val_loss']:.6f}" if r["val_loss"] is not None else "-"
        best = f"{r['best_val_loss']:.6f}" if r["best_val_loss"] is not None else "-"
        elo = f"{r['elo']:8.1f} {1.96 * r['se']:6.1f}" if r["elo"] is not None else f"{'-':>8} {'':>6}"
        lines.append(f"{r['net']:>4} {(r['run'] or '?')[:24]:<24} {r['superbatch'] or '-':>6} {loss:>10} {best:>10} "
                     f"{','.join(map(str, r['engines']))[:12] or '-':<12} {elo} {r['sprt_passes']}/{r['sprts']:<8}")
    pairs = [(r["val_loss"], r["elo"]) for r in rows if r["val_loss"] is not None and r["elo"] is not None]
    if len(pairs) >= 3:
        x, y = zip(*pairs)
        lines.append(f"correlation of val loss with Elo over {len(pairs)} nets: Spearman {spearman(x, y):+.2f}, "
                     f"Pearson {pearson(x, y):+.2f} (negative = lower loss, more Elo)")
    else:
        lines.append(f"{len(pairs)} net(s) have both a val loss and a rating; correlations need at least 3")
    return "\n".join(lines)


def play_match(engine, net_a, net_b, *, limits, label, pairs, concurrency=None, net_option="nnue_weight_file",
               net_value="path", fastchess=None, book=None, db=None, pin=True, results_dir=None):
    """Fixed-length match: the same engine with net A (Candidate) vs net B; stored in net_matches.

    Played on the match runner (utils/fixed_match.py): every game pinned to its own core, with its
    core type, cpu and clock in <results_dir>/<run>/games.jsonl (default logs/net_matches);
    concurrency None is the runner's default (every physical core but one).
    """
    from utils.fastchess_match import FASTCHESS
    from utils.fixed_match import Pairing, run_fixed_match
    from utils.match_runner import EngineSpec, probe_uci

    base = EngineSpec.from_path("Candidate", engine).resolve()
    if net_option not in probe_uci(base.cmd, base.args)["options"]:
        raise ValueError(
            f"{engine} does not declare the UCI option '{net_option}' on `uci`, so fastchess cannot load a net "
            "through it. Tomahawk declares nnue_weight_file only for DEV engines and takes a name under "
            "bin/nnue_wgts/: declare it on `uci` (and accept a path) to use net matches.")

    def value(net):
        net = Path(net).resolve()
        return str(net) if net_value == "path" else net.stem

    def side(name, net):  # one binary, two nets: told apart by name
        return EngineSpec(name, base.cmd, base.args, {"Threads": 1, net_option: value(net)}, workdir=base.workdir,
                          pin_option=base.pin_option)

    key = "net A vs net B"
    result = run_fixed_match([Pairing(side("Candidate", net_a), side("Baseline", net_b), pairs, key=key)],
                             tc_args=limits, book=book or DEFAULT_BOOK, plies=16, concurrency=concurrency, pin=pin,
                             fastchess=fastchess or FASTCHESS, results_dir=results_dir or LOGS_DIR / "net_matches",
                             extra_args=["-maxmoves", "100"], label="NET", time_control=label,
                             meta=dict(net_a=str(net_a), net_b=str(net_b), limit=label, engine=str(engine)))
    stats = result["pairings"][key]
    cnxn = connect_raw(db or RAW_DB)
    try:
        ids = [register_net(cnxn, net) for net in (net_a, net_b)]
        # net_matches has no penta_wl: penta_dd holds the combined DD + WL bin (stats["pentanomial"][2]).
        cnxn.execute("""INSERT INTO net_matches (net_a_id, net_b_id, engine_path, time_control, wins, draws, losses,
                            penta_ll, penta_ld, penta_dd, penta_wd, penta_ww, elo, elo_low, elo_high)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                     (*ids, str(engine), label, stats["candidate_wins"], stats["candidate_draws"],
                      stats["candidate_losses"], *stats["pentanomial"], stats["elo_diff"], stats["elo_low"],
                      stats["elo_high"]))
        cnxn.commit()
    finally:
        cnxn.close()
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description="NNUE net registry: nets, validation loss and Elo")
    parser.add_argument("--db", default=None, help="raw database (default: CHESS_RAW_DB / data.etl.paths.RAW_DB)")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="register one net (checkpoint dir or .bin)")
    add.add_argument("path")
    add.add_argument("--run-id")
    add.add_argument("--superbatch", type=int)
    add.add_argument("--notes")
    scan_p = sub.add_parser("scan", help="register every checkpoint under --root")
    scan_p.add_argument("--root", default=str(CHECKPOINTS))
    scan_p.add_argument("--run-id", default=None, help="only this training run")
    rep = sub.add_parser("report", help="validation loss vs Elo")
    rep.add_argument("--tc", default="blitz", help="rating category (default blitz)")
    m = sub.add_parser("match", help="fixed-length match between two nets in one engine")
    m.add_argument("--engine", required=True)
    m.add_argument("--net-a", required=True, help="candidate net file")
    m.add_argument("--net-b", required=True, help="baseline net file")
    m.add_argument("--nodes", type=int, default=None, help="nodes per move (the usual choice for nets)")
    m.add_argument("--tc", default=None)
    m.add_argument("--depth", type=int, default=None)
    m.add_argument("--pairs", type=int, default=200)
    m.add_argument("--concurrency", type=int, default=None,
                   help="Concurrent games (default: every physical core but one); each game gets its own core")
    m.add_argument("--net-option", default="nnue_weight_file")
    m.add_argument("--net-value", choices=["path", "stem"], default="path",
                   help="pass the net's full path, or its file name without extension")
    m.add_argument("--book", default=str(DEFAULT_BOOK))
    m.add_argument("--no-pin", action="store_true", help="Leave the placement of games to the OS")
    m.add_argument("--fastchess", default=None,
                   help="fastchess executable (default: utils.fastchess_match.FASTCHESS: $FASTCHESS, "
                        "libs/fastchess, or fastchess on PATH)")
    m.add_argument("--results-dir", default=None,
                   help="where the match's run folder goes (default logs/net_matches)")
    args = parser.parse_args(argv)

    if args.command == "match":
        from utils.fastchess_match import limit_args, limit_label
        if args.pairs < 1:
            parser.error("--pairs must be positive")
        limits = limit_args(args.depth, None, args.tc, args.nodes)
        label = limit_label(args.depth, None, args.tc, args.nodes)
        stats = play_match(args.engine, args.net_a, args.net_b, limits=limits, label=label, pairs=args.pairs,
                           concurrency=args.concurrency, net_option=args.net_option, net_value=args.net_value,
                           book=args.book, db=args.db, pin=not args.no_pin, fastchess=args.fastchess,
                           results_dir=args.results_dir)
        print(f"net A vs net B, {label}: {stats['pairs']} pairs, Elo {stats['elo_diff']:+.1f} "
              f"[{stats['elo_low']:+.1f}, {stats['elo_high']:+.1f}]")
        return
    cnxn = connect_raw(args.db or RAW_DB)
    try:
        if args.command == "add":
            print(f"net id {register_net(cnxn, args.path, args.run_id, args.superbatch, args.notes)}")
        elif args.command == "scan":
            print(f"registered {len(scan(cnxn, args.root, args.run_id))} net(s)")
        else:
            rows, fit_id = report_rows(cnxn, args.tc)
            print(format_report(rows, args.tc, fit_id))
    finally:
        cnxn.close()


if __name__ == "__main__":
    main()
