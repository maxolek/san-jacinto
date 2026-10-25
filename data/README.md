# data/

The data side of San Jacinto. It turns engine telemetry into an analytics
database and a public dashboard snapshot.

```text
engine logs (JSONL)            match-runner runs (games.jsonl + PGN, tests/sprt.py)
        \                         /
         etl/  ingest.py, match_games.py
                   |
                   v
   raw SQLite  (chess.db)                       databases/init_raw_db.py, migrate_schema.py
     also: match_results / rating_fits / ratings (ratings.py),
           nets / net_matches (nnue/registry.py)
                   |
                   v  databases/run_analytics_pipeline.py
   1 load_analytics        raw tables -> analytics tables (incremental or --full)
   2 transform_positions   position_features, Stockfish's top lines per position
                           (stockfish_moves, cached) and the sf_* columns
   3 transform_search      SQL views (transforms/sql/*.sql)
   4 aggregates            agg_* tables + search_sample
   5 validate, test_schema data quality and schema checks
                   |
                   v
   analytics DuckDB  (chess_analytics.duckdb)
                   |
                   v  databases/export_public.py
   dashboard/public/data/manifest.json + snapshots/<sha256>.duckdb
                   |
                   v
   dashboard/  (DuckDB-WASM + Mosaic, static site)
```

## Commands

Run these from the repository root. Install the dependencies first with
`pip install -r requirements.txt`.

The database paths default to `CHESS_RAW_DB` and `CHESS_ANALYTICS_DB`. If
those aren't set, `data/etl/paths.py` falls back to `F:/databases/` on Windows
and `~/Documents/databases/` elsewhere.

```sh
# raw database: create it, or add new columns to an existing one
python -m data.databases.init_raw_db [--db FILE_OR_DIR]   # also runs the migrations; a directory (or a path without an extension) means <dir>/chess.db
python -m data.databases.migrate_schema --dry-run         # preview; without --dry-run: apply
```

The base tables in `init_raw_db.py` are frozen: every schema change is a new
migration in `databases/migrate_schema.py` (its docstring has the rules;
[docs/databases.md](../docs/databases.md#how-the-schema-changes) lists the
migrations). Programs that write to the raw database apply pending migrations
themselves.

```sh
# analytics database (Stockfish is optional: --stockfish PATH, $STOCKFISH or PATH)
python -m data.databases.run_analytics_pipeline            # incremental
python -m data.databases.run_analytics_pipeline --full     # reload everything
python -m data.databases.run_analytics_pipeline --skip-load --skip-stockfish
python -m data.databases.run_analytics_pipeline --stockfish-depth 12 --stockfish-multipv 5 --stockfish-workers 6

# single steps
python -m data.databases.load_analytics [--full]
python -m data.transforms.transform_positions [--stockfish PATH | --skip-stockfish] [--depth 10] [--multipv 5] [--workers N] [--limit N]
python -m data.transforms.transform_search
python -m data.transforms.aggregates
python -m data.transforms.validate
python -m data.databases.test_schema [--db FILE]

# public snapshot for the dashboard (keeps the newest 3; --full adds per-search tables)
python -m data.databases.export_public --archive /tmp/analytics-public.tar.gz
python -m data.databases.publish [--refresh]     # export, GitHub release and Pages deploy in one step (needs gh)

# dashboard on http://localhost:3000 (Node.js 18+; installs its npm packages on first use)
python -m data.dashboard

# ratings over match_results (tournament and manual results; SPRT only with --include-sprt)
python -m data.ratings fit [--tc blitz] [--include-sprt]
python -m data.ratings show --tc blitz

# tests (synthetic data, no Stockfish needed); keep -t .
python -m unittest discover -s data -t . -p "test_*.py"
```

## Folders

| Folder | Contents |
|---|---|
| `etl/` | Log parsing and ingestion into the raw database, and path configuration (`paths.py`) |
| `databases/` | Raw schema and migrations, the loader, the pipeline runner, the public export and its allow-list (`public_schema.py`), and the schema check |
| `transforms/` | Position features, SQL views, pre-aggregated tables and data-quality checks |
| `tests/` | Unit tests with a synthetic log generator (`fixtures.py`) and a fake UCI engine |
| `dashboard/` | The static dashboard. See its README. |
| `openings/` | The ECO opening database |
| `ratings.py` | Global Elo fits over `match_results` (`rating_fits`, `ratings`, mirrored into `engine_ratings`) |

More detail:

- [docs/dashboard-data.md](../docs/dashboard-data.md): the tables and columns
  the dashboard reads.
- [docs/databases.md](../docs/databases.md): the schemas.
- [docs/public-dashboard.md](../docs/public-dashboard.md): publishing.
