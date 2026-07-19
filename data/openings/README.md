How to populate a comprehensive ECO/openings database

1) Obtain a canonical ECO PGN file (many public mirrors host `eco.pgn`).
   Save it as `eco.pgn` somewhere on disk — for example, your project `data/openings` directory.

2) Use the builder script to convert the PGN into `eco_db.json`.

Recommended (safe) ways to run the builder:

- If `eco.pgn` is already at `data/openings/eco.pgn` (project root):

```bash
# run from project root
python data/openings/build_from_pgn.py
```

- If the PGN is elsewhere, pass the absolute path and an explicit output path:

```bash
python data/openings/build_from_pgn.py --pgn /full/path/to/eco.pgn --out data/openings/eco_db.json
```

Notes and important caution:
- Do NOT run `mkdir -p data/openings` while your current working directory is already `data/` — that creates a nested `data/data` path. Use absolute paths or run commands from the project root instead.
- The builder requires `python-chess` (install with `pip install chess`).
- After creating/updating `data/openings/eco_db.json`, re-run the pipeline to repopulate `game_stats.opening`:

```bash
python data/run_analytics_pipeline.py
```
