## What I built

**1. Hosted dashboard (item 0)**
- **Pipeline in `data/`:** finished.
- Fixed a sign error in the engine-vs-Stockfish eval comparison, and removed a hard-coded Stockfish path.
- Added pre-aggregated tables, so the hosted "lite" snapshot is a few MB instead of about 113 MB.
- **Hosting:** `publish.py` exports a snapshot, attaches it to a GitHub release and runs the Pages workflow.
- **Dashboard:** rewritten.
- It only downloads what the open tab needs.
- Global engine, baseline and core-type filters; new Compare, Relationships, Hardware and Eval vs WDL tabs.
- Every opening view switches between family and variation.
- A light/dark design meant not to look generated.
- **Hardware labels:** every game records core type, CPU and measured clock, so compute differences can be separated from code differences.
- **Match-game ingestion:** the runner's games are the source of truth. Engine search logs are linked to them by matching positions, and that link was checked end to end (108 games, 216 engine logs, no mismatches).

**2. SPRT and a shared match runner (items 1 and 2)**
- **One core per game.** `match_runner.py` runs every game on a single core.
- Both engines start on that core and also get `setoption name Affinity`, so a game is always P vs P or E vs E.
- A guard throws out any pair where an engine leaves its core.
- The only setting is `--concurrency`, defaulting to physical cores − 1.
- Engines are killed if the runner dies.
- **Verified with a mock engine:** 0 off-core searches out of 5,931.
- **`sprt.py`:** pentanomial statistics everywhere (LLR, Elo, normalized Elo, LOS), also split per core type and per opening. Each game prints its core, CPU and clock.

**3. Opening register (item 5):** results by family and by name + variation, in the console, as CSV/JSON, and as a running file.

**4. Strength estimator (item 4):** `strength_estimator.py` finds the Stockfish `UCI_Elo` your engine matches.
- It also measures how compressed Stockfish's Elo scale is at short time controls.
- In simulation the estimate was within 7 Elo of the truth.

**5. SPSA:** `spsa_tuner.py` rewritten with Fishtest-style schedules, averaging, resume and a final verification match. On a mock engine it moved the parameter from 40 to 52, verified at +22.6 Elo.

**6. NNUE (item 3 and the tuning request)**
- **`train.py` curriculum:**
- Each stage now loads the previous stage's checkpoint.
- The learning-rate and WDL schedules run continuously across stages.
- A failed stage stops the run instead of the next stage silently training from scratch.
- **Best-so-far checkpoint:** saved on the wrapper side at whatever superbatch it happens. The temporary per-superbatch checkpoints are deleted afterwards.
- **`hpo.py`:** trains many settings briefly and only the best ones longer, with Optuna proposing new settings. It resumes after interruptions and handles curricula.

**7. Benchmarks:** reviewed, fixed four scripts (the build scripts now find the engine repo themselves), and gave you the rundown.

**8. Root moves vs Stockfish (today)**
- **Data:** Stockfish's top 5 lines at depth 10 for every position, plus each engine move's rank and centipawn loss. Results are stored once per position and survive full reloads.
- **Speed:** a small custom Stockfish client is 1.8× faster than python-chess, with identical results.
- **The tab:** now works in the hosted dashboard, with a phase selector and a "Largest misses" table.

**9. Tests and docs:** 37 unit, 58 data and 32 dashboard tests, all passing. A CI workflow runs them. Docs are in `testing.md`, `nnue.md`, `dashboard-data.md` and `public-dashboard.md`.

## Suggested next steps
1. **Commit in logical chunks and open a PR.** Before merging, delete or move the untracked `C:\Users\u407273\san-jacinto\benchmarks` folder in your main checkout, or git will refuse the merge.
2. **Let CI run once.** It has never run on Linux; everything here was tested on Windows.
3. **Try it on real engines and hardware:**
- a short SPRT with Tomahawk builds that support `Affinity`;
- a short curriculum training run;
- a first real `publish --refresh --watch`, which needs `gh` and a one-time Pages setting.
4. **Logging changes in the engine**, each of which would remove a workaround:
- log whether each root move's score is exact;
- give both engines one shared game id (today they differ, so logs are linked by matching positions);
- log games the engine didn't see finish.
5. **An automated test for the match runner.** It was only checked by hand with a mock engine.
6. **Core-type split for the remaining dashboard charts:** tree ply, timing, correlations, move order and openings.
7. **Small items:** add `optuna` to `requirements.txt` as optional; consider a deeper Stockfish "gold set" for key positions.

## Concerns
**Never run here**
- **Rust training scripts (highest risk).** The `v2.rs`/`v3.rs` changes were never compiled, because there's no Rust toolchain on this machine. They only use calls I read in the bullet fork's source.
- **GitHub publishing.** `gh` isn't installed here, so it has only been dry-run.
- **Engine `Affinity` handling.** Taken from the Tomahawk source and tested with a mock engine only.
- **Linux code paths.** Never run.## What I built

**Statistics and methodology**
- **Stockfish at depth 10 is itself a noisy reference.** For a strong engine, some "misses" may be Stockfish's mistakes.
- **Moves scored on their own aren't always consistent.** About 21% of moves outside the top 5 scored above Stockfish's 5th line; this shifts mean loss by only 2–4%.
- **The root "gap" chart understates the gap.** The engine only logs exact scores for its first move.
- **Stockfish stats count searches, not positions.** Repeated positions make the 95% intervals slightly optimistic.
- **The strength estimator's 95% interval is a little optimistic** at short time controls (about 88–92% coverage in simulation).
- **Laptop noise.** Clocks vary, and E-cores in the same cluster share cache. Clock is recorded per game, but run SPRTs plugged in on a high-performance power plan.

**Operational**
- **Tools live in my session folder, not the repo.** That covers cutechess, Stockfish and the Python environment; set `CUTECHESS_CLI` and `STOCKFISH` for your own installs.
- **The first Stockfish run analyses every position.** That's roughly 20 minutes per 100k positions, and it replaces the old depth-8 values.
- **Privacy.** Full snapshots publish positions and moves; lite snapshots don't.

**Code health**
- **The uncommitted change set is very large.** 63 tracked files (+8.6k/−5.5k lines), plus 42 new files or folders.
- **Agents wrote part of it.** It has been reviewed and tested, but give `match_runner.py`, `spsa_tuner.py` and `transform_positions.py` your own read-through.
- **The data test suite now takes about 100 s.** It starts real engine processes, and one test depends on timing, so it could be flaky on slow CI machines.

**Task completed:** Gave the session breakdown above: what was built, suggested next steps, and the main concerns (the Rust training scripts were never compiled, publishing has only been dry-run, CI has never run on Linux, and the uncommitted change set is very large).