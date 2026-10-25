# Testing: SPRT, strength estimation and SPSA

Every tool that plays games uses one runner, `utils/match_runner.py`, with
[fastchess](https://github.com/Disservin/fastchess) and the same statistics
(`utils/pentanomial.py`): SPRT, the strength estimator and SPSA directly; the
SMP ladder, net matches, tournaments and CMA-ES as fixed-length matches
(`utils/fixed_match.py`, below). So everything here (slots, pinning, jobs, the
environment recorded with every game) holds for all of them.

## How games are run

- **One game per core.** Each concurrent game gets a *slot*: one logical CPU of
  one physical core. A slot runs one fastchess process at a time (a *job*,
  `-concurrency 1`), so the two engines of a game take turns on that core and a
  game is always P-core vs P-core or E-core vs E-core, never mixed. Engines are
  kept on the slot three ways:
  1. the job's fastchess process starts with the slot's CPUs as its affinity
     mask, which the engines it starts inherit (so engines without an
     `Affinity` option, such as Stockfish, stay on the slot too);
  2. engines with a UCI `Affinity` option also get `option.Affinity=<cpus>`;
  3. on Windows the fastchess process starts suspended and is put in a job
     object of its own whose affinity limit (`JOB_OBJECT_LIMIT_AFFINITY`) is the
     slot. Every engine it starts joins that job and cannot leave the slot, not
     even an old Tomahawk build that pins itself to the P-cores at startup.

  Linux has no unprivileged equivalent of 3. There the runner checks every
  engine at the start and end of each game, and ten times a second while a
  job's engines start up. It moves a process found off its slot back. A game
  in which an engine was off its slot while playing is flagged (`core_mixed`),
  and SPRT excludes its pair. fastchess's own `-use-affinity` is not used: on
  Windows it places engines with CPU sets, which this check cannot see.
- **`--concurrency`** is how many games run at once (default: every physical
  core but one, divided by `--threads`). Cores are filled P-cores first, then
  E-cores spread over their L2 clusters. CPU 0 is left idle when its SMT sibling
  is free. Only a request for more games than physical cores puts a second
  game on P-cores' SMT siblings. `--threads N` (SPRT; the SMP ladder uses each
  step's higher thread count) gives each game N CPUs of one core type and both
  engines `Threads=N`. `--no-pin` leaves placement to the OS (macOS is never pinned).
- **Concurrent runs never share CPUs.** A run claims its slots' CPUs with lock
  files in the per-user temp folder (`%TEMP%\san-jacinto-cpus` on Windows,
  `/tmp/san-jacinto-cpus-<uid>` on Linux; `utils/cpu_slots.py`), and CPUs
  another run holds are skipped, so a second run of any tool takes other CPUs.
  The claims die with the Python process.
- **Jobs.** A slot plays a few opening pairs per fastchess process
  (`--pairs-per-job`), which spreads engine start-up cost. Openings come from the
  book shuffled once with a recorded seed (`--seed` reproduces a run). Each pair
  plays one opening with colours reversed (`-games 2 -rounds <pairs>`, book in
  sequential order). Jobs run with `-recover`: an engine that crashes is
  restarted and the job goes on.
- **Per-move limit:** exactly one of `--tc`, `--time` (seconds per move),
  `--depth` or `--nodes` (the strength estimator has no `--nodes`). With
  `--nodes`, every engine is first checked to honour `go nodes`; one that
  ignores it is refused.
- **Adjudication.** `--maxmoves N` (default 100; 150 in the strength estimator)
  ends a game as a draw after N moves per side played by the engines (book
  moves don't count). fastchess reports it as plain "Draw by adjudication", so
  it is counted as an adjudication. `--tb DIR` (SPRT) adds Syzygy adjudication
  (WDL files are enough).
- **Engines.** An engine path with a space can't be started by fastchess on
  Windows, so the runner passes its 8.3 short path. A volume without short names
  is refused before the run. A `.py` engine runs with the current Python
  interpreter.
- **Environment labels.** Every game records its CPU, core type, L2 cluster, the
  clock that CPU actually ran at during the game (mean/min/max MHz, sampled every
  second), utilisation, concurrency and power source. They go into
  `games.jsonl`, the `games` table and the dashboard, so speed differences can be
  told apart from compute differences.
- **Clean-up.** On Windows every fastchess process is in a kill-on-close job:
  if the runner dies, its engines die with it instead of stealing CPU from the
  next test. On Linux each job is its own process group, killed when the run
  stops.

fastchess is found in this order: `--fastchess PATH`, the `FASTCHESS` environment
variable, `libs/fastchess/fastchess.exe` (`libs/fastchess/fastchess` elsewhere),
then `fastchess` on PATH. The README explains how to get the binary.

## SPRT (`python -m tests.sprt`)

```powershell
python -m tests.sprt --engine-a new.exe --engine-b base.exe --tc 10+0.1 --elo0 0 --elo1 5
python -m tests.sprt --engine-a new.exe --engine-b base.exe --tc 10+0.1 --concurrency 10 --max-games 20000
python -m tests.sprt --engine-a new.exe --engine-b new.exe --param r_nmp=4 --tc 10+0.1   # a parameter change
python -m tests.sprt --engine-a new.exe --engine-b base.exe --nodes 20000 --no-db          # eval/NNUE change
python -m tests.sprt --engine-a new.exe --engine-b base.exe --tc 10+0.1 --threads 4 --tb D:\syzygy
python -m tests.sprt --retry-ingest logs/sprt_logs/<run>                                  # load kept telemetry
```

Defaults: H0 Elo 0, H1 Elo 10, alpha = beta = 0.05, at most 1000 games, the book
`bin/opening_books/8moves_v3.pgn` at 16 plies. `--param NAME=VALUE` sets a UCI
option for the candidate only (a `Threads` value there gives the candidate its
own thread count). A thread count an engine doesn't accept is refused before the
run.

Output:

- one line per game: number, core type, CPU, measured GHz, players, result,
  termination, duration, opening;
- a status line every `--status-every` seconds: pairs, LLR and progress to the
  bound, Elo and normalized Elo with 95% intervals, LOS, pentanomial counts,
  Elo per core type, games per minute, rough pairs to go;
- a final report: all of the above, candidate W/D/L overall and per colour,
  DD vs WL pairs, how much pairing reduced the variance, per core type
  (pairs, Elo, nElo, seconds per game, clock), terminations (time forfeits,
  crashes and illegal moves are flagged), and openings by family and by
  variation.

Statistics are pentanomial throughout: the unit is an opening pair (LL, LD,
DD/WL, WD, WW), which is what makes the LLR and the error bars correct when
openings favour one colour. Pairs enter the test in the order they started,
never the order they finished, so fast decisive games cannot bias an early stop.
A pair that did not finish, or in which an engine left its core, is excluded
(voided) and replayed. More than 20 exclusions, or 5% of the pairs `--max-games`
allows if that is more, stop the run, because something is then systematically
wrong.

Results go to `<results-dir>/<run>/` (`--results-dir`, default
`logs/sprt_results`): `summary.json`, `stats.jsonl` (one snapshot per tested
pair), `games.jsonl`, `games.pgn` (tagged with job, core type and CPU),
`openings.json/csv`, `openings_by_family.csv`, `config.json`, `sprt.png` (with
`--plot`) and, in `jobs/`, each job's fastchess command line and output, book and
PGN. Results per opening also accumulate across runs in `opening_register.csv`
in the parent of the results folder (`logs/opening_register.csv` by default).

The database upload (skip with `--no-db`; `--db` or `CHESS_RAW_DB` chooses the
file, which must exist) happens only when the test ended normally, not after an
interruption or a stop (too many excluded pairs, or jobs failing in a row; such a
stop ends the run with an error). Engines are identified by their build
(binary and net hash, see the README's "Engine identity"). It stores:

- the experiment and the `sprt` row with its pentanomial columns;
- every played game as a `games` row with `source = 'match'`: the runner's
  authoritative result, moves, opening and environment;
- the pairwise result in `match_results` (`source = 'sprt'`), unless both sides
  are the same build (a `--param` test of one binary): a build against itself
  would only distort the rating fits.

With `--log`, each side's engine writes its telemetry into its own folder,
`<logroot>/<run>/candidate` and `.../baseline` (`--logroot`, default
`logs/sprt_logs`). Its searches are linked to the played games by matching the
positions each search was made in. Only an engine game that matches no played
game gets a row of its own (`source = 'engine'`), so a game is never stored
twice. Each side loads in one transaction and is recorded in `telemetry_loads`,
so it can't be loaded twice; the side is matched to the played games before that
transaction opens, so other programs' writes don't wait for the matching. Loaded
logs are deleted. If loading fails, they
are kept: load them later with `--retry-ingest <logroot>/<run>`.

## Strength against Stockfish (`python -m tests.strength_estimator`)

```powershell
python -m tests.strength_estimator --engine tomahawk.exe --stockfish stockfish.exe
python -m tests.strength_estimator --engine tomahawk.exe --stockfish stockfish.exe --tc 10+0.1 --target-ci 40
```

Plays pairs against Stockfish with `UCI_LimitStrength` and adapts `UCI_Elo` as it
learns: half the pairs near the current estimate, a quarter each where the
candidate should score about 70% and 30%. It fits the level where the candidate
scores 50% together with the local scale of UCI_Elo (1.00 = calibrated), with
robust (sandwich) error bars, and stops when the 95% interval is within
`--target-ci` Elo. It reports the estimate, the scale, estimates per core type,
a per-level table and openings. In simulation the estimate is within a few Elo
of the truth; the 95% interval covers it about 95% of the time when UCI_Elo is
calibrated at the chosen time control and about 90% when the scale is strongly
compressed (the run stops as soon as the interval looks narrow enough, which
favours runs whose error was underestimated), so treat the interval as
slightly optimistic at short time controls.

Stockfish anchors UCI_Elo to CCRL Blitz (about +/-100 Elo). At short time
controls the scale is compressed (a run at 4+0.04 measured Stockfish 1800 as
equivalent to about 2000 on its own scale), so compare estimates only at the same
time control and hardware; the default is 60+0.6.

Stockfish comes from `--stockfish` or `$STOCKFISH` (extra options with
`--stockfish-option`; Threads=1 and Hash=16 by default). It has no `Affinity`
option, so its process mask, and on Windows the job's affinity limit, keep it on
the slot. Results go to `<results-dir>/<run>/` (default `logs/strength_results`):
`summary.json`, `games.jsonl`, `games.pgn`, `config.json` and `jobs/`. Nothing is
written to the database.

## SPSA (`python -m tuning.spsa_tuner`)

```powershell
python -m tuning.spsa_tuner --engine tomahawk.exe --tc 10+0.1 --pairs 8000
python -m tuning.spsa_tuner --engine tomahawk.exe --tc 10+0.1 --params r_nmp aspiration_window --verify-pairs 2000
python -m tuning.spsa_tuner --engine tomahawk.exe --tc 10+0.1 --resume logs/spsa_results/<run>/state.json
```

- One iteration is one opening pair, Plus (theta + c delta) against Minus
  (theta - c delta); pairs run asynchronously on every core. Both sides are the
  `--engine` build (there is no separate baseline). fastchess sets an engine's
  options once per process, so each job (`--pairs-per-job`, default 2) plays one
  perturbation and its pairs update theta together. `--pairs` defaults to 4000.
- Schedules follow Fishtest/OpenBench: `c_end` is the perturbation and `r_end`
  (default 0.002) the learning rate at the *end* of the tune, with A = 0.1 N,
  alpha 0.602, gamma 0.101. Rule of thumb: c_end about range/20, so that +-c_end
  changes strength by a few Elo.
- Integer parameters use stochastic rounding, so small perturbations are not
  lost to rounding.
- For each parameter the report gives the final and second-half-averaged value,
  how far it moved in units of c_end, and a signal t: |t| > 2 means the games
  clearly preferred one direction, |t| < 1 means the parameter barely matters at
  that time control.
- `--verify-pairs N` plays tuned vs start values afterwards (pentanomial Elo with
  95% CI), and the tuner prints the matching `tests.sprt --param ...` command.
- Parameters come from `DEFAULT_PARAMS` or `--params-file params.json` (same
  fields: value, min, max, c_end, integer, optional r_end). The tuner refuses to
  start if the engine does not declare every parameter as a UCI option, or if a
  range exceeds the option's own bounds (fastchess would skip such a value).
- `--nodes N` tunes at a fixed node count (the engine must honour `go nodes`);
  `--no-pin` leaves placement to the OS.
- Results go to `<results-dir>/<run>/` (default `logs/spsa_results`):
  `state.json`, `history.jsonl` (one line per job), `config.json`, `result.json`,
  `games.pgn` and `jobs/`. The state is saved after every job. `--resume
  <run>/state.json` continues the tune in the same folder (give the same engine
  and per-move limit); pairs that were running when it stopped are played again.
- `--optimizer adam` keeps the previous Adam variant as an experiment; plain SPSA
  is the default.

## Fixed-length matches (`utils/fixed_match.py`: SMP ladder, net matches, tournaments, CMA-ES)

```powershell
python -m tests.smp --engine tomahawk.exe --threads 1 2 4 --tc 10+0.1 --pairs 200
python -m nnue.registry match --engine X --net-a A.bin --net-b B.bin --nodes 50000 --pairs 200
python -m tests.tournament --engine engines/1.0.exe --tc blitz rapid --games 100 --log
python -m tuning.cmaes_tuner --engine tomahawk.exe --tc 10+0.1 --generations 50
```

- `run_fixed_match` plays one or more *pairings* (two engines and a number of
  opening pairs; the first engine is White in each pair's first game and its
  score is reported) on the match runner, with no early stopping: every Elo is a
  fixed-sample estimate. Each job plays a few pairs of one pairing; the idle slot
  gets the pairing furthest behind (ties: the one with fewer pairs on that core
  type), so pairings advance together and drift of the machine favours none.
  An engine name means one engine (command, arguments, options) in every
  pairing of a match: results, stored games and telemetry folders go by name, so
  two different engines under one name are refused.
- Each pairing has its own pentanomial state, fed in the order its pairs
  started, and a breakdown per core type. A pair with an unfinished game or a
  `core_mixed` game is excluded and its opening played again; as in the SPRT,
  too many exclusions or jobs failing in a row stop the match with an error.
  Ctrl-C keeps the results so far.
- The SMP ladder plays each step T_hi vs T_lo of one build on slots of T_hi CPUs
  of one core type (both sides get the slot as `Affinity`; on Windows it is the
  job's hard limit), default concurrency every physical core but one divided by
  T_hi (1 with `--no-pin`). The steps share one shuffled book, so each plays its
  own openings. `--option NAME=VALUE` values are checked against the engine's
  options first. Results: `<results-dir>/<run>/summary.json` (default
  `logs/smp_results`) with a folder per step.
- Net matches run one binary with two nets (`--net-option`, `--net-value`), store
  the result in `net_matches` and keep the run folder in `--results-dir`
  (default `logs/net_matches`); `--fastchess` chooses the binary and
  `--concurrency` defaults to every physical core but one.
- A tournament plays every pairing of a time control (the candidate against
  each opponent, or a round robin) as one match, interleaved; each time control
  is a run of its own, stored as one experiment in one transaction (all of it or
  nothing; `--results-dir`, default
  `logs/tournament_results`). `--log` gives every engine its own log folder
  `<logroot>/<run>/<engine name>/` and loads its telemetry linked to the games it
  played (`data/etl/telemetry.py`, shared with the SPRT). The run's ids
  (`engines.json`) are written as soon as it is stored, so telemetry that fails to
  load, or isn't loaded because the rating fit or the process failed, is kept for
  `--retry-ingest <logroot>/<run>`. `--pairs-per-job` and
  `--quiet` are the runner's.
- A CMA-ES generation is one match with a pairing per candidate (`Candidate<i>`
  vs `Baseline`); each pairing has its own copy of one shuffled book, so all
  candidates play the same openings. The CPUs are planned once per tune, and the
  run header's machine, slot and fastchess lines are printed for the first
  generation only (`run_fixed_match(..., show_setup=False)` after that).
  Generations whose candidates tie don't end the tune (pycma's flat-fitness
  stop is off). A tune that stops (a stopped match, Ctrl-C or any other error)
  saves its complete generations with the reason in `stopped`; the checkpoint's
  `cma_stop` holds pycma's own termination criteria. `--seed` is 1 to 2^32-1
  (pycma would take 0 as "seed from the clock").
  Results: `<results-dir>/<run>/` (default `logs/cmaes_results`) with a match
  folder per generation.
- A run folder holds `config.json`, `games.jsonl` (the SPRT's fields plus
  `pairing`, `pairing_pair`, `pairing_game`; `game`/`pair` number the whole run in
  start order), `games.pgn`, `jobs/` and `summary.json`.

## Unit tests

The same two commands run in CI (`.github/workflows/checks.yml`), from the
repository root:

```powershell
python -m unittest discover -s tests/unit -t .              # match tools, tuning, NNUE
python -m unittest discover -s data -t . -p "test_*.py"     # data pipeline
```

Keep `-t .`: without it, `data/` becomes the top-level package and the data
tests fail with import errors. pytest (not in `requirements.txt`) runs the
same tests with `python -m pytest data/tests tests/unit`; don't point it at
`data/` as a whole, where it would try to collect the schema checker
`data/databases/test_schema.py`. The dashboard has its own tests
(`npm test` in `data/dashboard`).

The `tests/unit` tests cover the pentanomial statistics, SPSA schedules and
convergence, the strength model, NNUE training orchestration, HPO, the match
runner's slot planning and fastchess commands, and the ingestion of runner
games, mostly without an engine. `tests/unit/mock_uci_engine.py` is a small python-chess
engine (with an `Affinity` option) for trying the tools end to end:
`python -m tests.sprt --engine-a tests/unit/mock_uci_engine.py --engine-b tests/unit/mock_uci_engine.py --param Strength=80 --tc 10+0.1 --no-db`.

Some tests skip themselves when something is missing:

- the tests that play real games with fastchess (match runner, SPRT, SPSA, fixed
  matches with the SMP ladder, net matches, a tournament and a CMA-ES tune): no
  fastchess binary (`libs/fastchess`, `$FASTCHESS` or PATH; CI has none), or no
  python-chess;
- job-object tests: not Windows; pinning tests: no CPU affinity (macOS); the
  best-checkpoint rename under a briefly held file (NNUE training): not Windows;
- the CMA-ES test: no `cma` package; the Stockfish-verified mistakes test: no
  Stockfish (`STOCKFISH_PATH`); the bullet manifest test: no `libs/bullet`
  submodule.
