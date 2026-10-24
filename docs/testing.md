# Testing: SPRT, strength estimation and SPSA

All three run games through the same runner (`utils/match_runner.py`) and use the
same statistics (`utils/pentanomial.py`).

## How games are run

- **One game per core.** Each concurrent game gets one physical core: its
  cutechess process starts with that core's CPU as its affinity (engines inherit
  it), and engines with a UCI `Affinity` option are also told
  `setoption name Affinity value <cpu>`. The two engines of a game take turns on
  that core, so a game is always P-core vs P-core or E-core vs E-core, never
  mixed. The OS never schedules a process outside its mask; as a safety net the
  runner checks every engine at the start and end of each game, moves back any
  process found off its core, and SPRT excludes a pair if that happens mid-game
  (and stops if it keeps happening).
- **The only setting is `--concurrency`**: how many cores to use (default: every
  physical core but one). Cores are filled P-cores first, then E-cores spread
  over their L2 clusters. Asking for more than the physical cores puts a second
  game on P-cores' SMT siblings.
- **Jobs.** A core plays a few opening pairs per cutechess process
  (`--pairs-per-job`), which spreads engine start-up cost. Openings come from the
  book shuffled once with a recorded seed (`--seed` reproduces a run), each pair
  plays one opening with colours reversed.
- **Environment labels.** Every game records its CPU, core type, L2 cluster, the
  clock that CPU actually ran at during the game (mean/min/max MHz, sampled every
  second), utilisation, concurrency and power source. They go into
  `games.jsonl`, the `games` table and the dashboard, so speed differences can be
  told apart from compute differences.
- **Clean-up.** On Windows every cutechess process is in a kill-on-close job:
  if the runner dies, its engines die with it instead of stealing CPU from the
  next test.

Use `CUTECHESS_CLI` (or `--cutechess-cli`) if cutechess-cli is not on PATH or in
the standard install folder.

## SPRT (`python -m tests.sprt`)

```powershell
python -m tests.sprt --engine-a new.exe --engine-b base.exe --tc 10+0.1 --elo0 0 --elo1 5
python -m tests.sprt --engine-a new.exe --engine-b base.exe --tc 10+0.1 --concurrency 10 --max-games 20000
python -m tests.sprt --engine-a new.exe --engine-b new.exe --param r_nmp=4 --tc 10+0.1   # a parameter change
```

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

Results go to `logs/sprt_results/<run>/`: `summary.json`, `stats.jsonl` (one
snapshot per tested pair), `games.jsonl`, `games.pgn` (tagged with job, core type
and CPU), `openings.json/csv`, `openings_by_family.csv`, `config.json` and the raw
cutechess logs in `jobs/`. Results per opening also accumulate across runs in
`logs/opening_register.csv`.

The database upload (skip with `--no-db`; `--db` or `CHESS_RAW_DB` chooses the
file) stores the SPRT with its pentanomial columns and every played game with its
authoritative result, moves, opening and environment. With `--log`, the engines'
own search telemetry is linked to those games by matching the positions each
search was made in.

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

## SPSA (`python -m tuning.spsa_tuner`)

```powershell
python -m tuning.spsa_tuner --engine tomahawk.exe --tc 10+0.1 --pairs 8000
python -m tuning.spsa_tuner --engine tomahawk.exe --tc 10+0.1 --params r_nmp aspiration_window --verify-pairs 2000
python -m tuning.spsa_tuner --engine tomahawk.exe --tc 10+0.1 --resume logs/spsa_results/<run>/state.json
```

- One iteration is one opening pair, Plus (theta + c delta) against Minus
  (theta - c delta); pairs run asynchronously on every core.
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
  start if the engine does not declare every parameter as a UCI option.
- `--optimizer adam` keeps the previous Adam variant as an experiment; plain SPSA
  is the default.

## Unit tests

```powershell
python -m unittest discover -s tests/unit -t .
```

They cover the pentanomial statistics, SPSA schedules and convergence, the
strength model, NNUE training orchestration, HPO and the ingestion of runner
games, without an engine. `tests/unit/mock_uci_engine.py` is a small python-chess
engine (with an `Affinity` option) for trying the tools end to end:
`python -m tests.sprt --engine-a tests/unit/mock_uci_engine.py --engine-b tests/unit/mock_uci_engine.py --param Strength=80 --tc 10+0.1 --no-db`.
