# NNUE training and hyperparameter search

## Training (`nnue/train.py`)

The training scripts `nnue/v2.rs` and `nnue/v3.rs` are cargo examples of the
bullet fork in `libs/bullet`: the fork registers them in
`libs/bullet/crates/bullet_lib/Cargo.toml`,

```toml
[[example]]
name = "v3"
path = "../../../../nnue/v3.rs"
```

so they build against the fork with its own `Cargo.lock`. There is no cargo
package in `nnue/`; a new script is registered the same way in the fork.
`train.py --example NAME` runs `cargo run --release --example NAME` in
`libs/bullet` (any NAME other than a script in `nnue/` is one of bullet's own
examples). Settings reach the script as environment variables.

```powershell
python nnue/train.py --example v3 --features cuda --train-data train.binpack --val-data val.binpack
python nnue/train.py --example v3 --net-id big --train-data s1.binpack s2.binpack s3.binpack --superbatches 200 400 800 --val-data val.binpack
python nnue/train.py --example v3 --net-id big --superbatch-start 601 --superbatches 200 --train-data s3.binpack   # resume from big-600

# the same script without the wrapper (no plot, best tracking or registry); settings as environment variables
cd libs/bullet
cargo run --release --example v3 --features cuda
```

### Defaults

The scripts own their defaults:

| | `v2` | `v3` |
|---|---|---|
| WDL blend (`wdl_start` -> `wdl_end`) | 0.25 -> 0.25 | 0.0 -> 0.0 |
| net id | `1024x16x32_25wdl` | `1024x16x32_0wdl` |
| training data | `data/T60T70wIsRightFarseer.binpack` | `F:/training_data/final_sets/2_LeelaFarsser-T78juntoaugT79marT80dec.binpack` |
| validation data | `data/test80-2023-12-dec-2tb7p.min-v2.v6.binpack` | `F:/training_data/validation_sets/fishpack32.binpack` |
| validation | 4,194,304 positions, end of every superbatch | 10,000,000 positions, end of every superbatch |
| training filter | ply >= 16, not in check, \|score\| <= 10000, normal non-capture moves | same |
| LR | cosine 0.001 -> 0.001 * 0.3^5, 10-batch warmup | same |
| length | 800 superbatches of 6104 batches of 16,384 | same |

Relative data paths are relative to `libs/bullet`, where cargo runs. The
validation positions are sampled once from the validation file at startup
(`TestDataset::positions`) and evaluated after the last batch of every
superbatch (`freq(0)`), so each checkpoint's validation loss is measured on the
weights it saved.

`train.py` needs the WDL blend (best tracking, plot) and the net id (output
folder) itself. Unless given (`--wdl-start`/`--wdl-end`, `--net-id`, or the
`wdl_start`/`wdl_end`/`net_id` environment variables), it reads them from the
script's source, which is why those defaults must stay literals, and always
passes them on: `train.py --example v3` trains exactly what
`cargo run --example v3` does. Every other setting it is not given is left to
the script.

### Curriculum

The fork trains one dataset over one superbatch range per `run()`. The wrapper
turns several datasets into stages, one process each:

```
stage 1  superbatches 1-200    data A   random init          -> saves <net>-200
stage 2  superbatches 201-600  data B   loads <net>-200      -> saves <net>-600
stage 3  superbatches 601-1400 data C   loads <net>-600      -> saves <net>-1400
```

- The training script loads the previous stage's checkpoint when
  `superbatch_start > 1`: weights and AdamW moments by default, weights only
  with `--reset-optimiser`.
- The LR cosine and the WDL blend are both indexed by the global superbatch and
  end at the curriculum's last superbatch (`CurriculumWDL` in the scripts;
  bullet's own `LinearWDL` would restart its taper every stage).
- Before a stage starts, the wrapper checks that the checkpoint it continues from
  exists; after it ends, that the stage wrote its final checkpoint. A failed stage
  stops the curriculum instead of the next stage silently training from scratch.
- Dataset paths are resolved to absolute paths (as given, else under
  `libs/bullet/data`), because cargo runs in `libs/bullet`.

### Live plot

The plot reads the loss logs bullet writes into each checkpoint
(`log.txt`, `val_log.txt`; `train.txt` is accepted too) and stitches the
stages together, so it advances whenever a checkpoint is saved.
`python nnue/plot.py nnue/checkpoints/<net>` plots a finished run.

### Best validation state

With a constant WDL blend (`--wdl-start` = `--wdl-end`, including 0; both
scripts' defaults are constant), the wrapper keeps the state with the
lowest validation loss in `<net>/best/` (full checkpoint plus `best.json` with
the superbatch and loss) at whichever superbatch it happens, not only at saved
ones. To do that the trainer is run with `save_rate=1`; every temporary
checkpoint is deleted as soon as the next one is saved, so what remains is
exactly your `--save-rate` checkpoints, each stage's last checkpoint (needed to
continue) and `best/` (hard links, no extra space). It also cleans up after a
failed or interrupted stage.
With a changing WDL blend, validation losses at different superbatches are
measured against different targets, so tracking is off. `--no-best` turns it
off too. Separate validation sets per stage get separate `best_<set>/` folders.
A `best/` from a run with other settings (net id, validation set or WDL) is
kept as `best_previous_<time>/`.

## Net registry (`nnue/registry.py`)

The registry connects each trained net to its run, its losses and the Elo of
the engines that use it. A net is identified by the sha256 of its
`quantised.bin`; rows go to the `nets` table of the raw database
(`CHESS_RAW_DB`, else the platform default in `data/etl/paths.py`, or `--db`).
Per net it stores the run id and superbatch, the validation and training loss
at that superbatch, the run's best validation loss, and the run's
`train_config.json` (written by `train.py`).

- `train.py` registers a run's checkpoints when it finishes; `--no-register`
  turns that off. `hpo.py` trials always pass it, since their runs are deleted
  after scoring.
- Losses are read from the logs bullet writes into every checkpoint
  (`log.txt`, `val_log.txt`), stitched across curriculum stages by
  `nnue/plot.py`'s loader (which does not need matplotlib): the training loss
  is the mean over the superbatch, the validation loss the superbatch's last
  validation.
- Best folders (`best/`, `best_<set>/`, `best*_previous_<time>/`) are hard links
  to a checkpoint's files, described by `best.json`. A scan skips one while its
  checkpoint still exists (same net) and otherwise registers it as that
  checkpoint (run id and superbatch from `best.json`). An archived best of an
  earlier run gets only `best.json`'s loss, since the folder's logs and
  settings are the later run's.
- `report` lists every net's losses next to the Elo of the most-played engine
  using it in the latest rating fit of a category (`data/ratings.py`), its SPRT
  passes, and the correlation of validation loss with Elo.
- `match` plays one engine with two nets and stores the result in
  `net_matches`. That table has no `penta_wl`: its `penta_dd` is the combined
  DD + WL pentanomial bin. The games run on the match runner
  (`utils/fixed_match.py`), one per core with its core type and clock
  (`--concurrency`, default every physical core but one; `--no-pin` leaves
  placement to the OS). Exactly one per-move limit: `--nodes` (the usual choice
  for nets), `--depth` or `--tc`; `--pairs` opening pairs (default 200) from
  `--book`. The net reaches the engine through the UCI option `--net-option`
  (default `nnue_weight_file`) as its full path or, with `--net-value stem`, its
  file name without extension. The run folder goes to `--results-dir` (default
  `logs/net_matches`), and `--fastchess` chooses the fastchess binary.

```powershell
python -m nnue.registry scan --run-id 1024x16x32_0wdl     # one run under nnue/checkpoints
python -m nnue.registry add nnue/checkpoints/1024x16x32_0wdl/best --notes "candidate"
python -m nnue.registry report --tc blitz
python -m nnue.registry match --engine X --net-a A.bin --net-b B.bin --nodes 50000 --pairs 200
```

## Hyperparameter search (`nnue/hpo.py`)

Each training run costs hours, results are noisy, and the loss after a short
run predicts the loss after a long one only roughly. Options considered:

| Approach | Verdict |
|---|---|
| Grid / manual guess-and-check | wastes most runs on bad regions |
| Random or quasi-random (Sobol) search | a solid baseline, no learning |
| Bayesian optimisation (Gaussian process or TPE) | learns where good settings are; one full run per evaluation is still expensive |
| Successive halving / Hyperband / ASHA | spends most compute on promising settings by training many briefly and few long |
| BOHB (TPE + successive halving) | both at once; the choice here |
| Learning-curve extrapolation (freeze-thaw) | pruning runs mid-way from partial curves; unreliable with a cosine LR, whose early loss depends on the schedule length |
| Population-based training | needs many parallel GPUs |
| SPSA on training settings | possible, but each gradient step needs full runs |

What `hpo.py` does:

- Every trial is a complete short run (`train.py`), with its whole LR schedule
  compressed into the budget, so short runs are faithful miniatures.
- Rungs with budgets `min * eta^i` (default 25, 75, 225 superbatches). Every
  configuration trains at the first budget; the best third moves up. With a
  curriculum (`--train-data` resolving to several files) the budget is per
  stage, as with `train.py --superbatches`, and the score is taken at the end of
  the last stage.
- New configurations come from Optuna's multivariate TPE if Optuna is installed
  (`pip install optuna`), trained on the first-rung results, otherwise from a
  scrambled Sobol sequence.
- The objective is the mean of the last three validation losses; `--score-cmd`
  ranks the last rung by any number a command prints (e.g. Elo from a short
  match) instead.
- Results go to `nnue/hpo/<study>/trials.jsonl`; rerunning the same `--study`
  resumes without repeating finished trials. `summary.json` holds the best
  settings and, with Optuna, the parameter importances (fANOVA).

```powershell
python nnue/hpo.py --study lr1 --train-data train.binpack --val-data val.binpack --features cuda
python nnue/hpo.py --study lr1 --initial 12 --brackets 2 --max-hours 24
python nnue/hpo.py --study demo --simulate     # exercise the search without training
```

The default search space is `lr_start` and the final/initial LR ratio (JSON via
`--space`, e.g. `{"lr_start": {"type": "log", "low": 2e-4, "high": 4e-3}}`).
Validation loss is only comparable when the WDL blend is the same for every
trial, and larger layers nearly always lower it while costing speed, so WDL and
layer sizes are only searchable with `--score-cmd`. Treat the winner as a
proposal: train it at full length and confirm it with games (`tests.sprt`).
