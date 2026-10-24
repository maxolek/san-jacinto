# NNUE training and hyperparameter search

## Training (`nnue/train.py`)

`train.py` runs one of san-jacinto's training scripts (`nnue/v2.rs`, `nnue/v3.rs`)
against the bullet fork in `libs/bullet`, through the small cargo package in
`nnue/Cargo.toml`; any other `--example` name runs a bullet workspace example.
Settings reach the script as environment variables.

```powershell
python nnue/train.py --example v3 --features cuda --train-data train.binpack --val-data val.binpack
python nnue/train.py --example v3 --net-id big --train-data s1.binpack s2.binpack s3.binpack --superbatches 200 400 800 --val-data val.binpack
python nnue/train.py --example v3 --net-id big --superbatch-start 601 --superbatches 200 --train-data s3.binpack   # resume from big-600
```

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

With a constant WDL blend (`--wdl-start` = `--wdl-end`, including 0; the
scripts' default is a constant 0.25), the wrapper keeps the state with the
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
