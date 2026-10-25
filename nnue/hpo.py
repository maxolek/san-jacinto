#!/usr/bin/env python3
"""
Hyperparameter search for NNUE training runs (successive halving + TPE).

Training a net is expensive, so trials are cheap, *complete* short runs:
every configuration first trains for a small budget (e.g. 25 superbatches,
with its whole LR schedule compressed into that budget); only the best third
(eta = 3) is retrained with 3x the budget, and so on (successive halving, the
core of Hyperband/ASHA). New configurations are proposed by Optuna's TPE
sampler from the results so far when Optuna is installed (BOHB-style: the
model learns from the cheap first rung, where most evaluations happen), and
by a scrambled Sobol sequence otherwise. Each trial is one `nnue/train.py`
run; results go to nnue/hpo/<study>/trials.jsonl, so an interrupted search
resumes where it stopped.

Objective: the validation loss at the end of the run (mean of the last few
validations, to damp the noise of a single one). That makes three rules:
  * only compare configurations whose loss is measured against the same
    target: the WDL blend (and eval scale) must be the same for every trial,
    so WDL is not searched here unless you rank by games (--score-cmd);
  * validation loss is a proxy: retrain the winner at full length and confirm
    it with games (tests/sprt.py) before adopting it;
  * architecture sizes (L1/L2/L3) nearly always lower the loss when larger;
    their real cost is inference speed, so search them only with --score-cmd.

--score-cmd "python my_match.py {net}" replaces the loss with a number the
command prints last (higher is better, e.g. Elo from a short match), for
the last rung only; {net} is the trained quantised.bin.

Examples:
    python nnue/hpo.py --study lr1 --train-data train.binpack --val-data val.binpack --features cuda
    python nnue/hpo.py --study lr1 --rungs 3 --min-superbatches 25 --initial 12 --brackets 2
    python nnue/hpo.py --study demo --simulate          # exercise the search without training
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HPO_ROOT = HERE / "hpo"

DEFAULT_SPACE = {
    "lr_start": {"type": "log", "low": 2e-4, "high": 4e-3},
    "lr_final_ratio": {"type": "log", "low": 1e-4, "high": 0.1},
}
# Search-space names that are not train.py flags of the same name.
DERIVED = {"lr_final_ratio"}
FLAGS = {"lr_start": "--lr_start", "lr_final": "--lr_final", "batch_size": "--batch-size",
         "L1": "--L1", "L2": "--L2", "L3": "--L3", "wdl": None}


# ----------------------------------------------------------------------------
# Search space and proposals
# ----------------------------------------------------------------------------

class Space:
    def __init__(self, spec):
        self.spec = spec
        for name, s in spec.items():
            if s["type"] not in ("log", "float", "int", "choice"):
                raise SystemExit(f"{name}: type must be log, float, int or choice")

    def from_unit(self, name, u):
        s = self.spec[name]
        if s["type"] == "log":
            return math.exp(math.log(s["low"]) + u * (math.log(s["high"]) - math.log(s["low"])))
        if s["type"] == "float":
            return s["low"] + u * (s["high"] - s["low"])
        if s["type"] == "int":
            return int(min(s["high"], s["low"] + math.floor(u * (s["high"] - s["low"] + 1))))
        values = s["values"]
        return values[min(len(values) - 1, int(u * len(values)))]

    def optuna_distributions(self):
        import optuna.distributions as d
        out = {}
        for name, s in self.spec.items():
            if s["type"] == "log":
                out[name] = d.FloatDistribution(s["low"], s["high"], log=True)
            elif s["type"] == "float":
                out[name] = d.FloatDistribution(s["low"], s["high"])
            elif s["type"] == "int":
                out[name] = d.IntDistribution(s["low"], s["high"])
            else:
                out[name] = d.CategoricalDistribution(s["values"])
        return out


class SobolProposer:
    """Space-filling proposals (no model); used when Optuna is not installed."""

    name = "sobol"

    def __init__(self, space, seed):
        self.space = space
        try:
            from scipy.stats import qmc
            self.engine = qmc.Sobol(d=len(space.spec), scramble=True, seed=seed)
        except ImportError:
            self.engine = None
        self.rng = random.Random(seed)

    def ask(self):
        units = self.engine.random(1)[0] if self.engine is not None else [self.rng.random() for _ in self.space.spec]
        return {name: self.space.from_unit(name, u) for name, u in zip(self.space.spec, units)}

    def tell(self, params, loss):
        pass


class TPEProposer:
    """Optuna's multivariate TPE, fed with the first-rung results."""

    name = "tpe"

    def __init__(self, space, seed, history=()):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.optuna = optuna
        self.space = space
        self.dists = space.optuna_distributions()
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=6)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.pending = {}
        for params, loss in history:
            self.study.add_trial(optuna.trial.create_trial(params=params, distributions=self.dists, value=loss))

    def ask(self):
        trial = self.study.ask(self.dists)
        key = json.dumps(trial.params, sort_keys=True)
        self.pending[key] = trial
        return dict(trial.params)

    def tell(self, params, loss):
        trial = self.pending.pop(json.dumps(params, sort_keys=True), None)
        if trial is not None:
            state = self.optuna.trial.TrialState.COMPLETE if loss is not None else self.optuna.trial.TrialState.FAIL
            self.study.tell(trial, loss if loss is not None else None, state=state)

    def importances(self):
        try:
            return self.optuna.importance.get_param_importances(self.study)
        except Exception:
            return None


def make_proposer(space, seed, history, force_random=False):
    if not force_random:
        try:
            return TPEProposer(space, seed, history)
        except ImportError:
            print("[hpo] Optuna not installed (pip install optuna); using Sobol proposals without a model")
    proposer = SobolProposer(space, seed)
    for _ in history:  # keep the sequence reproducible across resumes
        proposer.ask()
    return proposer


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def train_flags(params):
    """train.py arguments for one configuration."""
    params = dict(params)
    if "lr_final_ratio" in params:
        params["lr_final"] = params["lr_start"] * params.pop("lr_final_ratio")
    flags = []
    for name, value in params.items():
        if name == "wdl":
            flags += ["--wdl-start", f"{value:g}", "--wdl-end", f"{value:g}"]
            continue
        flag = FLAGS.get(name, f"--{name}")
        flags += [flag, f"{value:.6g}" if isinstance(value, float) else str(value)]
    return flags


def final_val_loss(net_dir, net_id, budget, last=3):
    """Mean of the last `last` validation losses of a finished run (None if missing)."""
    path = Path(net_dir) / f"{net_id}-{budget}" / "val_log.txt"
    try:
        rows = [line.strip().split(",") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        losses = [float(r[2]) for r in rows if len(r) == 3]
    except (OSError, ValueError):
        return None
    losses = [x for x in losses if math.isfinite(x)][-last:]
    return sum(losses) / len(losses) if losses else None


class TrainEvaluator:
    def __init__(self, args, study_dir):
        self.args, self.study_dir = args, Path(study_dir)
        self.runs = self.study_dir / "runs"
        # A curriculum (--train-data resolving to several files) trains every stage for the
        # budget, so a run ends at stages * budget: that is the checkpoint to score.
        self.stages = 1
        if args.train_data:
            import train
            try:
                self.stages = max(1, len(train.expand_train_data(args.train_data, train.DATA_ROOT)))
            except FileNotFoundError as exc:
                raise SystemExit(f"[hpo] {exc}")

    def __call__(self, trial_id, rung, budget, params, last_rung):
        net_id = f"t{trial_id:04d}_r{rung}"
        final = budget * self.stages
        # --no-register: trial runs are deleted after scoring, so they stay out of the net registry.
        cmd = [sys.executable, str(HERE / "train.py"), "--example", self.args.example, "--net-id", net_id,
               "--output-dir", str(self.runs), "--superbatches", str(budget), "--save-rate", str(budget),
               "--no-plot", "--no-best", "--no-register", *train_flags(params)]
        if self.args.train_data:
            cmd += ["--train-data", *self.args.train_data]
        if self.args.val_data:
            cmd += ["--val-data", *self.args.val_data]
        if self.args.features:
            cmd += ["--features", self.args.features]
        cmd += self.args.train_args
        log = self.study_dir / "logs" / f"{net_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8", errors="replace") as out:
            code = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=str(HERE.parent))
        net_dir = self.runs / net_id
        if code != 0:
            print(f"[hpo] trial {trial_id} rung {rung} failed (exit {code}); see {log}")
            return None
        if self.args.score_cmd and last_rung:
            net = net_dir / f"{net_id}-{final}" / "quantised.bin"
            output = subprocess.run(self.args.score_cmd.format(net=net), shell=True, capture_output=True,
                                    text=True, cwd=str(HERE.parent)).stdout.split()
            try:
                return -float(output[-1])  # higher score is better; the search minimises
            except (IndexError, ValueError):
                print(f"[hpo] --score-cmd printed no number for trial {trial_id}")
                return None
        loss = final_val_loss(net_dir, net_id, final)
        if not self.args.keep_runs:
            import shutil
            shutil.rmtree(net_dir, ignore_errors=True)
        return loss


class SimulatedEvaluator:
    """A synthetic loss surface with noise, to exercise the search without a GPU."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def __call__(self, trial_id, rung, budget, params, last_rung):
        lr_error = math.log(params.get("lr_start", 1e-3) / 1e-3)
        ratio_error = math.log(params.get("lr_final_ratio", 3e-3) / 3e-3)
        # Longer runs are better; badly chosen LRs hurt more when the run is short.
        return (0.020 + 0.03 / math.sqrt(budget) + 0.004 * lr_error ** 2 * (1 + 10 / budget)
                + 0.0015 * ratio_error ** 2 + self.rng.gauss(0, 0.0006))


# ----------------------------------------------------------------------------
# Successive halving
# ----------------------------------------------------------------------------

class Study:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trials.jsonl"
        self.records = []
        if self.path.exists():
            self.records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                            if line.strip()]

    def add(self, record):
        self.records.append(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def find(self, bracket, rung, trial):
        for r in self.records:
            if (r["bracket"], r["rung"], r["trial"]) == (bracket, rung, trial):
                return r
        return None

    def next_trial_id(self):
        return max((r["trial"] for r in self.records), default=0) + 1


def run_search(args, study, space, evaluator):
    budgets = [args.min_superbatches * args.eta ** i for i in range(args.rungs)]
    history = [(r["params"], r["loss"]) for r in study.records if r["rung"] == 0 and r["loss"] is not None]
    proposer = make_proposer(space, args.seed, history, force_random=args.random)
    print(f"[hpo] study {study.dir.name}: {args.brackets} bracket(s) of {args.initial} configurations, "
          f"budgets {budgets} superbatches, eta {args.eta}, proposals {proposer.name}")
    try:
        _brackets(args, study, proposer, evaluator, budgets)
    finally:
        importances = proposer.importances() if hasattr(proposer, "importances") else None
        return report(args, study, budgets, importances)


def _brackets(args, study, proposer, evaluator, budgets):
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None

    for bracket in range(args.brackets):
        # Rung 0: new configurations (reused from the study file after a resume).
        entries = []
        existing = [r for r in study.records if r["bracket"] == bracket and r["rung"] == 0]
        for record in existing:
            entries.append((record["trial"], record["params"], record["loss"]))
        while len(entries) < args.initial:
            if deadline and time.time() > deadline:
                print("[hpo] time budget reached")
                return
            trial = study.next_trial_id()
            params = proposer.ask()
            loss = evaluate(study, evaluator, bracket, 0, trial, budgets[0], params, args.rungs == 1)
            proposer.tell(params, loss)
            entries.append((trial, params, loss))
        # Promotions: the best 1/eta of each rung trains again with eta times the budget.
        for rung in range(1, args.rungs):
            ranked = sorted((e for e in entries if e[2] is not None), key=lambda e: e[2])
            survivors = ranked[:max(1, math.ceil(len(ranked) / args.eta))]
            promoted = []
            for trial, params, _ in survivors:
                if deadline and time.time() > deadline:
                    print("[hpo] time budget reached")
                    return
                done = study.find(bracket, rung, trial)
                loss = done["loss"] if done else evaluate(study, evaluator, bracket, rung, trial, budgets[rung],
                                                           params, rung == args.rungs - 1)
                promoted.append((trial, params, loss))
            entries = promoted


def evaluate(study, evaluator, bracket, rung, trial, budget, params, last_rung):
    started = time.time()
    loss = evaluator(trial, rung, budget, params, last_rung)
    record = dict(bracket=bracket, rung=rung, trial=trial, budget=budget, params=params, loss=loss,
                  seconds=round(time.time() - started, 1))
    study.add(record)
    shown = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
    print(f"[hpo] bracket {bracket} rung {rung} ({budget} sb) trial {trial}: "
          f"{'failed' if loss is None else f'{loss:.6f}'}  {shown}")
    return loss


def report(args, study, budgets, importances=None):
    top_rung = max((r["rung"] for r in study.records if r["loss"] is not None), default=None)
    if top_rung is None:
        print("[hpo] no successful trials")
        return None
    finalists = sorted((r for r in study.records if r["rung"] == top_rung and r["loss"] is not None),
                       key=lambda r: r["loss"])
    best = finalists[0]
    print(f"\n[hpo] best at the longest budget ({budgets[top_rung]} superbatches): trial {best['trial']} "
          f"loss {best['loss']:.6f}")
    for r in finalists[:5]:
        print(f"      trial {r['trial']:4d}  loss {r['loss']:.6f}  {r['params']}")
    if importances:
        print("[hpo] parameter importance (fANOVA on the first rung): " +
              ", ".join(f"{k} {v:.2f}" for k, v in importances.items()))
    flags = " ".join(train_flags(best["params"]))
    print(f"\n  Train the winner at full length, then confirm it with games:\n"
          f"    python nnue/train.py --example {args.example} {flags} --train-data ... --val-data ...")
    summary = dict(best=best, finalists=finalists[:10], budgets=budgets, importances=importances,
                   train_flags=train_flags(best["params"]))
    (study.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", required=True, help="Name; results in nnue/hpo/<study>/ (resumable)")
    p.add_argument("--space", default=None, help="JSON search space (default: lr_start, lr_final_ratio)")
    p.add_argument("--train-data", nargs="+", default=None)
    p.add_argument("--val-data", nargs="+", default=None)
    p.add_argument("--example", default="v3")
    p.add_argument("--features", default=None, help="cargo features, e.g. cuda")
    p.add_argument("--min-superbatches", type=int, default=25, help="Budget of the first rung")
    p.add_argument("--eta", type=int, default=3, help="Keep the best 1/eta and multiply the budget by eta")
    p.add_argument("--rungs", type=int, default=3)
    p.add_argument("--initial", type=int, default=9, help="Configurations per bracket")
    p.add_argument("--brackets", type=int, default=1)
    p.add_argument("--max-hours", type=float, default=None, help="Stop starting trials after this long")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--random", action="store_true", help="Sobol proposals even if Optuna is installed")
    p.add_argument("--score-cmd", default=None,
                   help="Rank the last rung by this command's last printed number ({net} = quantised.bin)")
    p.add_argument("--keep-runs", action="store_true", help="Keep every trial's checkpoints")
    p.add_argument("--simulate", action="store_true", help="Synthetic losses instead of training (dry run)")
    p.add_argument("train_args", nargs=argparse.REMAINDER,
                   help="Extra train.py arguments after --, e.g. -- --batch-size 16384")
    args = p.parse_args(argv)
    args.train_args = [a for a in args.train_args if a != "--"]
    if args.eta < 2 or args.rungs < 1 or args.initial < 1:
        p.error("need eta >= 2, rungs >= 1, initial >= 1")
    space = Space(json.loads(Path(args.space).read_text(encoding="utf-8")) if args.space else DEFAULT_SPACE)
    if any(name in space.spec for name in ("wdl", "L1", "L2", "L3")) and not args.score_cmd and not args.simulate:
        p.error("WDL and layer sizes change what validation loss measures; search them only with --score-cmd")
    study = Study(HPO_ROOT / args.study)
    evaluator = SimulatedEvaluator(args.seed) if args.simulate else TrainEvaluator(args, study.dir)
    run_search(args, study, space, evaluator)


if __name__ == "__main__":
    main()
