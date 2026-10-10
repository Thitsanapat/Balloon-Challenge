# Scenario 4: selected-controller route-ML collection (8 Oct 2026)

The qualification agent remains
`BalloonPoppingGymEnv/agents/submission_scenario4_guarded_innovation_v1.py`
(SHA-256 `dab27623c797c70c6fc5c13b48e4faa2e146a694c7de737f6cae8138485fb333`).
No learned model is connected to that agent yet; the current known seed-0 score
remains 5 pops.

## Collection protocol and initial check

`scripts/collect_guarded_route_shadow.py` runs the exact selected config on the
official Scenario-4 simulator twice per seed in fresh processes. Only the
second flight observes the agent's committed route after each action. Its
observation-only feature extractor never changes actions. The gate compares
every action and observation hash, termination and score, then audits every
observed pop and first-leg outcome. No unflown proposal is labelled a failure.
The source/configuration hashes are frozen per run. GPU devices are hidden and
only one seed is processed at a time.

The already-used seed 0 served **only as a protocol anchor**, not independent
ML validation: 5 pops in each flight, exact action/observation parity, 6
executed attempts (5 hits, 1 miss), and a complete event audit. Its archive is
`BalloonPoppingGymEnv/evaluation/results/guarded_route_anchor_seed0_20261008.zip`.

## New data run

Twelve fresh training seeds were generated before seeing outcomes and recorded
in `qualification/used_seed_manual.jsonl`:

`313734695 48700076 470928925 257808910 164593090 300439883`
`93120163 767678882 230652457 333239953 839958652 724686099`

Four separate seeds are reserved for a **future live-policy comparison** and
must not enter model fitting or hyperparameter selection:

`761692544 234703015 439862696 130752622`

The training run began locally at about 15:02 Bangkok time on 8 Oct 2026, one
CPU worker and no GPU. It is deliberately long because every fresh Scenario-4
field runs the simulator's 100-balloon Monte Carlo initialization twice.
Progress and errors are in the ignored local directory
`BalloonPoppingGymEnv/evaluation/results/guarded_route_train12_20261008/`;
external stdout/stderr are the neighboring files with `_stdout.log` and
`_stderr.log` suffixes. On success, the runner creates a `.zip` archive and
`route_capture_fit.json` in the directory. The fit is a strongly regularized
six-feature logistic prototype evaluated by leaving one *entire seed* out of
each fold, then fit on all twelve training fields. It is CPU-only.

Use `progress.json` to see completed seed pairs; do not interpret a partial
batch as a model result. `failed.json` means the run stopped and no trained
model should be used. The selected qualification runner and JSON submission
are unchanged.

## Promotion gate

The learned model predicts **first-leg capture probability**, not the outcome
of a route never flown. A lower Brier and log loss than the train-fold hit-rate
prior on seed-held-out folds is a prerequisite, not proof of a higher pop score.
Only then build a separate experimental reranker for physically feasible
routes, with the reviewed controller as fallback. Freeze that candidate, run
the four reserved fields paired against the selected agent, and inspect both
mean score and low-score failures. Do not promote on a single favorable seed.
The source/training data path must remain reproducible for organizer review.

The separate, **not yet flight-tested** prototype is
`BalloonPoppingGymEnv/agents/scenario4_learned_route_agent.py` with its
experimental config `scenario4_learned_route_launch42.yaml`. It adds at most
one second of preference penalty for a low-probability *first leg* while the
existing beam search still supplies feasible candidates and all low-level
control. It does not alter launch-axis selection. After the collection run
finishes, create a seed-free weight artifact only if both held-out Brier and
log loss beat the prior:

```powershell
.\.venv\Scripts\python.exe -m scripts.package_guarded_capture_model BalloonPoppingGymEnv/evaluation/results/guarded_route_train12_20261008/route_capture_fit.json BalloonPoppingGymEnv/evaluation/results/guarded_route_train12_20261008/capture_weights.json
```

Failure of that command is a deliberate no-deployment gate, not a reason to
force the weights into the live runner. Unit tests match the runtime feature
formula against the collector's logged training feature formula. No score gain
has been measured for this prototype yet.

`scripts/validate_guarded_route_ml.py` also ran as a hidden, idle
watcher. It waited for the full audited training report, applied the same
weight-packaging gate, and (only if it passes) evaluates the selected and
experimental agents sequentially on the four reserved seeds. The result is
`evaluation/results/guarded_route_ml_holdout4_20261008.json`; its neighboring
`_stdout.log` and `_stderr.log` show progress/errors. A failed calibration
would create a `model_rejected` result with zero holdout flights. Even a favorable
holdout does **not** automatically change `run_qualification.cmd` or the
submission agent. Source/model/config hashes are frozen for the comparison.

## Completed first holdout and next gate

The twelve-field model passed its narrow calibration gate: seed-held-out
Brier score 0.25310 versus prior 0.25416, and log loss 0.69964 versus prior
0.70152. These improvements are small. On the four reserved official fields,
the selected agent scored 2/2/6/4 and the frozen learned-route agent scored
4/2/6/5, totaling 14 versus 17. That is **not** a 17-pop single flight and
does not justify changing the live runner yet.

Eight additional independent seeds were reserved and started in a serial,
paired second holdout. See `doc/scenario4_route_ml_stage2_20261008.md` for the
predeclared reliability screen and resumable result path. The old four-field
outcome was known before this second set was chosen.
