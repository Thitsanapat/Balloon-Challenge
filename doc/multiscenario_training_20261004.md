# Mixed-scenario residual PPO pilot, 2026-10-04

**Update 2026-10-05:** the pilot finished at its two-hour wall budget. The final
checkpoint is backed up locally; deterministic paired validation has started.
See [completion and validation report](multiscenario_validation_20261005.md).
The launch/progress notes below describe observations made during training.

## Scope and rules

Train one shared residual policy on released official Scenarios 0--4, with
episode probabilities 2/8/15/25/50 percent. Scenario 0 is only a small simple
case; Scenario 4 remains the main target. These weights are an initial design
choice, not an experimentally established optimum. Do not invent parameters
for unreleased scenarios/final-round changes or assume this ensures robustness.

The organizer's supplied September 18 email encourages training on Scenarios
2/3 for qualification preparation. The public rules require observation-only
execution, unchanged evaluation physics, and reviewable training code. We found
no prohibition on training across released scenarios; this is an interpretation
of the published scope, **not organizer certification of this implementation**.
No organizer email/post was sent. References:
[rules clarification](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165),
[competition README](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/blob/main/README.md).

Rationale: a conventional controller plus learned residual can reduce the
amount of control behavior learned from scratch; mixed dynamics expose the
policy to more than one set of conditions. Related research:
[Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201),
[Sim-to-Real Transfer with Dynamics Randomization](https://arxiv.org/abs/1710.06537).
These robotics results do not establish a score gain for this competition.
This experiment selects official presets, rather than changing their physics.

## Reproducible design

- Base controller: full-route joint derivative refresh, shortening disabled.
  Its prior development scores are 6/3 on Scenario-4 seeds 0/1. Those are
  **controller** results, not learned checkpoint results.
- Actions: bounded acceleration residual, 2 m/s^2 per component, held for
  ten official physics steps. A missing/popped target clears the correction;
  target changes also clear stale residuals. Existing thrust/tilt allocation
  remains in effect. Zero correction preserves base-controller commands.
- Actor input: schema `multiscenario_residual_v1_112`; 96 active-target-first
  observation/history features plus public sensor/actuator/vehicle context
  and live plan timing. No scenario ID, seed, future trajectory, hidden wind,
  simulator object, or replay lookup enters the policy. This is incompatible
  with older 96-feature checkpoints; the trainer deliberately has no resume
  option and starts a new model.
- Train seeds: [1,000,000, 1,000,000,000); validation:
  [1,000,000,000, 1,500,000,000); holdout: [1,500,000,000, 2,000,000,000).
  Pools are for this new experiment. Seed/scenario draws are externally
  logged for reproduction and excluded from actor features. Known invalid
  mass RESET errors alone get bounded retries, also logged; flight errors
  are not hidden and official physics is not patched.
- PPO actor/critic: separate [256,256] MLPs, LR 5e-5, rollout/batch 128,
  gamma .995, five epochs, target KL .015. Initial mean action is zero with
  small stochastic exploration (log standard deviation -2.3). Training
  returns use shaping; the separate `popped_count` remains actual score.
- One simulator worker avoids prior multiworker reset timeout problems.
  One explicitly visible P40 GPU only; BLAS/OpenMP threads limited to one.
  Small MLPs are not expected to saturate the GPU; simulator CPU throughput
  remains important. No other machine jobs or GPU power settings were changed.

## Authorized remote run

Host: `192.168.1.135`, user `rtcmspaceweather`. Password authentication through
an interactive SSH prompt succeeded (the earlier BatchMode key failure did
not mean the password was unusable). No password or team secret is stored here.

Isolated directory: `/home/rtcmspaceweather/balloon_multiscenario_20261004_v1/`.
Existing September projects and their environments remain untouched. The
existing September-22 Python environment is reused, but `PYTHONPATH` pins
RocketPy to the new snapshot's exact source, commit
`3b3d6c02feafab8e0ef29908c75a8190a6cdc268`.

The allowlisted 182-file archive contains code, official parameters/weather
data, and pinned RocketPy runtime source; no submissions, credentials, Git
metadata, or old models. It is preserved locally at
`BalloonPoppingGymEnv/evaluation/results/multiscenario_training_20261004_v1.zip`.
Archive SHA-256: `0d42bd17674e06f817f1ef3bc1873ff50a1e7fde9ba72a0d2beb7f5b204fc153`.
Remote archive and all 182 extracted hashes were verified; nine wrapper/model
tests passed there. Local related tests passed before launch.

Run PID at launch: `1969545`. GPU UUID:
`GPU-f32768cf-a4cf-c057-c466-db71f18efafa` (Tesla P40, physical index 0).
An actual CUDA tensor operation passed with exactly one visible device.
After launch, nvidia-smi attributed this run only to that P40; the other P40
was unused by this job. Existing RTX 3060 Ti jobs were not touched.

Command, from the snapshot's `source` directory with its PYTHONPATH:

```sh
CUDA_VISIBLE_DEVICES=GPU-f32768cf-a4cf-c057-c466-db71f18efafa \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/rtcmspaceweather/balloon_champion_20260922/.venv/bin/python \
  -u -m scripts.train_multiscenario_residual_ppo \
  --output /home/rtcmspaceweather/balloon_multiscenario_20261004_v1/pilot \
  --device cuda --hours 2 --steps 20000 --n-steps 128 --batch-size 128 \
  --checkpoint-every 128
```

The actual process was launched with `nohup` and `nice -n 10`; its stdout is
`train.log`. It stops at the decision budget, a soft two-hour wall budget,
three consecutive official truncations, source-hash changes, or a `pilot/STOP`
file / SIGTERM request. Wall/STOP checks occur between decisions and can be
delayed by an official reset. It is not an indefinite training run.

`pilot/manifest.json`, `started.json`, `progress.json`, `episodes.jsonl`,
`episode_seeds.jsonl`, `worker.monitor.csv`, `initial_model.zip`, checkpoints,
and `final_model.zip` describe the run. Initial model is not a trained result.
The first official balloon reset took 37.6 s. Checkpoint 256 was loaded on
CPU for inspection: it records five PPO optimizer epochs and its actor output
weight norm is 0.0037380466, versus zero initially. Thus actual learning
updates occurred, not merely a launched process or a saved initial model.
At 339 decisions the callback recorded one completed Scenario-3 training
episode: 10 actual pops, no truncation, simulation time 75.84 s, unchanged
source. This is a stochastic **training** episode, not deterministic held-out
evaluation, not a Scenario-4 score, and not evidence that learning beats the
base controller. The process was still running and had begun a second
Scenario-3 episode; short runs need not yet sample every mixture component.

A checkpoint-256 snapshot (model, manifest, progress, monitor and external
seed log) was copied back into the ignored local results directory:
`multiscenario_checkpoint_256_snapshot_20261004.zip`. Local and remote SHA-256
match: `97c77f77b87a4ae057328a0ea9e10ff3e217fd13f4d0a2d9b700f6dc77ca9a89`.
An extracted validation copy is in `results/multiscenario_checkpoint256/`,
also ignored by Git. The source archive is already backed up locally. This
is not a submission artifact. Further checkpoints remain in the remote pilot
folder; they must be collected after the bounded run completes.
The downloaded checkpoint loaded successfully through the CPU validation
adapter with matching source/schema and finite bounded predictions. This
checks portability/loading only, not official flight score. SSH was closed
after inspection; the authorized `nohup` pilot continues with its own budget
and stop controls, not an open interactive shell.
Last inspected progress: 713 decisions, 7,121 controlled physics steps,
two completed Scenario-3 training episodes (10 and 8 pops), zero truncations,
unchanged source, no error file, process still alive. These remain training
results, not paired evidence of improvement on Scenario 4.

## Promotion gate

Do not replace the reviewed agent based on training return or one lucky seed.
Compare deterministic checkpoint actions against the frozen zero-residual
full-route controller on identical previously unused validation seeds, then
one untouched holdout set. Report each scenario separately and prioritize
Scenario 4 average score, low-score cases, truncations, and wall time. Mixing
can cause negative transfer; retain the base if it does not improve.

`MultiScenarioPPOPolicyAgent` is a local evaluation adapter for a trusted
checkpoint plus matching manifest/controller source. It checks schema,
observation/action sizes and controller hashes. It is **not** a standalone
submission file. No learned score improvement, source-review certification,
20-pop result, or final-round robustness is claimed yet.
