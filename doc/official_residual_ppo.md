# Official-environment residual PPO (2026-09-24)

## Scope and safety

The user requested a large PPO training run. The previous cached-field/replay
PPO experiments are not reused. This trainer wraps the unchanged official
Scenario-1 environment and draws a new external random seed at each episode.
The policy sees neither seeds nor future trajectories nor private simulator
state. No expert replay, target-ID bonus or fixed balloon field is used.

The frozen 10-point `submission_time_allocation_v1.py` remains the base planner.
PPO chooses a bounded residual world-frame acceleration (maximum 2 m/s^2 per
component) every ten official time steps. The same physical force allocation,
TVC limits and actuator controller still apply. A zero residual bypasses the
extra allocator exactly. This is learned GNC correction, not a learned route
selector or an end-to-end replacement of the baseline agent.

The 96 input features contain only current observation and controller history:
position, velocity, gyro, estimated orientation, burn time remaining, previous
controls, reference position/velocity errors, estimated disturbance, previous
residual and up to eight currently released targets. No target ID is encoded.

Reward shaping is training-only: official pops times ten, discounted nearest-
target potential difference, and a small residual-effort penalty. It does NOT
change the official evaluator or submitted score. Training stochastic episode
scores are not comparable to deterministic held-out leaderboard scores.

## Resource and run limits

Server: authorized instructor host, dual Xeon E5-2660 v4, roughly 62 GiB RAM.
Only Tesla P40 GPU 0 (UUID GPU-f32768cf-a4cf-c057-c466-db71f18efafa) is made visible
using CUDA_VISIBLE_DEVICES. The trainer refuses CUDA if it sees more than one
device or if an explicit one-device restriction is missing. No DataParallel,
multi-GPU process, system power changes or use of the other P40/desktop GPU.
BLAS/OMP/MKL each use one CPU thread; at most eight simulator workers.

The pilot has four environments, 2,048 PPO decisions, a 30-minute wall budget,
128-step rollouts, 128-sample minibatches, five epochs and checkpoints each
512 decisions. The planned larger run has eight environments, a target of
2,000,000 additional decisions and a six-hour wall limit, whichever comes first.
This is not a promise to finish two million decisions in six hours.

Each PPO decision holds its action for up to ten official physics steps
(0.1 seconds). Physics steps during launch warmup are reported separately at
reset and excluded from the progress counter. Do not relabel decisions as
physics steps or claim an uncompleted 100M-step run.

## Architecture and reproducibility

Separate actor/critic MLPs: 512, 512, 256 hidden units, Tanh activations;
learning rate 5e-5, gamma 0.995, GAE lambda 0.95, clip 0.15, target KL 0.015.
The actor's final layer starts at zero, log standard deviation -1.6, so initial
deterministic actions match the baseline and exploration starts small.

`scripts/train_official_residual_ppo.py` creates a NEW output directory, records
source hashes and library versions, saves progress and full checkpoints, and
stops on a wall-time limit, changed source, SIGTERM/SIGINT, or a `STOP` file in
the run directory. A requested stop is acted on at the next callback; an active
simulator step/reset can delay it. Checkpoints include optimizer state.

`scripts/export_residual_ppo.py` exports the actor plus frozen planner to a
standalone NumPy agent. It compares deterministic actor outputs against SB3
before writing and refuses to overwrite an existing source. The export embeds
learned weights only, not training trajectories or observations. Such an export
must still be evaluated on fresh official episodes before being submitted.

Primary references:
https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165

SB3 notes MLP PPO often benefits more from CPU environment parallelism than
GPU use. This experiment uses a single GPU for network updates, but simulator
throughput is measured before any claims about training speed or score.

## Validation and pilot status

The zero-policy controller independently reproduced **10 points** on Scenario 1,
seed 0, with exactly the frozen reference's pop events. Report:
`BalloonPoppingGymEnv/evaluation/results/residual_ppo_zero_s0.json`.
This validates zero-residual parity, not learned PPO performance.

Five focused tests pass: zero-residual actuator parity, finite observation-only
features, physical force bounds, truncation handling/external seed isolation,
and narrow handling of invalid reset samples.

The first pilot failed before any training update because the official random
balloon generation sampled a negative rocket/balloon mass and raised
`InvalidParameterError`. No simulator or sampled parameter was changed.
The training wrapper now retries only that exact reset-time invalid-mass error,
at most eight times, recording every rejected seed and message in
`reset_failures_<worker>.jsonl`. Other errors still propagate. Consequently
training samples the valid-reset subset of the official distribution; it is
not an unconditional sample of every seed. The failed pilot and original source
archive are preserved for audit. The second pilot uses a new output directory.

Source archives: `official_ppo_source_20260924.tgz` followed by
`official_ppo_reset_fix_20260924.tgz` in evaluation/results. They reproduce the
updated trainer without modifying official environment files.

## Completed pilot v2

Remote directory: `/home/rtcmspaceweather/balloon_champion_20260923/ppo_pilot_v2_20260924`.
The run completed 2,048 decisions / 20,458 controlled physics steps in 609.85 s
(3.36 decisions/s), with 20 optimizer epochs recorded in the final checkpoint.
Four completed stochastic training episodes scored 6, 8, 7, 8; none was truncated.
These are training episodes on different seeds, NOT evidence of an improvement
over the frozen reference. Source hashes remained unchanged.

Final checkpoint SHA256:
`f6b85b798569adbb9b2325faac6e957ef127e4d87481d726781d91d5ddc015f0`.
Standalone NumPy export SHA256:
`067b5b0c1d0ded50fe3991041215c8fd7bce4b20cac28c3cce8e249f2f293ad8`.
Maximum actor-output difference against SB3 on 32 test inputs was 4.66e-9.
The exported pilot is experimental, not a promoted submission.

Fresh deterministic evaluations of the exported NumPy agent scored **10 on
seed 0 and 7 on seed 3**, versus the frozen baseline's 10 and 9 respectively.
Both terminated normally without truncation and with unchanged source hashes.
Thus the pilot has NOT improved score and is NOT promoted. Continued training
is an experiment, not a claim that more training necessarily fixes this regression.
Reports: `eval_seed0.json` and `eval_seed3.json` in the pilot artifact directory.

At the measured four-worker pilot rate, two million decisions would take about
6.9 days, not six hours. Eight-worker throughput must be measured separately;
the six-hour run is a bounded first training installment with resumable weights.

Six focused PPO tests and a selected 67-test controller/planner/audit unittest
suite pass after adding a deployment action-hold cadence test. `pytest` is not
installed locally; this is not a claim that the full repository test suite ran.
Official simulator, evaluator, packer and verifier were rechecked unchanged
against `bd9bb51`.

### Operational notes

`progress.json` is refreshed about every 30 seconds at an environment callback;
long resets/plans can delay it. `started.json` records PID and start time.
`manifest.json` records source/library hashes and the one visible GPU UUID.
`episodes.jsonl` contains stochastic training scores, never submission scores.
`checkpoint_<steps>.zip` includes optimizer state. Checkpoints are saved inside
the rollout callback, so a checkpoint exactly at a rollout boundary precedes
that rollout's optimizer update. Prefer `final_model.zip` after a completed run.

To request a graceful stop, create an empty file named `STOP` in the specific
run directory on the instructor server. The trainer saves `final_model.zip`
on the next callback and reports `requested_stop`. Do not kill unrelated Python
jobs. Always export to a new agent filename and evaluate before promoting it;
do not replace the frozen 10-point submission based on training reward.

## Larger run launched

**Status superseded:** this eight-worker run was subsequently stopped through
its STOP file after 36/41 episodes truncated. Its final model is preserved but
not promoted. A single-environment recovery and truncation guard are documented
in `doc/exit_aware_and_training_health.md`. Historical start checks below do not
represent current run health.

Run directory: `/home/rtcmspaceweather/balloon_champion_20260923/ppo_large_20260924`.
PID: `2127180`; started Unix time `1790259124.6520548` (2026-09-24, 21:12 ICT).
Log: `/tmp/ppo_large_20260924.log`. Started detached with `nohup` and closed stdin,
so the SSH connection is not needed to keep it running. It resumes the completed
pilot's full checkpoint; no result from this larger run is yet claimed.

Exact training arguments, from the remote repository root:

```sh
scripts/train_official_residual_ppo.py --output ../ppo_large_20260924 \
  --resume ../ppo_pilot_v2_20260924/final_model.zip \
  --steps 2000000 --envs 8 --device cuda --hours 6 \
  --n-steps 128 --batch-size 128 --checkpoint-every 4096 --seed 240927
```

The launcher explicitly sets the single GPU UUID above and BLAS/OMP/MKL threads
to one. Eight simulator workers are used. `nvidia-smi` confirmed this trainer's
PID only on GPU 0; unrelated desktop GPU processes were left untouched.
Six-hour expiry is approximately 03:12 ICT on September 25, plus any in-progress
simulator callback delay. Two million decisions is a target ceiling, not a claim
about completion. The first action on resuming work should be to inspect
`progress.json` / `error.json` and export/evaluate a new checkpoint on paired seeds.

Completed pilot artifacts have been downloaded to
`BalloonPoppingGymEnv/evaluation/results/official_ppo_20260924/ppo_pilot_v2_20260924/`.
The local standalone agent is `BalloonPoppingGymEnv/agents/submission_ppo_pilot_v2.py`;
its local SHA256 and the downloaded final checkpoint hash match the remote export.
The config `residual_ppo_pilot_v2.yaml` explicitly disables leaderboard submission.

Initial large-run health check: at 161.25 s, 1,120 **additional** decisions
(3,168 cumulative including pilot) and 11,200 controlled physics steps had run.
The first 1,024-decision rollout and optimizer update were complete, and the
next rollout was collecting data. No error file, no source changes; eight worker
processes plus the multiprocessing resource tracker were present. No complete
new training episode or larger-run evaluation was yet available at that check.
The 6.95 decisions/s startup average is preliminary, not a sustained throughput
guarantee. The job remains running in the background; inspect the live remote
progress file for current status. A local snapshot is historical, not live.

Before handoff, the log reached 4,096 cumulative decisions (2,048 additional).
The next logged optimizer diagnostics were finite: approximate KL 0.00293,
value loss 8.87, optimizer epochs 25 (including pilot). A local launch/progress
snapshot is preserved in `evaluation/results/ppo_large_start_20260924.tgz`.

The downloaded pilot's four recorded training-source hashes match local source.
Static review of its standalone export found no flagged imports/private-state
access; this is supporting evidence, not organizer certification of compliance.
