# Bounded continuation after positive ten-seed evidence

**Update 2026-10-06:** continuation finished normally and its final checkpoint
is backed up. See [parent-vs-continuation comparison](multiscenario_comparison_20261006.md).

The user requested results and further work. The frozen pilot improved average
Scenario-4 score from 2.9 to 3.8 on ten holdout seeds, winning five and tying
five. Both still failed on one seed. Keep the tested parent unchanged; do not
claim the architecture has reached 20 pops or that continued training must help.

## Decision

Continue this exact policy for another bounded two-hour pilot before changing
architecture. This tests additional learning time as the main intervention.
The mixture remains 2/8/15/25/50 percent Scenarios 0/1/2/3/4; all network,
feature, controller, optimizer and rollout settings are retained. This is not
a rollout of the proposed attention route-selector yet.

`scripts/resume_multiscenario_ppo.py` validates parent SHA, feature schema,
source, mixture, batch and rollout sizes before restoring. It preserves model
and optimizer state, timestep and optimizer-epoch counters. It never zeros the
restored output layer. A new external RNG stream (seed 261006) and a fresh
official reset are used. This is not bitwise continuation of a partial rollout;
that incomplete rollout is discarded and no simulator trajectory is loaded.

Input checkpoint:
`/home/rtcmspaceweather/balloon_multiscenario_20261004_v1/pilot/final_model.zip`.
SHA-256 `8aad07e2ac6b834601b1d992f0fbb3ce9bcce1d1bc845ea179623b9e98577b3b`.
The tested parent, its training artifacts and both evaluation sets stay intact.
All new files are isolated under the same remote project root:

- Runtime snapshot: `source_resume_20261005/` (copy, new runner/helper/tests only).
- Artifacts: `continuation_20261005/`.
- Launcher log: `continuation_20261005.log`.
- Existing September-22 venv reused; PYTHONPATH pins this snapshot's RocketPy.

Limits: 20,000 **additional policy decisions** or two hours (soft wall budget),
whichever first; one simulator worker; one visible Tesla P40 GPU UUID
`GPU-f32768cf-a4cf-c057-c466-db71f18efafa`; numerical threads one; nice level 10.
Existing jobs on other GPUs are untouched. Save checkpoints every 512 new
decisions. Stop on three consecutive official truncations, source/parent hash
change, signals or a `STOP` file in the new artifact directory. Reset can delay
a wall-limit check. No indefinite training or submission upload.

Launch verified: PID `352465`, device CUDA on the sole authorized P40;
`started.json` records 13,038 restored policy decisions and 505 optimizer
epochs. No error file was present, and the first fresh official reset was
progressing. No completed continuation flight or new score improvement yet.
NVIDIA process inspection attributed 152 MiB to this PID on the P40 only.
An independent CPU load of the actual parent and resumed `initial_model.zip`
confirmed bit-exact policy parameter tensors and optimizer state tensors.

Tests: local parent-provenance contract passed. Remote suite: 9 tests passed,
including exact synthetic parameter and optimizer restoration, new RNG seed,
counter continuation, mixture/controller/adapter checks. Synthetic tests are
not official flight scores. `initial_model.zip` here is the restored trained
parent, not an untrained network.

## Evaluation gate after completion

Compare the new final checkpoint **against the frozen learned parent**, not just
the weaker zero-residual base, on the same newly predeclared seed set. Keep both
models fixed and report paired differences/low-score cases/failures. The earlier
ten-seed set informed this decision and must not be called untouched again.
Do not automatically choose a checkpoint by its stochastic training score.

If continuation does not help, diagnose route vs tracking errors and then test
the attention-based multi-target planner described in the holdout document.
Larger networks alone cannot replace the current model's limited action role.

The shared zero-pop seed 1818045955 deserves separate development analysis:
mean tracked-position error fell from about 3.923 m (base) to 0.719 m (PPO),
but both popped zero. Navigation RMS was similar (0.457/0.452 m). The counters
recorded 29/22 no-feasible-plan events and zero accepted joint-refresh rescues
for either agent. This is a diagnostic hint that better tracking alone may not
solve this field, not a causal proof of which planner component failed. Do not
tune the continuation mid-run using this seed or silently exclude it.
