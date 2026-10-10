# Frozen PPO ten-seed holdout, 2026-10-05

User authorized the ten-seed paired test and suggested a more capable model if
the present approach fails. Do not infer a score gain from network size alone.
This stage performs inference/evaluation only; no additional training or upload.

## Completed result

All 20 official flights completed normally in 5,485.00 seconds (91.4 minutes).
Checkpoint, source and runtime remained unchanged. No score is a training return.

| Seed | Base | PPO | Difference |
| --- | ---: | ---: | ---: |
| 1793692987 | 2 | 2 | 0 |
| 1576325320 | 5 | 5 | 0 |
| 1931191098 | 4 | 5 | +1 |
| 1735313215 | 2 | 3 | +1 |
| 1889085391 | 2 | 4 | +2 |
| 1877745918 | 2 | 6 | +4 |
| 1818045955 | 0 | 0 | 0 |
| 1968662391 | 3 | 3 | 0 |
| 1726021446 | 6 | 6 | 0 |
| 1717629685 | 3 | 4 | +1 |
| Mean per flight | 2.9 | 3.8 | +0.9 |

PPO wins/ties/losses: 5/5/0. Median base/PPO scores: 2.5/4.0;
median paired difference: +0.5. Both have one zero-pop flight, minimum 0 and
maximum 6. Mean wall time: 266.18/271.66 seconds. Positive evidence on this
small set, not proof of 20-pop ability or a guarantee on live random seeds.
Do not confuse 38 total across ten PPO flights with a single-flight score.

Full reports/plan/configs/status/summary backed up locally in
`BalloonPoppingGymEnv/evaluation/results/multiscenario_holdout_complete_20261005.zip`.
Remote/local SHA-256 match:
`33f8c6dad310dc0c45c817d5967f5fff060567b8f7895bcb71eeaa8cc39f7bb1`.
Keep this checkpoint as the reference for further learned candidates; it is
not yet a packaged/reviewed competition submission. The authorized next step
is a bounded continuation, documented in
[continuation plan](multiscenario_continuation_20261005.md).

## Fixed experiment before viewing outcomes

- Same Scenario-4 zero-residual base and final PPO as the three-seed validation.
- Final checkpoint SHA-256:
  `8aad07e2ac6b834601b1d992f0fbb3ce9bcce1d1bc845ea179623b9e98577b3b`.
- Ten seeds drawn from the reserved holdout identifier range, without simulating
  fields to select them. Reproduce selection using
  `random.Random(261005).sample(range(1500000000,2000000000),10)`:
  `1793692987, 1576325320, 1931191098, 1735313215, 1889085391,
  1877745918, 1818045955, 1968662391, 1726021446, 1717629685`.
- Reject duplicates, wrong-pool seeds, overlap with the training episode log,
  and overlap with prior evaluation plan(s). No replacing low-scoring seeds.
- Twenty fresh official flights, alternating base/PPO for each seed. CPU only,
  one worker, native numerical threads limited to one, nice level 10.
- Identical deterministic policy/controller settings; preserve full source,
  runtime, weather, checkpoint and prior-plan hashes throughout all flights.
- At most 1,200 seconds per subprocess; fail visibly on an exception, timeout,
  official truncation, or source change. No failed run becomes a positive delta.
- Estimate 80--100 minutes based on the earlier 25.4-minute six-flight run;
  actual duration may vary. The finite job exits after twenty flights.
- Report paired mean/median differences, wins/ties/losses, lowest scores and
  zero-pop frequency, completion/truncation counts and runtime. Do not promote
  merely because the single best seed improves. Ten seeds still give limited
  uncertainty evidence; avoid claiming guaranteed performance on live seeds.
- Once outcomes are used to change architecture or tune the agent, this set is
  development evidence. Use a fresh untouched set for the next final decision.

## Isolated execution and tests

Remote root `/home/rtcmspaceweather/balloon_multiscenario_20261004_v1/`.
Original `source/`, `pilot/`, and `validation_20261005/` are preserved. A copy
`source_holdout_20261005/` has the updated evaluation runner and its tests only;
all controller, official environment, and pinned RocketPy files remain exact.
New results: `holdout_20261005/`; launcher log `holdout_20261005.log`.
Launch PID `1783186` (verify command before acting on a PID).

The runner `scripts/validate_multiscenario_pilot.py` now takes `--split holdout`
and requires `--prior-plan`; passing no explicit seeds uses the frozen list.
It saves configs, `plan.json`, per-flight reports/logs, `status.json`, and a
`summary.json` only on full completion. Existing output directories are refused.

Local runner/cadence/summary tests: 10 passed. Remote runner/adapter/mixture
tests: 11 passed. Mock test scores printed during tests are synthetic fixtures,
not official flight scores. No GPU was made visible to this evaluation job.

## Conditional architecture direction (not yet implemented or trained)

The current policy is a [256,256] MLP making a three-component bounded
acceleration correction. Its features include at most eight balloon entries
(active target first, then nearby ones). The hand-designed planner still owns
launch time and route selection. A larger residual MLP alone cannot directly
make all of those discrete planning decisions.

If held-out performance is insufficient, inspect missed-target distances,
route changes, tracking error and saturation before choosing the intervention.
One proposed extension is a two-level learned agent:

1. A masked attention encoder over all publicly observed eligible balloons,
   plus rocket state and remaining burn-time context. Learn relations between
   reachable groups instead of relying only on the nearest eight entries.
2. A learned route/target selector considering multi-hit continuation and
   observation-based predicted interception timing, including feasible single
   targets rather than always preferring groups. Add history/memory only if
   state-estimation diagnostics support it; test it separately.
3. Retain physical action bounds and a feasible low-level controller. The
   learned module makes planning choices rather than only perturbing the base
   command. Check causal feature use, action masking and source-review rules
   before deployment; never expose seeds, hidden state or recorded futures.

Research motivation: [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)
uses attention-based learned heuristics for routing, including orienteering and
prize-collecting problems. It does **not** validate this rocket task: moving
targets, actuator dynamics, sensor noise and short powered flight require new
modeling and official-simulator evaluation. The above architecture is our
proposed adaptation, not a reproduced result or a guarantee of 20 pops.

Use a separate bounded pilot and ablations (route selector vs residual-only vs
base) before committing to long training. Any future GPU training must retain
the user's one-card limit. No such new training was launched in this stage.
