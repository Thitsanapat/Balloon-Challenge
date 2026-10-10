# Continued PPO vs frozen learned parent

The continuation stopped normally at its soft two-hour budget: 7,330.78 seconds,
10,220 new decisions, 23,258 cumulative decisions, 28 completed episodes, no
official truncations, unchanged source, no error file. Reset can delay the wall
limit check. Scenario-4 **training** scores: 18 episodes, mean 4.111, range 1--7;
these are not a paired comparison or proof of improved evaluation performance.

Continued final checkpoint SHA-256:
`79d588b6ccab6c2a9b9951febc9a42a4c9b77af6347f8a626e5570ff889cd0f6`.
Parent SHA-256:
`8aad07e2ac6b834601b1d992f0fbb3ce9bcce1d1bc845ea179623b9e98577b3b`.

Checkpoint/provenance/episode logs backed up locally at
`BalloonPoppingGymEnv/evaluation/results/multiscenario_continuation_final_20261006.zip`.
Remote/local archive SHA matches:
`7c53d0abddac80b87a3c1a5d8fcd21639709f34af05b441ba695500102f21e5d`.

## Predeclared validation

Three new seed identifiers sampled without inspecting fields/results using
`random.Random(261006).sample(range(1000000000,1500000000),3)`:
`1121306046, 1017421605, 1462280736`.
Disjoint from both training episode logs and both preceding evaluation plans.
This small set is validation, not proof of generalization. No seed replacements.

The baseline is now the **learned parent**, not the zero-residual controller.
Both agents use the same deterministic adapter and controller, differing only
in frozen checkpoint/manifest paths. The runner now supports
`--baseline-checkpoint`, guards both source manifests and checkpoint hashes,
and rejects comparison of identical checkpoints. Seven remote runner/adapter
tests passed before launch.

Six fresh official flights, CPU only, one worker, one numerical thread, nice
level 10; no new GPU training. Same timeout/error/source-change handling as the
prior tests. Both former runtime snapshots stay untouched. New runtime:
`/home/rtcmspaceweather/balloon_multiscenario_20261004_v1/source_compare_20261006/`.
Results: sibling `comparison_20261006/`, log `comparison_20261006.log`.
Launch PID `1462906`; inspect its command before acting on a PID.

## Status

Comparison completed: all six flights terminated normally in 1,494.35 s,
with unchanged source/model hashes. No truncations, no upload.

| Seed | Learned parent | Continued PPO |
| --- | ---: | ---: |
| 1121306046 | 4 | 5 |
| 1017421605 | 3 | 4 |
| 1462280736 | 3 | 2 |
| Mean per flight | 3.333 | 3.667 |

Continued model won two pairs and lost one; mean difference +0.333 pops, total
difference +1 across three flights. This small validation sample does not
establish a reliable advantage over the parent. Retain the ten-seed-tested
parent as the reference; preserve the continued checkpoint as a candidate, not
an automatically promoted submission. Do not compare these means directly to
the older 10-seed means as if fields were identical. No new training is running.

Full reports, configs and provenance are backed up locally in
`BalloonPoppingGymEnv/evaluation/results/multiscenario_comparison_complete_20261006.zip`.
Remote/local SHA-256:
`98fe6ab4afb6bfa5c20f818b5f80e113dfb53106072e9ed3259c149bbcedba00`.

The separate [route-selection design](route_attention_design_20261006.md) includes
an isolated, untrained attention prototype and tests. It does not change this
comparison's agents and has no measured competition score yet.
