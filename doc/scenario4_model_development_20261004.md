# Scenario 4 model development, 2026-10-04

The established comparison policy is `Scenario4WindProfileAgent` with the
42-second launch config. It uses only `given_parameters` and current/past
observations. Scenario 4 uses a new independently drawn seed during the live
round, so development scores on a small set of seeds are not a guarantee of
qualification performance. No tested candidate below replaces the established
policy or the existing submission artifact.

## Time-aligned swept-corridor route model

`Scenario4SweptCorridorAgent` changes beam ordering only after the normal
rocket-feasibility checks. For each sampled spline interval it minimizes the
*relative* rocket-balloon distance, so two objects at the same place at
different times do not count as a collision opportunity. Each unplanned,
released balloon contributes at most once, with proximity and time discounts.
It is a planning heuristic, not a calibrated hit probability.

| Fresh official Scenario 4, seed 0 | Actual pops | Wall time |
| --- | ---: | ---: |
| Established wind-profile policy | 5 | 315 s |
| Two-leg corridor bonus | 3 | 280 s |
| Conservative first-leg multi-balloon bonus | 5 | 266 s |

The aggressive variant lost the first three established pop opportunities
when its route timings changed; it is rejected. The conservative variant
selected exactly the same route events as the established policy. Its selected
first legs offered no reliable additional multi-balloon sweep, so there is no
evidence for promoting it. Reports:
`scenario4_wind_profile_launch42_seed0_20261003.json`,
`s4_dev_swept_corridor_seed0_20261004.json`, and
`s4_dev_swept_corridor_conservative_seed0_20261004.json`.

## Observation-triggered launch timing

`Scenario4AdaptiveLaunchAgent` checks whether a group of *released* balloons
has a time-aligned, constant-acceleration first-hit proxy. It cannot read a
seed, unreleased state, hidden wind, or a future simulator trajectory. If the
proxy is not strong enough it keeps the 42-second launch. Normal trajectory
search still decides the actual route after launch.

| Fresh official Scenario 4 | Seed 0 | Seed 1 | Early trigger? |
| --- | ---: | ---: | --- |
| Established 42-second policy | 5 | 2 | Not applicable |
| Group radius 8 m, horizon 7.5 s | 5 | 2 | Neither |
| Group radius 20 m, horizon 7.5 s | 5 | 2 | Neither |
| Group radius 20 m, horizon 10 s, mid-burn thrust proxy | 5 | Not run | No |

The initial 7.5-second horizon was too pessimistic: the established seed-0
first hit takes about nine seconds after launch. Correcting that and using
the public mid-burn mass schedule still found only one or two reachable
balloons in the prelaunch window, not a four-balloon group. Expanding radius
further merely to cause a trigger would no longer be evidence of a capturable
group. This experiment is parked, not promoted. Reports are named
`s4_dev_adaptive_launch_seed{0,1}_20261004.json`,
`s4_dev_adaptive_launch_group20_seed{0,1}_20261004.json`, and
`s4_dev_adaptive_launch_group20_horizon10_seed0_20261004.json`.

## Residual learning foundation

The existing official residual-PPO wrapper accepted `--scenario 4` but still
constructed the Scenario-1/24-second controller. It now selects a separate
`Scenario4ResidualController` with exactly the established Scenario-4 settings.
The residual is bounded, passes through the ordinary thrust/tilt allocator,
and exactly bypasses correction when zero. Scenario-4 features keep the active
target before other nearby released balloons in the fixed 96-value vector;
Scenario-1 features retain their old behavior. The Scenario-4 training
potential follows the selected target instead of an unrelated nearest
balloon. A target switch suppresses that step's potential-shaping reward and
clears the previous residual; no active plan means no residual force.

The untrained zero-output `Scenario4ResidualPolicyAgent` completed fresh
official seed-0/seed-1 evaluations with **5/2 pops and identical pop IDs/times**
to the established policy, with unchanged source files during each run.
Reports: `s4_dev_residual_zero_seed{0,1}_20261004.json`. The target-switch
safety guard was added after these runs and tested at unit level; it is inert
when the residual is zero. This verifies a neutral starting point, **not** a
learned score improvement. Unit tests cover zero-action
equivalence, bounded force, feature shape/active target, wrapper selection,
target-switch clearing, and route/launch geometry.

Next model experiment: train a small correction predictor from multiple fresh
observation trajectories, using the *later observed target position* as a
training label. Keep train/validation/holdout seeds disjoint; then, only if
forecast or terminal hit metrics improve, fine-tune bounded residual control
with official episodes. A single GPU can update the network, but official
RocketPy resets and flights are CPU/simulator-bound and take several minutes
per seed here. A large PPO step count or a numerically large seed does not
itself establish robust performance. Do not promote a checkpoint until it
beats the established policy on paired, previously unseen seeds without
official time truncations.

Local PyTorch reports no CUDA device. The professor's host answered SSH but
batch key authentication failed; its GPU state was not inspected and no remote
job was started. Use a verified, secure login and check that exactly one GPU
is free before any training there. The trainer rejects CUDA unless exactly
one GPU is explicitly exposed.
