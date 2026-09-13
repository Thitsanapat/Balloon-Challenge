# Multi-target GNC experiments (2026-09-11)

The requested target is **20 popped balloons**. It has not been demonstrated.
The existing `spline_route_launch24.yaml` reference is preserved unchanged.
No simulator dynamics, balloon radius, scoring, or submission credentials were
modified by these experiments. No submission was uploaded.

## Models implemented

- `BeamInterceptAgent`: depth-limited search carrying arrival time, position,
  velocity, and acceleration between targets. Each branch blends a
  position-only minimum-jerk fly-through with a stopped arrival. For normalized
  time, the free-endpoint displacement coefficients are `(5/3, -5/6, 1/6)`;
  they satisfy natural endpoint conditions `jerk(T) = snap(T) = 0`.
- `JointBeamAgent`: SLSQP refinement of leg durations and endpoint velocities
  and accelerations, followed by receding-horizon tracking. Predicted targets
  use only observed positions and velocities. The first implementation forced
  a stopped final arrival and accepted no joint plans; preserving the feasible
  beam endpoint states fixes that initialization problem.
- `CaptureSplineAgent`: plans toward a point inside the near side of the
  balloon sphere, using a private copy of the observations. This leaves the
  real collision geometry unchanged, but did not improve the reference.
- `OptimizedSplineAgent`: applies the same SLSQP refinement to the reference
  two-target route instead of a beam-selected route. This matched, but did not
  exceed, the best seed-0 score in the first test.

All planners retain the original sensor-based attitude/tracking controller.
The optimizer constrains available thrust from the known burn/mass model,
axis tilt and rotation rate, altitude, throttle slew, and remaining burn time.
Optimization checks 33 temporal samples per leg; acceptance checks 65, with
normalized numerical tolerance `1e-6`. If a boundary solution fails the denser
check, parameters are backtracked toward the feasible seed. Squared rate
constraints avoid the nondifferentiability of a norm at zero angular rate.

These are reduced translational models with sampled constraints, not full
6-DOF trajectory optimization or proofs of continuous-time feasibility. The
simulator remains the actual closed-loop test. This implementation uses SLSQP,
not the successive-convexification algorithm described in the related
[6-DOF guidance paper](https://arxiv.org/abs/1811.10803).

## Completed development benchmarks

Scenario 1, seed 0, identical full balloon-field cache. This cache is injected
by the development harness only; the agent never receives future trajectories.
The cache previously reproduced the fresh reference's score and impact times.

| Policy | Launch (s) | Score |
|---|---:|---:|
| Preserved two-target reference | 24 | 7 |
| Beam fly-through | 12 | 5 |
| Beam fly-through | 24 | 5 |
| Joint beam, corrected initialization | 23 / 24 | 5 / 5 |
| Joint beam, free derivatives + dense validation | 23 / 24 | 5 / 5 |
| Capture sphere, radius fraction 0.7 / 0.9 | 24 | 6 / 6 |
| Optimized reference pair | 23 | 6 |
| Optimized reference pair | 24 | 7 |
| Optimized reference pair | 24.25 | 4 |
| Optimized pair, 0.5 s grid and five next-target candidates | 24 | 6 |

The optimized 24-second pair policy hit balloons
`58, 27, 37, 93, 80, 8, 56` at times
`30.59, 33.20, 38.24, 41.64, 47.69, 49.41, 52.68` seconds.
Its first hit was 1.32 seconds earlier than the reference, but its changed
downstream route did not yield an extra hit. Matching a score is not an
improvement, and these results do not establish that 20 is impossible.

## Verification and reproduction

Fresh, uncached evaluations of `optimized_spline.yaml` completed normally:

| Seed | Preserved reference | Optimized pair |
|---:|---:|---:|
| 0 | 7 | 7 |
| 1 | 6 | 7 |

The seed-0 fresh impact IDs and times match the cached benchmark. Both fresh
runs terminated without truncation. The improvement is one extra hit on seed
1, not progress to 20 or evidence of broad generalization from just two seeds.
The optimized configuration stays experimental because its seed-0 score drops
from 7 to 4 when launch time changes from 24 to 24.25 seconds.

Twenty targeted tests passed, covering polynomial boundary/natural conditions,
continuous chain derivatives, finite-radius observation isolation, reduced
travel time on a simple feasible route, burnout rejection, and existing
physics/two-target/three-target planner tests.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_beam_intercept tests.test_joint_beam tests.test_capture_spline tests.test_physics_guidance tests.test_spline_route tests.test_three_intercept -v
.\.venv\Scripts\python.exe -m scripts.evaluate_guidance BalloonPoppingGymEnv/evaluation/configs/optimized_spline.yaml --seeds 0 1 --report BalloonPoppingGymEnv/evaluation/results/optimized_spline_fresh_validation.json
```

The second command regenerates the balloon field in the unmodified official
environment and does not package or upload a submission. Per-run metrics are
under `BalloonPoppingGymEnv/evaluation/results/`; they are development reports,
not uploadable submission objects.

## Public workshop and replay audit (2026-09-11)

The official `ws/coscup-2026` branch confirms that the one-balloon example is
an intentionally reduced GNC loop. It predicts the target with measured
acceleration, applies near-target zero-effort-miss correction, and uses a
cascaded attitude/body-rate controller capped at 1.35 rad/s. Its documentation
also says the unreleased competition implementation included multi-target
planning, replanning, and opportunity targets; that full implementation is not
present in the public Git history.

Public leaderboard replay geometry was used only for diagnostics, never as a
submission policy. The 16-point route climbs through nine balloons to a
315.2-m apogee, then collects seven on descent. The 17-point PPO route travels
only 433.2 m horizontally, climbs through its first sixteen targets to about
380 m, and takes its final target while descending. In contrast, the verified
seven-point seed-0 route reaches apogee around its third hit and remains near
100--132 m. This supports energy management through route choice, not a forced
vertical-acceleration clip.

A state-tracking diagnostic following an already feasible public trajectory
improved from 1 to 12 points after adding position/velocity feedback; an
iterative correction experiment reached 13. An offline inverse-dynamics solve
then reconstructed the actuator history with median derivative residual below
`3e-6`; closed-loop replay of those controls reproduced all 17 public impacts
in a fresh run of the local 6-DOF simulator. These numbers are not claimed as
our competition score because the diagnostic reads an external replay/control
history. The
same direct rate controller applied to our online two-target optimizer scored
0, while forcing a minimum climb acceleration scored 1. The cause is reference
feasibility: aggressive control cannot repair a route whose requested turns
and vertical-energy constraint conflict.

The closest unused targets to the 17-point path are #75 (12.0 m near 60.6 s)
and #28 (18.9 m near 38.5 s). Smooth insertion of #75 between #20 and #18
requires roughly 44--49 m/s^2 in a minimum-jerk perturbation; the more general
free-velocity cubic estimate remains above 90 m/s^2. Simulator detours reduced
the #75 miss to 3.1 m but lost the following targets and scored 11. This rules
out that local insertion, not all possible 18-point routes.

A fast point-mass Gymnasium curriculum and Stable-Baselines3 PPO harness now
run at about 830--1,200 environment steps/s on the current 8-logical-CPU
machine. Zero-residual tracking reproduces 16 of the 17 expert impacts in the
surrogate (the simplified ballistic tail misses #26). Two 131,072-step PPO
fine-tuning trials retained deterministic score 16; neither improved the #28
closest approach. At the measured throughput, 100 million local steps would
take about 33 hours before 6-DOF transfer/validation. These short runs validate
the pipeline but are not evidence that PPO has converged.

The best submission-eligible result remains 7. The public leaderboard evidence
available during this audit tops out at 17, from a PPO policy described as
trained for 100 million steps. Reaching 20 therefore requires a new training
track (most plausibly PPO imitation/curriculum plus a fast surrogate or GPU),
not another hand-tuned spiral gain.
