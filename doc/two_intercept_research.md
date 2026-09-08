# Two-intercept guidance: repository audit and experiment

## Source audit (2026-09-08)

- [RocketPy](https://github.com/RocketPy-Team/RocketPy) is the underlying variable-mass
  six-DOF flight simulator, not an out-of-the-box balloon route planner.
- [ActiveRocketPy](https://github.com/ARRC-Rocket/ActiveRocketPy) adds active TVC,
  throttle and roll control. Its documented limitation schedules mass properties
  and cutoff independently of throttle. Local `flight.py` confirms that throttle
  scales thrust, while mass flow is obtained separately from the motor.
- [BalloonPoppingChallenge](https://github.com/ARRC-Rocket/BalloonPoppingChallenge)
  supplies the observation/action contract, balloon generation, collision scoring,
  and pinned ActiveRocketPy dependency. Launch time and direction are agent choices;
  changing timestep, balloon radius, vehicle properties or scoring is not a valid
  way to improve comparable scores.

Inspected local challenge commit: `4a8abaa0fc9b63ae819dc3892986132324edaa26`.
Its recorded ActiveRocketPy pin and actual submodule HEAD both equal
`3b3d6c02feafab8e0ef29908c75a8190a6cdc268`; the submodule worktree is clean.
This verifies local pin consistency, not equality to today's remote HEAD. No pull,
dependency upgrade, or official simulator/scenario modification was performed.

## Hypothesis

The single-target matched-velocity planner slows to balloon drift at each hit.
Instead jointly choose intermediate velocity and acceleration for two consecutive
hits. This tests route continuity/throughput; it does not require a spiral.
The previous 5-point empirical agent and submission remain untouched.

For each candidate pair and two candidate durations, predict both positions using
only currently observed balloon velocities. Construct two quintics with common
position, velocity and acceleration at their join. Start position/velocity/
acceleration are measured; final velocity matches the second balloon's drift and
final acceleration is zero. Only the interior velocity and acceleration are free.

Write the normalized coefficients as `c1 = b1 + D1*q`, `c2 = b2 + D2*q`, where q
contains interior velocity and acceleration, independently for each spatial axis.
Integrated squared jerk is a positive quadratic with entries

    Q[i,j] = f[i]*f[j] / ((i+j+1)*H^5), f = [6,24,60].

Thus solve the small linear system

    (D1.T Q1 D1 + D2.T Q2 D2) q
      = -(D1.T Q1 b1 + D2.T Q2 b2).

This is the exact minimum-jerk interior knot for fixed endpoints/times, **not** a
global minimum-time or maximum-pop solution. Search up to eight nearby first
targets and three nearby successors over a bounded duration grid, minimizing total
time to the two predicted hits. Both segments must pass the inherited 33-sample
thrust, tilt, axis-rate, throttle-rate, ground and burnout constraints. Infeasible
minimum-jerk pairs are rejected rather than being projected into a different
constrained optimum. If none pass, use the original single-intercept planner.

Only the first segment is flown before replanning. The second hit is therefore a
prediction, not a credited score. First-target locking reduces target switching,
but it does not guarantee that receding-horizon replans preserve an entire route.
No future field cache or environment internals are available to the agent.

## Predefined comparison

- Launch at 4 seconds: comparable early-flight opportunity to the baseline.
- Launch at 12 seconds: more balloons observed before ignition, at the cost of
  chasing a higher, drifting field. Same controller/search settings otherwise.

Both use Scenario 1 cached seed 0 and the unmodified rocket/scoring simulation.
The cache is a development acceleration, not an official submission. No result
from it establishes changed-wind robustness or a score above 15.

```powershell
$env:MPLCONFIGDIR = Join-Path $env:TEMP 'bpc-mpl-cache'
$env:PYTHONWARNINGS = 'ignore'
.\.venv\Scripts\python.exe -m unittest tests/test_spline_route.py tests/test_physics_guidance.py -q
.\.venv\Scripts\python.exe scripts/benchmark_cached_field.py BalloonPoppingGymEnv/evaluation/configs/spline_route_scenario_1.yaml BalloonPoppingGymEnv/evaluation/results/20260907T144737Z_trajectory.json --set launch_time=4 --report BalloonPoppingGymEnv/evaluation/results/spline_route_launch4_seed0_metrics.json
.\.venv\Scripts\python.exe scripts/benchmark_cached_field.py BalloonPoppingGymEnv/evaluation/configs/spline_route_scenario_1.yaml BalloonPoppingGymEnv/evaluation/results/20260907T144737Z_trajectory.json --set launch_time=12 --report BalloonPoppingGymEnv/evaluation/results/spline_route_launch12_seed0_metrics.json
```

Unit tests verify endpoint conditions, C2 continuity, and increased jerk cost under
perturbations of the solved interior knot, alongside existing physical checks.

## Initial results

| Cached seed 0 variant | Score | Pop times (s) | Accepted pair replans |
| --- | ---: | --- | ---: |
| Launch 4 s | 5 | 13.93, 16.75, 21.66, 24.56, 28.58 | 25 |
| Launch 12 s | 6 | 19.86, 23.74, 27.15, 32.84, 35.76, 40.03 | 29 |

These improve on the four-point single-intercept physics baseline. The 12-second
variant is the first observation-only candidate here to exceed the five-point
empirical reference on the cached seed-0 field, but is still far below 16 points.
It evaluated 4,864 pair candidates, of which 34 passed both-segment checks across
the episode; those are planning diagnostics, not hit counts or upper bounds.

Fresh generation confirmed the 12-second variant's 6 points; an official-packer
JSON was produced locally. Follow-up 24- and 40-second launches tested the
delayed-launch hypothesis further without changing guidance settings. These
require fresh evaluation because their flights extend beyond the original
cache's 49.88-second horizon.

| Fresh seed 0 variant | Score | Pop times (s) |
| --- | ---: | --- |
| Launch 12 s | 6 | 19.86, 23.74, 27.15, 32.84, 35.76, 40.03 |
| Launch 24 s | 7 | 31.91, 34.68, 39.61, 43.52, 47.33, 49.18, 52.38 |
| Launch 40 s | 5 | 49.10, 53.62, 56.45, 63.24, 64.32 |

The best measured policy is therefore `spline_route_launch24.yaml`. Waiting longer
is not monotonically beneficial. The 7-point result is a complete fresh local
simulation, not a leaderboard-verified score. A repeated run is used to generate
its submission and seed 1 is evaluated without retuning. The requested score
greater than 15 has **not** been achieved or proven feasible/impossible.

For the measured 7-point route, the first hit is 7.91 seconds after the configured
launch time and subsequent hit gaps average 3.41 seconds. Reaching 16 hits before
the approximate 54.01-second cutoff, with that same first-hit time, would require
15 further hits averaging about 1.47 seconds apart. This quantifies the gap for
this route, not a physical upper bound: different first-hit times/routes and
post-burnout hits could change the calculation. No claim of a 16-point solution is
made. The 24-second fresh run took about 116.5 wall seconds, below the scenario's
600-second limit on this machine during the experiment.

## Final verification for this iteration

The repeated fresh seed-0 run reproduced **7 points** and the official local
packer generated
`20260907T171617.055Z_TASTI_Cool_Ba_Malaew_84b67800899143869cc387780a2477b0_submission.json`.
It was parsed strictly as a JSON object (format 2); its final trajectory contains
seven popped balloons and its embedded agent source matches the current file.
It contains team credentials and is ignored by Git; do not publish its contents.
The packer's advisory remote `evaluate.py` integrity check was blocked by local
network permissions. No leaderboard upload or server acceptance is claimed.

Fresh seed 1, with the same 24-second configuration and no retuning, scored **6**
(32.54, 35.36, 38.18, 42.09, 46.65, 51.42 s). Two seeds do not establish robustness
to different wind profiles. Eighteen focused geometry/control/optimization tests
passed. The requested >15 score remains unmet; best verified local seed-0 score
for this iteration is 7, up from the previous 5.
