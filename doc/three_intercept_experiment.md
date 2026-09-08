# Three-intercept follow-up toward a 14-point target

This experiment preserves the existing seven-point policy and submission. The
requested 14 points are an objective, not an achieved result.

## Method

`ThreeInterceptAgent` first obtains the existing best feasible two-target plan.
It then tests up to four nearby third targets, adjusting the second segment by
up to half a second and searching the third duration on the existing one-second
grid. The first target and its arrival deadline are preserved.

For each three-target chain, solve jointly for velocity and acceleration at both
interior knots. Coefficients are linear in those unknowns, so minimizing integrated
squared jerk is a small positive-definite quadratic problem. All three segments
must pass the inherited thrust, tilt, axis-rate, throttle-rate, ground-clearance
and pre-burnout checks. If no extension passes, retain the two-target plan.

This is a bounded extension of the best pair, **not** exhaustive three-target
route optimization. It can miss a better triple whose first pair was not chosen.
Minimum jerk is not maximum pop count, and sampled nominal feasibility does not
guarantee exact flight tracking. Only actual simulator pops count.

The new chain solver agrees with the previous two-segment solver, maintains
position/velocity/acceleration and natural jerk continuity, has lower jerk energy
than stopping at every waypoint, and rejects invalid durations.

## Controlled comparisons

- Three-target planner, launch 24 s: compare with the measured seven-point pair
  planner at identical launch time and other control/search settings.
- Existing pair planner, launch 18 and 30 s: bracket the previous best 24-second
  time; change no guidance gains or physical parameters.

The 18-second run reuses the previously recorded seed-0 field through 58.51 s.
The other two use fresh field generation. The fresh evaluation tool now supports
`--set KEY=VALUE` for agent parameters and records the effective configuration in
the report without editing official scenario YAML.

```powershell
$env:MPLCONFIGDIR = Join-Path $env:TEMP 'bpc-mpl-cache'
$env:PYTHONWARNINGS = 'ignore'
.\.venv\Scripts\python.exe -m unittest tests/test_three_intercept.py tests/test_spline_route.py tests/test_physics_guidance.py -q
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/three_intercept_launch24.yaml --seeds 0 --report BalloonPoppingGymEnv/evaluation/results/three_intercept_launch24_fresh_seed0_metrics.json
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/spline_route_launch24.yaml --set launch_time=30 --seeds 0 --report BalloonPoppingGymEnv/evaluation/results/spline_route_launch30_fresh_seed0_metrics.json
.\.venv\Scripts\python.exe scripts/benchmark_cached_field.py BalloonPoppingGymEnv/evaluation/configs/spline_route_launch24.yaml BalloonPoppingGymEnv/evaluation/results/20260907T171601Z_trajectory.json --set launch_time=18 --report BalloonPoppingGymEnv/evaluation/results/spline_route_launch18_seed0_metrics.json
```

## Completed initial results

| Variant (seed 0) | Evaluation | Score | Pop times (s) |
| --- | --- | ---: | --- |
| Previous pair planner, launch 24 | Previous fresh run | 7 | 31.91, 34.68, 39.61, 43.52, 47.33, 49.18, 52.38 |
| Three-intercept extension, launch 24 | Fresh run | 7 | 31.91, 34.74, 39.68, 43.57, 47.41, 49.25, 52.49 |
| Pair planner, launch 18 | Cached field | 5 | 26.92, 29.76, 33.66, 38.38, 41.63 |
| Pair planner, launch 30 | Fresh run | 6 | 38.89, 40.93, 47.78, 49.61, 54.48, 56.89 |

The triple extension does not increase score and finishes its seventh hit later
than the existing policy. Keep the original seven-point controller as the best.
These comparisons do not prove that 14 points are impossible.
The triple run tested 1,231 extensions, found 27 feasible triples, and applied a
triple first-segment plan on 25 replans. Thus its unchanged score is not merely a
case of never activating the extension. Its full fresh evaluation took 130.14
wall seconds on this machine.

## Follow-up: continuous segment timing

`TimedSplineAgent` keeps the two chosen targets but refines both durations with
SLSQP after the discrete pair search. For each candidate duration pair it jointly
refits the intermediate velocity/acceleration with the existing minimum-jerk
solver, then imposes physical constraints on both segments. The objective is total
predicted time plus a penalty for delaying a held first target. Duration changes
are bounded to [-1.5,+0.5] seconds around each seed, with positive floors.

The optimizer has a 20-iteration cap and a small interior feasibility margin. Its
output is independently rechecked and only improving feasible trajectories replace
the baseline pair. This tests whether the one-second duration grid is losing
throughput; it does not optimize the entire route or guarantee maximum score.
`planned_seconds_saved` sums overlapping forecast improvements and must **not** be
reported as actual episode time saved. A regression test verifies that an invalid
optimizer result cannot replace a feasible baseline plan.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/timed_spline_launch24.yaml --seeds 0 --report BalloonPoppingGymEnv/evaluation/results/timed_spline_launch24_fresh_seed0_metrics.json
```

### Timing result and final status

The fresh seed-0 timing run scored **4** (30.51, 32.77, 43.63, 50.37 s), despite
accepting 20 local improvements from 22 solver calls. It reached the first two
balloons earlier but had a 10.86-second gap before its third hit. Optimizing a
short-horizon timing objective therefore did not improve total episode score.
The reported 17.95 planned seconds saved is an overlapping prediction diagnostic,
not realized time savings. Evaluation took 127.81 wall seconds.

All 23 focused tests passed; compile and diff-whitespace checks passed. The best
verified policy remains the previous two-target planner at launch 24 s, with its
existing seven-point submission. No new submission was generated or uploaded in
this iteration because none of the candidates beat it. The requested **14 points
remain unachieved**; no physical impossibility claim is made.
