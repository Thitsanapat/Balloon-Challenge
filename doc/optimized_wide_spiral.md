# Wide wind-relative spiral and local trajectory optimization

## Experiment definition

Both new configurations start with a **25 m radius**, expanding at 0.35 m/s,
without waiting for a first pop. The geometric axis follows smoothed observed
balloon drift, with a 45-degree tilt cap. This is a drift proxy, not a direct
measurement of wind at the rocket. Rocket attitude remains independently
controlled and limited. The center is offset so the initial reference position
coincides with the rocket; starting at radius 25 m does not teleport it.

- `wide_spiral_scenario_1.yaml`: capture future spiral waypoints only.
- `optimized_spiral_scenario_1.yaml`: use the spiral as a **soft reference**,
  choose observed balloon targets, then optimize the intercept trajectory.

The latter may leave the spiral substantially to reach a balloon. It is not a
strict helical trajectory with guaranteed revolutions. Neither experiment uses
hidden wind, future cached states inside the agent, or target IDs fixed in code.

## Optimization actually implemented

The general finite-parameter objective/constraint formulation is motivated by
[MIT's trajectory optimization notes](https://underactuated.csail.mit.edu/trajopt.html).
For up to twelve nearby released balloons, generate feasible quintic seeds with
matched drift velocity or a 4 m/s tangential exit component. Predict target
position from current observed velocity. Choose among those discrete candidates
and refine up to three seeds with
[SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).

Continuous decision variables are arrival duration H and three components of
terminal velocity relative to balloon drift. Quintic endpoint constraints fix
initial position/velocity/acceleration and intercept position/velocity with zero
terminal acceleration. The local cost is

    J = H + 0.4 * mean(||p-p_spiral||^2)/radius_now^2
          + 0.002 * integral(||jerk||^2 dt)
          + 4 * max(0, H-previous_remaining_arrival_time).

The last term applies only to a held target, discouraging indefinite arrival
delays. The weights are explicit engineering choices, not learned optima or
proof that the proxy objective maximizes competition score. Jerk energy is
integrated analytically; spiral deviation uses nine samples. SLSQP runs with a
15-iteration cap, when changing target or every two seconds. Replanning between
optimizer calls still recomputes feasible quintics using measured state.

Constraints use the existing controller's 33-sample thrust budget, pointing cone,
thrust-axis rate, throttle rate, ground clearance, and pre-burnout horizon.
Terminal relative-velocity components are bounded to +/-6 m/s. A 1e-5 normalized
interior margin helps avoid solver-tolerance solutions being rejected by the
strict feasibility checker. Every candidate is independently rechecked; if no
improvement passes, keep the best feasible seed. Solver success alone is not
treated as validity, and a feasible improving iterate may be used even when the
iteration limit was reached. Diagnostics distinguish calls, accepted objective
improvements, and solver non-success statuses.

This is local one-intercept receding-horizon optimization, not a globally optimal
multi-balloon route, a full six-DOF optimal-control solve, or a continuous-time
feasibility certificate. The force/mass model and feedback are inherited from
`PhysicsGuidanceAgent` (see `physics_guidance.md`).

## Reproduce and verification

```powershell
$env:MPLCONFIGDIR = Join-Path $env:TEMP 'bpc-mpl-cache'
$env:PYTHONWARNINGS = 'ignore'
.\.venv\Scripts\python.exe -m unittest tests/test_optimized_spiral.py tests/test_wind_spiral.py tests/test_physics_guidance.py -v
.\.venv\Scripts\python.exe scripts/benchmark_cached_field.py BalloonPoppingGymEnv/evaluation/configs/wide_spiral_scenario_1.yaml BalloonPoppingGymEnv/evaluation/results/20260907T144737Z_trajectory.json --report BalloonPoppingGymEnv/evaluation/results/wide_spiral_seed0_metrics.json
.\.venv\Scripts\python.exe scripts/benchmark_cached_field.py BalloonPoppingGymEnv/evaluation/configs/optimized_spiral_scenario_1.yaml BalloonPoppingGymEnv/evaluation/results/20260907T144737Z_trajectory.json --report BalloonPoppingGymEnv/evaluation/results/optimized_spiral_margin_seed0_metrics.json
```

Tests check exact jerk integration, agreement between optimizer margins and the
existing feasibility checker, a wide start before any pop, rejection of an
infeasible solver output, and a real solver improving a feasible seed. Existing
geometry/control checks remain in the suite. Cached-field reports are development
results, not submission files or changed-wind robustness validation.

The initial no-margin optimized run scored 3 (pops at 13.49, 23.11, 29.95 s), but
accepted **zero SLSQP refinements**. That is a target-aware seed-planner result,
not evidence of benefit from continuous optimization. Its original report is
`optimized_spiral_seed0_metrics.json`; the corrected-margin run is separate.
The wide-waypoint-only run scored 0 with no accepted plans, so it never flew a
successful wide spiral. Comparing it with the target-aware planner does not
isolate the effect of SLSQP alone.

## Completed cached-field results (2026-09-07)

| Variant, Scenario 1 seed 0 | Score | Pops (seconds) | Accepted SLSQP improvements |
| --- | ---: | --- | ---: |
| Wide tilted spiral waypoints only | 0 | none; no accepted flight plan | n/a |
| Target-aware wide spiral, original no-margin solver | 3 | 13.49, 23.11, 29.95 | 0 |
| Target-aware wide spiral, strict-checked margin solver | 3 | 20.82, 25.06, 29.59 | 10 |
| Previous physics interception baseline | 4 | 13.49, 17.88, 23.60, 31.15 | n/a |
| Previous seed-specific empirical reference | 5 | 9.48, 10.34, 15.89, 21.11, 27.57 | n/a |

The margin solver made 18 calls, accepted 10 objective improvements, and reported
4 non-success statuses. It changed the trajectory but did not improve pop count
over the seed-based variant; its first pop was later. Thus improving the surrogate
objective is not equivalent to improving the competition score. The final observed
geometric tilt was about 37 degrees. Sixteen unit checks passed. Keep the empirical
reference as the best known seed-0 score; do not promote this experiment on score.
