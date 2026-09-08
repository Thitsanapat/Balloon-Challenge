# Wind-relative expanding spiral experiment

## Hypothesis and scope

Test whether tilting an expanding spiral's geometric axis toward observed balloon
drift improves popping compared with the same spiral around a vertical axis.
`WindSpiralAgent` is a separate experiment; the existing interception agents and
official simulator/scenario files are unchanged. This does not spin the rocket
about its own longitudinal axis.

No direct wind measurement is assumed. The median velocity of released, moving
balloons is smoothed with a one-second time constant. This is a cloud-drift proxy,
not an exact estimate of air velocity at the rocket (especially with wind shear).
The axis azimuth follows horizontal drift; its tilt is

    beta = min(beta_max, atan2(norm(drift_xy), max(drift_z, 0.1))).

The experiment uses beta_max=45 degrees, versus zero in the control. Rocket
thrust-axis tilt remains independently limited to 55 degrees by the inherited
controller. If the drift direction changes, the geometric frame updates on the
next replan. This behavior is unit-tested under a 90-degree rotation, not yet
validated in a full simulator run with changed wind.

## Geometry and tracking

First use the existing feasible intercept planner until the first balloon pops.
Start the spiral at that rocket position, offsetting its center so there is no
position jump at entry. Thereafter advect the center with the observed drift.
Let u and v span the plane perpendicular to the axis, c be that advected center,
and d be the estimated drift. Over a candidate horizon h:

    r(h) = r_now + growth*h
    theta(h) = phase_now + omega*h
    e_r = cos(theta)*u + sin(theta)*v
    e_t = -sin(theta)*u + cos(theta)*v
    p(h) = c + d*h + r(h)*e_r
    velocity(h) = d + growth*e_r + r(h)*omega*e_t
    acceleration(h) = 2*growth*omega*e_t - r(h)*omega^2*e_r

Radius starts at 2 m on spiral entry and expands at 0.35 m/s. Angular speed is capped at 0.4 rad/s
and additionally limited by a quarter of the ideal level-flight lateral reserve
using the largest radius remaining before burnout. This heuristic limit alone
does not certify feasibility; capture quintics are checked for thrust, attitude
rate, throttle rate, ground clearance, and burnout using the inherited checks.

Each replan builds a smooth quintic from the measured rocket state to a future
spiral waypoint, matching its position, velocity, and acceleration. This is
**receding-horizon spiral-waypoint capture**, not exact tracking of a single fixed
helix. The advected center, observed drift, frame, and angular speed can change. Plans
are sampled for feasibility, not continuously certified. No-feasible-plan cases
retain a still-current plan, otherwise use the inherited descent-arrest fallback.
Until the first pop and enough moving balloons are visible, the ordinary intercept planner is used.

This first ablation deliberately does not snap waypoints to individual balloons:
it measures the expanding-sweep idea, which can miss sparse targets even when its
geometric path is flyable. The spiral is not evidence that 20 points are reachable.

## Reproduce

From the project root with the existing virtual environment:

```powershell
$env:MPLCONFIGDIR = Join-Path $env:TEMP 'bpc-mpl-cache'
$env:PYTHONWARNINGS = 'ignore'
.\.venv\Scripts\python.exe -m unittest tests/test_wind_spiral.py tests/test_physics_guidance.py -v
.\.venv\Scripts\python.exe scripts/benchmark_cached_field.py BalloonPoppingGymEnv/evaluation/configs/wind_spiral_scenario_1.yaml BalloonPoppingGymEnv/evaluation/results/20260907T144737Z_trajectory.json --set spiral_axis_tilt=45 --report BalloonPoppingGymEnv/evaluation/results/wind_spiral_capture_tilted_seed0_metrics.json
.\.venv\Scripts\python.exe scripts/benchmark_cached_field.py BalloonPoppingGymEnv/evaluation/configs/wind_spiral_scenario_1.yaml BalloonPoppingGymEnv/evaluation/results/20260907T144737Z_trajectory.json --set spiral_axis_tilt=0 --report BalloonPoppingGymEnv/evaluation/results/wind_spiral_capture_vertical_seed0_metrics.json
```

Only the axis tilt limit differs between the two spiral runs. Both use the same
recorded Scenario 1, seed 0 balloon field with current rocket/scoring dynamics.
Cached-field metrics are development results, not uploadable submission objects
or leaderboard validation. Unit tests check frame orthogonality/handedness,
wind-direction rotation, the vertical-axis control, and analytic derivatives and
radius growth, and capture/spiral entry continuity and endpoint derivatives,
alongside the six existing physics/controller checks.

## Initial entry failure

The first implementation targeted the median of all released balloons from the
pad. Both axis settings scored 0, with 0 accepted plans and 80 failed replans.
Therefore these runs do **not** measure the effect of flying a tilted spiral:
neither rocket actually entered one. At t=4.01 the cached field's released-cloud
median was approximately [-75.2, 74.4, 32.3] m, far from the launch site.
Those reports are retained as `wind_spiral_{tilted,vertical}_seed0_metrics.json`
with the original source hash. The revised entry strategy above uses one feasible
intercept first; its reports have the distinct `wind_spiral_capture_` prefix.

## Revised-entry results (2026-09-07)

| Controller, Scenario 1 cached seed 0 | Score | Pop events | Accepted plans |
| --- | ---: | --- | ---: |
| Capture then vertical-axis spiral | 1 | balloon 17 at 13.49 s | 72 |
| Capture then drift-tilted spiral (45-degree cap) | 1 | balloon 17 at 13.49 s | 72 |
| Previous physics intercept baseline | 4 | 13.49, 17.88, 23.60, 31.15 s | see baseline report |

Both spiral runs executed accepted capture plans after initial entry but produced
**zero additional pops during the spiral phase**. The tilted frame's final
reported angle was 37.05 degrees; it is observation-dependent, not fixed at 45.
Neither experiment improves on the existing baseline. This does not prove all
wind-relative spirals are ineffective: only this radius/speed/entry policy in
one cached field was tested. There was no full changed-wind or fresh-field
validation and no leaderboard upload. Eleven unit checks passed.

The repeated replanning locally resets polynomial tracking error; the diagnostic
error sum is not a global helix-tracking accuracy measure. Accepted plans are
sampled nominal feasibility, not a guarantee of exact flight. Saturation counts
include post-burnout flight. A target-aware arc/waypoint planner is a possible
follow-up, but is not implemented or credited with a score here.
