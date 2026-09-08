# Launch-at-45 inclined cloud-corridor experiment

## Hypothesis

At 45 seconds, 91 of Scenario 1's 100 balloons have been released (the final
release is at 49.5 seconds). Fit the observed cloud's horizontal displacement as
a function of altitude, then prefer dense targets near an inclined central core.
This tests the proposed picture of a roughly quarter-area central group rising at
different times and leaning with drift. It does not alter release times or fields.

`CloudCorridorAgent` computes the axis solely from currently released balloon
positions. Its core radius is half the observed 80th-percentile radial distance;
for a circular cross-section, half radius corresponds to one-quarter area. This is
a soft target-ranking preference, not a claim that the board or actual cloud is a
circle. Neighbour density uses a 30 m Gaussian kernel. Descending along the fitted
axis is penalized so the planned pair generally climbs with the cloud.

The underlying two-intercept minimum-jerk planner and physical feasibility checks
remain unchanged. If no corridor-ranked pair is feasible, it falls back to the
ordinary single-intercept planner. No hidden wind, future trajectory, balloon ID,
or environment private state is read by the agent.

## Fresh seed-0 result (2026-09-08)

| Launch 45 s controller | Score | Pop times (s) |
| --- | ---: | --- |
| Ordinary two-intercept comparison | 5 | 54.41, 57.28, 64.96, 66.81, 71.15 |
| Dense inclined cloud corridor | 5 | 54.90, 57.76, 61.68, 66.12, 70.23 |
| Best retained launch-24 controller | 7 | 31.91, 34.68, 39.61, 43.52, 47.33, 49.18, 52.38 |

The corridor changed the route and obtained its third hit 3.28 seconds earlier
than the ordinary launch-45 comparison, but did not increase final score. It is
therefore not promoted over the seven-point controller. This one seed does not
prove the idea cannot help another field.

Four new tests cover recovery of a known inclined axis, equivariance under a
rotated horizontal drift direction, finite flat-cloud handling and density order,
and observation-preserving feasible planning. Together with the existing spline
and physics tests, all 12 selected checks passed. No submission was generated or
uploaded for either five-point result.

## Reproduce

```powershell
$env:MPLCONFIGDIR = Join-Path $env:TEMP 'bpc-mpl-cache'
$env:PYTHONWARNINGS = 'ignore'
.\.venv\Scripts\python.exe -m unittest tests/test_cloud_corridor.py tests/test_spline_route.py tests/test_physics_guidance.py -q
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/spline_route_launch24.yaml --set launch_time=45 --seeds 0 --report BalloonPoppingGymEnv/evaluation/results/pair_launch45_fresh_seed0_metrics.json
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/cloud_corridor_launch45.yaml --seeds 0 --report BalloonPoppingGymEnv/evaluation/results/cloud_corridor_launch45_fresh_seed0_metrics.json
```
