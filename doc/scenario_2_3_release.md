# Scenario 2/3 release validation (2026-09-22)

The repository is merged with the official `v0.2.1` tag.  The competition
simulator and scoring logic are unchanged.  The agent changes in this iteration
use only observations and parameters exposed through `given_parameters`.

## Released differences

- Scenario 2 changes the atmosphere data and adds GNSS, gyro, and accelerometer
  noise.  Its balloon-position standard deviation is also slightly smaller.
- Scenario 3 contains the Scenario 2 changes and publishes first-order actuator
  time constants for gimbal, roll, and throttle.

`PhysicsGuidanceAgent` now applies an observation-only alpha filter when the
published sensor accuracies are nonzero.  Clean Scenario 1 observations remain
passthrough.  Scenario 3 commands invert the published first-order actuator lag
subject to the original command range and rate limits.  Controller bandwidth is
selected from the presence of the published lag, while explicit configuration
values still override the defaults.

## Fresh official-environment results

These runs regenerate the balloon field and use the ordinary six-degree-of-
freedom `BalloonPoppingEnv`; no cached field or replay is substituted.

| Scenario / seed | Before this iteration | Current score | Notes |
|---|---:|---:|---|
| 1 / 0 | 7 | 7 | Regression retained |
| 2 / 0 | 1 | 7 | Sensor filtering enabled |
| 2 / 1 | not measured | 4 | No retuning for seed 1 |
| 3 / 0 | 3 | 6 | Filtering, lag compensation, lag-aware gains |
| 3 / 1 | not measured | 5 | No retuning for seed 1 |

Two seeds are useful regression evidence, not a robustness guarantee.  Reports
are stored in the ignored evaluation-results directory and record the complete
configuration, agent hash, pop events, and diagnostics.

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.mplconfig'
.\.venv\Scripts\python.exe scripts\evaluate_guidance.py `
  BalloonPoppingGymEnv\evaluation\configs\optimized_spline.yaml `
  --scenario 3 --seeds 0 1 `
  --report BalloonPoppingGymEnv\evaluation\results\scenario3_validation.json
```

## Route-learning track

The optional Stable-Baselines3 environment is installed locally and can select
its device with `--device`.  A 200-epoch behavior-cloning smoke test and a longer
DAgger run did not produce a self-contained useful policy (scores 0 and 0 in the
surrogate after the final iterations).  A policy that follows an external public
replay can reproduce a high surrogate score, but it is not submission eligible
and is not promoted.

Public replay analysis also shows why seed-specific leaderboard performance is
misleading: the 17-point trajectory launches at 36.31 seconds and its first
targets include balloons that have not yet been released at launch in seed
304729.  Their future release order cannot be inferred from a different random
seed.  `EnergyOpportunityAgent` is an observation-only experiment that preserves
fly-through velocity, but its first benchmark scored only 4; it remains a
research candidate rather than the selected agent.
