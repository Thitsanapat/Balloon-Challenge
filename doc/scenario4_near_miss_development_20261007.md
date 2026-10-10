# Scenario 4 near-miss development, 7 October 2026

The reviewed 42-second wind-profile policy remains the selected qualification
agent. Changes below are agent-only experiments using observations and public
given parameters; the official simulator and its collision rule are unchanged.
The evaluator's true rocket state is read **after** each action solely for
offline distance diagnostics, never by the agent.

## Why focus on terminal capture?

Scenario 4 has a 1.5 m balloon radius, 2 m configured GNSS position/altitude
accuracy, 64 ms gimbal lag and 80 ms throttle lag. The collision check uses a
swept relative path, not visual overlap in a rendered frame. On seed 0, the
reviewed agent popped five balloons but passed two other released balloons
with minimum sampled true center distances of 2.03 and 2.28 m. The 16-flight
earlier holdout (seeds 100–107 and 200–207) contains 26 unpopped approaches
between 1.5 and 3.0 m across 14 flights. Thus the miss problem generalizes
beyond one picture, but a near miss is not automatically a capturable extra pop.

The seed-0 guidance telemetry separates target forecast error, spline-tracking
error and alignment. Near the two late misses the target-forecast errors were
about 3.3 and 3.4 m, while tracking error was about 3.7 and 2.8 m. The selected
wind-profile forecast's existing offline 2-second horizontal RMSE was about
1.98 m. Both prediction and control matter; changing only a final TVC gain is
unlikely to be robust.

## Bounded terminal tracking-residual experiment

`Scenario4TerminalResidualAgent` predicts the part of measured spline error
that would remain under the baseline critically damped PD loop. It combines
that residual with a fresh observed-balloon endpoint forecast and adds a
small three-dimensional force correction during the last 3.5 seconds. It
retains the original route, nominal spline, baseline feedback, actuator limits
and attitude-rate feedforward; an infeasible extra force is reduced or skipped.

Seven focused unit tests passed. In the full official Scenario-4 seed-0 flight,
the default gain `0.35` scored **4** versus the reviewed policy's **5**. A lower
gain `0.15` also scored **4**. The default missed balloon #64 by a sampled
1.59 m after the baseline had popped it; another missed balloon #86 became
closer (1.75 m versus 2.28 m), but no extra pop resulted. Both runs ended
normally and kept their source files unchanged. Reports:
`s4_terminal_residual_dev_seed0_20261007.json` and
`s4_terminal_residual_g015_dev_seed0_20261007.json` in ignored results.

**Decision:** do not promote this terminal correction. Chasing another gain
against seed 0 would overfit a live qualification with independently drawn
seeds.

## Measured balloon-acceleration forecast

`ObservedAccelerationForecast` filters finite differences of the *observed*
balloon velocities. It subtracts the instantaneous acceleration already
represented by the existing altitude-wind profile, then decays and bounds the
remaining acceleration over the prediction horizon. Nothing reads the
simulator's hidden gust grid or future trajectories. The offline diagnostic
records a forecast before stepping the official environment and compares it
only against a later observation.

On each of seeds 0 and 1, at 1/2/3-second horizons, the new horizontal forecast
had lower RMSE than the incumbent. At 2 seconds the results were:

| Seed | Incumbent RMSE | New RMSE | Within 1.5 m, old → new |
| ---: | ---: | ---: | ---: |
| 0 | 1.71 m | 1.48 m | 61.5% → 71.8% |
| 1 | 1.89 m | 1.62 m | 60.0% → 70.0% |

Each row comprises 1,364 paired, released-balloon forecasts. These are
correlated snapshots within one simulated field, **not 1,364 independent
seeds**. Report: `s4_acceleration_forecast_dev_20261007.json`.

Giving the augmented forecast to **every** route candidate did not translate
to higher score: official seed-0/1 flights scored **4/0**, compared with the
reviewed policy's **5/2**. The altered forecast reordered launch/route choices
and lost early opportunities. This full-route variant is rejected. Report:
`s4_acceleration_wind_dev_20261007.json`.

An active-target-only variant left initial route scoring unchanged, then used
the extra acceleration term only for a committed target. Without a forecast
horizon limit it scored **5/2** on development seeds 0/1, tied with the
reviewed policy while changing several pop IDs. Restricting use to the
**1–3 s horizon actually validated above** scored **5/3** on the same two
development seeds. Seed 1's extra pop was balloon #38, a near miss in the
reviewed run; seed 0 still missed the reviewed policy's first balloon.

The 3-second version was frozen before a four-seed paired holdout. Seeds
`1652635469`, `1504633979`, `1154366829`, `1630657292` were predeclared
from a deterministic random draw and checked absent from the local used-seed
registry. Every cell below is a *fresh*, complete official Scenario-4 flight
with unchanged agent source; no cached field or submitted JSON was used.

| Holdout seed | Reviewed agent | Active-target acceleration |
| ---: | ---: | ---: |
| 1652635469 | 1 | 3 |
| 1504633979 | 0 | 0 |
| 1154366829 | 4 | 3 |
| 1630657292 | 3 | 3 |
| **Total** | **8** | **9** |

The candidate wins one pair, ties two and loses one: a +0.25 pop/flight mean
over just four holdout fields, including a zero-pop field for both agents. It
does not yet show a dependable qualification advantage. The immutable plan,
all eight full metrics and scores are in the ignored local report
`s4_acceleration_holdout4_20261007.json`, SHA-256
`08d79448fd002aac80cac9c1004b76daef6f0af0ae5e94bf40435cd01578ec4e`.

**Decision:** keep the reviewed standalone wind-profile agent as the default
qualification submission. Do not promote the acceleration agent or its
multi-file source without a larger, fresh holdout and a reproducible source
package. Improving average target prediction is useful, but planning and
first-leg capture are still the limiting interactions. A next experiment
should predict *capture probability of the committed leg* from observation-
derived forecast/tracking uncertainty, preserving the reviewed route unless
the alternative's probability gain exceeds a conservative threshold. That
would need its own untouched paired seeds and full official evaluation.
