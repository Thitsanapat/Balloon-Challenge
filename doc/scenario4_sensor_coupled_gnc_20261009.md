# Scenario 4: sensor, control and guidance ablation

The guidance-only pure joint refresh completed four predeclared fresh pairs:
selected 4/1/4/3 versus joint 4/5/6/4, total 12 versus 19. It passes the fixed
screen with three wins, one tie and no regressions. These are totals across
flights, not one 19-pop flight.

Two separate additions to that guidance candidate are now fixed for tests on
already-used development seeds 2 and 0:

* Navigation: propagate position with trapezoidal integration of adjacent
  GNSS velocity samples. Keep the existing position correction and filtered
  velocity used by control. This removes velocity-filter delay from position
  propagation without differentiating noisy GNSS position.
* Control: estimate a scalar angular response on each pitch/yaw axis from
  gyro derivatives and the controller's own predicted actuator outputs.
  Match gyro filtering on the nominal response, then apply a common 0.10 s
  derivative filter. Fit a three-second exponentially weighted regression
  with a prior toward unity. Require 1 s of useful excitation, limit inferred
  response to 0.85–1.15 and its change to 0.03/s. Only the inner attitude
  controller receives this compensation; route feasibility retains the
  original public model. No IMU/GNSS fusion model is claimed here.

This is a bounded adaptive-gain experiment inspired by the measurement-delay
and control-effectiveness concerns in [Smeur et al., Adaptive Incremental Nonlinear Dynamic Inversion for Attitude Control of Micro Air Vehicles](https://repository.tudelft.nl/record/uuid%3A31536cfa-89e1-4d44-873e-f2f398bd69ca).
It does not reproduce that paper's full INDI law. Predicted actuator outputs
are imperfect, and aerodynamic torque, variable inertia, cross-axis effects
and sensor timing can contaminate the fitted response. Score and tracking
must therefore be tested rather than inferred from fit error.

The runner `scripts/validate_sensor_coupled_gnc.py` waits for the existing
four-pair holdout to complete, then evaluates four policies per field:
joint-only, navigation-only, control-only, and both additions. Every flight
uses a fresh official simulator process and read-only offline telemetry.
Maximum two CPU workers; GPU visibility disabled. Used seeds are explicitly
recorded. All agent sources and evaluator sources are frozen for the run.

Development screen fixed before scores: total gain at least two over
joint-only across the two fields, with no regression greater than one on
either. Passing permits a separate independent test, not live promotion.
The live standalone, simulator and launcher are unchanged.

Rule check: [organizer Discussion #165](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165)
requires runtime decisions to use observations/public parameters and explicitly
clarifies that model reference adaptive control is allowed. This experiment
is a scalar response adaptation, not an implementation of MRAC. Its state
starts afresh for each agent; it estimates response from observations and
past commands only. It contains no seed-dependent trajectory or lookup table.
The organizer remains the authority on source-review acceptance.

Seven focused tests pass, including synthetic excitation/convergence,
no-excitation behavior, matched filter delay, bounded gain/slew,
constant-acceleration position integration and disabled-mode parity.

Results: `evaluation/results/sensor_coupled_gnc_dev_20261009/`.

Diagnostic interpretation: the existing evaluator compares the latest agent
position estimate with simulator truth after `env.step`, so its navigation
error includes a one-step timing offset. Different policies also visit
different states and may fly for different durations. Navigation/path-error
means are useful diagnostic comparisons, not matched-state causal proof.
The `missed_targets_within_3m` summary counts pursuit episodes, which can
include repeated pursuit of an ID; pop score remains the official count.

## Near-miss evidence from the joint-only development baseline

`2_joint_misses.json` records target 59 at sampled true distance 1.57090 m,
only 0.07090 m outside the public 1.5 m radius, at its planned deadline.
Along the miss direction, reference-to-balloon error contributes +2.47101 m
(alignment plus constant-velocity prediction residual), while rocket-to-
reference tracking contributes -0.90011 m. Target 9 is at 1.94557 m, with
reference-to-balloon +2.12173 m and tracking -0.17616 m. On seed 0, target
77 is at 3.08800 m: reference contribution +3.75674 m, tracking -0.66874 m.

These vector decompositions describe samples, not counterfactual proofs.
They show why reducing tracking error alone need not capture these particular
misses: the reference point is also displaced from the balloon, and tracking
error happens to partially cancel it. Future experiments should assess an
earlier feasible update of the intercept point/forecast together with control.
The prediction component uses a diagnostic constant-velocity forecast and
must not be interpreted as the exact wind-profile model's isolated error.
All truth-based decomposition stays in offline reports, never in the agent.

## Completed development ablation and decision

All eight fresh official flights terminated normally and retained their
source hashes. Scores:

| Policy | Seed 0 | Seed 2 | Two-flight total |
| --- | ---: | ---: | ---: |
| Pure joint guidance | 6 | 4 | 10 |
| Joint + trapezoidal navigation | 6 | 3 | 9 |
| Joint + adaptive angular control | 5 | 4 | 9 |
| Joint + both additions | 6 | 4 | 10 |

None of the three additions passes the predeclared development screen.
Keep pure joint refresh as the preferred development candidate. Do not
promote the sensor/control variants or spend new independent seeds on these
frozen variants. Retain their implementation and diagnostic reports for review.

The adaptive controller was exercised, not dead code: 3,255 and 3,359 steps
on seeds 2 and 0 for control-only. The combined policy had fewer allocation-
saturated steps on seed 2 (984 versus 1,177), but more on seed 0 (889 versus
674), and mean path error rose there from 0.504 to 0.811 m. Better aggregate
sensor error therefore did not establish a better end-to-end policy.

Control-only's pre-update, filtered angular-response MSE improved from
0.13189 to 0.12583 on seed 0 and from 0.34310 to 0.32231 on seed 2. These
are correlated online samples, not held-out identification scores. Even
that improved response fit did not improve pops, another reason not to
optimize the response-fit metric in isolation or widen the gain bounds now.

The useful confirmed progress in this round is the prior independent
guidance screen: 12 to 19 pops summed over four flights. A separate packaged
pure-joint parity run on development seed 2 is queued/completing under
`evaluation/results/pure_joint_standalone_parity_20261009/`. The two packaging
unit tests also pass, making nine focused tests across these changes.

### Packaging dispatch correction

The first standalone flight finished with 3 pops versus the modular agent's
4 on seed 2, so it failed parity and was rejected. Semantic-source comparison
found that the reviewed older bundled ChainBeamAgent lacked the calls to
the newer `_accept_committed_curve` / `_rescue_committed_leg` hooks. Appending
the override alone left it unreachable in a flight. The direct numerical
rescue test did not exercise this dispatch and therefore missed the defect.

The v2 builder replaces current ChainBeamAgent and ChainLaunchAgent class
definitions inside the old AST before constructing derived classes. A new
behavioral regression test forces a failed committed refresh and verifies
that the packaged planner actually invokes joint rescue. All three packaging
tests pass (ten focused tests including sensor/control tests). A new official
seed-2 flight is under `pure_joint_standalone_parity_v2_20261009/`; v1 and its
failed report remain intact. None of the modular score results was changed.

V2 flight parity is now **passed**: seed 2 scores 4 and all compared fields
match the modular reference exactly (pop times/IDs, target/route events,
termination, diagnostics, navigation error, sampled true distances and full
guidance telemetry). Its SHA256 is
`9ff7f93d21a945d0479289be57bd893d8f33ab169fb1a6bfeb153ea442b6efb9`.
A fourth packaging test checks every inherited method against the current
modular source, ignoring docstring indentation and the deliberate profile
class alias; total focused tests are now eleven. The source/config bundle
is `review/scenario4_pure_joint_source_v2.zip`, generated only after parity
and source/config identity checks. It is a source-review artifact, not a
packed evaluation JSON. The live launcher is still unchanged.
