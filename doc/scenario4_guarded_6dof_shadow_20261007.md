# Guarded agent: observation-only 6-DOF shadow validation

The repository already contains `agents/scenario4_shadow_predictor.py`, a
reduced six-state/attitude dynamics rollout using only public rocket/motor and
actuator parameters plus the agent's filtered observation history. It is not
the official 6-DOF simulator and omits aerodynamic moments and future wind.
Earlier validation checked a few post-launch anchors with the older wind
profile agent. This experiment tests whether it is accurate enough during
the selected guarded agent's active target pursuit, before using it to alter
actions.

Run one fresh official-environment replay on already-used development seed 2
with the unchanged guarded agent/config. At post-launch offsets
8/12/16/20/24/28 seconds, predict 0.25/0.5/1/2 seconds ahead. The offline
evaluator may compare later official truth with the frozen prediction but
must never pass truth, seed, or future balloon positions to the agent or
predictor. Split results by whether the nominal spline or target changed
before each horizon. A provisional usability gate is <=0.75 m mean rocket
position error at 0.5 seconds and <=2 m at 1 second among comparisons with
no plan/target change, with at least three samples at each horizon. If this
gate fails or no enough samples exist, leave the live controller untouched.
Even passing the gate would justify only a bounded, separately tested
experimental controller, not immediate promotion.

## Sparse-anchor result and follow-up

The guarded flight reproduced 3 pops on seed 2. Of six requested anchors,
one had no suitable active plan. Completed forecasts by horizon were 5, 4,
3 and 2; forecasts without an intervening plan change were only 4, 2, 1
and 0. Mean rocket position errors over *all* completed forecasts were
0.376, 0.392, 0.567 and 0.597 m at 0.25, 0.5, 1 and 2 seconds. The 0.5-
and 1-second no-plan-change counts fail the predeclared minimum of three, so
the model has **not** passed the usability gate. Longer-horizon means mix in
changed-route effects and are not evidence that the fixed-plan physics is
accurate there. Report: ignored
`s4_guarded_shadow_seed2_20261007.json`.

Before using these forecasts for any control, repeat the same diagnostic on
the already-used development seed 2 with denser post-launch anchors at each
integer second from 8 through 28. Keep the predictor, horizons and agent
unchanged. This increases the number of matched fixed-plan comparisons; it
does not create an independent test seed or a new score claim.

## Dense seed-2 diagnostic

The denser replay again scored 3. Completed forecasts at horizons 0.25,
0.5, 1 and 2 seconds numbered 20/19/16/10; without a plan change they
numbered 17/15/10/4. On those fixed-plan comparisons, mean rocket-position
errors were 0.382/0.410/0.502/0.870 m, respectively. The predeclared 0.5-
and 1-second thresholds therefore pass on this *one used development seed*.
Closest-approach distance mean absolute errors on fixed-plan comparisons
were 0.208/0.247/0.371/0.977 m. Near one unpopped target, the model
predicted 1.55 m against an actual 1.95 m, enough to alter a decision near
the 1.5 m capture boundary. Thus passing mean position error is not evidence
of reliable binary capture prediction. Report: ignored
`s4_guarded_shadow_dense_seed2_20261007.json`.

Before adding the predictor to any agent, repeat the same frozen dense-anchor
diagnostic on already-used development seed 1. This is cross-field model
validation, not a new score test or promotion; keep the selected submission
untouched. Require both the earlier fixed-plan position-error gates and
inspect near-radius closest-distance errors rather than claiming capture
probabilities from a single seed.

## Seed-1 confirmation and arrival-sampling plan

The frozen dense diagnostic on seed 1 reproduced 4 pops. With no intervening
plan change, mean rocket-position errors at 0.25/0.5/1/2 seconds were
0.307/0.322/0.444/0.894 m (15/14/10/5 comparisons). Mean absolute
closest-distance errors were 0.147/0.133/0.231/1.390 m. The position-error
gates pass on both development seeds, but there were **no predicted captures
among the fixed-plan anchor comparisons**. Therefore these sparse regular
anchors cannot calibrate hit/miss classification, and the 2-second range is
especially uncertain. Report: ignored `s4_guarded_shadow_dense_seed1_20261007.json`.

The diagnostic evaluator now supports bounded arrival-focused anchors that
still leave the agent and official environment unchanged. On already-used
development seed 2, sample at most 120 anchors spaced 0.1 seconds apart
during the last 1.5 seconds of an active planned leg; predict 0.25/0.5/1
second horizons. Count false predicted captures and near-radius distance
errors separately for unchanged plans. This is a model-calibration test, not
an independent score test or authorization to control the rocket.

## Arrival-focused result and control decision

The unchanged guarded agent again scored 3 on the same development seed.
Arrival-focused sampling created 88 total anchors. At 0.25/0.5/1-second
horizons, completed fixed-plan comparisons numbered 77/63/33. Their mean
rocket-position errors were 0.442/0.478/0.646 m; mean absolute closest-
distance errors were 0.261/0.297/0.381 m. The 90th-percentile closest-
distance errors reached 0.501/0.549/1.033 m. These near-arrival samples
are more relevant to collision decisions than the earlier integer-second
anchors and show a wider error tail.

Crucially, a fixed-plan prediction at time 53.52 s said target 9 would pass
at 1.07 m over the next second, inside the 1.5 m balloon radius. The actual
minimum over that horizon was 1.95 m, a miss. Another 0.5-second prediction
was 1.48 m versus an observed 1.57 m, very close to the boundary. The
pointwise position-error gate alone is therefore insufficient to authorize
a capture/no-capture switch. The report is ignored local file
`s4_guarded_shadow_arrival_seed2_20261007.json`. The new offline-only
summary tool `scripts/summarize_scenario4_shadow.py` exposes these counts
without printing credentials; its tests and the diagnostic-runner tests pass.

**Decision:** do not wire the shadow predictor into the live controller yet.
It can still guide further model calibration or conservative feasibility
checks, but action changes need a calibrated uncertainty margin and paired
fresh-seed evidence. The qualification default and standalone source are
unchanged, and no submission JSON was generated or uploaded by this work.
DRL training/integration was intentionally deferred: feeding an uncalibrated
hit prediction into a learned controller would compound, not resolve, this
measured failure mode.
