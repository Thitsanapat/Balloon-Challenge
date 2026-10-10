# Scenario 4: shadow calibration and across-depth route-value trial

## Decision

No score increase was established. Keep the selected standalone pure-joint
agent and qualification launcher unchanged. Do not enable an MPCC controller
on the basis of this predictor. No GPU, remote machine, upload, simulator
change, or new seed was used. Seeds 0/2 were already used and the new uses are
recorded in `qualification/used_seed_manual.jsonl`; the registry was refreshed
(445 distinct local/planned seeds, six pre-existing coverage warnings).

## 1. Predictor check and distance calibration

Ran a complete fresh official Scenario-4 flight with the selected standalone
on seed 0: six pops, 74 prediction anchors, horizons 0.25/0.5/1 second,
including arrival-focused anchors every 0.15 seconds. The actual agent was
unchanged; later official truth is used only by the diagnostic evaluator.

Fit per-horizon closest-distance residual medians using earlier guarded-policy
seed-1/2 reports, weighting target/field clusters rather than time rows for
the bias. Freeze those estimates, then evaluate on current-policy seed 0,
which was excluded from this calibration fit. This is development validation,
not an unseen competition field or a generalization guarantee. The change
from guarded to pure-joint policy is also a distribution shift.

| Horizon | Fixed-plan comparisons | Raw distance MAE | Bias-corrected MAE |
| --- | ---: | ---: | ---: |
| 0.25 s | 57 | 0.176 m | 0.230 m |
| 0.5 s | 43 | 0.180 m | 0.255 m |
| 1.0 s | 22 | 0.303 m | 0.332 m |

These are errors in predicted **closest distance**, not overall navigation
or rocket position errors. Calibration worsened them. At 0.25 seconds the
raw model falsely predicted one of its two captures; the empirical upper
distance envelope still falsely predicted its one capture. At 0.5 seconds
the fitted envelope was exceeded in 30.2% of held-out comparisons, not the
roughly 10% its training quantile might suggest. The controller-experiment
gate failed; no learned correction was put into the flight agent.

An additional methodological weakness emerged: filtering out changed plans
also removes many successful-capture windows. At 0.5 seconds the evaluator
had 53 completed forecasts, including three actual-capture windows; all
three were absent from the 43 fixed-plan comparisons. At 0.25 seconds, six
of seven actual-capture windows were excluded. These are correlated forecast
windows, not counts of distinct popped balloons. Consequently, fixed-plan
mean errors cannot certify live hit/no-hit decisions. This is evidence about
coverage, not proof that all observed errors arise from replanning.

Artifacts:

- `evaluation/results/crossdepth_shadow_dev_20261010/0_shadow.json`
- `evaluation/results/crossdepth_shadow_dev_20261010/calibration_coverage_heldout0.json`
- `scripts/assess_shadow_calibration.py`

## 2. Across-depth route selection

Implemented an isolated subclass of the frozen standalone controller:
`agents/scenario4_crossdepth_value_agent.py`, with its own config. It collects
up to three proposals per depth through the existing priority hook, leaving
beam expansion order and pre-launch comparisons unchanged. After launch,
short and long proposals compete under a discounted capture-value surrogate;
an alternative must beat the incumbent by 0.05 and pass dense validation of
every leg. Unsupported first-leg features preserve the incumbent.

The scorer reuses the existing six-feature capture model trained on executed
guarded-policy first legs. For later legs it uses planned reference states;
those probabilities and the discounted multi-leg value are NOT independently
calibrated. No unflown alternative receives a fabricated success label. This
is a structural routing ablation, not successful integration of the failed
shadow calibration.

| Development seed | Incumbent | Experimental selector | Adopted alternatives |
| --- | ---: | ---: | ---: |
| 0 | 6 | 6 | 0 |
| 2 | 4 | 4 | 0 |

Both candidate flights were complete official-environment runs with source
fingerprints unchanged throughout. The seed-0 incumbent was also freshly
reproduced by the shadow run; the seed-2 incumbent reference is the existing
verified four-pop run. Candidate reset/flight times were approximately
127/123 seconds and 127/121 seconds. Proposal pools peaked at 24/23 nodes.
The scorer never found a sufficiently better supported alternative: this
tests the new boundary but does not establish that changing routes helps or
that across-depth selection cannot help. The declared development screen
required at least one total pop gain and no regression worse than one, so
it failed and no fresh holdout was consumed.

Artifacts under `evaluation/results/crossdepth_value_dev_20261010/`:
`experiment_plan.json`, `0_2_candidate.json`, and `summary.json`. These JSONs
are credential-free experiment reports, NOT uploadable competition results.

## What remains before MPCC

The next useful model must evaluate execution with the actual controller's
forecast correction and replanning behavior, or use explicitly defined
short/censored prediction windows. It must not silently label an altered
reference as a fixed-plan prediction. Capture calibration needs enough
successful and failed target/field groups near the collision boundary.

Likewise, summing a weak first-leg model over nominal later states is not a
validated route-value learner. Collect outcomes from the current policy,
including attitude, angular velocity, actuator estimates, remaining burn,
and reference-tracking state, then validate by field before allowing these
estimates to override routing. Do not enlarge a network merely to compensate
for missing execution-state inputs or biased labels. The proposed moving-
target MPCC remains gated, not implemented/enabled by this trial.

## Verification and preservation

33 focused tests passed: calibration separation/coverage, cross-depth short-
route selection, dense-physics rejection, unchanged launch comparisons,
development-screen audit, existing shadow diagnostics, and byte-identical
standalone rebuild. The selected standalone SHA256 remains
`9ff7f93d21a945d0479289be57bd893d8f33ab169fb1a6bfeb153ea442b6efb9`.
`run_qualification.cmd`, its default config, the official simulator, and the
tested source-review ZIP were not changed. No training/evaluation job from
this trial was left running.
