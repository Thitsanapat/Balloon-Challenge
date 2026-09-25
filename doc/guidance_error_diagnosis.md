# Scenario 1: separate prediction and tracking error (2026-09-24)

## Diagnostic boundary

`scripts/guidance_telemetry.py` is a read-only observer in the development
evaluation wrapper. It receives copies of the agent's polynomial reference and
the public observations. It does not read private simulator state, alter
observations/actions, or provide information to the agent. Official evaluator,
simulator and competition payload remain unchanged.

At a common time sample, the observer records the vector identity:

`rocket - balloon = (rocket - reference) + (reference - predicted_balloon) + (predicted_balloon - balloon)`

These are tracking, reference/prediction alignment, and balloon-prediction
errors respectively. Their **vectors** add exactly; their norms generally do
not. Alignment can be nonzero before the intended intercept even with perfect
prediction and tracking. At each target episode's closest sampled approach,
the time remaining, error components and control command are retained.

The observer also checks constant-velocity forecasts and planned endpoints
at their deadlines for still-unpopped targets. It drops checks for targets
already popped, avoiding false failure diagnoses. Deadline comparisons use a
short back-interpolation from the first sample at/after the deadline; the
sample offset is recorded explicitly. These distances are not replacements
for the official swept-path collision criterion.

## Paired reproduction

```powershell
$env:MPLCONFIGDIR = Join-Path $PWD '.mplconfig'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m unittest tests.test_guidance_telemetry -v
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/chain_launch.yaml --seeds 0 3 --set max_tilt=75 --set max_axis_rate=1.2 --telemetry --report BalloonPoppingGymEnv/evaluation/results/telemetry_reference.json
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/constrained_beam.yaml --seeds 0 3 --set beam_repair_depth=8 --telemetry --report BalloonPoppingGymEnv/evaluation/results/telemetry_constrained.json
```

Expected control scores before instrumentation: reference 10/7 and constrained
9/8 for seeds 0/3. Any discrepancy must be investigated before attributing
effects to a controller change. These seeds were selected for diagnosis, not
for an unbiased assessment of generalization.

## Findings

Instrumentation reproduced all four control scores (10/7 reference, 9/8
constrained). All completed normally with unchanged agent sources. Reports:
`BalloonPoppingGymEnv/evaluation/results/telemetry_reports_20260924/`.

Two distinct mechanisms were observed:

1. **Seed 3, target 88, reference agent:** closest sampled approach before
   target abandonment was 1.7034 m, with 0.12 s remaining to the deadline.
   Tracking error was only 0.032 m and prediction error approximately 0.001 m.
   Reference-to-prediction alignment was 1.6915 m: the reference had not yet
   reached the balloon. The planner's `duration > .12` commitment condition
   caused it to abandon this live final intercept just outside the 1.5 m sphere.
2. **Seed 0, target 12, constrained agent:** closest distance was 4.018 m at
   t=53.80 s, 0.02 s before the deadline. Tracking error was 3.945 m, prediction
   error 0.035 m, and reference alignment 0.612 m. Commanded throttle was 1.0.
   Here prediction is not the dominant error at this sample; tracking is.
   This does not by itself identify the full dynamic cause of tracking loss.

## Implemented final-approach fix

`FinalApproachMixin` retains the current reference for a still-released target
in the last 0.15 seconds. It does not postpone the intercept. If the target is
not popped, it schedules a review at the original deadline, instead of waiting
another complete replanning interval. Expired/popped targets still go through
the original planner. Guard zero disables the change exactly.

`final_approach.yaml` uses the reference chain; `final_approach_constrained.yaml`
uses depth-8 constrained beam search. Both are compared on seeds 0–3 with
telemetry, without tuning target identifiers or loading any recorded fields.

## Tracking-allocation hypothesis

`projected_tracking_agent.py` tests Euclidean projection of requested thrust
onto the feasible tilt cone and magnitude ball, instead of vertical-priority
clipping. It preserves maximum thrust and tilt limits, but minimizes total
squared instantaneous acceleration error under saturation. This mathematical
property does not guarantee higher flight scores or safe ground clearance;
only complete simulator runs establish the observed outcome. The existing
`saturated_steps` counter still describes the inherited allocator's check,
so it must not be interpreted as a new projection-specific metric.

The projected variants also include final-approach commitment. Their diagnostic
comparison uses seeds 0 and 3; independent additional tests of the basic
final-approach fix use seeds 4–7. Those extra seeds are not given to the agent.

Implementation tests cover the vector error identity, nonmutation, popped-target
forecast exclusion, deadline scheduling, disabled-guard equivalence, force/tilt
limits and projection error versus the original feasible allocator.

## Completed results (2026-09-24)

All runs below used fresh official environments, completed normally without
truncation, and reported unchanged agent source dependencies during execution.

| Agent | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
| --- | ---: | ---: | ---: | ---: |
| Previous reference | 10 | 8 | 9 | 7 |
| Final-approach fix | 10 | 8 | 9 | 8 |
| Depth-8 constrained + final approach | 9 | 8 | 9 | 8 |
| Projected tracking + final approach | 10 | not run | not run | 8 |
| Projected constrained + final approach | 9 | not run | not run | 7 |

The corrected basic agent popped seed-3 target 88 at 34.92 s, only 0.02 s
after the old planner had abandoned it. The paired four-seed mean increased
from 8.50 to 8.75; this is a small measured improvement, not evidence of a
20-point agent or an unbiased generalization estimate. The additional basic
tests on seeds 4, 5, 6 and 7 scored 7, 8, 7 and 6 respectively. Across these
eight seeds the corrected basic agent averaged 7.875, with maximum 10.

Projection did not improve these diagnostic scores and reduced the constrained
seed-3 result. It remains an experimental option, not the promoted submission.
Reports are in `evaluation/results/final_approach_reports_20260924/` and
`evaluation/results/tracking_extra_reports_20260924/` under BalloonPoppingGymEnv.
Remote jobs completed and the SSH session was closed; no GPU was used.

## Reproduced submission

Standalone agent: `BalloonPoppingGymEnv/agents/submission_final_approach_v1.py`.
Configuration: `BalloonPoppingGymEnv/evaluation/configs/submission_final_approach_v1.yaml`.
The unmodified official evaluator independently reproduced **10 points, seed 0**.
The official packer generated:

`BalloonPoppingGymEnv/evaluation/results/20260924T023250.314Z_TASTI_Cool_Ba_Malaew_5df48f1ab723401dadbbefac8ba1a57a_submission.json`

Submission SHA256:
`4c4235ada11e9f1c03e177857c8afffad0e6b5ea3a7243576c83f42d79ceefa5`.
Embedded agent SHA256:
`e6ddde0857883e8cea782d943925c21433b8928cbadcd0ffa0c44d0ac5470b9b`.

The official verifier passed all 19 checks. The embedded source matches the
standalone agent, the source scan raised no review flags, and simulator,
evaluator, packer and verifier remain unchanged against bd9bb51. These are
evidence checks, not organizer certification. The packer's network integrity
lookup was blocked by local socket permissions; the independent local git
comparison above passed. Receipt and audit: `submission_final_approach_v1_receipt.json`
and `submission_final_approach_v1_audit.json` in the results directory.
The prior verified 10-point payload was preserved. Nothing was uploaded.

Validation: 49 selected unittest tests passed, covering the new features and
related chain/planner/source-audit regressions. The full pytest suite was not
run because pytest is not installed in this virtual environment; the existing
pytest-based submission-runner module was excluded from that unittest count.
