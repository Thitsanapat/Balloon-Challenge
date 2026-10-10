# Scenario 4 causal capture-confidence experiment

The candidate is `scenario4_capture_confidence_agent.py`, using the separate
`scenario4_capture_confidence_launch42.yaml` config. The qualification default
remains `submission_scenario4_wind_profile_launch42.yaml`.

Hypothesis: the observed-acceleration forecast can reduce average balloon
prediction error yet hurt an individual committed intercept. The candidate
records forecasts before a two-second interval, scores them only against later
balloon observations, and switches the active target's forecast only when a
local radial-capture proxy improves by at least 0.015. It uses public balloon
radius, a common 1.5 m tracking-error floor, and rolling altitude-nearby
examples. This proxy is **not** a calibrated probability of a successful pop.
No simulator truth, seed, future state, or trajectory file enters the agent.

Development evaluation is fixed to Scenario 4 seeds 0 and 1, where the reviewed
agent scored 5 and 2 pops. If development results justify a holdout, use the
predeclared fresh paired seeds `190038660` and `629093901`, drawn from
`numpy.random.default_rng(2026100702)` and absent from the local seed registry
before this experiment. A paired holdout means a fresh official simulator reset
for each agent and seed, with source/config hashes checked and no replay cache.

Do not promote the candidate based solely on development-seed score or a small
holdout. Qualification seeds will be independently generated, so stability
across fields matters more than a best-case single flight.

## Completed development results

Both candidate variants were evaluated with fresh official Scenario-4 resets
on development seeds 0 and 1. All four runs terminated normally without
truncation; the agent source dependency hashes were stable during each run.

| Policy | Seed 0 | Seed 1 | Development total |
| --- | ---: | ---: | ---: |
| Reviewed incumbent | 5 | 2 | 7 |
| Continuous confidence gate | 3 | 2 | 5 |
| Target-latched confidence gate | 5 | 2 | 7 |

The continuous gate lost seed-0 balloons after its third pop. Switching the
forecast while a target is already committed changes the commanded intercept
enough to affect later routing. The latched variant instead chooses once when
each target becomes active and preserved the incumbent's five seed-0 pops,
including balloon #44 on its first leg. It chose the acceleration model for
three of seven committed targets on each development seed, but did not gain an
extra pop. The confidence proxy was therefore useful as a stability guard,
not an observed score improvement.

Reports (ignored local results): `s4_capture_confidence_dev_20261007.json`,
`s4_latched_capture_dev_seed0_20261007.json`, and
`s4_latched_capture_dev_seed1_20261007.json`.

**Decision:** do not run the predeclared two-seed holdout or promote either
variant: neither beats the incumbent on these development fields. The
predeclared seeds remain unused/planned, not training or testing seeds. The
qualification runner retains the reviewed standalone wind-profile agent.
The best previous acceleration-only candidate scored 5/3 on these development
seeds but 9 versus 8 *total across four distinct holdout flights*, with one
loss and a shared zero-pop field. That remains too weak to justify promotion.
