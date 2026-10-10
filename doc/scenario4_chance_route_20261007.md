# Scenario 4 chance-aware route experiment (7 October 2026)

The qualification default is still `submission_scenario4_wind_profile_v1.py`.
This experiment did **not** improve the same-seed development scores and is not a
submission agent. It uses only current/past observations and given parameters;
official Scenario 4 physics, weather, and collision checks were not changed.

## Model

`scenario4_chance_route_agent.py` extends the reviewed wind-profile planner.
The parent still proposes joint, time-allocated splines and validates thrust,
tilt and control-rate limits. For each feasible beam node, the experiment:

1. Forecasts moving balloons using the current observed position/velocity.
2. Samples the planned spline and computes the closest *time-aligned relative*
   approach to each unplanned released balloon. Each incidental balloon can
   contribute at most once.
3. Estimates forecast spread from 1/2/3-second forecasts recorded before their
   due time and checked only against later balloon observations. A fixed
   navigation spread is combined with this empirical spread.
4. Uses the probability that an isotropic 3-D Gaussian falls inside the public
   balloon radius as a route-ranking proxy. This is mathematically checked
   against the noncentral chi-square CDF, but its isotropic error assumption
   and fixed navigation spread are **not calibrated to observed pop frequency**.

The calculation does not obtain the scenario seed, balloon Monte Carlo field,
future trajectory, simulator truth or hidden `info`. The offline evaluator
records truth only after the agent has acted, for diagnostics.

## Verification and result

The 13 focused tests (new model plus existing corridor/confidence tests) passed.
Two full, fresh official Scenario-4 evaluations used the unchanged experimental
source/config on already-known development seeds. The incumbent values below
come from earlier saved evaluations, not simultaneous reruns; numeric-library
parity between the two dates was not independently proven. Results are in ignored
`evaluation/results/s4_chance_route_dev_20261007.json`.

| Seed | Reviewed default | Chance-aware route | Chance flight wall time after reset |
| ---: | ---: | ---: | ---: |
| 0 | 5 | 4 | 124.0 s |
| 1 | 2 | 2 | 125.5 s |

Seed 0 popped #44, #74, #1 and #7; the reviewed agent also popped #64.
Both agents popped two on seed 1. These two seeds are development cases, not
an independent qualification forecast. The new ranking processed about 6,700
beam nodes on seed 0 and 6,400 on seed 1. Flight runtime was not prohibitive
in this local run, but a slower event machine has not been measured.

## Decision and next model question

**Reject this variant for live use.** It discounts all planned waypoints using
uncertainty extrapolated beyond the 1–3-second horizon actually checked
against observations. That makes the long route score hard to interpret;
receding-horizon guidance will observe and replan before those late arrivals.
The changed seed-0 route/timing lost a previously captured balloon. This is a
plausible failure mechanism, not a causal proof from two flights.

The next experiment should keep the reviewed long-horizon route and evaluate
capture risk only on an already committed near-term leg, first checking the
probability estimate against actual pop/near-miss frequencies on held-out
flights. Do not tune weights repeatedly on seeds 0/1 or promote a candidate
without paired independent-seed evidence and a reviewed standalone package.

## Near-term-only follow-up

`scenario4_near_term_chance_agent.py` is a separate conservative variant. It
leaves the reviewed ordering intact whenever the first leg exceeds 3 seconds,
the longest horizon validated for the forecast. Inside 3 seconds it scores
only that first leg and nearby incidental balloons, never a whole future
chain. The gate has a unit test proving that a long first leg keeps the exact
parent priority. Both complete development flights ran with unchanged source:

| Seed | Reviewed default (earlier run) | Near-term chance | Near-term ranked nodes | Flight wall time after reset |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5 | 5 | 251 | 109.1 s |
| 1 | 2 | 2 | 468 | 112.4 s |

Reports: ignored `s4_near_term_chance_dev_seed0_20261007.json` and
`s4_near_term_chance_dev_seed1_20261007.json`. Seed 0 again popped #44, #74,
#1, #64 and #7; seed 1 popped #0 and #30. The near-term model was exercised,
but the observed score merely tied the incumbent on two known development
fields. It is **not promoted**, and no new-seed holdout is claimed. A useful
next step is to calibrate its stated probabilities against actual outcomes and
focus on correcting the committed trajectory rather than changing route order.
