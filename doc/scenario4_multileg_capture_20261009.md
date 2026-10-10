# Scenario 4: discounted multi-leg capture ranking, experimental only

The existing six-feature capture model was trained on 75 executed leg
attempts across twelve fields; seed-held-out calibration improved only
slightly. The plain learned route improved the mean on eight independent
fields but lost three pops on two. A first-leg acceleration-headroom guard
also failed its predeclared development screen. Neither replaced the live
qualification agent.

This candidate asks a different question: among routes already accepted by
the physical beam-search feasibility checker, which route's *first three*
legs appear more likely to be captured? It projects visible balloon position
to each leg start, samples the nominal spline start position and velocity,
evaluates the same frozen capture model, and adds a discounted mean failure
penalty of at most 0.5 seconds to the original route priority. Later legs
weigh 1/2 and 1/4 as much as the first because their projected states are
more uncertain. An invalid forecast falls back to the original priority.
All runtime inputs are observations, frozen model weights, and public
parameters. Future-leg probabilities are an *unvalidated extrapolation* of
a model trained on executed first-leg outcomes; this experiment tests that
hypothesis, not an established probability guarantee.

Screen **one fixed candidate** on already-used development fields in this
order: `140200341`, `568300771`, `178420909`, `820431975`. The reviewed
agent scored 4/2/6/6 (18 total); the plain learned agent scored 8/8/3/3.
Require this candidate to score at least 19 total with no field more than
two below the reviewed agent, and exercise the multi-leg ranker. If it fails,
do not tune discount, risk weight or horizon on these fields. If it passes,
freeze code/model/config before independent paired seeds. The live runner
and reviewed standalone source remain untouched.

Report, once run: ignored local
`evaluation/results/s4_multileg_capture_dev4_20261009.json`.

## Completed development screen

All four official Scenario-4 flights completed normally with the candidate
source unchanged (`ec788aab96db8aad967ba6778b1b6f48c1a27a8b00f124a75a5fd2afae91133c`).
The ranker was exercised on 7,199 / 7,309 / 3,602 / 9,721 nodes. In the
predeclared seed order, the candidate scored **7/2/3/5**, totaling **17**
versus the reviewed agent's **4/2/6/6**, totaling **18**. It also lost three
pops on seed `178420909`. It fails both the aggregate and the worst-case
screen, so there is no independent holdout and no live promotion.

The first two experiments together show that neither a single-leg model with
a simple headroom gate nor an uncalibrated projection of that model across
later legs solves the route-choice instability. The multi-leg candidate's
post-reset simulator time ranged from 115 to 159 seconds, so runtime was not
the primary rejection reason. Do not tune its weight or discount on these
four familiar seeds. A next data-driven attempt would require executed-route
outcomes across more independent fields and a predictor explicitly checked
for later-leg calibration before it influences a live route. The reviewed
standalone source and `run_qualification.cmd` remain unchanged.
