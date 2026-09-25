# Capture-region optimization (2026-09-23)

The frozen 10-point submission is preserved. This experiment changes only agent
planning; official physics, scoring, observations and evaluation are unchanged.
The controller only uses given parameters and observation history.

## Motivation and model

The balloon has a nonzero hit radius. Passing through its center is sufficient,
but not necessary. We optimize the hit point inside a smaller sphere to leave
room for tracking error. Fractions 0.25 and 0.5 of the given balloon radius are
tested; the resulting spare radius is a design margin, not a proven bound on
tracking error under every scenario.

Joint polynomial-segment optimization is informed by
[Richter, Bry and Roy, Polynomial Trajectory Planning](https://groups.csail.mit.edu/rrg/papers/Richter_ISRR13.pdf).
Their quadrotor formulation is not assumed to be a rocket model. Here the
objective is integrated **squared jerk**, using our existing quintic-chain model
and separate rocket thrust, tilt, attitude-rate, throttle-rate and ground checks.

For fixed segment durations, free-chain coefficients are affine in waypoint
offsets. Substituting that affine relation into the jerk integral gives a convex
quadratic. A bounded projected-gradient solve constrains each 3D offset to the
shrunken hit sphere. We then backtrack and check 65 samples per segment against
the physical planning limits before replacing the reference. This sampled check
is not a proof of continuous feasibility; actual simulation decides the score.

The selected offsets persist during observation-based deadline refreshes. The
agent does not mutate the observation or save an offline balloon trajectory.

`CompressedCaptureAgent` additionally attempts to shorten later legs by a small
fraction (default 6%, then 3% and 1.5%). It preserves the first intercept time,
recomputes balloon predictions and all knot derivatives, and accepts only a
dense-check-feasible candidate. It is a bounded timing search, not a globally
time-optimal solver. Its diagnostic saved seconds sum changes to planned routes;
they are **not** measured fuel savings or guaranteed extra flight time.

Reducing throttle does not extend the official motor burn time. These methods
aim to reduce control demand and time spent between hits, not change fuel physics.

## Files and tests

- `agents/capture_chain_agent.py`: convex spatial-offset refinement.
- `agents/compressed_capture_agent.py`: optional bounded timing compression.
- Matching configs: `evaluation/configs/capture_chain.yaml` and
  `evaluation/configs/compressed_capture.yaml`.
- `tests/test_capture_chain.py`: sphere bounds, jerk reduction, C2 continuity,
  fixed initial state, affine/full-solve equivalence, zero-radius control, and
  persistence without observation mutation.

Initial tests: offset fractions 0.25/0.5 on seeds 0–3 (eight remote CPU jobs).
A local disabled-offset control reproduced the reference score of **10**.

| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
| --- | ---: | ---: | ---: | ---: |
| Frozen launch-chain reference | 10 | 8 | 9 | 7 |
| Hit offsets up to 0.25 of radius | 9 | 7 | 9 | 8 |
| Hit offsets up to 0.5 of radius | 8 | 8 | 8 | 6 |

All eight offset runs terminated normally. The local spatial-plus-timing
compression run scored **7** on seed 0. These methods are not promoted: a lower
jerk objective or a shorter reference schedule does not necessarily increase
realized hits in closed-loop flight.

## Near-path waypoint insertion

`OpportunisticChainAgent` instead searches for released, unplanned balloons near
the existing trajectory. Relative-motion projection estimates a prospective
intercept time between existing waypoints. It inserts that waypoint, jointly
re-solves all derivatives and checks physical limits before acceptance. The last
arrival time does not move. At most two insertions are accepted per new plan.
The projection is approximate; it is not the official collision detector.

The associated unit test checks a moving target that crosses between two path
samples, and verifies observation immutability. Config:
`evaluation/configs/opportunistic_chain.yaml`. Trials use candidate search radii
4 and 8 m on seeds 0–3. These are *search* distances; the balloon hit radius is
unchanged. Both radii reproduced **10/8/9/7**, exactly the original scores.
Only one waypoint was inserted on seed 0 (without increasing the realized score),
and no candidates were attempted on seeds 1–3. Checking only when a route is
created can miss balloons released later during the committed flight.

A second version therefore accepts an optional `insertion_interval`. At one
second intervals it inspects the current committed route again, preserving
existing offsets and deadlines. Physical checks still gate every change, and
zero interval preserves the previous behavior. Both radii again scored
**10/8/9/7**, with one insertion/attempt on seed 0 and none on the other seeds.
All eight runs terminated normally. In the tested reference routes, checking
more frequently did not expose extra eligible near-path insertion candidates.
The periodic scheduling test also confirms that committed routes can be revisited
without extending the final deadline or mutating observations.

## Control-authority reserve

A separate ablation applies the existing planner's thrust-acceleration reserve
of 0.25 m/s², tapered from zero at each segment's initial condition. This is a
planning margin; it does not alter simulator thrust limits or motor parameters.
The local seed-0 run scored **10**, matching rather than beating the reference.
Additional baseline/reserve comparisons on seeds 20 and 72 completed:

| Seed | No reserve | Reserve 0.25 m/s² |
| ---: | ---: | ---: |
| 0 | 10 | 10 |
| 20 | 9 | 7 |
| 72 | 9 | 8 |

The reserve variant is not promoted.
Those two seeds were selected because an earlier baseline performed well on
them, so they are *not* an unbiased holdout set. Seed assignment stays outside
the agent in the official-environment evaluation wrapper.

## Outcome

This session completed **31 fresh simulation runs**: 28 remote and 3 local.
The best remains **10 points**. No new submission was packed or uploaded, no GPU
was used, and all remote workers finished. The original submission hash remains
`b6ff58aac9ad159357a1f33f155546bf2cb4d42a8636a609570b9c7c6a280afe`.

Remote reports: `evaluation/results/capture_reports_20260923/` and the matching
`.tgz` archive. Local reports: `capture_disabled_control_s0.json`,
`compressed_capture_s0.json`, and `chain_reserve025_s0.json` in evaluation/results.
Tested source snapshots: `capture_chain_source.tgz`, `capture_followup_source.tgz`
and `opportunistic_live_source.tgz`. Six tests were added for capture geometry,
affine optimization, observation immutability and periodic route insertion;
sixteen relevant numerical/bundle tests passed in the final checks.

These results support moving the next investigation toward whole-route target
ordering, rather than more smoothing or local geometric refinement of the same
route. They do not establish a score ceiling or guarantee that a new route
optimizer will reach 20. A learned high-level selector would require a clean
observation-only training pipeline; the older replay-based training script is
not used by any agent evaluated or submitted in this session.

No GPU, replay route, hard-coded target list, seed input to the agent, or changed
simulator is used. Any new best result must be reproduced with the official
evaluator/packer and verified before replacing the existing submission.
