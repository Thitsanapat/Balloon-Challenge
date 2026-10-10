# Scenario 4: remaining structural gaps and relevant research

Research/diagnosis on 2026-10-10. No agent, launch config, simulator, model
weights, source-review ZIP, or live submission was changed by this review.
No new seeds were run. These observations do not establish a higher score.

## Evidence from the selected pure-joint policy

Existing four-pair holdout reports are under
`BalloonPoppingGymEnv/evaluation/results/pure_joint_holdout4_recovered_20261009/`.
Use `*_candidate.json` for the current pure-joint policy; `*_selected.json`
is the older guarded policy, despite that historical filename.

| Seed | Actual pops | Targets in initial route | First pursued target popped before switch? | Powered pursuit time without the active target popping |
| --- | ---: | ---: | --- | ---: |
| 605131410 | 4 | 9 | No | 8.0 s |
| 435857031 | 5 | 9 | No | 14.2 s |
| 927297526 | 6 | 8 | Yes | 0.0 s |
| 298628555 | 4 | 9 | No | 12.0 s |

Method: partition each report's `target_events` into pursuit intervals,
clip intervals to [42,72] seconds (launch plus the public 30-second burn),
and test whether the active target ID appears in `pop_events` within that
interval. A later successful revisit does not retroactively make an earlier
interval successful. Incidental and post-burn pops still count in the score.
These intervals are NOT proven recoverable wasted time: an unsuccessful leg
may still position the rocket for later captures. Four fields are a small
sample, and their initial predicted route lengths are not calibrated hit
probabilities, especially after replanning.

The frozen pure-joint policy totals 19 across these four flights, mean 4.75,
versus 12/mean 3.0 for the old guarded policy. Development seeds 0/2 score
6/4. There is no new 19- or 20-pop single flight here.

## Structural gaps in the current implementation

1. **Route count is not expected realized reward.**
   `agents/chain_beam_agent.py::_node_priority` ranks time within a depth;
   `_make_plan` replaces `best` at each deeper successful expansion. Thus a
   longer nominal chain wins over a shorter one, regardless of reliability.
   `agents/chain_launch_agent.py::_launch_candidate_key` likewise ranks
   `(-count, finish)`. Changing only `_node_priority` cannot compare shorter
   and longer routes at the final selection stage. Retain executable
   proposals from multiple depths before ranking expected realized reward.

2. **Feasible reference is not necessarily a trackable capture.**
   `agents/momentum_beam_agent.py::_valid` checks sampled translational
   curves, thrust, tilt, desired force-axis rate, throttle slope, and ground
   clearance. It does not roll out the current quaternion, angular velocity,
   and actuator states. The lower-level controller DOES compensate actuator
   lag/rate limits; the mismatch is between planning and execution, not a
   total absence of actuator handling. `submission_sqp_v2.py` also uses
   fixed inertia/lever approximations in torque allocation. Their causal
   effect on misses remains to be measured, not assumed.

3. **The entry leg can dominate the flight.**
   First-target failures in three of four held-out fields make reliable
   entry direction, arrival state, and opportunity cost worth testing.
   This does not justify another arbitrary launch-time shift or premature
   target abandonment. Compare reliable entry into a reachable group with
   reliable isolated captures, including the useful departure velocity.

4. **Candidate pruning can hide good alternatives.**
   `_candidate_ids` keeps eight nearest projected targets at each branch.
   Cluster payoff and departure geometry cannot rescue a target removed
   before search. Add a bounded mixture of nearest, forward-aligned, and
   group-entry proposals rather than merely increasing all beam dimensions.

Existing shadow dynamics are a starting point, not a collision oracle:
`doc/scenario4_guarded_6dof_shadow_20261007.md` records a predicted 1.07 m
pass versus actual 1.95 m at a 1.5 m capture boundary. Mean position error
alone is not a valid promotion gate. Forecast and tracking error vectors can
also cancel; reducing one error need not increase pops.

## Primary papers and what transfers

- [Romero et al., Model Predictive Contouring Control for Time-Optimal Quadrotor Flight, T-RO 2022](https://rpg.ifi.uzh.ch/docs/Arxiv21_MPCC_Romero.pdf):
  jointly optimize path progress, timing, and controls under platform input
  constraints, rather than tracking a fixed time schedule. For us, adapt to
  moving target positions at absolute arrival times and finite remaining
  burn. Simply slowing progress along the old geometric path is insufficient.

- [Krinner et al., MPCC++, RSS 2024](https://rpg.ifi.uzh.ch/docs/RSS24_KrinnerRomero.pdf):
  combine constrained predictive control with a small learned residual
  dynamics model. This motivates identifying repeatable acceleration/torque
  prediction errors before increasing neural-network size. Its track and
  periodic-terminal-set safety result does not directly apply to a rocket
  intercepting moving balloons with a finite engine burn. We cannot inherit
  its safety guarantees or claim a 20-pop guarantee.

- [Beul et al., Visually Guided Balloon Popping with an Autonomous MAV at MBZIRC 2020](https://www.ais.uni-bonn.de/papers/SSRR_2020_Beul_MBZIRC_Balloon_Popping.pdf):
  turning over balloons caused misses; straight-through approaches with an
  added intermediate waypoint improved completion despite longer segments.
  Borrow the importance of attack/departure geometry, not the literal
  return-to-center strategy. Their hover-capable MAV, tentacle collision
  mechanism, and stationary pole-mounted balloons differ from this task.

## Recommended sequence, with falsifiable gates

1. Freeze the live policy. Use existing observation-only logs to measure
   first-leg prediction error, capture-margin error near 1.5 m, and model
   errors conditioned on attitude, actuator saturation, and burn age.
   Split by flight/seed, not random time rows. Check signed bias and false
   predicted captures, not only average position RMSE.
2. Calibrate a bounded actuator-aware rollout with public mass/geometry and
   a small regularized residual model. Reuse the shadow predictor but require
   arrival-focused validation before it can override route selection.
3. Implement a separate entry-and-route selector that retains proposals
   across depths, includes straight-through entry/departure alternatives,
   and scores expected realized captures and time cost. A short reliable
   chain must be allowed to beat a long fragile one. Preserve the incumbent
   when the learned estimate is unsupported.
4. Only after predictive calibration, prototype moving-target MPCC on the
   next one or two intercepts. Include attitude/actuator dynamics and an
   explicit solver-time budget/fallback. Compare paired official flights;
   report mean, lower tail, first-leg hit rate, and wall-clock duration.
   Model accuracy or planned route count is not a substitute for score.

Already-tried widening of beam search, density bonuses, velocity diversity,
time shifts, trackability penalties, acceleration forecasts, and sensor
filter tweaks have mixed or negative evidence in the project notes. These
proposals change the selection/execution architecture; they are not claims
that those earlier variants worked. There is insufficient evidence to spend
the final hours on a giant end-to-end PPO run.

All agent-side predictors/selectors must use only observations and public
`given_parameters`; do not read simulator internals, seeds, future balloon
trajectories, or replayed commands. Offline future outcomes may label
training data but must not enter runtime features. Retain training scripts,
artifacts and all used seeds for review. Do not modify the official
environment. The organizer has final authority; see
[Discussion #165](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165).
