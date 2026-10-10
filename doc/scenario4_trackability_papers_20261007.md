# Scenario 4 trackability-ranking experiment, 7 October 2026

## What the suggested papers actually transfer

- Guadagnini et al., [*End-to-End GNC Solution for Reusable Launch Vehicles*](https://doi.org/10.3390/aerospace12040339), design onboard reference trajectories under dynamics/control constraints and assess robustness under dispersed conditions. Its reusable first-stage landing mission, numerical solver, and vehicle model are not directly transferable to this balloon-ascent scenario. The useful principle here is that a nominally feasible reference can still be hard to track near control limits.
- Beul et al., [*Visually Guided Balloon Popping with an Autonomous MAV at MBZIRC 2020*](https://www.ais.uni-bonn.de/papers/SSRR_2020_Beul_MBZIRC_Balloon_Popping.pdf), generate closed-loop trajectories constrained by acceleration and jerk, synchronizing axes for a relatively straight approach. Their vision pipeline, hover-capable multirotor dynamics, and physical spiked tentacles do not apply to this rocket or its supplied balloon-state observation. The useful principle is to rank intercept trajectories by trackability, not only shortest nominal time.

The current agent already checks nominal thrust, tilt, axis rate, and throttle slew, and uses quintic minimum-jerk-like splines. `scenario4_trackability_agent.py` adds a small penalty for *first-leg utilization close to these same public limits* when sorting feasible beam nodes. It does not implement SCP, the MAV's triple-integrator MPC, hidden simulator information, or a pretrained seed-specific policy. The original guarded terminal control and public-parameter feasibility checks remain unchanged.

## Test plan frozen before scores

Use already-known development seeds, not live-seed stand-ins. First run one fresh complete Scenario-4 flight on seed 0 against the selected guarded agent's saved score of 5. Stop the variant if it scores below 4 or errors/truncates. Otherwise run seeds 1 and 2 against saved guarded scores 4 and 3. Require a better three-seed total and evidence that route ranking changed before spending new untouched paired seeds. The qualification launcher and standalone source remain untouched until a separately predeclared multi-seed paired holdout and source-package check passes.

The result report is local, credential-free, and ignored under `evaluation/results/`. Runtime agent inputs remain only current observations and `given_parameters`. Offline score/truth fields are for evaluation after actions, never passed to the agent.

## Initial development result and narrower follow-up

The `trackability_time_weight=1.0` candidate completed seed 0 with 4 pops
versus the selected guarded agent's 5. It ranked 7,187 beam nodes and took
131.1 s after reset, versus 133.3 s for the selected run. Its mean observed
spline-tracking error was 0.707 m, above the selected run's 0.529 m. It
completed seed 1 with only 1 pop versus 4, despite *lower* mean tracking
error (0.584 versus 0.726 m). It ranked 3,622 nodes. Thus the proxy affects
route decisions but does not reliably rank realized hits; a trackable route
can reach a less fruitful balloon region. Full reports are ignored local
`s4_trackability_dev_seed0_20261007.json` and
`s4_trackability_dev_seed12_20261007.json`. The latter contains only seed 1:
seed 2 was stopped during Monte Carlo reset after the three-pop seed-1 loss,
before any score or complete flight, to save resources. No result is claimed
for that aborted seed-2 attempt.

One conservative follow-up changes only the config argument to
`trackability_time_weight=0.25` seconds, less than the 0.5-second planner
time grid. This aims to affect near-ties instead of dominating route time.
Test known development seed 1 first. If it still scores below 3 versus the
selected 4, stop the variant; otherwise compare seed 0 before considering
any untouched paired holdout. This is exploratory tuning on known seeds, not
an independent demonstration of generalization.

## Conservative follow-up result and decision

The `trackability_time_weight=0.25` seed-1 episode completed without
truncation, with unchanged agent source. It again scored **1 pop, target 0**,
versus the selected guarded agent's **4 pops** on this already-known seed.
The candidate ranked 3,622 nodes, had mean observed tracking error 0.584 m,
and took 107.0 s after reset. Its route and score were effectively unchanged
from the stronger-risk version. The report is the ignored local
`s4_trackability_tiebreak_seed1_20261007.json`.

**Reject both trackability weights for live use.** The weaker penalty did not
restore the baseline route because many beam candidates can tie on discrete
arrival times; even a small nonzero risk changes their ordering. Measured
tracking error is not the same as expected pop count: seed 1 illustrates a
route that is easier to track but reaches a less productive balloon group.
Do not run new held-out seeds or tune another weight on these same fields.
The selected standalone guarded-innovation source still has SHA-256
`dab27623c797c70c6fc5c13b48e4faa2e146a694c7de737f6cae8138485fb333`.

The papers remain useful as *architectural guidance*, not drop-in algorithms.
The next meaningful model would estimate tracking/capture likelihood from
the rocket's actual observation and actuator history, then jointly rank
reachable target groups under that calibrated likelihood. Before affecting
flight commands, such a model needs a causal offline prediction-error test
and fresh paired official-simulator seed evaluation. A nominal force-margin
surrogate alone has now failed the development screen.
