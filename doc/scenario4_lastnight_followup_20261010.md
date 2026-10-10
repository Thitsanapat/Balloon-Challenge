# Scenario 4 follow-up: route, forecast and replanning screens

All scores here are complete local official-simulator Scenario-4 flights, not
leaderboard submissions. Development seeds were already used and logged in
`qualification/used_seed_manual.jsonl`; none is eligible for an organizer live
evaluation without redraw. Experiments use current/past observations and
public `given_parameters`, not hidden simulator state, seed-dependent routes,
or prerecorded commands. No GPU was used. The organizer has final authority
on rule compliance; see [Discussion #165](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165).

| Isolated experiment | Development result | Reference | Decision |
| --- | ---: | ---: | --- |
| Same-route timing allowed without extra planned hits | seed 0/2: 6/4 | 6/4 | Tie; timing adopted zero times |
| Beam width doubled 12→24 | seed 0: 4 | 6 | Stop after pilot |
| Arrival-time-specific observed-wind correction | seed 0/2: 5/2 | 6/4 | Stop after development |
| 0.2 s reference refresh near committed target | seed 2: 4 | 4 | Stop after pilot |

The wind correction has a real mathematical motivation: the existing planner
converts a correction computed for one estimated horizon into an effective
velocity, then evaluates candidate routes with different arrival times. The
new branch calculates the correction at each candidate time and passes a
synthetic linear-shear test. Nevertheless, it changed route selection and
lost three pops over the two development flights. Correcting one model
inconsistency did not establish a better end-to-end policy. The exact rejected
source and test are retained in `review/scenario4_horizon_profile_experiment_20261010.zip`
(SHA256 `1d641ccd3e25a0bd35ce3acac4289d5e384f6c80376c43bef3bccf4186bc2181`).
Inspect or extract that rejected experiment only in a separate checkout; it
contains modified planner modules and must not overwrite the live checkout.
Afterward, the three shared planner files were restored to their original
hashes so rebuilding the selected standalone again matches byte-for-byte.

The terminal-cadence variant was exercised, but increased reference refresh
did not add a pop on the seed with a documented close miss. The four-pair
fresh holdout still favors the selected standalone pure-joint agent over the
previous guarded agent (19 versus 12 total, 4.75 versus 3.0 per flight).
There is no new single-flight score of 20, and those holdout totals must not
be presented as a 19-pop run.

**Live decision:** leave `run_qualification.cmd`, its default config, and
`submission_scenario4_pure_joint_v2.py` unchanged. Keep the tested source ZIP
in `review/scenario4_finalnight_joint_source_20261010.zip`. Do not rebuild it
from these rejected experimental planner modules. A future improvement should
first calibrate *joint* target-forecast, spline-tracking, and capture
probabilities from observation-only flight logs, then test route choices on
new paired seeds before replacing the submitted policy.

Raw reports and predeclared gates are in
`evaluation/results/timing_same_route_dev_20261010/`,
`beam24_dev_20261010/`, `horizon_profile_dev_20261010/`, and
`terminal_cadence_dev_20261010/`.
