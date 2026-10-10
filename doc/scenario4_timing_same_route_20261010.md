# Scenario 4: allow a faster feasible version of the same route

The current agent runs a continuous time-allocation optimizer after a new
route is found. With `timing_only_more_hits: true`, its output is adopted only
when the optimizer can also insert another balloon into that route. Four
inspected completed flights each spent 9–17 timing solves and adopted **zero**
optimized timings. The solver's results were thus unused on those flights.

This experiment changes one existing agent argument to
`timing_only_more_hits: false`. The optimizer may then adopt a shorter version
of the same observed route by at least 0.05 seconds, provided it passes the
existing dense thrust, tilt and rate feasibility checks. The solver can still
insert an extra observed balloon as before. All other model, route, launch,
forecast, guidance, navigation and control settings are identical.

No code in the official simulator/evaluator or live launcher is changed. The
agent receives only observation and `given_parameters`; the optimizer has no
seed, future field, truth or cached commands. This is a test of online time
allocation, not a claim of global trajectory optimality.

The plan was written before evaluation to
`evaluation/results/timing_same_route_dev_20261010/plan.json`. Development
seeds 0 and 2 already have verified pure-joint scores 6 and 4. Both candidate
flights use fresh official simulator processes, two CPU workers and no GPU.
Predeclared screen: at least +2 combined pops, no seed worse by more than one,
and `timing_adopted` positive. Any passing result still requires fresh paired
seeds and a standalone/full JSON check before switching the live agent.

The untouched source to inspect is `agents/time_allocation_agent.py`; the
selected qualification config remains
`evaluation/configs/submission_scenario4_pure_joint_launch42.yaml` until
independent validation and packaging succeed.

## Result

Both complete official flights ended normally with the expected unchanged
source hash. Seed 0 scored 6 and seed 2 scored 4, exactly matching the selected
agent. Both ran 17 timing solves and adopted **zero** new timings. The one-line
gate change was therefore behaviorally inactive on these fields and failed
the predeclared screen. Do not promote it or infer that faster schedules are
impossible on other fields. Reports are `0_candidate.json` and
`2_candidate.json` in the plan directory.
