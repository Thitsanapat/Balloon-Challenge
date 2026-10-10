# Scenario 4: same-deadline joint refresh, simplified candidate

The selected live agent and `run_qualification.cmd` are unchanged. The
existing `Scenario4ReachableRefreshAgent` previously gained one pop across
four independent paired fields (7/5/5/2 versus 7/4/5/2); its separate
actuator-preview screen never flagged a leg in those flights. The supported
mechanism is therefore the full-route derivative refresh after a committed
leg becomes infeasible, **not** the actuator preview. Four fields are far too
few to claim a reliable gain.

The same frozen reachable-refresh source/config was then checked on two
*already-used development* fields where a learned route selector had regressed:

| Seed | Selected live agent | Reachable-refresh candidate |
| ---: | ---: | ---: |
| 178420909 | 6 | 6 |
| 820431975 | 6 | 7 |

The candidate kept the whole observed route and original deadlines when it
found a feasible joint spline. On the 7-pop field its actuator-preview gate
again flagged zero cases; the joint refresh was accepted ten times. These
two fields were **not** independent holdouts for selecting this candidate,
because their baseline/learned scores were already known. The 6-pop tie also
had a different final balloon ID and more allocation-saturated steps, so a
score tie does not prove the trajectories are equivalent.

`scenario4_pure_joint_refresh_agent.py` removes the untriggered preview and
implements only the same-deadline joint derivative refresh. It uses current
observation and public `given_parameters`, never seed or hidden simulator
state. Its config is `scenario4_pure_joint_refresh_launch42.yaml`.
Focused unit tests verify that a feasible repair preserves target IDs and
deadlines and that a non-active target cannot trigger rescue. A complete
official-simulator flight on the already-used seed `820431975` scored **7**,
the same as the older candidate. It also had identical pop events/times,
route events, target events, mean navigation error and ten accepted joint
rescues. This is strong behavioral evidence on *one* seed, not an action-hash
proof for every trajectory.
Neither candidate should replace the reviewed live agent without fresh,
predeclared paired scores and a standalone source-review bundle.

## Fresh paired reliability screen, predeclared before flights

Four previously absent seeds were checked against the local seed registry and
recorded in `qualification/used_seed_manual.jsonl`: `605131410`, `435857031`,
`927297526`, `298628555`. The registry was regenerated. The holdout runner
`scripts/validate_pure_joint_holdout.py` waits for the ongoing audited data
recovery to finish, then flies the selected standalone and simplified joint
candidate in fresh official processes, sequentially, on each seed. GPU
visibility is disabled. Source/config hashes are frozen in its plan and
checked between flights. It writes only local diagnostic reports under
`evaluation/results/pure_joint_holdout4_20261009/`; it never uploads or
changes `run_qualification.cmd`.

Predeclared screening rule: candidate total at least **2 pops above** selected
across four pairs, no field more than **1 pop below**, and candidate wins at
least **2 fields**. Passing this small screen is necessary but not sufficient
for live promotion; a standalone source-review bundle and an additional
environment-parity check would still be required. A failure keeps the
reviewed agent. No fresh-pair scores were known when these seeds and criteria
were fixed.

## Holdout-runner recovery

The four-field test initially stopped after the first **selected** flight on
seed `605131410` (4 pops) because the runner expected a JSON object while
`evaluate_guidance` writes a one-element JSON array even for one seed. The
candidate was not flown on that seed; no pair was counted. The original
`pure_joint_holdout4_20261009/` folder and `failed.json` remain untouched.
The corrected runner validates that array format, verifies the original
source/config hashes and completed selected flight, then copies that one
result into `pure_joint_holdout4_recovered_20261009/` and continues. Its
unit tests reject object and multi-flight shapes. The predeclared seeds and
promotion screen are unchanged; no live policy is automatically replaced.

## Completed fresh paired screen

All four pairs finished successfully in the recovered directory, with
unchanged agent sources and normal termination:

| Seed | Reviewed guarded | Pure joint refresh | Difference |
| ---: | ---: | ---: | ---: |
| 605131410 | 4 | 4 | 0 |
| 435857031 | 1 | 5 | +4 |
| 927297526 | 4 | 6 | +2 |
| 298628555 | 3 | 4 | +1 |
| Total over four flights | 12 | 19 | +7 |

Mean score rises from 3.0 to 4.75 on this small set; three wins, one tie,
no regressions. The predeclared screen passes. This is not a 19-pop single
flight, nor a guarantee on an organizer's fresh seed. Sources remain frozen
for reproducibility and the reviewed live launcher has not changed. The
next navigation/control experiments use this candidate as their comparison
baseline and use old development fields rather than tune on these four fields.

The v2 standalone in `review/scenario4_pure_joint_candidate_v2.py` now also
passes a complete seed-2 packaging replay: 4 pops with identical events and
all compared telemetry. Use its matching review YAML for a separate candidate
test. The first standalone attempt failed because its older embedded planner
did not call the new recovery hook; that v1 is retained only for audit and
must not be used. The v2 builder includes current planner dispatch methods.
