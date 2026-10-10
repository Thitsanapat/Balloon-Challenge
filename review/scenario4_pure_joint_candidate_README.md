# Pure joint route-refresh v2

`scenario4_pure_joint_candidate_v2.py` is a reproducibly assembled standalone
agent for Scenario 4. Entry class: `Scenario4PureJointRefreshAgent`.

Its ready-to-test config is `review/scenario4_pure_joint_candidate_v2.yaml`.
It has the same agent parameters as the tested modular config
`BalloonPoppingGymEnv/evaluation/configs/scenario4_pure_joint_refresh_launch42.yaml`.
On 10 October the byte-identical copy
`BalloonPoppingGymEnv/agents/submission_scenario4_pure_joint_v2.py` became the
qualification default with `submission_scenario4_pure_joint_launch42.yaml`,
after successful local official-runner/JSON rehearsal. The old guarded source
is unchanged and remains available through `run_qualification_guarded_backup.cmd`.

The candidate retains an observed route and its intercept deadlines while
re-solving its joint spline derivatives when the active committed leg is no
longer feasible. It uses current observations and public `given_parameters`.
It does not read a random seed, hidden simulator state, stored flight paths,
or diagnostic truth. It does not need a trained model or GPU.

Predeclared four-field test with fresh official simulator processes:

| Seed | Guarded | Joint refresh |
| ---: | ---: | ---: |
| 605131410 | 4 | 4 |
| 435857031 | 1 | 5 |
| 927297526 | 4 | 6 |
| 298628555 | 3 | 4 |

These are four separate flights. Total 12 versus 19, mean 3 versus 4.75.
No claim of 19 pops in one flight or of guaranteed performance on a new seed.

Rebuild with `python -m scripts.build_scenario4_joint_submission_agent
--output <unused-output.py>`. The builder refuses to overwrite a file.
Its generated SHA256 is
`9ff7f93d21a945d0479289be57bd893d8f33ab169fb1a6bfeb153ea442b6efb9`.

AST-import, inherited-method equivalence, numerical spline parity and
committed-leg dispatch tests pass. V2's complete official seed-2 flight scored
4, with identical pop events, time, target/route events, diagnostics, navigation
metrics and guidance telemetry to the modular flight. This is one-seed
packaging verification, not another independent score sample. This artifact
is source code, not a packed submission JSON, and no result was uploaded. Same-environment
packaging parity must not be confused with testing in an environment pinned
to the official lockfile.

The fresh local packaging rehearsal on seed 741093826 scored 4, produced a
validated official JSON object and restored the scenario file. It was not
uploaded and is not an organizer evaluation. The final source package is
`review/scenario4_finalnight_joint_source_20261010.zip`; the older source-v2
archive is retained unchanged as a historical artifact.

## Rejected packaging attempt

Do not use `scenario4_pure_joint_candidate_v1.py`. Its archived seed-2 flight
scored 3 versus the modular agent's 4 and failed behavior parity. That first
builder appended the recovery override to an older standalone planner that
never called it. The corrected builder replaces the current ChainBeamAgent
and ChainLaunchAgent definitions before constructing their subclasses, then
adds the joint repair class. The old file/report are retained for audit.
