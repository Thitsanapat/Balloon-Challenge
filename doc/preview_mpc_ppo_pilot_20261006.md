# Observation-based preview MPC + frozen PPO: bounded pilot

## Why this is a separate experiment

The established full-route controller already recomputes feasible waypoint
chains roughly every 0.4 seconds and the frozen PPO adds a bounded acceleration
residual every 0.1 seconds. It is a receding-horizon planner, but its low-level
tracking correction is a fixed feedback law. This pilot asks whether a small
finite-horizon optimization of tracking error adds value **before** the PPO
residual. It does not replace the existing route selector or train a new
policy; changes in score are attributable to this tracking layer alone.

Official README development rules require decisions based on observations and
given parameters, without accessing other simulator information. Organizer
Discussion #165 permits mathematical models reconstructed from given
parameters and distinguishes an adaptive reference model from a stored
seed-specific trajectory. The trial uses only sensor-derived state, a reference
plan generated online from observations, and public thrust/mass properties.
This is an interpretation of the published rules, not organizer certification.
Training and agent source remain available for review. Sources:
[README](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/blob/main/README.md),
[Discussion #165](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165).

## Algorithm and scope

At most every 0.2 simulated seconds, sample eight future reference points from
the current observed-state route at 0.12-second spacing. Hold the controller's
currently allocated acceleration constant inside an approximate point-mass
model and solve for small additive accelerations. Minimize position/velocity
tracking error, correction effort and inter-step change, subject to ±1.5 m/s²
per axis. Apply only the first correction, then re-estimate on the next call.
The existing thrust/tilt allocation runs again before the frozen PPO residual
and again in the parent's PPO correction. Near the end of a committed spline,
where fewer than two preview points remain, the MPC correction is zero.

This model omits aerodynamic moments, changing wind and actuator lag within
the short horizon. It is an optimization of a deliberately approximate
predictor, so a score improvement is a hypothesis, not a consequence of the
method name. No seed, official truth state, precomputed balloon path or
prerecorded controls enter the agent. Default MPC gain zero should reproduce
the learned parent exactly; the runner checks full action and observation
hashes before evaluating the active variant.

The decomposition follows the idea of learning a bounded residual around a
model-based controller, as studied in [Residual Policy Learning](https://arxiv.org/abs/1812.06298)
and [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201).
NASA's [Small Body GN&C work](https://ntrs.nasa.gov/citations/20080036078)
also illustrates constrained receding-horizon guidance, although its vehicle
and dynamics differ from this challenge. These are conceptual precedents, not
evidence that this particular rocket controller scores better. The current
experiment uses a **frozen** PPO trained on the earlier baseline; joint
training with MPC would be a later, separately validated experiment.

Files:
`agents/preview_mpc.py`, `agents/preview_mpc_ppo_agent.py`,
`scripts/compare_preview_mpc.py`, `tests/test_preview_mpc.py` under
`BalloonPoppingGymEnv/` or the project scripts/tests folders as appropriate.
Three local and three remote optimizer tests pass. A one-thread remote CPU
microbenchmark of 100 synthetic 8-step solves took 0.686 ms per solve on
average; this excludes feature/trajectory generation and full simulation.

## Predeclared official comparison

Frozen learned parent checkpoint SHA-256:
`8aad07e2ac6b834601b1d992f0fbb3ce9bcce1d1bc845ea179623b9e98577b3b`.
Anchor seed `261006001` is a previously used training-field sanity check for
exact zero-gain parity. Active MPC development seeds are `261006020`,
`261006021`, `261006022`, selected before seeing their fields or scores and
checked disjoint from pilot/continuation training logs and the ten-field
route-shadow collection. Each development seed compares frozen PPO vs the
same frozen PPO with MPC gain one, in two fresh official flights. All run CPU
only, one simulator worker/one numerical thread, GPUs hidden, nice level 10.
Any missing normal termination or source hash change fails the comparison.

Remote isolated runtime:
`/home/rtcmspaceweather/balloon_multiscenario_20261004_v1/source_preview_mpc_20261006/`.
Results:
`/home/rtcmspaceweather/balloon_multiscenario_20261004_v1/preview_mpc_pilot_20261006/`.
Outer log: `preview_mpc_pilot_20261006.log`; launch PID `3486974`.
Older runtime/checkpoints and the reviewed submission remain unchanged.

Zero-gain parity **passed** on anchor seed `261006001`: frozen parent and
zero-gain wrapper each scored 6, with identical hashes of every action and
observation over 8,037 steps. Both episodes terminated normally. The runner
has advanced to the first active-MPC development seed.

First active development pair `261006020` finished normally: frozen parent
**1 pop**, preview MPC + frozen PPO **5 pops**. The candidate solved its short
optimization 143 times and returned nonzero corrections on all 143 solves,
so the comparison exercised the new path. Its first popped balloon was the
same ID as the parent's, at nearly the same time; the four additional pops
occurred later. This one-field gain is exploratory and cannot establish a
general advantage. Pair `261006021` finished **4 vs 4** with 141 active MPC
solves; the same four balloon IDs were popped, although their times differed.
The final predeclared development seed `261006022` finished **5 vs 3**. All
three development pairs terminated normally, with frozen-parent scores
**1/4/5** and MPC+PPO scores **5/4/3**. Paired differences are **+4/0/-2**,
means **3.333 vs 4.000**; one win, one tie and one loss. This is mixed evidence,
not grounds to promote the candidate, especially given the two-pop regression
on the final field. On that field both variants first popped balloon #17 at
almost the same time, then chose different reachable routes: the parent added
four pops, the candidate two. Although the code intervention is only at the
tracking layer, its trajectory changes can alter downstream replanning, so
the lost pops cannot be attributed to a single missed control command from
the pop-event log alone. The audit recorded 143/141/138 active MPC solves. The
pilot finalizer passed the frozen-source and parity gates, wrote `rollup.json`
and backed up the result and exact agent source to
`BalloonPoppingGymEnv/evaluation/results/preview_mpc_pilot_complete_20261006.zip`.
Remote and local SHA-256 match, and ZIP integrity passed:
`efb396da2257d48e7bcbf752bd00763e871509c9aecc54c8e3d50e0d78db32aa`.

Five later validation seeds were fixed *before* inspecting any validation
fields or scores: `1141149390`, `1165092567`, `1410104461`, `1399585391`,
`1352273614`, sampled from the reserved validation pool with
`random.Random(261007)`. `scripts/validate_preview_mpc.py` checks these are
disjoint from pilot and earlier validation/comparison/holdout plans and runs
fresh parent/MPC pairs only after the pilot completes. The five-pair validation
was started after the completed pilot because its total gain was positive but
inconsistent. It runs one CPU simulator worker, one numerical thread, hidden
GPUs and nice level 10. Isolated output is `preview_mpc_validation_20261006/`
on the remote project; launch PID `3820869`. The finalizer audits and archives
the completed run as `preview_mpc_validation_complete_20261006.zip`; watcher
PID `3821525`. Both scripts compile locally and remotely; seed-disjointness
passed remotely. No validation result has been used for tuning.

Decision rule fixed before any validation pair completed: advance this frozen
variant to a larger, fresh holdout only if it gains pops in total, wins at
least three of the five paired seeds, and does not lose more than two pops on
any seed. Otherwise retain the existing parent and investigate model mismatch
on the development seeds; do not adjust the MPC against these validation
fields and reuse them as unseen evidence. Even passing this screening rule
would not guarantee a better live score.

## Completed unseen validation and further holdout

All five predeclared validation pairs terminated normally. Frozen-parent
scores were **1/3/3/2/2**; MPC+frozen PPO scores were **2/3/6/4/2**. Paired
differences **+1/0/+3/+2/0** give totals **11 vs 17**, means **2.2 vs 3.4**,
three wins, two ties and zero losses. This passed the screening rule above,
but a five-field improvement is not a tournament-score guarantee, and its
absolute mean is still far below 20 pops. No validation seed was used to tune
the agent. The remote finalizer checked full source/model hashes, normal
terminations and nonzero MPC activity on every field. Results/source archive
copied locally to
`BalloonPoppingGymEnv/evaluation/results/preview_mpc_validation_complete_20261006.zip`.
Remote/local SHA-256 and ZIP integrity passed:
`65fcc4b87ff05c7d64d7d2067b8dd4457bc9ec6cdebcdc66ab05538abe84e71c`.

To check whether the improvement survives more fresh fields, ten holdout seeds
were sampled *before inspecting their fields/scores* from the separate
1.5--2.0-billion pool using `random.Random(261008)`:
`1638895464, 1584252769, 1722428225, 1859368188, 1570091972,
1824032641, 1945086148, 1533739991, 1676901674, 1830415931`.
`scripts/holdout_preview_mpc.py` checks disjointness against all prior plans,
requires the completed validation and its positive predeclared screening
result, freezes source/model/checkpoint hashes, and runs ten new matched
official pairs. `scripts/finalize_preview_mpc_holdout.py` will audit and
archive it. Both compile locally/remotely; the remote seed-disjointness check
passed. One CPU simulator worker, one numerical thread, hidden GPUs and nice
level 10. Isolated remote result `preview_mpc_holdout10_20261006/`, runner PID
`61628`, archive watcher PID `63334`. No holdout result has yet been used for
tuning or agent promotion.

Status: ten-field holdout running. Do not infer a score benefit from unit tests
or synthetic timing. If the holdout regresses, investigate using development
seeds; do not tune to holdout fields and then call them unseen again.
