# Paired-successor route search and completed PPO validation (2026-09-25)

## Current evidence and scope

The unchanged verified submission scores 10 on Scenario 1, seed 0. Recent
exit-aware and wider-beam experiments did not improve it. This work does not
claim 20 points or assume PPO training reward is a submission score.

Organizer clarification was rechecked:
https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165
Only observations and given parameters enter agents. No prerecorded routes,
seed inference, future field loading, simulator changes, or leaderboard uploads.
Static source review and reproducible tests are evidence, not organizer approval.

## Completed single-environment PPO run

The two-hour recovery stopped normally at its wall budget (7,219.20 seconds,
including the active callback delay). It completed 18,481 additional decisions,
20,529 cumulative including the pilot; 52 completed episodes, zero truncations,
unchanged source hashes and no error file. Its final model is saved remotely.
The per-episode health problem observed with eight synchronous workers did not
recur in these 52 single-worker episodes. No GPU training is currently active.

The final deterministic actor is exported to
`agents/submission_ppo_recovery_final.py`. Actor output agreement with SB3 is
2.24e-8 maximum absolute error on the exporter's 32 test inputs. Fresh official
evaluations on development seeds 0/1/2/3 compare it with baseline 10/8/9/9.
This is checkpoint evaluation, not a reason to promote based on training scores.

Completed deterministic scores are **8/9/9/8** on seeds 0/1/2/3, compared with
the reference **10/8/9/9** (paired means 8.5 versus 9.0). All four runs completed
without truncation and with unchanged source dependencies. The final residual
PPO is NOT promoted and training is not extended automatically on this evidence.
These 20,529 cumulative decisions do not establish PPO convergence or a limit
on what another learned architecture could achieve.

The unchanged reference is also evaluated on additional seeds 16/17/18/19.
Seeds remain external to the agent. Any better score from this sweep is field
selection, not an algorithmic improvement or unbiased generalization estimate.

Separate acquisition-time controls use the frozen reference at launch times
12/16 seconds on seeds 0/3. The reference's seed-0 first hit is at 32.41 seconds
after launch near 24 seconds: about eight powered seconds before scoring.
Earlier launch may trade fewer released targets for less initial climb/travel.
Older launch tests near 20--32 seconds and legacy-controller early launches do
not settle these particular stronger-controller settings. All other constraints
and search settings stay fixed; this is a four-run hypothesis test, not a claim
that earlier is necessarily better.

These controls completed without truncation: launch 12 scored **6/7**, launch
16 scored **6/6** on seeds 0/3. Neither is promoted. Additional unchanged-agent
seeds 16/17/18/19 scored **7/8/6/8**, also below the existing best of 10.

The completed PPO artifacts and four official reports are downloaded under
`evaluation/results/ppo_recovery_final_20260925/`. The exported NumPy agent is
available locally but remains experimental; no new submission is generated.

## New structural search hypothesis

The frozen joint-chain search rejects a minimum-jerk prefix immediately if its
sampled physical checks fail. However, `free_chain` recomputes all prior waypoint
velocities and accelerations when a successor is appended. Failure of this
particular prefix curve therefore need not imply failure of every longer curve
through those targets. A bounded paired-successor proposal can bridge one
rejected intermediate prefix, but the complete extended chain must still pass
all physical checks before execution. No infeasible prefix is commanded.

The experimental agent keeps the frozen planner as incumbent, limits extra
search work, and accepts only fully checked proposals. Disabled mode must match
the baseline. This is not an exhaustive optimizer or a proof of reachability.

## Independent first-leg hypothesis

The frozen `ChainBeamAgent._make_plan` starts the first leg at 2 seconds and
later legs at 1 second. Optional shorter leg times apply only after the first
leg. Fresh seed-0 routes start at 24.02, 32.41 and 44.86 simulation seconds, so
the first-leg restriction also applies while already moving. Timing optimization
can only reduce a 2-second nominal leg to 1.2 seconds, and faster equal-count
plans are not adopted by the promoted setting. The momentum fallback already
searches shorter legs, so this is not a universal two-second lockout.

The new proposal search optionally uses a 1-second first leg only more than
0.5 seconds after launch. Default remains 2 seconds; launch/physical/actuator
constraints are unchanged. Compare identical bridge budgets with minima 2/1
on seeds 0/3, retaining the baseline for inferior proposals. Prior experiments
with short *later* legs are not evidence for this distinct change.

## Implementation, controls, and launch-selection correction

`agents/bridge_chain_agent.py` subclasses the frozen submission agent. It uses
bounded root and incumbent-prefix frontiers; each rejected prefix gets at most
one recovery successor. Complete candidates pass 17-sample physical screening
and improving proposals pass 65-sample screening. These are sampled model
checks, not a continuous-time feasibility proof. Only additional planned hits
are adopted by default. A larger plan is not an observed score increase.

The budget counts additional curve fits, not baseline work or dense checks.
`bridge_recovered` counts recovered improving proposals, not every feasible
recovery; `bridge_extra_planned_hits` must not be reported as actual hits.

Disabled seed 0 reproduces the baseline score **10 and identical pop events**:
`evaluation/results/bridge_disabled_s0_20260925.json` versus
`evaluation/results/residual_ppo_zero_s0.json`.

Initial enabled tests (source SHA256
`5c44e665bf8a7a5445908bf22fabc37b50e47a5bfb143d01b7f648a95a2f82cb`)
scored **7/9** on seeds 0/3 for BOTH first-leg minima 2/1. All completed, none
truncated. Seed 0 reported no bridge adoption but differed from the baseline:
the inherited launch-axis comparison invokes the overridden planner on
temporary deep-copied state. Bridge proposals changed the selected launch axis;
restoring that temporary state discarded the corresponding diagnostics. Thus
zero retained adoption counters did NOT imply unchanged behavior.

The correction skips bridge search until the existing launch-axis selector has
committed its choice. No frozen reference code was edited. A regression test
checks that even a parent beam search during launch probing cannot invoke extra
bridge search. The corrected source SHA256 is
`25b40aadf727b221697d5e0fefe25a90f873afd832144c51419e825d070b2228`.
Twelve bridge tests and 43 combined focused tests passed locally; the twelve
bridge tests also passed remotely before evaluations. Four follow-up runs use
the fixed 2-second minimum, budgets 1,024/4,096, and seeds 0/3, CPU only.

## Completed corrected results

| Extra-fit budget per search | Seed 0 | Seed 3 | Total fits (seed 0 / 3) | Adopted proposals |
| --- | --- | --- | --- | --- |
| 1,024 | 10 | 9 | 807 / 1,060 | 0 / 0 |
| 4,096 | 10 | 9 | 2,997 / 3,100 | 0 / 0 |

All four episodes terminated normally, without truncation or source changes;
reported wall times were 154.0--167.4 seconds under four-process concurrency.
Both seed-0 runs have exactly the disabled control's pop events. Launch
comparison counts/durations also match; axis components differ by less than
1e-12 between the local and remote platforms, so raw nested-list equality is
not an appropriate cross-platform axis test.

No recovered improvement was found on these runs. Increasing the cap produced
more fits but no additional hits, so the experimental bridge is not promoted.
The synthetic prefix counterexample establishes a search limitation, not its
practical importance on these two fields. First-leg minima 2/1 were compared
only in the earlier launch-confounded version; their isolated effect remains
unresolved. Do not infer from those runs that a shorter first leg cannot help.

All sixteen bridge, early-launch, and additional-seed reports are downloaded
under `evaluation/results/bridge_completed_20260925/`, alongside the archived
original and corrected sources. Remote evaluation/training processes were
checked after completion: none remained active. No GPU was used in this batch.

The existing 10-point submission was rechecked: JSON object, source matches the
frozen local agent, all 19 verifier checks passed, and official components
remain unchanged versus `bd9bb51`. Evidence is saved in
`evaluation/results/bridge_continuation_best_audit_20260925.json`.
No submission was uploaded or overwritten, and no result of 20 is claimed.

Next useful experiment: isolate short in-flight first legs with the corrected
launch guard, or measure why the candidate frontier dies before spending its
budget. Merely increasing the budget or extending the underperforming PPO run
is not supported by these results. Any additional experiment should retain
paired baseline controls and promote only verified actual score gains.
