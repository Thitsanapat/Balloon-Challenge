# Exit continuation and PPO run-health experiment (2026-09-24)

Update September 25: the single-worker recovery finished its two-hour budget,
with 52 complete episodes and zero truncations. No training job remains active.
See `doc/bridge_and_ppo_validation.md` for final-checkpoint validation; the
running-status notes below are historical snapshots.

## Diagnosis before changing the planner

Baseline fresh Scenario-1 reports score 10 (seed 0) / 9 (seed 3). Seed 0 has
17 distinct IDs across overlapping/superseded proposals, but only 10 selected
target IDs; all 10 were eventually popped. The seven never-selected targets
are not seven tracking failures. Seed 3 selected eight IDs and popped nine,
including an incidental hit. Mean sampled tracking errors were 0.0377 / 0.0775 m.
These are conditional tracking averages, not worst-case errors or proof that
guidance can never improve. Reports: `exit_baseline_diagnosis_v2_20260924.json`.
The original v1 diagnostic is superseded: its saturation fraction mixed two
different counting populations. v2 reports the two raw counts instead.

`scripts/summarize_guidance_outcomes.py` analyzes saved reports only; it is never
an input to any agent. It deduplicates route IDs, separates selected/unselected
proposals, and summarizes observation-only telemetry. Error component magnitudes
are descriptive and do not establish causality.

## Bounded exit-aware search

`ExitAwareChainAgent` subclasses the frozen `submission_time_allocation_v1.py`;
that source and all official components are unchanged. It changes only the
beam-node ordering: elapsed time plus a weighted estimate of time to reach an
additional released target from the current planned exit position/velocity.
The estimate samples 0.5-second times up to six seconds, using current target
positions/velocities, gravity, estimated disturbance and force/tilt bounds.
It excludes used, unreleased and cooldown targets. No feasible continuation
gets a finite capped cost, not a hard rejection: the full chain can reshape
earlier derivatives. No remaining candidates or useful horizon gives zero cost.

This is a constant-acceleration heuristic, not an exact reachable set or an
admissible bound; it omits transient attitude/actuator effects. Existing complete
joint-trajectory constraints and dense validation still control acceptance.
Groups do not earn imaginary points, and isolated targets remain eligible.
Zero weight takes the original ordering exactly.

Paired development tests: weights 0 (disabled control), 0.25, 0.75; seeds 0/3;
fresh official episodes with telemetry. These seeds are not held-out evidence
of generalization. At most four independent evaluation processes, CPU only.
All selected scores must come from normal complete episodes, not plan counts.

First paired results: disabled control **10/9**, weight 0.25 **9/9**. All four
episodes terminated normally with unchanged agent sources; disabled seed-0
pop events reproduce the frozen reference. Telemetry shows no unpopped targeting
episodes in these four runs. The 0.25 variant's tracking averages were lower
(0.0242/0.0260 m), but that did not increase actual hit count. Improved tracking
alone is therefore not a sufficient score-selection criterion.

Alongside the 0.75 pair, a separate zero-weight control doubles beam width from
12 to 24 on both seeds, isolating aggressive search pruning from the new exit
heuristic. This is two additional full evaluations, not a change to the default.

Weight 0.75 scored **9/8**; neither exit weight improves the 10/9 control.
The exit-order experiment remains disabled by default in any promoted agent.
This is evidence against these bounded heuristics on these two seeds, not proof
that continuation-aware planning cannot help in general.

## Completed route results

| Route search | Seed 0 | Seed 3 |
| --- | ---: | ---: |
| Frozen behavior, weight 0, beam 12 | 10 | 9 |
| Exit weight 0.25, beam 12 | 9 | 9 |
| Exit weight 0.75, beam 12 | 9 | 8 |
| Exit weight 0, beam 24 | 7 | 9 |

All eight episodes completed normally without truncation. No tested variant is
promoted and no leaderboard upload occurred. Increasing beam width is not a
monotonic quality guarantee for this pruned heuristic search and launch selector.
The wider seed-0 run's maximum retained route depth was eight, and there were no
unpopped selected-target episodes. These data do not support blaming its lost
score on failure to hit selected targets. More promising subsequent work would
compare fundamentally different route proposals, rather than larger exit bonuses.

## PPO health intervention

The eight-worker run `ppo_large_20260924` was stopped gracefully through its STOP
file after repeated official truncations: **36 of 41** completed episodes were
truncated. Final cumulative PPO decisions 8,656 (6,608 beyond the pilot).
`final_model.zip` was saved; the trainer exited. This is not a reliable score
comparison against the baseline. No unverified model replaces the 10-point file.

Installed SB3 `SubprocVecEnv` resets a finished environment before sending its
step response; its parent waits for all worker responses. Official episodes
enforce 600 wall-clock seconds, including waiting while another worker performs
a costly fresh reset/planning warmup. Monitor files show successive truncated
episodes roughly 650 seconds apart with falling decision counts. This supports
cross-worker reset waiting as a cause; prior logs lacked per-episode wall/sim
times, so the original truncation reason was not individually recorded.

After the old process exited, the training wrapper gained external elapsed-wall
and public simulation-time diagnostics (not passed to the policy). The trainer
now stops and saves on three consecutive truncated episodes. Simulator clocks,
parameters, physics, scoring, and observations are not modified. The guard does
not retroactively remove collected data; it limits further unhealthy training.

A one-environment recovery experiment resumes the completed 2,048-decision pilot,
NOT the heavily truncated run. This eliminates waiting on other vector workers.
It retains one P40, caps training at 20,000 additional decisions or two hours,
and saves every 512 decisions. It is a throughput/health experiment first, not
a claim that fewer workers guarantee better policies. Frozen old source archives
and all failed-run checkpoints are retained.

Recovery directory: `/home/rtcmspaceweather/balloon_champion_20260923/ppo_single_recovery_20260924`.
PID `2400956`, log `/tmp/ppo_single_recovery_20260924.log`; detached with nohup,
single visible GPU UUID `GPU-f32768cf-a4cf-c057-c466-db71f18efafa`, BLAS/OMP/MKL
one thread. Trainer options: `--resume ../ppo_pilot_v2_20260924/final_model.zip
--steps 20000 --envs 1 --device cuda --hours 2 --n-steps 128 --batch-size 128
--checkpoint-every 512 --seed 240928`. Snapshot source archive:
`official_ppo_health_fix_20260924.tgz`, applied after prior source archives.

74 selected unittest checks pass, including exit-velocity ranking, finite capped
unreachable estimates, observation/cooldown exclusion, exact disabled ordering,
deduplicated proposal accounting, action cadence, and the truncation guard.
Official simulator/evaluator/packer/verifier still match `bd9bb51`.

The recovery monitor recorded its first complete episode: 357 PPO decisions,
7 actual pops, `official_truncated=False`, at 139.98 s since monitor creation.
The PPO callback can report this later because SB3 waits for the next reset to
finish before forwarding the terminal transition. This is a stochastic training
result on a fresh seed, not a paired improvement or a submission score.

The terminal callback subsequently confirmed public simulation time 59.63 s,
episode wall time 96.57 s, score 7, no truncation. The next episode is underway.
This validates the first complete single-worker episode only, not long-run
stability. Recovery began at Unix time 1790262988.1086094 (~22:16 ICT); its two-hour
budget ends around 00:16 ICT September 25, subject to active-step/reset delay.

Before handoff the recovery completed two episodes, both score 7 and neither
truncated, with 656 additional decisions and unchanged source hashes. The first
new checkpoint is `checkpoint_2560.zip` (cumulative step count including pilot).
The recovery remains running detached; no route-evaluation processes remain.

Downloaded reports, stopped-run checkpoints and a historical recovery snapshot:
`BalloonPoppingGymEnv/evaluation/results/exit_health_20260924/`.
All eight reports verify normal termination and in-run source immutability.
Remote agent dependencies match local source modulo Windows/Linux line endings
in `base_agent.py`; raw hashes are retained in the reports. The disabled seed-0
control's pop events exactly match the frozen reference. No new submission is
generated or promoted; the existing verified 10-point payload is unchanged.
