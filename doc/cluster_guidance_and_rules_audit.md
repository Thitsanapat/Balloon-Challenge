# Cluster-aware guidance and submission review (2026-09-23)

## Rules interpretation, not organizer approval

Rechecked the current organizer [clarification #165](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165)
and [README](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/blob/main/README.md).
Selecting groups using current observations is consistent with the documented
input boundary. It is not an exemption to the prohibition on precomputed balloon
trajectories, seed inference, stored replay routes or simulator-internal access.
The organizers retain final judgment; an official trajectory verifier passing
does not certify the provenance or legality of every algorithm.

The group planner:

- Uses only currently released balloons' observed positions/velocities.
- Predicts short-horizon positions by constant-velocity extrapolation.
- Scores coherent nearby neighbors with a smooth distance/relative-velocity
  kernel, and discounts that benefit by estimated travel time.
- Keeps nearest-single candidates in half the search slots, so distant groups
  cannot eliminate all cheap nearby opportunities.
- Still checks each selected trajectory against thrust, turn, tilt, slew and
  ground constraints. The actual simulator, not the heuristic, determines hits.

This is a physics-constrained search heuristic, not PPO, not a learned policy,
and not a guarantee of the globally optimal trajectory or of 20 points.

## Existing 10-point submission audit

File: `evaluation/results/20260923T074855.512Z_TASTI_Cool_Ba_Malaew_3d64859bc1cc4647876b30e2f8928efd_submission.json`

- SHA-256 of the entire payload:
  `b6ff58aac9ad159357a1f33f155546bf2cb4d42a8636a609570b9c7c6a280afe`.
- JSON object, format version 2, scenario 1, measured score 10.
- Embedded agent source exactly matches `agents/submission_chain_launch_v1.py`.
- Only imports: NumPy, SciPy optimization, `functools`, `copy`, official BaseAgent.
- Manual inspection: physical model derived from given parameters; balloon
  targets extrapolated from observations; routes initially empty and planned
  online; own attitude inferred from commanded launch angles and gyro history.
- Mathematical coefficient caches are not balloon-data caches. The launch
  comparison copies only the agent's own state, not the environment.
- Static source scanner found no flagged I/O, seed key or private simulator access.
  This scanner is deliberately only an aid, not a complete security/rules proof.
- Environment/evaluator/packer/verifier have no diff from release `bd9bb51`.
- The official verifier was freshly rerun, exit code **0**; all 19 findings passed.
  All 10 hits are reachable; worst closest approach 0.392 m versus radius 1.5 m.

Credential-free machine-readable evidence:
`evaluation/results/submission_10_rules_audit.json`.
The payload itself contains team credentials and must not be published.

Re-run the audit from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/audit_agent_submission.py BalloonPoppingGymEnv/evaluation/results/20260923T074855.512Z_TASTI_Cool_Ba_Malaew_3d64859bc1cc4647876b30e2f8928efd_submission.json --agent BalloonPoppingGymEnv/agents/submission_chain_launch_v1.py --report BalloonPoppingGymEnv/evaluation/results/submission_10_rules_audit.json --verify
```

The audit reads but never executes the embedded agent. It does not print any
team secret, environment credential or complete submission payload.

## Experiment status

The original frozen standalone submissions are unchanged. Development-only
`ChainBeamAgent` gained two ranking hooks whose default behavior is unchanged.
`ClusterChainAgent` overrides those hooks; its config is
`evaluation/configs/cluster_chain.yaml`.

Initial experiment grid: cluster radius 20 m, attraction weight 2/5 seconds,
beam-density bonus 0/2 seconds, seeds 0/1 (eight CPU jobs). A separate local
control disables both bonuses to check equivalence with the 10-point reference.
The disabled-cluster control reproduced **10 points**, with the same pop times
as the frozen reference. Initial results:

| Candidate bonus | Beam bonus | Seed 0 | Seed 1 |
| ---: | ---: | ---: | ---: |
| 2 | 0 | 10 | 6 |
| 2 | 2 | 9 | 8 |
| 5 | 0 | 9 | 6 |
| 5 | 2 | 9 | 6 |

The initial group heuristic has **not** improved the best score; larger group
weights sometimes hurt. Further checks separate smaller coherent groups
(8/12 m radius, mild candidate bonus) from pure beam-priority changes
(12/30 m radius, original nearest candidates retained). No new submission has
yet been promoted. The 10-point JSON remains unchanged.

Follow-up results:

| Strategy | Seed 0 | Seed 1 |
| --- | ---: | ---: |
| Radius 8 m, candidate bonus 1.5, no beam bonus | 10 | 8 |
| Radius 12 m, candidate bonus 1.5, no beam bonus | 10 | 8 |
| Original candidates, beam bonus 2, radius 12 m | 9 | 8 |
| Original candidates, beam bonus 2, radius 30 m | 9 | 8 |

An additional experiment retains all original timing choices but permits
0.25/0.5/0.75-second inter-target segments. This removes the search's arbitrary
one-second minimum for close neighbors without weakening physical constraints.
It is being tested with candidate bonuses 0 and 1.5, radius 8 m, on seeds 0–3.
The mixed short/long-segment numerical test preserves a constant-velocity path
to 1e-7 tolerance. Both short-leg variants scored **9/8/9/7** on seeds 0/1/2/3,
versus the original launch-selection agent's **10/8/9/7**. All runs terminated
normally (no wall-time truncation); dependency hashes remained unchanged during
each of these eight runs. The short timing options are not promoted.

In total, **24 remote development runs** plus one local disabled-feature control
were completed in this session. The best score remains **10**, not 20. No new
submission replaces the audited 10-point file. These tests show that simply
adding a density preference or extra short timing options does not improve this
controller on the tested seeds; they do not establish that clustering can never
help. A future investigation should measure achievable multi-hit trajectories
and model/tracking error rather than increasing cluster weight blindly.

The remote report archive is `evaluation/results/cluster_reports_20260923.tgz`,
with extracted reports under `evaluation/results/cluster_reports_20260923/`.
The tested source snapshots are retained as `cluster_chain_source.tgz` and
`cluster_shortleg_source.tgz` in `evaluation/results/`. Local development later
adds rejection of non-finite cluster settings; valid tested settings are unchanged.
All workers are finished; no GPU was used, and no submission was uploaded.

The evaluation wrapper now records hashes of imported agent source dependencies,
not only the entrypoint, and checks those files remained unchanged during a run.
It does not pass any of this diagnostic information to the agent.
