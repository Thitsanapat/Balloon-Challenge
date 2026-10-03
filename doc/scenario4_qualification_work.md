# Scenario 4 qualification work (2026-10-03)

## Target and rule boundary

The organizer's v0.2.2 tag provides Scenario 4. Qualification uses two live
evaluations with independently drawn seeds; the official score is their **sum**.
Thus a target total of 20 means about 10 pops per flight. It is not a verified
winning threshold. The agent may use `given_parameters` and observation history,
but not a seed, hidden gust field, stored balloon trajectory, or simulator
internals. All experiments here keep the official environment and official
evaluator unchanged. `scripts/evaluate_guidance.py` is a separate development
runner; its use of `info['rocket_states']` is strictly offline diagnostics and
is never passed to an agent.

The official gust is a random altitude-dependent profile. Sampling that gust
also advances the environment RNG *before* release ordering and balloon
Monte Carlo. Consequently, Scenario 3 seed 0 and Scenario 4 seed 0 are **not**
matched balloon fields; the Scenario 3 score of 9 versus Scenario 4 score of 0
does not isolate a causal gust effect. Within-Scenario-4 comparisons using the
same seed are the valid paired experiments.

## Confirmed local results

All rows below are fresh official Scenario 4 environment runs; no balloon
trajectory cache, replay, or field substitution is used. Reports in the
ignored `BalloonPoppingGymEnv/evaluation/results/` directory record config,
source hashes, and telemetry. `reserve` is an agent-side thrust feasibility
margin in m/s², not a simulator modification.

| Variant | Seed | Pops | Finding |
|---|---:|---:|---|
| Frozen time-allocation baseline | 0 | 0 | Predicted a long route but missed targets. |
| Frozen time-allocation baseline | 1 | 1 | Same source and config; different field. |
| Baseline with `reserve=0.8` | 0 | 1 | Tracking error 1.29 → 0.47 m and saturated steps 1,286 → 687, versus seed-0 baseline. |
| Baseline with `reserve=0.8` | 1 | 1 | No gain versus the unreserved seed-1 baseline. |
| Baseline with `reserve=0.8` | 2 | 1 | Holdout control. |
| `reserve=0.8`, launch 16 s / 28 s | 0 | 1 / 1 | Launch timing alone did not solve the misses. |
| Per-balloon short-horizon acceleration forecast | 0 | 0 | No score gain over baseline. |
| Same forecast with `reserve=0.8` | 0 | 1 | No gain over the margin-only control. |
| Corrected IMU/GNSS fusion with `reserve=0.8` | 0 | 3 | Better than margin-only, but not yet cross-seed validated. |
| Observed altitude–wind profile with `reserve=0.8` | 0 | **5** | Better than margin-only on this seed; climbed to 279 m versus 171 m. |
| Observed altitude–wind profile with `reserve=0.8` | 1 | **2** | Better than the paired baseline's 1, but still far below the target. |
| Observed altitude–wind profile with `reserve=0.8` | 2 | **3** | Better than the paired holdout baseline's 1. |
| Observed altitude–wind profile with `reserve=0.8` | 3 / 4 | **2 / 2** | Additional fresh seeds; source and config unchanged. |
| Wind profile + corrected fusion with `reserve=0.8` | 0 | 4 | One pop worse than the wind-profile-only control. |
| Corrected IMU disturbance + fusion with `reserve=0.8` | 0 | 2 | One pop worse than fusion without this term. |
| Risk-ranked short-horizon route with wind profile | 0 / 1 | 3 / 1 | Below wind-profile-only 5 / 2 on both paired seeds. |
| Wind profile, launch 36 s | 0 / 1 | 1 / 0 | Later launch strongly underperformed 24 s. |
| Wind profile, launch 48 s | 0 / 1 | 2 / 2 | Below launch 42 s (5 / 2); exploratory sweep stopped early. |
| Wind profile, launch 42 s | 0 / 1 / 2 / 3 / 4 | 5 / 2 / 4 / 2 / 3 | +2 total over launch 24 s on these five development seeds; no loss in pop count, but seed-0 last pop was later (64.20 s vs 50.72 s). |
| Wind profile, launch 42 s vs 24 s | 100–107 | **23 vs 18 total** | Predeclared holdout: 5 wins, 2 ties, 1 loss; all episodes complete. |
| Wind profile, 8 rather than 3 launch-axis candidates | 0 / 1 / 2 | 5 / 2 / 4 | Identical pop counts to the selected 42 s policy on these development seeds; not promoted. |
| Wind profile, first-hit-aware launch ranking | 0 / 1 / 2 | 5 / 3 / 2 | One-pop gain on seed 1 but two-pop loss on seed 2; 10 vs incumbent 11 total, rejected. |
| Wind profile, lower spline-tracking frequency 1.0 | 0 / 1 / 2 | 5 / 3 / 2 | Seed-1 gain offset by two-pop seed-2 loss; 10 vs 11 total. Seed-0 final hit was later (71.61 vs 64.20 s) with more saturated steps (843 vs 748). Rejected. |
| Wind profile + bounded active-leg deadline rescue | 0 / 1 | 4 / 2 | Worse than incumbent 5 / 2; recovery was accepted 9 / 11 times, so this is a meaningful negative test. |
| Wind profile + reserve-aware active refresh | 0 / 1 | 3 / 2 | Rejected three nominally feasible refreshes per seed for reserve; lost two seed-0 pops. |
| Wind profile + climb-recovery beam ranking (weight 0.5) | 0 / 1 | 2 / 1 | Physically motivated climb proxy, but fewer actual pops than 5 / 2. |

The 42 s launch improved seeds 2 and 4 by one pop each, tied the other three,
and had a worse seed-0 last-pop time. The predeclared untouched-seed comparison
below supports using 42 s for the live qualification, although the earlier
24 s JSON remains a valid historical local artifact.

Offline telemetry points to launch-direction reliability as another bottleneck:
the first planned target was missed on six of the eight 42 s holdout flights,
while the initial search often planned eight or nine targets but only two to
four were actually popped. Between 18% and 46% of powered steps saturated the
available acceleration in those flights. This motivates an isolated
first-hit-aware launch selector that ranks the *planned* first leg by duration
and late thrust headroom. It scored 5/3/2 on development seeds 0/1/2 versus
the selected agent's 5/2/4, so it is not promoted. The seed-1 extra pop was
not the first planned target; the hypothesis remains unproven.

The deadline-rescue variant keeps the official simulator unchanged and uses
only observed targets. It attempted recovery 22 times on each of seeds 0/1 and
accepted 9/11 alternatives after dense feasibility checks. It reduced seed-0
command saturation but lost one pop; this shows a smoother feasible path is
not necessarily a better collision route. Do not promote it on current evidence.

The active refresh did have a real consistency gap: its ordinary feasibility
test ignores the 0.8 m/s² planning reserve. A separate variant enforced the
same reserve at 33 and 65 samples. It rejected three updates per tested seed,
but scored 3/2 versus the incumbent's 5/2. This fix is physically conservative
yet not a score improvement and is not promoted.

The climb-recovery beam ranking penalized a feasible route's loss of vertical
speed by an optimistic time to regain the observed target's climb rate. Its
weight-0.5 test scored 2/1 on seeds 0/1, below 5/2. A plausible physical
proxy can still prune the useful collision route, so this variant is not
selected.

A ridge-regularized, smooth-edge altitude/wind forecast was also checked
against later *observations* of the same balloons in fresh, no-launch official
Scenario 4 episodes (seeds 0/1/2). The incumbent's 4-second horizontal RMSE
was 5.73/7.07/5.45 m, whereas the regularized version was
6.99/8.39/6.32 m. It was worse at every tested 2/4/6/8-second horizon on
all three seeds, despite correcting additional sparse-coverage cases. This
offline metric is not a pop score; it is sufficient to reject the candidate
before spending full-flight evaluations on it. The credential-free report is
`s4_regularized_forecast_dev_20261003.json` in the ignored results directory.

The short-horizon forecaster corrected some non-target balloon states, but its
route and flight on seed 0 remained effectively unchanged. It is not selected
as the qualification agent. Offline telemetry from the margin-only run shows
2–6 m error in constant-velocity target projection near several intercepts,
compared with a 1.5 m pop radius. The initial 7.5 s lookahead for target #8
has about 12.5 m target-motion prediction error. This is diagnostic evidence,
not evidence that a replacement predictor will necessarily score better.

## Earlier development experiments

- `Scenario4WindProfileAgent` estimates horizontal drift versus altitude from
  *currently observed released balloons*, anchored to each target's own
  measured velocity. It integrates a bounded correction over a short horizon;
  sparse altitude coverage reverts to constant velocity. Its first seed-0
  trial improved 1 → 5 pops, seed 1 improved 1 → 2, and seed 2 improved
  1 → 3. The first two-seed total improved 2 → 7, still well short of 20.
  Further unseen seeds must still confirm robustness. This is a generic observation-only estimator, not reconstruction
  of a hidden seed's gust field.
- `Scenario4FusionAgent` fuses observed GNSS, gyro, and accelerometer. A sensor
  convention error was found and corrected: the official accelerometer reports
  **specific force**, so gravity must be added after body-to-world rotation
  before inertial navigation propagation. The earlier fusion scores are not
  valid evidence for the corrected variant. With reserve 0.8, the corrected
  variant scored 3 on seed 0. Combining it with the wind profile scored 4,
  below the wind-profile-only score of 5, so that combination is not selected.
- `Scenario4DisturbanceAgent` estimates unmodelled acceleration from specific
  force minus expected thrust. Its first test double-subtracted gravity and
  crashed; that defect is fixed in source and unit tests. On seed 0 with
  reserve 0.8 it scored 2, below corrected fusion's 3, so it is not selected.
- `Scenario4RiskRouteAgent` changes only route ranking and lookahead depth,
  penalizing speculative long or high-thrust first legs. Its unit tests pass;
  fresh Scenario 4 seed-0/1 evaluations scored 3/1 versus wind-profile-only
  5/2. Do not select it on current evidence.

Before promoting any variant, test at least seeds 0 and 1 against the same
baseline/config and then holdout seeds not used for tuning. Record both
individual pops and two-seed totals. A controller that tracks a spline better
but still fails to pop balloons is not a score improvement.

## Larger ideas, ordered by evidence and compute cost

1. Improve the moving-balloon forecast and deadline rescue. The existing
   planner extrapolates each balloon at constant velocity and may keep an old
   committed route when replanning becomes infeasible. Use measured forecast
   innovation to trigger a feasible intercept-time update, not an arbitrary
   turn or a prerecorded route.
2. Include capture uncertainty in route selection: discount a speculative
   8–10-balloon chain when the first intercept is 5–8 s away and gust-driven
   position uncertainty is metres. Preserve single-target options when a
   cluster would cost too much thrust or vertical energy.
3. Improve vertical energy management. The launch vehicle has little initial
   upward acceleration margin, so large lateral turns or repeated saturation
   can squander climb. Compare a smooth, climb-preserving path and lower
   feasible axis-rate limits on paired seeds before adopting them.
4. Use official Scenario 4 residual PPO only after a competitive nonlearning
   base is established. The existing official wrapper can draw fresh seeds,
   but direct 6-DoF training is slow; a previous single-worker recovery rate
   suggests millions of steps would require days. Do not train/deploy from a
   fixed prerecorded balloon field or public replay: that is not a credible
   qualification-generalizing policy under organizer Discussion #165.

## Reproduction

### Multi-seed selection protocol (2026-10-03)

Seeds 0–4 have already been used for exploratory development and are not a
holdout set. Compare each candidate with the incumbent on exactly the same
seeds and only count complete, nontruncated official episodes. Before seeing
any candidate results, reserve seeds **100–107** as the next untouched holdout
set. Freeze the selected agent and its config before running those seeds; do
not tune it to the holdout outcomes. The read-only
`scripts/summarize_paired_evaluations.py` checks source/config consistency and
reports per-seed pop deltas. A 100,000-seed official sweep is not currently
practical: the measured wind-profile episodes take roughly 4–7 minutes each,
so one candidate alone would need around ten thousand core-hours. A fast
surrogate may screen ideas, but any score claim must come from the unchanged
official simulator with fresh independently seeded episodes.

The 42 s launch candidate was selected on seeds 0–4 before running any of
100–107. The candidate's frozen config is
`evaluation/configs/scenario4_wind_profile_launch42.yaml`; the incumbent is
`evaluation/configs/scenario4_wind_profile_reserve08.yaml`. The same
`Scenario4WindProfileAgent` source is used for both, changing only launch
time and the descriptive name. Neither config was tuned to the holdout set.

All eight holdout pairs (100–107) completed without truncation or agent-source
changes. The 24 s scores were **3/1/0/2/5/2/1/4** (sum 18); 42 s scored
**3/2/2/4/2/3/3/4** (sum 23). Thus 42 s won 5, tied 2 and lost 1 paired
seed, improving the mean from 2.25 to 2.875 pops per flight. Seed 104 is a
material regression (5→2), so the result is a modest average gain, not a
guarantee on every draw. Across the five development and eight holdout seeds,
the totals were 32 for 24 s and 39 for 42 s. The qualification score still
comes from two independent live seeds; at that stage, twice the observed
holdout mean was only 5.75 pops, far below the target of 20.
`s4_holdout_launch24_20261003.json` and `s4_holdout_launch42_20261003.json`
are the ignored credential-free reports used for this comparison.

An exploratory 48 s launch scored 2/2 on development seeds 0/1 versus 5/2
for 42 s; the rest of that sweep was stopped early for lack of improvement.
The 36 s launch was already tested with this agent on seeds 0/1 (1/0 pops),
so a duplicate run was also stopped rather than consuming further simulation
time. Increasing launch-axis candidates from 3 to 8 tied the selected policy
on all three tested development seeds (5/2/4). First-hit-aware launch ranking
regressed from 11 to 10 total pops on the same three seeds. These development
runs do not change the frozen
100–107 holdout result; any later promotion must use new untouched seeds.

Seeds 200–207 were declared as a second fresh validation set after the
first-hit-launch candidate was implemented but before seeing any scores on
that set. The candidate subsequently regressed on development seed 2 and was
rejected, so these seeds were run with the selected 42 s policy only to
estimate its robustness, not to claim a paired improvement. They all completed
without truncation or agent-source changes, scoring **1/6/2/2/4/4/1/2** (sum
22, mean 2.75). The two distinct eight-seed holdouts together yielded 45 pops
over 16 flights (mean **2.8125**), implying approximately 5.625 pops for a
two-flight total if future draws resembled them; this is an empirical estimate,
not a guarantee or a competition result. An independent
development check tested `timing_candidates=0` because the original timing
optimizer was adopted zero times on all eight first-holdout runs; this is a
compute-efficiency experiment, not yet an accepted score improvement.
Development runs with `timing_candidates=0` reproduced the selected agent's
seed-0/1 scores and pop times (5/2) exactly, without any timing solves.
Fresh timing-enabled controls also reproduced 5/2 and identical pop events.
Post-reset flight wall times were 106.3/106.7 s with timing disabled versus
115.3/121.8 s with timing enabled, saving 9.0/15.1 s locally while skipping
17/20 timing solves. This is a measured two-seed compute reduction, not a
guarantee for every machine or seed. Keep the reviewed qualification config
unchanged until any resource-optimized variant is validated on new seeds.

From the repository root, with the official v0.2.2 dependencies installed:

```powershell
$env:MPLCONFIGDIR = Join-Path $env:TEMP 'balloon_mpl_s4'
.\.venv\Scripts\python.exe -m unittest tests.test_scenario4_fusion tests.test_scenario4_disturbance tests.test_scenario4_forecast tests.test_scenario4_wind_profile -q
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/scenario4_time_baseline.yaml --seeds 0 --set reserve=0.8 --report BalloonPoppingGymEnv/evaluation/results/scenario4_baseline_check.json --telemetry
```

## Standalone source and local JSON artifact

`scripts/build_scenario4_submission_agent.py` combines the frozen standalone
time-allocation source with the selected observation-only wind-profile class
into `agents/submission_scenario4_wind_profile_v1.py`. The generated agent has
no imports from our development agent modules. The official evaluator and
packer ran this standalone file on Scenario 4 seed 0 and reproduced **5 pops**.
The credential-free receipt is
`evaluation/results/scenario4_standalone_official_seed0_20261003.json`; the
official-format JSON is in the same ignored results directory. It is roughly
64 MB, embeds the team secret, and must not be committed or publicly shared.
The official packer's optional online integrity check could not connect from
this machine; the file had already been saved. A local v0.2.2 comparison and
submission verifier provide the independent checks. The credential-free
`scenario4_submission_audit_20261003.json` reports a JSON object (format 2),
score 5, embedded source matching the standalone file, no static source flags,
unchanged official components versus v0.2.2, and **19/19 verifier findings
passing**. The submission SHA-256 is
`28e3a03d4749272c9c8b5902226a44917dd20f7c8dce37f64095c75322ad2ac8`.

The promoted 42 s standalone config also completed the unmodified official
evaluator on seed 0 with **5 pops**. Its credential-free receipt is
`s4_standalone_launch42_official_seed0_20261003.json`; the official-format JSON
is `20261003T021910.264Z_TASTI_Cool_Ba_Malaew_fe3e7d50c87b4528a1a793233faf988a_submission.json`
in the ignored results directory, with SHA-256
`77723417cef612dcb3f49f6bdb69e4763bea7ef79eeab9f59220c7f2fe513b5f`.
The credential-free audit `s4_standalone_launch42_audit_20261003.json` reports
matching embedded/local source, zero static source flags, unchanged official
components versus v0.2.2, and the official verifier passing all 19 findings.
This JSON likewise contains team credentials and must not be committed or
shared publicly. The 42 s candidate is selected for its multi-seed improvement,
not a seed-0 advantage over the earlier 24 s JSON.

The qualifier on October 10 uses two *live* evaluations and source-code
review, not a repeat of the closed Scenario-1 leaderboard upload. Therefore
this JSON is a reproducible local artifact, not a guarantee that an upload
endpoint will accept a Scenario-4 result or that it will be the qualifier's
actual score. Supply the standalone source and build script to organizers
when requested; use no seed-specific lookup or prerecorded flight.

Sources: [v0.2.2 release](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/releases/tag/v0.2.2),
[official README and qualification rules](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/blob/main/README.md),
[organizer clarification #165](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165).
