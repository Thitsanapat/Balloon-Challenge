# Scenario 4 qualification work (2026-10-02)

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
| Wind profile + corrected fusion with `reserve=0.8` | 0 | 4 | One pop worse than the wind-profile-only control. |
| Corrected IMU disturbance + fusion with `reserve=0.8` | 0 | 2 | One pop worse than fusion without this term. |
| Risk-ranked short-horizon route with wind profile | 0 / 1 | 3 / 1 | Below wind-profile-only 5 / 2 on both paired seeds. |
| Wind profile, launch 36 s | 0 / 1 | 1 / 0 | Later launch strongly underperformed 24 s. |
| Wind profile, launch 42 s | 0 / 1 / 2 | 5 / 2 / 4 | One more pop on seed 2 than 24 s, tied seeds 0/1; seed-0 last pop was later (64.20 s vs 50.72 s). |

The 42 s launch improved one of three tested seeds but did not improve the
other two. Because the seed-0 tied score had a worse last-pop time and the
qualification seeds are unknown, the standalone submission artifact remains
the previously validated 24 s version. More unseen-seed comparisons would be
needed before promoting the timing change.

The short-horizon forecaster corrected some non-target balloon states, but its
route and flight on seed 0 remained effectively unchanged. It is not selected
as the qualification agent. Offline telemetry from the margin-only run shows
2–6 m error in constant-velocity target projection near several intercepts,
compared with a 1.5 m pop radius. The initial 7.5 s lookahead for target #8
has about 12.5 m target-motion prediction error. This is diagnostic evidence,
not evidence that a replacement predictor will necessarily score better.

## Experiments in progress

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

The qualifier on October 10 uses two *live* evaluations and source-code
review, not a repeat of the closed Scenario-1 leaderboard upload. Therefore
this JSON is a reproducible local artifact, not a guarantee that an upload
endpoint will accept a Scenario-4 result or that it will be the qualifier's
actual score. Supply the standalone source and build script to organizers
when requested; use no seed-specific lookup or prerecorded flight.

Sources: [v0.2.2 release](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/releases/tag/v0.2.2),
[official README and qualification rules](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/blob/main/README.md),
[organizer clarification #165](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165).
