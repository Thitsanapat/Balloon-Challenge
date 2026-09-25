# Group versus unrestricted route comparison (2026-09-24)

## User intent

Enter coherent groups when several hits are worth the travel and turns, but
retain isolated-target routes when they are better. No claim is made about
another team's implementation based on its visible flight alone.

The prior density-bonus agent did not improve the best score. This experiment
instead compares complete feasible route proposals against the best existing
observation-only timing agent. The baseline is not purely greedy: it can already
choose groups or isolated targets without an entry restriction.

## Implementation

`GroupPortfolioAgent` builds small coherent neighborhoods of currently released
balloons using predicted positions four seconds ahead and relative velocities
within 4 m/s. Radii of 8/16 m are tested, with two distinct neighborhoods per
search; overlap suppression limits redundant searches.

For each neighborhood, an independent copy of the agent's own controller state
searches a complete route. The first up to three targets are sought in that
neighborhood; after exhausting it, or reaching the entry-depth limit, unrestricted
candidate search resumes. The original joint-derivative beam, continuous timing
and insertion optimizer, and dense physical checks remain in force.

There is no density bonus in the final comparison. More distinct planned hits
win first; earlier finish breaks ties. The conservative mode accepts only more
hits, while the other mode also accepts equal counts finishing at least 0.1 s
earlier. A worse proposal restores the entire baseline controller state. Focus
IDs are cleared after comparison so a selected route does not permanently lock
the agent into the original group. Group size is never counted as actual reward.

Comparisons run only when the parent builds a new route, not during committed
tracking or final approach. Added search work is counted even if rejected.
As with the original launch selector, diagnostics from discarded launch-axis
trials are not all retained, so these counters are not a complete CPU-work total.

## Rules and controls

Only observations and given parameters are used. No seed, saved trajectory,
hidden future state or environment object is passed into the agent. Copies are
of the agent's own state only. Official simulator/scoring/evaluator/packer are
unchanged. Organizer interpretation remains authoritative:
https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165

Reference timing-agent scores: seed 0 = 10, seed 3 = 9. A local disabled control
uses `group_trials=0`. Initial comparison: 8/16 m radii x extra-only true/false
x seeds 0/3 = eight remote episodes, CPU only with at most eight workers.
The frozen 10-point standalone agents and submission payloads are preserved.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLCONFIGDIR = Join-Path $PWD '.mplconfig'
.\.venv\Scripts\python.exe -m unittest tests.test_group_portfolio
.\.venv\Scripts\python.exe scripts/evaluate_guidance.py BalloonPoppingGymEnv/evaluation/configs/group_portfolio.yaml --set group_radius=8 --set group_extra_only=true --seeds 0 3 --report BalloonPoppingGymEnv/evaluation/results/group_portfolio_reproduction.json
```

This is a bounded online search, not a global optimum or a guarantee of higher
score. Planned count and actual hits must be compared in complete simulations.

## First comparison results

All four radius/acceptance combinations reproduced **10 / 9** on seeds 0 / 3,
matching the timing reference. No group proposal was adopted in the retained
in-flight comparison diagnostics. Radius 8 recorded 1/4 comparisons; radius 16
recorded 5/9. Discarded launch-axis trial work is not included in those counts.
Complete episode wall times were 146–162 seconds on the instructor server.
All eight runs terminated normally with unchanged source hashes.

Reports: `evaluation/results/group_portfolio_reports_20260924/` under
BalloonPoppingGymEnv. The local disabled control reproduced **10**, including
the reference pop events, and completed without truncation/source changes:
`evaluation/results/group_portfolio_disabled_s0.json`.

The follow-up expands to four group proposals and permits unrestricted exit
after just one or two entry targets, rather than three. This tests whether
forcing several initial hits within a neighborhood is over-restrictive.
Radii 8/16 m, entry depth 1/2, seeds 0/3, extra-only acceptance: eight runs.

64 selected unittest tests passed, including six new group-comparison tests:
observation isolation, group rejection/acceptance, singleton-route preference,
disabled behavior and removal of entry restrictions after group exhaustion.

## Follow-up and conclusion

All four radius/entry-depth combinations also scored **10 / 9** on seeds 0 / 3.
None adopted a group route in the retained in-flight diagnostics. Expanding
group alternatives and allowing earlier exit did not improve these two seeds.
Reports: `evaluation/results/group_exit_reports_20260924/`.

In total: 16 remote episodes plus one local disabled control. All episodes
terminated normally without truncation; source dependencies remained unchanged.
All remote workers finished and the SSH session was closed. No GPU was used.

This result does not establish that group routes can never help: only bounded
neighborhood proposals and two development seeds were tested. It does show that
the added computation did not outperform the unrestricted timing planner here.
The implementation remains experimental; no new submission is promoted. The
previous verified 10-point payload and standalone agent are untouched, and
nothing was uploaded. More planned hits or dense neighborhoods must not be
reported as actual score gains without simulator evidence.
