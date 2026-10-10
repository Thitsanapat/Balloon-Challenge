# Scenario 4: feasible terminal correction, experimental only

The reviewed standalone qualification agent and `run_qualification.cmd` are
unchanged. This development run is **not** a submission or a claim of 20 pops.

## Why this trial

The currently selected guarded controller discards an entire observed-balloon
endpoint correction if adding it to the nominal force would exceed the public
thrust/tilt limits. In the four recent selected-agent held-out flights,
`guarded_skipped_saturated` was 298, 198, 40, and 26 steps. The development
seed-2 controller also had true closest approaches of 1.95 and 1.74 m to two
unpopped targets, versus the public 1.5 m radius. Merely increasing tracking
frequency to 1.6 raised both average reference error and saturation despite
one extra pop on that seed, so this trial does **not** just turn up feedback.

A new read-only diagnostic in `scripts/guidance_telemetry.py` can compare the
current spline with the *true rocket position only after* `env.step()`. That
simulator `info` is never passed to the agent, and the diagnostic does not
change the official environment or scoring. On the already-used seed
761692544, the selected agent reproduced 2 pops. Its unpopped targets had
sampled true closest distances of about 2.28, 1.59, and 2.96 m (another
target was far away). At the 1.59 m approach, both reference tracking and
balloon forecast errors contributed, so the evidence does not establish that
controller tracking alone caused the miss. These are sampled distances, not
the simulator's swept collision test.

## Change and predeclared screen

`Scenario4BudgetedTerminalAgent` keeps all selected-agent routes, spline
planning, tracking gains, and actuator compensation. Only when the *full*
forecast correction saturates, it binary-searches the largest unsaturated
fraction. If no meaningful feasible fraction exists, it returns exactly to
the nominal force. The same tracking-error gate still applies. This is a
bounded line search in the public thrust cone, **not** a new 6-DOF model.

The exact development screen was started on the already-used seeds 0, 1, 2,
for which the selected agent's scores are 5, 4, 3. The candidate should have
at least 13 total pops, no two-pop regression on any seed, and exercise the
new partial-correction branch before it merits *new* paired validation seeds.
These are screening thresholds, not a statistical guarantee. The source and
config remain experimental; no live launcher or submission file is switched.

Unit tests cover the thrust bound, input non-mutation, unsaturated behavior,
and command parity when there is no innovation. The three-flight official
simulator report, when finished, is ignored local
`evaluation/results/s4_budgeted_terminal_dev012_20261008.json`.

## Completed development screen

The three official flights finished with **5, 3, 3** pops versus the frozen
selected agent's **5, 4, 3**: 11 versus 12 in total. The partial-correction
branch ran 0, 23, and 39 times respectively. Seed 1's mean spline-tracking
error and allocator saturation also increased. This misses the predeclared
13-pop gate. **Reject this candidate**; do not replace the qualification
agent or tune this controller against the same three seeds.
