# Scenario 4 shorter, weaker terminal correction experiment

The reviewed guarded controller remains the qualification default. Its
endpoint correction starts five seconds before the selected intercept and is
applied on roughly two thousand simulation steps in the development flights.
The comparison on seed 1458234498 was a one-pop regression; the target order
changed and a later third target was missed. This observation motivates a
less intrusive terminal-only variant, but does not prove the correction caused
the regression.

Before any new scores are inspected, evaluate exactly one variant with
`innovation_start_seconds=4.0` and `innovation_strength=0.45`; all other
parameters, source and official Scenario 4 simulator are unchanged. The
physics rationale is to reduce acceleration spent on correcting uncertain
longer-horizon drift, leaving more of the nominal route intact. First run
development seeds 0, 1 and 2, where the guarded incumbent scored 5, 4 and 3.
Only if this variant improves the three-seed total without a regression of
two or more pops on one seed should it advance to independently drawn paired
seeds. No previously evaluated holdout seed will be used to tune it. The
existing submission source, config and qualification launcher are not changed
by this experiment.

## Result and decision

Fresh official-environment runs on the already-used development seeds scored
6 on seed 0 and 2 on seed 1, versus 5 and 4 for the selected agent. Seed 1
therefore regressed by two pops, triggering the predeclared rejection rule.
The seed-2 run had just started its Monte Carlo setup and was stopped to save
resources; there is no seed-2 result for this variant. The partial report is
ignored local file `s4_terminal_window_dev_seed012_20261007.json` and contains
the two completed flights. No source, config or qualification default was
changed. A seed-0 gain alone is not sufficient evidence of improvement.
