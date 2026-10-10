# Scenario 4 navigation-filter time constant experiment

The official Scenario 4 parameters specify 2 m GNSS position accuracy,
0.05 m/s GNSS velocity accuracy and 100 Hz sampling. The current guarded
controller integrates measured velocity, then blends in measured position
with a 0.5-second time constant. Offline diagnostics of four prior paired
flights measured mean position-estimation error around 0.37–0.44 m; the
balloon collision radius is 1.5 m. These truth-based diagnostics are never
provided to the agent.

Before seeing any new score, test a single physically motivated candidate:
set `position_filter_tau=1.0` seconds and leave every other agent parameter
unchanged. A more slowly blended position measurement may reject more of the
2 m GNSS noise while retaining low-noise velocity integration. It may also
increase lag or drift, so neither navigation accuracy nor score improvement
is assumed. Evaluate already-used development seeds 0, 1 and 2 with fresh
official Scenario 4 resets, record score and navigation mean/RMS error, and
stop early if a seed regresses by two or more pops against the selected
guarded controller (5, 4, 3). Only a development gain without such a severe
regression merits later testing on new paired seeds. No holdout seed is used
for parameter choice; the live submission and launcher remain unchanged.

## Result and decision

The variant scored 3 on development seed 0 versus 5 for the selected agent,
triggering the predeclared two-pop stop rule. Offline mean navigation error
also rose from 0.412 to 0.430 m, and RMS error from 0.438 to 0.471 m. This
suggests prediction lag or accumulated velocity error outweighed the extra
position-noise smoothing in this flight; that mechanism is an inference, not
a demonstrated cause. Seed 1 had only begun its Monte Carlo reset and was
stopped to save resources; no seed-1 or seed-2 result exists for this variant.
The ignored local report is `s4_navigation_tau1_dev_seed012_20261007.json`.
No source, config or qualification default was changed.
