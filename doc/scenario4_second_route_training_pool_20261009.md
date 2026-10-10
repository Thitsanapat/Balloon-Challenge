# Scenario 4: second audited route-data pool (9 Oct 2026)

The first learned-route model was fitted on only 75 executed leg attempts
from twelve independent simulator fields. Its leave-one-field-out Brier
improvement was just 0.00106 and log-loss improvement just 0.00187 over a
training-fold hit-rate prior. It gained four pops in the second eight-seed
holdout but lost three pops in two fields, failing the predeclared reliability
screen. Two later, fixed development variants also failed. The reviewed
standalone qualification agent and `run_qualification.cmd` remain unchanged.

Before collection, twelve new training seeds were checked absent from the
conservative local seed registry, and recorded in
`qualification/used_seed_manual.jsonl`:

`10298371 68394312 50192733 482617990 297410583 700126354`
`846320117 119657428 337854091 914708236 592461073 781235409`

`scripts/collect_guarded_route_shadow.py` runs the *selected* agent twice
per seed in fresh official Scenario-4 processes, one plain flight and one
observation-only route-shadow flight. It gates on exact action/observation
hash parity, official score, observed pop-event audit and frozen source hashes
before retaining executed-leg decisions. Only one CPU worker runs at a time,
and GPU visibility is disabled. The seed is archive provenance, never an
agent feature. These twelve fields will never be treated as independent
validation for a model trained on them.

The collection output is an ignored local directory
`evaluation/results/guarded_route_train12_b_20261009/` with `progress.json`,
per-seed audits, a ZIP archive, and a group-held-out fit if all pairs pass.
The archived data can be pooled with the first twelve-field archive **only
after** both collections pass their audits, yielding 24 training fields.

Predeclared next gate: with the fitting method and six existing features
unchanged, a leave-one-entire-field-out 24-field model must beat the
training-fold prior by at least 0.003 Brier and 0.006 log loss. If it does
not, do not connect new weights to an agent. If it does, freeze a separate
candidate and compare it with the reviewed agent on *new* paired seeds,
including a maximum-regression check. Better calibration is a prerequisite,
not proof of more pops or rule approval.

After collection, `scripts/fit_combined_guarded_route.py` loads both ZIP
archives, independently rechecks every field audit and exact seed set, then
performs the fixed six-feature/regularization fit. It always writes an offline
group-held-out report and writes a seed-free frozen weight artifact *only*
if the above gate passes. The intended new outputs are ignored local
`evaluation/results/guarded_route_combined24_20261009_fit.json` and
`evaluation/results/guarded_route_combined24_20261009_weights.json`.
Neither file changes a live agent or config automatically.

An idle background watcher, `scripts/watch_combined_guarded_route.py`, runs
that fit after the entire second collection passes and creates its own fit
stdout/stderr logs. If the collector writes `failed.json`, the watcher stops
without fitting. Check `progress.json` for completed seed pairs; an absent
file means no pair has completed yet, not that the run is hung. The collection
is serial and may take hours because every seed has two fresh official flights.

## Recovery note, 9 Oct

The original pool completed five audited pairs and then correctly stopped the
sixth baseline flight when a development edit changed the hash of a script
listed in its source manifest. That script was restored byte-for-byte; every
manifest hash again matches the original plan. The incomplete sixth flight
has no retained result. The original directory, its `failed.json`, and logs
remain intact. `scripts/recover_guarded_route_pool.py` re-audits/copies the
five completed pairs to a **new** directory
`guarded_route_train12_b_recovered_20261009/`, then reruns seeds six through
twelve with two fresh flights each. The new recovery watcher writes separate
combined-fit and disjoint feature-validation reports; it does not alter the
selected qualification agent. Use the recovered output rather than the
original failed directory for subsequent results.
