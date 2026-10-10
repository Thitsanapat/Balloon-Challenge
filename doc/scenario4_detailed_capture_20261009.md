# Scenario 4: detailed capture calibration (9 Oct 2026)

The reviewed live launcher and standalone guarded agent are unchanged. This
experiment is offline only. It uses only observation-derived decisions from
the parity-audited route-shadow archive and later observed outcomes as labels.
Unflown alternatives are never labelled. The official simulator is unchanged.

The incumbent six-feature first-leg logistic model uses time, range, closing
speed and acceleration-demand proxies. This experiment tested seven additional
features available at the first committed decision: remaining burn after the
deadline, relative altitude, lateral relative speed, observed 10/25 m target
density, planned departure-velocity mismatch, and change from the current
measured acceleration. These last quantities are *proxies*, not a 6-DOF
actuator or impact-probability model.

On the original twelve **development** fields (75 executed attempts), adding
all seven features worsened leave-one-field-out Brier score by 0.00653 and log
loss by 0.01380 relative to the incumbent model. This rejects a broad,
high-dimensional fit on the current small sample. An exploratory one-feature
ablation found relative altitude was the only addition with a material gain:
Brier improved by 0.00358 and log loss by 0.00725; eight of twelve held-out
development fields had lower Brier error. The next best, 25 m
density, changed the metrics by less than 0.00006. These are *development*
results, not an independent claim of better score.

The altitude-only seven-feature choice is now frozen for one disjoint
validation: train on the original 12 fields and evaluate calibration on the
second 12-field audited collection, without fitting or selecting anything
using that second set. Predeclared gate: the altitude model must improve
Brier by at least 0.003 and log loss by at least 0.006 against the six-feature
model on exactly the same second-pool executed attempts. Passing would only
justify a separate flight-agent experiment; it would **not** promote a live
policy or establish a pop-score gain. A hidden, idle watcher writes the
credential-free result to
`evaluation/results/guarded_route_altitude_holdout12_recovered_20261009.json`
after the audited second collection completes. The training pool is currently
still incomplete.

During development, I initially added an optional argument to the old
`diagnose_route_capture_ml.py` loader while collection was active. That file
is in the collector's frozen source manifest, so its fail-closed hash gate
stopped the sixth baseline flight. No result from that partial flight was
retained. I restored the old file byte-for-byte and verified **every**
manifest hash against the original plan. The five completed seed pairs were
not altered; `scripts/recover_guarded_route_pool.py` re-audits and copies
them into a new output, then runs the seven remaining seeds afresh under the
same manifest. It keeps the original failed directory and marker for review.
The new feature code now has its own independent audited reader, outside the
collector manifest, so there is no further source edit to the running job.
The recovery output is `guarded_route_train12_b_recovered_20261009/` and an
idle watcher evaluates both predeclared offline models only after all twelve
pairs pass. Neither watcher changes the live agent or config.

Scripts: `scripts/diagnose_detailed_route_capture.py`,
`scripts/validate_detailed_capture_holdout.py`,
`scripts/package_altitude_capture_model.py`,
`scripts/recover_guarded_route_pool.py`, and
`scripts/watch_recovered_route_models.py`. The old audited loader is again
byte-identical to the original frozen version. Five focused
feature/calibration tests and two combined-fit tests passed. Neither the
seed registry nor the qualification runner was changed by this feature
experiment.

An isolated candidate `scenario4_altitude_route_agent.py` and its
`scenario4_altitude_route_launch42.yaml` config are ready for **later** score
testing if the disjoint calibration gate passes. Its frozen first-pool model
has SHA-256 `081c34e31130b255227492edfc0b012b56ad494aa57bfa87b1f4f9fceee4b5a3`.
The artifact contains seven feature normalizers, an intercept and seven
feature coefficients, plus archive provenance,
not any seed-indexed instruction or hidden field. The candidate preserves
the reviewed physical planner and controller. No official flight of this
candidate has yet been claimed, and the live launcher still selects the old
standalone agent.

## Completed disjoint result

The recovered second pool finished all twelve parity-audited pairs, with 80
executed attempts; the combined pool has 24 fields and 155 attempts. Training
the original six-feature model on the first twelve fields and evaluating it
on the **different** second twelve fields gave Brier 0.22689 and log loss
0.64590. The altitude-augmented seven-feature model was worse: Brier 0.23188
and log loss 0.65611. It failed both predeclared improvement thresholds.
**Reject the altitude candidate for live use**; no score flight is warranted
from this evidence.

Separately, refitting the original six-feature model on all 24 fields passed
its leave-one-field-out calibration gate: Brier 0.23399 versus 0.24433 for
the training-fold prior, and log loss 0.65958 versus 0.68191. Its seed-free
weight artifact was produced, but this measures prediction, **not** a pop-score
increase or a safe route-selection policy. The reviewed live agent remains
unchanged pending paired official flight tests.
