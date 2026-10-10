# Scenario 4 learned-route second paired holdout (8 Oct 2026)

The frozen first four new fields favored the learned route selector 17–14
total pops, but four paired flights are too little evidence to change the
qualification agent. The selected standalone agent and `run_qualification.cmd`
are unchanged.

Before further evaluation, eight new seeds were verified absent from the
local conservative seed registry and reserved in
`qualification/used_seed_manual.jsonl`:

`140200341 1098270439 1904235602 568300771 178420909 820431975`
`1126043508 662504311`

`scripts/validate_guarded_route_ml_stage2.py` sequentially flies the exact
selected and learned-route configs on each seed, using the official Scenario 4
simulator. It freezes source/config/model SHA-256 digests in its resumable
result and checks them before and after each flight. GPU visibility is disabled.
No seed or simulator truth enters either agent; the experiment may read the
official score afterward. The learner was trained on twelve different seeds,
not these eight.

Predeclared promotion screen for this stage: learned-route total at least four
pops above selected across eight pairs, no seed more than two pops worse,
and learned-route score no lower on at least four of eight seeds. This is a
pragmatic reliability screen, not a statistical proof. Even if it passes,
we must make the model artifact submission-ready, verify a standalone source
package, rehearse the Windows live runner, and keep an untested final seed
for validation before changing the qualification policy. A failure leaves
the selected agent in place.

Result: pending in ignored local artifact
`evaluation/results/guarded_route_ml_stage2_20261008.json`. The script can
resume after interruption; no result should be inferred from an incomplete
report.

Use `python -m scripts.assess_guarded_route_ml_stage2` with the result path
to verify paired-flight integrity and apply the exact predeclared screen.
The assessor intentionally returns `pending` without a pass/fail verdict
until all 16 official flights have finished. It also reports pop IDs,
tracking error, and allocation saturation as diagnostics, not proof that
either quantity caused a score change.
