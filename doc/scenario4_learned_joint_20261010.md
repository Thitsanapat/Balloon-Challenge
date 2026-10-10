# Scenario 4 learned-route + joint-refresh experiment (10 Oct 2026)

This is an experimental policy, not the live qualification agent. The live
runner and reviewed standalone source remain unchanged. The candidate combines
the frozen six-feature first-leg capture model with the existing joint repair
of an infeasible committed leg. It reads observations and given parameters;
the seed is supplied only to the simulator, not the agent.

## Why this test

The learned-route second holdout scored 39 versus 35 for the older guarded
controller, but regressed by three pops on two of eight seeds. On one such
seed its mean trajectory-tracking error was 2.59 m; on both failures the
actuator allocator saturated more often than for the guarded controller.
These observations motivate a check of whether the already-tested joint
refresh can recover selected routes. They do not prove that joint refresh
will solve the learned-route failures.

## Frozen development screen

Compare `scenario4_learned_joint_launch42.yaml` with the currently selected
`submission_scenario4_pure_joint_launch42.yaml` on already-used seeds
`140200341, 568300771, 178420909, 820431975`. The candidate uses the
original 12-field frozen capture weights so that the new factor is the joint
refresh, not an updated model fit. Do not tune its risk penalty on these
fields. Score complete fresh official simulator flights, CPU only.

Frozen SHA-256 for the valid candidate: candidate agent
`0bbf2e43389c857c66e95f80b2602aac1538d41634959ecdce49bffa22d61090`,
candidate config
`5fcaf8cfd61a61f718368ef33ff2253c5154c79c08f9c0672c4b356f2962603`,
and 12-field capture weights
`cb2a039f0763032c0699714c4fe52ee0b99062e2367a0c97e9467af767afd506`.

Before seeing these paired scores, require: candidate total at least two
pops above selected pure-joint, no field more than one pop below selected,
at least two wins, and actual exercise of joint repair. A pass permits an
independent paired holdout; it is not live promotion. A failure leaves the
live submission unchanged. Development fields are already familiar and are
not independent evidence.

Independent promotion would additionally require fresh-seed paired score and
worst-case gates, source packaging/parity, and a rehearsal of the Windows
live runner. The two organizer-drawn seeds cannot be used for post hoc model
selection.

## Predeclared follow-up if the 12-field model fails

One separate, fixed model candidate uses the already-audited 24-field
six-feature weights without any change to risk penalty or controller. Its
leave-one-field-out calibration Brier/log loss were 0.23399/0.65958 versus
the training-fold prior's 0.24433/0.68191. Calibration is not a pop-score
claim. The 24-field weights were fitted before the four development outcomes
of this experiment were observed. If tested, use the same four-field screen,
do not tune any weights on these fields, then require independent paired
validation before considering promotion. No automatic live switch. Test the
known two regression fields first, `178420909` then `820431975`; stop the
development screen early if either is more than one pop below the freshly
verified standalone selected agent. An early stop is a rejection, never a
promotion or a claim about the unflown fields.

An initial multiple-inheritance prototype used the older standalone guarded
class and the modular joint class. A startup smoke test was insufficient:
its first completed flight scored 8, but its diagnostics showed no joint
repair hooks were initialized. That result is **invalid for this hypothesis**.
The run was stopped during the next seed, and its partial `candidate.json`
is retained as an audit record, not a score report. The valid candidate now
subclasses the reviewed standalone pure-joint controller directly and
wraps only its route-priority method. A constructor smoke test confirms both
joint-repair counters and the risk-ranking counter exist. Valid development
flights will use a separate `candidate_v2.json` report.

## Completed 12-field model screen

Four complete official flights of the corrected candidate scored, in the
declared order, **6/8/3/4**. Each flight verified unchanged agent-source
dependencies. Joint repair was accepted **5/5/0/2** times respectively.
The selected standalone pure-joint agent was freshly flown on both known
regression fields: **6** versus candidate **3** on `178420909`, and **7**
versus candidate **4** on `820431975`. Thus the corrected candidate fails
the predeclared maximum single-field regression by three pops on each of
these two fields. The selected live runner stays unchanged. No baseline
flights on the two favorable development fields are needed for rejection;
do not claim a complete four-pair comparison.

On `178420909` the candidate attempted 44 joint repairs and accepted none;
its mean reference-tracking error was 2.587 m and allocator saturation
fraction 0.575. This is evidence of a route-tracking problem, not proof that
repair absence alone caused the three-pop shortfall. Raw credential-free
reports are in `s4_learned_joint_dev4_20261010/`.

## Completed 24-field follow-up and decision

The fixed 24-field model was tested on the two known regression fields in
the declared order. On `178420909` it scored **6**, tying the selected
standalone **6**; on `820431975` it scored **4**, versus selected **7**.
The second result breaches the no-more-than-one-pop regression gate, so the
screen was stopped early. The other two development fields were not flown
with this model, and no new holdout fields were consumed. Do not infer a
four-field total for this candidate.

The 24-field model improves offline first-leg calibration but not robust
route choice. Its success on one known weak field and failure on the other
indicate that the missing quantity is likely route-level *trackability*
under observed wind, actuator lag/saturation, and target sequence, not only
the first-leg hit probability. This is a diagnostic inference, not a proven
causal decomposition. The selected live runner remains the reviewed pure
joint v2; neither learned candidate is submission-ready or promoted.

## One structural actuator-margin screen (separate candidate)

The first-leg model scores success among routes previously flown; it has no
direct measurement of whether each alternative spline will leave enough TVC
and thrust authority for disturbances. `Scenario4MarginLearnedJointAgent`
therefore leaves the selected pure-joint priority intact whenever the
*proposed first-leg spline* has less than 10% normalized headroom in thrust
magnitude, thrust-axis slew, or throttle slew over 17 samples. It uses the
same public parameters as the existing feasibility check and the same frozen
24-field weights. It does not change the official simulator or use a seed at
runtime. This 10% value is an engineering pilot margin, not a calibrated
capture-probability guarantee. It is distinct from the previously failed
constant-acceleration headroom proxy because it checks the actual planned
spline and actuator rates.

Before testing: agent SHA-256
`569fbfd9796b522b2b1c0052eee96591c841dd322ab9d97f475092d711efbf36`,
config SHA-256
`370494c2072e50f3d4da244c3f115403405a566ce8ea8bf2e82a027cfb8cf45a`.
One focused margin-gate unit test and the inherited-policy unit test pass.
Test known regression seed `820431975` first: candidate must score at least
6 versus selected 7 and show that the margin branch was exercised. If it
fails, stop immediately, without adjusting 10% or trying more familiar
seeds. If it passes, test `178420909` (candidate at least 5 versus selected
6), then the other two development fields, then separate fresh paired
holdouts and source packaging before any promotion. Live launcher remains
unchanged throughout.

### Completed margin screen

All four already-used development flights completed with unchanged source.
In the order `140200341, 568300771, 178420909, 820431975`, selected and
margin candidate both scored **4/3/6/7**. Every paired flight also had
identical pop events and route events. The margin branch was exercised
(`4525/5228/4165/6937` rejected learned rankings); the learned ranking
still ran on `1291/1174/1884/601` candidate nodes. Thus the policy is not
an inert implementation, but it made no observed route or score difference
on these fields. It fails the predeclared requirement of two wins and a
two-pop aggregate gain. No fresh holdout or packaging is justified, and the
selected qualification agent remains unchanged. Do not tune the 10% margin
on these known fields.

The result isolates a useful tradeoff: unguarded first-leg risk ranking can
produce eight-pop flights, but also three-pop regressions; a stringent
actuator-margin gate protects these four known fields by restoring the
selected routes, losing the eight-pop upside. The next model should learn
route-level *execution* outcomes from multiple independent fields and
prospective candidate policies. It should include planned spline thrust,
axis-rate and throttle margins, recent observed tracking error, and the
next target transition. Evaluate by whole-field holdout and fresh paired
flights; never label unflown alternatives as misses. Logged deterministic
actions have limited counterfactual coverage, so offline first-leg
calibration alone cannot establish the best alternate route.
