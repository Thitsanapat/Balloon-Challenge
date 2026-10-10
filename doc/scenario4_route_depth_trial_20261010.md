# Scenario 4 route-depth experiment (10 October 2026)

Status: development only. The reviewed qualification source, launcher, and
official simulator are unchanged. The user-facing score is the official
`BalloonPoppingEnv` pop count from complete local flights, not a predicted
route length.

## Hypothesis and initial screen

The incumbent's initial route can contain eight or nine nominal targets, but
several first pursuits missed on observed fields. A shorter beam horizon
could change the entry target and avoid planning credit for distant,
low-certainty legs. The isolated candidate changes only
`agent_kwargs.search_depth` from 14 to 6, preserving the same standalone
controller, public parameters, launch time, and observations.

Use previously evaluated development seeds 2 then 0. Their incumbent scores
are 4 and 6. Continue to a fresh paired test only if the candidate total is
at least 11 and neither seed regresses by more than one. A gain on these known
fields is a screen, not evidence of generalization.

Seed 2 candidate finished with 5 pops, versus incumbent 4. The first route
changed from `[17,36,80,24,26,65,30,98]` to `[77,44,94,9,37]`; thus this
is a route-choice change, not merely a later controller correction. Seed 0
finished with 5 versus incumbent 6. The two-field total tied 10 to 10,
failing the predeclared 11-pop development screen. No fresh seed was spent on
this full-flight depth-6 variant, and it is not the live agent.

## Fresh paired gate, fixed before those scores

If the development screen passes, test two stages in order, each with one
short and one long integer seed:

1. Stage 1: `6818`, `933878928`.
2. Stage 2, only if stage 1 candidate total is not below the incumbent:
   `1990`, `282227154`.

These four numbers were drawn with NumPy `default_rng(202610101455)`, two
draws from `[1000,10000)` and two from `[100000000,1000000000)`, then checked
absent from the 445-seed local registry before any evaluation. Number of
digits is not assumed to measure field difficulty. Record every seed that is
actually run in the manual registry. Run both policies on each field with
fresh official reset, no cached field or submission JSON. A candidate can be
considered for live promotion only if the four-field total exceeds incumbent
by at least two, no individual pair loses more than one, all flights finish,
and a standalone build/source-review and launcher parity test pass. Ties do
not establish an improvement. This is a practical gate, not statistical
proof for all possible seeds.

The full-flight depth-6 variant did **not** pass its development screen, so
the four fresh seeds above remain unrun. They may be used only under a new
explicit plan, not retroactively counted as a successful validation.

## Entry-only horizon follow-up, fixed before scores

The depth-6 seed-2 initial route changed and gained one pop. On seed 0 the
initial target was still #44 under either depth, but the later route and
final pops differed, losing one. This suggests a narrower test: use depth 6
only for the launch-axis comparison and initial plan, then restore depth 14
for all airborne replans. `Scenario4EntryHorizonAgent` implements that
one-call override while retaining the frozen standalone controller.

Initial source SHA-256: `598626f6eefa541c09d447746d6e2a0d8b52491a7974c345ebc8b5d9d789b476`.
Config SHA-256: `eae3a19ffd59a0189fe98562b508c435e96f6360482031f49ae04a6b73171ec2`.
Two focused tests pass. First run already-used seed 2 (require at least 5 vs
incumbent 4); if it passes, seed 0 (require at least 6 vs incumbent 6).
Only if both pass, test the unrun fresh stage-1 pair `6818` and `933878928`
with both policies. Require stage-1 total nonnegative before stage 2
`1990` and `282227154`; require the same full four-field promotion gate
above. This choice of fresh seeds was made before any entry-only score.

The initial source scored 4 on seed 2, exactly matching incumbent, but its
`entry_route` diagnostic was empty: the launch-axis comparison used depth 6,
while the first *executable* route was only created on the next sensor step
after depth 14 had been restored. That did not test the intended entry-only
policy. Corrected source SHA-256:
`344fe463533bf7801a1a96c0c99529b6449e32e459cf014c64e5a387df8f234f`.
The corrected version keeps depth 6 until the first nonempty route is built,
then restores depth 14. Two focused tests pass for this revised behavior.
The seed-2/seed-0 development gate above applies to this corrected version;
the earlier 4-pop score is retained only as a diagnostic.

Corrected entry-only seed 2 completed with **5 pops** versus incumbent 4.
Its `entry_route` and first route both were `[77,44,94,9,37]`, proving the
short depth applied through the first executable plan. Agent source and all
loaded agent dependencies were unchanged during flight. Corrected seed 0
completed with **6 pops** versus incumbent 6, using entry route
`[44,76,32,86,87,94]`; the full-flight depth-6 candidate on the same seed
had only 5. Both complete flights terminated normally without truncation.
Thus the development gate passed 11 versus 10 across seed 2/0, but this is
still known-field tuning, not a validated live improvement. Fresh stage-1
seeds `6818` and `933878928` were then reserved in the manual registry before
their paired flights; the local registry now has 447 distinct entries.

Fresh stage 1 in progress: seed `6818` completed as incumbent **5** and
entry-only candidate **4** (one-pop regression). The candidate's first two
pops were earlier, but the final score was lower; its initial nominal route
was `[99,30,72,17,25,90]` versus incumbent
`[99,24,65,92,2,69,13]`. On seed `933878928`, incumbent completed with
**5**; candidate flight is pending. Stage 2 remains forbidden unless the
stage-1 aggregate is nonnegative.

Stage 1 completed: seed `933878928` scored incumbent **5**, entry-only
candidate **5**. Across the two predeclared fresh fields, incumbent totaled
**10** and candidate **9**. The stage-1 gate failed; stage-2 seeds `1990`
and `282227154` were **not run**, and the entry-only agent is rejected for
live use. The current live launcher remains on the original pure-joint v2.

## Adaptive overlong-entry pilot, fixed before its scores

The failed stage-1 pair is now development evidence. In those four known
fields (0, 2, 6818, 933878928), the maximum nominal launch-comparison count
was 9, 9, 8, 8 respectively. The entry-only variant won/tied/took a loss/
tied. A simple capacity hypothesis is that a 9-target promise within the
public 30-second burn is too optimistic, while an 8-target plan may be worth
retaining. This exact threshold was chosen **after** seeing those fields, so
their apparent benefit is not independent evidence.

`Scenario4AdaptiveEntryAgent` computes the incumbent launch comparison on a
copy of its own state from the current observation. It restores the state
before issuing any action and uses entry depth 6 only if the nominal maximum
count is at least 9; otherwise it uses the incumbent depth 14. No seed,
future target trajectory, hidden environment state, or offline outcome is
read at runtime. It is an experimental heuristic, not a trained probability
model. Source SHA-256:
`11307fbbca6d9b389914c3b3abcd757ed699148f2e342837ee1f1f8eaab57509`.
Config SHA-256:
`daf59953a76101a4623bee0300e16b22c7f19770323219f1a71747a45d510bac`.
One focused branch/restoration test passes.

Before any adaptive score: development parity must show seed 6818 selects
incumbent and scores 5, and seed 2 selects short entry and scores 5. Stop if
either fails. Only then reserve the still-unrun `1990` and `282227154` in the
used-seed registry and run fresh paired official flights. Consider promotion
only if the adaptive two-field total beats incumbent by at least two, no
individual loss exceeds one, at least one field exercises the short-entry
branch, all sources stay unchanged during complete runs, and standalone
packaging/official launcher parity pass. Two fields remain a noisy estimate;
if any gate fails, keep the original live agent.

Development branch parity passed. On already-used seed 6818, the adaptive
policy saw nominal count 8, kept the incumbent, and matched all five pop
events and route events. On already-used seed 2, it saw nominal count 9,
selected short entry, and matched the entry-only five-pop trajectory and
route events. Both agent dependency fingerprints remained unchanged during
flight. Seeds `1990` and `282227154` were then reserved before paired
testing; the local registry contains 449 distinct local/planned seeds.

Fresh adaptive gate completed with full official local evaluations. On seed
`1990`, incumbent and adaptive both scored **4**; adaptive observed a nominal
count of 8 and correctly retained the incumbent branch. On seed `282227154`,
incumbent scored **7** and adaptive scored **6**; adaptive observed a nominal
count of 9 and selected the short-entry branch. All four flights terminated
normally, and the agent files were unchanged during each run. Thus the
predeclared paired total was incumbent **11**, adaptive **10**, failing the
required improvement of at least two. **Do not promote this agent or change
the qualification launcher.** Reports are in
`evaluation/results/route_continuation_current_20261010/` under the four
`1990_*` and `282227154_*` names. The adaptive and incumbent policies popped
the same first five targets on seed `282227154` before diverging; the one-pop
loss is a measured outcome, not proof that a particular late maneuver caused
it. This also weakens the proposed nominal-count threshold as a robust
selector. There is no verified score gain from this trial.

An offline telemetry rerun of the unchanged incumbent on the already-used
seed `1990` reproduced its **4** pops. It found two unpopped active targets:
`#31` had a sampled closest approach of 18.54 m near its deadline, with
32.19 m of target-forecast error, 11.67 m of path-alignment error, and
2.00 m of spline-tracking error; `#79` had a 6.59 m closest approach with
11.30 m forecast error, 2.86 m alignment error, and 2.29 m tracking error.
These are post-action diagnostics, not information available to the flying
agent or official collision calculations. This field's dominant miss signal
is forecast/routing, not simply a final TVC tracking error. The existing
observed-acceleration and chance-aware route experiments in this workspace
already failed promotion, so this observation alone does not justify
changing the live forecast model or fitting to seed `1990`.

## Route-learning audit in parallel

The observation-only audit of the older guarded policy's two 12-field
archives found 137 planned two-leg attempts but only 52 factual observed
second legs. The rest are censored, not failed counterfactual routes. A
four-feature regularized logistic pilot had worse first-hit Brier score than
the training-set prior in both cross-archive transfer directions; conditional
second-hit results were mixed. Those weights are not put into any agent.
Source: `scripts/audit_route_continuation.py` and
`scripts/fit_route_continuation.py`; reports are ignored under
`evaluation/results/route_continuation_audit_20261010/`.
