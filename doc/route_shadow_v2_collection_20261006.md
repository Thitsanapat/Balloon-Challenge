# Route shadow v2: complete observed pop events

Scenario 4 data collection now records every transition to popped status in
the observation stream. For each pop it records its time, balloon ID and the
currently committed first-leg attempt, if any. The relation is one of
`active_target`, `incidental`, or `unassigned`. The relation says what the
controller was doing at that moment; it does **not** estimate what an unflown
route would have hit. Route prefixes continue to have null outcomes.

The stream begins with balloon count and any initially popped IDs, and ends
with final popped IDs and counters. The audit rejects duplicate/missing pops,
invalid attempt associations, hidden scene data, invented labels, incomplete
episodes, mismatched stream hashes, and any difference from the official score.
It accepts a genuine zero-pop flight with no planned attempt. Old v1 records
remain readable for historical audit, but new runs require the complete v2
stream. The collector remains outside the agent; it does not change the
simulator, official evaluator or frozen PPO policy.

Local unit suite: 24 passed (route collector, audit, feature masks, proposal
selection, policy cadence). Remote CPU suite: 23 passed (collector, audit,
feature masks, attention model). The runtime copies have the same SHA-256 as
the local collector/runner/auditor files. Existing v1 runtime/data was copied
to a **new** `source_route_shadow_v2_20261006/` snapshot; the v1 data backup
remains untouched.

## Predeclared data fields

The parity anchor repeats development seed `261006001`; this repeats the
previous six-pop field only to validate the event stream, not to count as a
fresh training field. Ten additional collection seeds are:

`261006010, 261006011, 261006012, 261006013, 261006014, 261006015,
261006016, 261006017, 261006018, 261006019`.

They lie in the training seed pool below one billion. The set was checked
against pilot and continuation training logs before launch: no overlap.
Validation and holdout pools begin at one billion and are not part of this
collection. Do not replace a difficult/zero-pop field with a more favorable
one. Each field gets two fresh official flights in separate processes. Both
streams of actions and observations must match exactly and terminate normally
before the field enters the dataset. The runner audits each pair before the
next seed; on failure it stops and writes a failure record. The runner stores
progress after each completed pair.

CPU only, one numerical thread, one simulator worker, nice level 10, all GPUs
hidden. No leaderboard upload or source promotion. A training dataset is not
evidence of an improved score. After collection, inspect how often incidental
pops occur and how many independent episodes were obtained, then train a
route-ranking model and compare on **new** paired fields.

Remote project root:
`/home/rtcmspaceweather/balloon_multiscenario_20261004_v1/`.
Anchor results: `route_shadow_v2_anchor_20261006/`.
Anchor log: `route_shadow_v2_anchor_20261006.log`.
Anchor launch PID `2213154`; verify the command before acting on it.

## Verified anchor and collection status

The anchor completed with exact action/observation parity over 8,037 steps.
Both flights scored **6** and terminated normally. The baseline's action and
observation hashes also match the earlier v1 pilot. The new audit records all
**6** popped balloons: **5** active targets, **1** incidental pop, and **0**
unassigned pops. It groups 33 planner decisions into 7 attempted first legs
(5 hits, 2 deadline misses). The v1 dataset had seen only the five active
target hits; v2 closes that missing-reward gap without guessing outcomes for
unflown alternatives.

Anchor dataset SHA-256:
`bc94da708719fc39313db5cb4ac645afc77152ed19a5a836b787460b966691e5a`.
The completed anchor and exact source snapshot are backed up locally at
`BalloonPoppingGymEnv/evaluation/results/route_shadow_v2_anchor_complete_20261006.zip`.
Remote/local archive SHA-256 matched; ZIP integrity passed:
`36380fe694f2c35b633c5fc4e3cb758adc59302441f1f5c91e45ade3be4f3a60`.

After the anchor passed, the ten predeclared seeds were launched in one
sequential job (PID `2303179`) with the frozen v2 runtime. Output:
`route_shadow_v2_train10_20261006/`; log:
`route_shadow_v2_train10_20261006.log` in the same remote project root. The
runner saves `progress.json` after each complete pair and stops on the first
parity or audit failure. The run completed all ten fields with no failed gates
or truncations. Scores in predeclared order were **7/1/1/3/4/3/5/5/5/2**:
sum 36, mean 3.6, median 3.5, minimum 1, maximum 7. There were **36 observed
pops**, including **2 incidental pops**, 61 executed first-leg attempts and
304 planner decisions. These are training-pool fields of the frozen parent,
not a score improvement or a new submission.

The finalizer created `rollup.json` and
`route_shadow_v2_train10_complete_20261006.zip`. The archive has been copied
to `BalloonPoppingGymEnv/evaluation/results/` in this workspace. Remote/local
SHA-256 agrees and ZIP integrity passed:
`0c0d0a86f64a06ca2007caf2346d8c49837c8a299db8d116385e4a843b3f4619`.

First new field `261006010` completed with exact parity and all-event audit:
7 official pops, 7 observed pop events, 7 first-leg hits, 0 missed deadlines,
52 planner decisions. The frozen parent scored 7 on this field. This is one
training-field observation, not an improvement over the same parent or a
generalization estimate. The runner advanced to `261006011`.

The read-only summarizer `scripts/summarize_route_shadow_v2.py` checks each
saved pair, its dataset hash and the full pop count before computing mean,
median, low-score frequency, incidental hits and attempted legs. It was run
against the completed anchor and reproduced its 6-pops/1-incidental/7-attempt
audit. It will not treat an unfinished ten-seed progress file as a final result.
`scripts/finalize_route_shadow_v2.py` waits for the completed summary, rechecks
every field, writes `rollup.json`, and creates an integrity-tested remote ZIP.
It was exercised on the completed anchor; for the ten-seed collection it ran
as background PID `2345735` and exited normally after the backup. A failure
marker would have stopped it without announcing an incomplete dataset as
complete.
