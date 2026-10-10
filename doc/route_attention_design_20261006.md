# Learned route selection: design and isolated prototype

**Integration update:** an external observation-only decision collector and
full-flight bypass-parity runner are implemented. See the
[shadow collection pilot](route_shadow_collection_20261006.md). This connects
the feature encoder to actual committed choices, not the learned scorer to
flight control; counterfactual routes still have no observed outcome labels.

**Comparison completed:** continued PPO vs learned parent scored 5/4/2 vs
4/3/3 on three new seeds, a small mean improvement of 0.333 with one regression.
See [paired report](multiscenario_comparison_20261006.md). Keep the parent as
the reference while implementing and validating the separate route-selection
experiment; neither more training nor this prototype is a proven 20-pop agent.

## Why change planning, not just the control network size?

The ten-seed Scenario-4 test gave the frozen residual PPO 3.8 actual pops per
flight against 2.9 for its zero-residual controller. However, recorded maximum
route lengths were 8--9 for all ten PPO flights, while actual pops ranged 0--6.
Those are maximum plans over different moments, not a calibrated promised hit
count. The discrepancy is useful evidence to investigate route robustness.
On the zero-pop field, PPO reduced tracking error without making any actual hit.

Current code has two distinct restrictions:

1. `ChainBeamAgent._candidate_ids` sorts by a short constant-velocity proximity
   projection and takes only `branch_targets` (8 in this policy). A more useful
   distant chain may be excluded before search. The residual actor separately
   observes only eight balloons, active target first.
2. `_make_plan` sets `best = beam[0]` at each succeeding depth. Final selection
   favors the deepest feasible chain, with time ordering within a depth. Merely
   changing `_node_priority` cannot teach a preference for a shorter but more
   reliable route across depths. An explicit across-depth proposal comparison
   is required for a genuine learned route selector.

These are code/design observations, not proof that either alone explains the
score shortfall. Forecast error, actuator saturation and execution timing still
matter. Preserve the tested parent until fresh paired flights show a benefit.

## Architecture and responsibility boundaries

```text
current/past permitted observations + public vehicle parameters
                    |
      all-released-balloon attention encoder
                    |
    target ranking + candidate route proposals
                    |
  joint timing/arrival-velocity solve + physical rejection
                    |
   learned expected actual-hit value across route lengths
                    |
  selected first leg -> frozen low-level GNC/residual policy
                    |
        actual observations -> replan at events
```

### Scene encoding (prototype implemented)

Files: `agents/route_attention_features.py`, `agents/route_attention_model.py`
under `BalloonPoppingGymEnv/`. Feature schema `observed_route_set_v1_12_9`:

- One 12-feature token for every observed balloon row. Only released, finite
  balloons enter attention: relative position and velocity, distance, closing
  speed, drift speed, relative local density within 10/25 m, candidate flag.
- Nine query-context features: controller velocity, acceleration, remaining
  burn time, partial-route elapsed time, available acceleration. All are
  observation/history/public-model derived; no hidden flight state is read.
- Optional partial-route projection uses the current observed velocity only.
  It is a fallible prediction, not prerecorded or simulator future positions.
- Separate visibility and candidate masks. Popped/unreleased/nonfinite rows
  are not context. Visited/blocked/infeasible rows cannot be selected. Density
  is a soft feature, not a command to always pursue the densest group.
- Two attention blocks, width 128, four heads, feed-forward width 512, zero
  dropout. No row-ID or positional index embeddings; permutation equivariance
  prevents arbitrary balloon numbering from becoming a geometric shortcut.
- Prototype outputs N target logits plus a defer-to-validated-planner action,
  and a scalar value head. A permanent context token handles empty/all-masked
  scenes. The fallback is NOT a zero-throttle/unsafe command.

**Status:** tested untrained tensor prototype only. It is not wired into the
flight agent, not a complete route-value model, and not a submission. No score
improvement or trained reasoning about groups is claimed for this prototype.
Measured default size: 449,282 parameters. A one-thread CPU synthetic benchmark
on the professor host (100 visible tokens, batch 1, 10 warmups, 100 forward
passes, no gradients) averaged 5.42 ms per call. This excludes proposal solving,
feature generation and flight control, and is not a real-time guarantee.

### Candidate generation and across-depth selection (next implementation)

A pure selection-boundary prototype is implemented in
`agents/route_proposal_selection.py`: given externally predicted expected-hit
values, it compares proposals from all lengths, breaks ties by time, rejects
malformed/budget-exceeding sequences, and calls the controller's supplied dense
physics validator before accepting anything. Tests show a shorter route can
win and a high value cannot override rejection. It does not yet create curves,
learn those values or install a route; no synthetic utility is a measured score.

Use the target head to broaden candidate coverage, not replace physics:

- Keep a quota of the current near-target candidates while adding attention-
  ranked alternatives from the complete visible set. Include singles and group
  entries; don't hard-code balloon IDs, cluster count or a seed-specific route.
- At every expansion carry arrival position, velocity, acceleration and elapsed
  time, not just Euclidean distance. Predict moving targets at arrival time.
- Reuse `free_chain` / timing optimization to solve proposed chains, retaining
  validated complete candidates from **every depth**, not only the deepest.
- Check thrust-to-mass limits, tilt, axis rate, throttle slew, ground clearance
  and remaining powered time, then densely validate the selected whole route
  (existing 65-sample check). Sampled model feasibility is not a guarantee in
  the full noisy nonlinear simulator; keep reserve and feedback.
- Reject an infeasible proposal before any flight action. Always retain a
  validated base proposal or its existing recovery behavior. Do not declare a
  distance-only or attention-derived mask to be physical feasibility.
- A route-value head must explicitly encode sequence/order, durations, exit
  velocity, clearance to limits and predictive uncertainty. The current target
  head is insufficient for this across-depth decision and must not be relabeled
  as a trained route selector. Prefer expected *actual* future pops over planned
  waypoint count, with time/turn effort as small tie-breakers rather than an
  incentive to hover instead of scoring.

Inference budget: one global ranking per real replan (existing 0.4 s interval
or hit/failure event), not at every 0.01 s physics step or every beam node.
The initial prototype recomputes its context-dependent attention; do not cache
it across changed contexts and assume equivalence. A later fast implementation
can factor a scene-only encoder from a lightweight route-state decoder, tested
separately. Profile one-thread CPU inference before setting wall-time budgets.

### Commitment and launch time

Preserve the current final-approach guard near a committed intercept; otherwise
rank noise could switch targets just before a hit. Replan after hit/miss or a
material state/feasibility change. Evaluate switching costs and hysteresis as
separate ablations, not unmeasured improvements.

Initial planner pilot keeps launch time 42 s and the frozen low-level policy,
so a gain/loss can be attributed to route selection. Adaptive launch belongs
in a later separate experiment: compare launching now vs waiting using only
currently released balloons and allowed history, with a bounded wait fallback.
Never inspect unreleased schedules or future wind. Claiming the present
prototype already learns launch timing would be incorrect.

## Training plan and data boundary

1. Instrument **agent-side** candidate proposals, current permitted features,
   masks and chosen decisions on training-pool seeds. Keep seed identifiers
   in external provenance only. Existing JSON reports contain diagnostics,
   not the full actor input stream: they cannot train this network by themselves.
2. Warm-start rankings from the current planner's online choices if useful,
   while recognizing imitation alone inherits its biases. Unexecuted proposals
   do not have observed hit outcomes: never label them as failed ground truth.
3. Train the decision policy on fresh official flights. Use actual hit outcomes
   as the main reward; any time/effort shaping is training-only, never a changed
   competition score. A policy that merely reproduces high planned counts is
   not the goal. Preserve source and data-generation code for review.
4. High-level decisions occur at variable intervals. Either fix the decision
   interval for the first PPO integration or explicitly handle duration-based
   discounts/advantages. Do not silently reuse the 0.1-second residual PPO
   training clock for variable-length route options.
5. Freeze the low-level learned parent initially. If joint training is later
   attempted, compare it separately because simultaneous route/control changes
   alter both the task distribution and attribution of score gains.
6. One GPU maximum for a bounded pilot. Compare base, residual-only parent,
   near-target ranking ablation, global attention ranking, and full route-value
   selection using identical predeclared unseen seeds. Report mean, low-score
   frequency, misses, truncations and wall time, not best-seed performance.

## Tests and boundaries checked so far

Three local feature tests and six combined remote CPU tests passed: released-
only filtering, nonmutation of observations, permutation consistency, mask
semantics, empty inputs, all-masked fallback, finite gradients and NaN rejection.
The completed checkpoint comparison used another immutable snapshot; these new
prototype files do not change either of its agents.

Three additional local proposal-selection tests passed: across-depth utility
comparison, rejection despite high predicted value, and malformed/budget-invalid
route filtering. Nine component tests cover the present prototype, not a full
learned-planner flight. No simulator/controller implementation was changed.

Practical acceptance checklist for the next implementation:

- Explain each rejection using agent-side features and physical model limits,
  without reading evaluator truth.
- Baseline-proposal bypass must reproduce the current agent on the same seed
  before enabling learned routing, to avoid attributing unrelated changes to it.
- Record chosen-leg outcomes as hit, miss, cancelled or censored (episode ended
  before deadline). No unexecuted candidate gets a fake outcome label.
- Beat the frozen parent on fresh paired fields within the same official wall
  budget; the present 5.42 ms tensor timing is not sufficient evidence.
- First integrate route selection alone. History, learned wind prediction and
  adaptive launch should be separately measured experiments.

## Sources and limitations

[Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)
demonstrates learned attention heuristics for routing including orienteering
and prize-collecting variants. Our moving-target rocket proposal is an adaptation,
not a reproduction of its results or evidence that 20 pops is achievable.

[Organizer Discussion #165](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165)
requires observation-only execution, fixed evaluation components and reviewable
agent/training source. The architecture is designed around those boundaries;
this is not organizer certification. Do not use precomputed seed-specific routes,
stored commands, hidden simulator state, or prerecorded balloon trajectories.
