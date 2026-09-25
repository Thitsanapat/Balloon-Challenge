# Learned route-selector pilots (2026-09-25)

The current 10-point controller tracks selected targets well. These pilots move
learning to the first target of each new chain route; the inherited controller
continues to produce trajectories and commands. All model inputs are current
observations and given parameters: up to eight released target rows, rocket
sensor features, and remaining burn time. Target index, random seed, future
balloon state, stored trajectory, and simulator internals are excluded.

Three pilots share `SelectorChainAgent` and `OfficialSelectorEnv`:

| Pilot | Learning signal | Model |
| --- | --- | --- |
| XGBoost ranker | Random selected rank receives the actual terminal pop count from its completed episode | LambdaMART tree ranker |
| Maskable PPO | Official pops plus a training-only nearest-target potential | Discrete masked rank selector |
| Attention REINFORCE | Same official reward and potential | Attention scores over released target rows |

The XGBoost signal is contextual-bandit feedback. It does not claim that an
unselected candidate would have produced zero score. Attention and PPO begin as
small pilots; their checkpoints are not submitted. Any score increase must be
retested with a standalone observation-only agent, official evaluator and
submission verifier before it could replace the preserved 10-point payload.

Models use fresh external seeds in the training wrapper. The final agent cannot
read the seed, a data file, a stored field, or future balloon states. Training
source and all learned weights are retained for organizer review. Rule reference:
https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165

## Completed first pilots

The remote environment used XGBoost 2.1.4, Stable-Baselines3 2.7.0 and
sb3-contrib 2.7.0. Attention used the one authorized Tesla P40; its inference
evaluation and the tree ranker used CPU. No second GPU was made visible.

The random contextual-bandit collection completed six full episodes, with
scores 4, 4, 3, 5, 4, 5 and no truncations. It produced 550 decisions with
eight candidates each. The XGBoost LambdaMART model had 4,400 candidate rows,
one observed-return positive row per decision. This pilot data is too small for
a claim of convergence and the feedback does not identify unchosen
counterfactuals.

Attention REINFORCE completed six full training episodes with scores 5, 5, 5,
5, 0, 7; none was truncated. Maskable PPO completed its requested 512 decisions
in 640 seconds. The checkpoint is retained even though it is not a submission
candidate.

Three models were then evaluated deterministically on the same three fresh
external evaluation seeds. The baseline value below is the unmodified chain
controller running through the training wrapper; it is a paired pilot control,
not the leaderboard seed-0 score.

| Controller | Episode scores | Mean actual pops |
| --- | --- | ---: |
| Baseline | 5, 6, 5 | **5.33** |
| XGBoost ranker | 4, 5, 3 | 4.00 |
| Attention pilot | 3, 5, 5 | 4.33 |
| Maskable PPO pilot | 3, 5, 4 | 4.00 |

All 12 evaluation episodes terminated normally. The trained selector forces one
root target, which is substantially more restrictive than the reference beam
that explores up to eight root candidates. The result therefore says that these
small pilots do not compensate for that lost search breadth; it does not show
that learned route ranking cannot help after a different integration and much
larger training set.

Artifacts, including models, dataset and score reports, are stored under
`evaluation/results/route_selector_20260925/route_selector_20260925/`; their source snapshot is
`evaluation/results/route_selector_source_v3_20260925.tgz`. The preserved
verified 10-point submission was not changed or uploaded.
