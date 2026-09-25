# Exit velocity diversity and short in-flight first legs (2026-09-25)

The frozen Scenario 1 submission scores 10 on seed 0 and remains preserved.
This experiment changes only an observation-based agent planner. The official
simulator, scenario parameters, scoring, packer, and verifier are unchanged.

## Velocity diversity experiment

The joint-chain search currently keeps at most two paths sharing the same first
and last balloon IDs. It ranks by completion time; different exit velocities
are not a diversity key. This can discard a path pointing toward other targets.
The older momentum planner had velocity bins, but the current stronger joint
chain search does not. `VelocityDiverseAgent` adapts velocity bins to this joint
search. A node enters at most once per first/last/velocity-bin key, with at most
three per first/last pair and the same total beam width 12. Physical checks,
timing optimizer, and guidance are inherited from the frozen reference. The
launch-axis comparison uses the original planner, avoiding temporary-state
selection effects found in the earlier bridge experiment.

Tests on remote official Scenario 1 episodes, seeds 0/3, CPU only:

| Exit velocity bin (m/s) | Seed 0 | Seed 3 | Pruning changes (seed 0 / 3) |
| --- | ---: | ---: | ---: |
| 0 (disabled; seed 0 control) | 10 | — | 0 / — |
| 4 | 9 | 9 | 12 / 13 |
| 8 | 9 | 6 | 8 / 6 |

The changed frontier is observable, but neither tested bin improves the baseline
10/9 score. These two development seeds are not an independent generalization
test. The five reports were produced from the source in
`evaluation/results/velocity_diverse_source_20260925.tgz`.

## Isolated in-flight first-leg experiment

The same agent can leave the velocity bin disabled while changing only the
minimum duration of the first searched leg, after launch is established. The
initial launch-axis comparison and first launch plan retain the original 2 s
minimum. The in-flight threshold is more than 0.5 s after launch. Every
candidate still passes the same sampled physical checks. Settings 1.0 and
0.5 s were paired against seeds 0/3:

| In-flight first-leg minimum | Seed 0 | Seed 3 |
| --- | ---: | ---: |
| Original 2.0 s | 10 | 9 |
| 1.0 s | 10 | 8 |
| 0.5 s | 10 | 8 |

The extra duration choices did not increase actual hits on these fields. The
0.5 s seed-0 run has exactly the original pop events. This tests only the
in-flight first leg, not the benefit of a different launch schedule.

Reproduction config: `evaluation/configs/velocity_diverse_chain.yaml` with
`--set exit_velocity_bin=0 --set first_leg_min=1` or `=0.5`. Source snapshot:
`evaluation/results/short_first_leg_source_20260925.tgz`. Fourteen focused
local tests and five remote agent tests passed before these runs.

The nine complete reports are in `evaluation/results/velocity_and_shortleg_20260925/`.
All nine terminated normally, none truncated, and each reported unchanged agent
source dependencies during the run. The disabled seed-0 control reproduced the
reference pop events exactly. The original verified 10-point submission still
has SHA256 `0f1d78c805630dd29c243d21003a765859e6c01e5775bca01d82cbdbb2d5d46b`.
No variant is promoted and no upload was made. No simulator worker or training
job remains active on the authorized server after this batch.

Neither planned route length nor early progress is an official score. If a
variant improves actual hits, it still needs a fresh official submission,
standalone source packing, and verifier checks before replacing the 10-point
payload. No current experiment is approved by the organizer merely from static
source review; see https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165.

The next distinct research direction is a high-level selector that ranks
observation-based route candidates and learns from *actual completed pops*.
The previous PPO model only perturbed acceleration and its paired mean was
lower than baseline; continuing that checkpoint is not evidence-based.
