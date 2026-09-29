# Source review: leaderboard submission 10976510-05c7-40cc-b08f-a970a907c9aa

This archive is the minimal, credential-free source package for the submitted
Scenario 1 result.

| Field | Value |
| --- | --- |
| Score | 10 pops |
| Scenario seed | 0 |
| Submitted JSON SHA-256 | `b6ff58aac9ad159357a1f33f155546bf2cb4d42a8636a609570b9c7c6a280afe` |
| Embedded agent-source SHA-256 | `86795ef3e0afdf17dc894b99a57dc6082446ef4bf674a745ce299eabea26aa57` |

## Included source

The canonical submitted agent is:

```
BalloonPoppingGymEnv/agents/submission_chain_launch_v1.py
```

Its file bytes exactly equal `agent_info.agent_module_file` embedded in the
submitted JSON (the JSON itself is deliberately not included because it holds a
team credential). The supplied evaluation config is:

```
BalloonPoppingGymEnv/evaluation/configs/submission_chain_launch_v1.yaml
```

It preserves the scenario, agent class, agent name, and all agent keyword
arguments from the submitted configuration. It intentionally sets
`leaderboard_submission: false` and has no team fields: that prevents creating
another credential-bearing result JSON and does not change the simulation or
the agent's actions.

## Reproduction

1. Start from the organizer's [v0.2.1 release](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/releases/tag/v0.2.1), initialize its `ActiveRocketPy` submodule, and install the locked dependencies as documented there.
2. Extract this archive over that checkout, preserving its paths.
3. From the repository root, run:

   ```powershell
   uv run python BalloonPoppingGymEnv/evaluation/evaluate.py BalloonPoppingGymEnv/evaluation/configs/submission_chain_launch_v1.yaml
   ```

   With pip rather than `uv`, activate the environment and use `python` in the
   same command. Scenario 1's released parameter file specifies random seed 0.
   The expected score is 10 pops.

The result was re-run locally with this exact standalone agent and config on
2026-09-29: 10 pops, 5,728 simulation steps, ending at 57.28 seconds.
`audit_report.json` records the credential-safe source scan and the official
data-verifier results for the original submitted JSON.

## Agent scope and data use

`ChainSubmissionAgent` is a deterministic online guidance algorithm. At
construction it receives only the scenario `given_parameters`; while flying it
uses the supplied observations, specifically simulation time and currently
observed balloon states, to select and re-plan interception chains. It contains
no trained model, checkpoint, stored route, precomputed balloon positions,
seed lookup, reference trajectory, or external data file. It does not read
private simulator attributes or modify the provided simulator.

There was therefore no training code or learned artifact for this submitted
agent. The standalone source file above is the complete executable agent used
for the submitted result; its only non-standard dependency is the organizer's
provided `BaseAgent` class.
