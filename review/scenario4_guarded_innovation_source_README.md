# Scenario 4 guarded endpoint-innovation source

This archive is a credential-free source package for the selected
observation-only Scenario 4 agent. It contains no precomputed balloon routes,
seed-conditioned commands, trained checkpoint, submission JSON, or private
team credential. The source is provided for reproducibility and review; the
competition organizer remains the authority on rules.

The runnable entry point is
`BalloonPoppingGymEnv/agents/submission_scenario4_guarded_innovation_v1.py`,
class `Scenario4GuardedInnovationAgent`. The associated configuration is
`BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_guarded_innovation_launch42.yaml`.
Place these paths in the official Balloon Popping Challenge v0.2.2 checkout
and run a credential-free evaluation from the repository root:

```powershell
python -m scripts.run_submission BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_guarded_innovation_launch42.yaml --dry-run
```

`scripts/build_scenario4_guarded_submission_agent.py` demonstrates exactly how
the standalone source was composed from the included source files. Run it with
a *new* output path if regenerating: it refuses to overwrite an existing
agent. `doc/scenario4_guarded_innovation_20261007.md` records the tests and
scores. Four paired test seeds yielded totals 17 versus 13 across four flights,
not 17 pops on any one flight and not a live-seed guarantee.
