# Scenario 4 qualification source: 42-second launch

This archive contains the source of the deterministic, observation-based agent
selected for the Scenario 4 qualification round. It contains no trained model,
run data, team credentials, or submission JSON. The agent uses the official
`given_parameters` and observations to plan during each flight.

The runnable entry point is
`BalloonPoppingGymEnv/agents/submission_scenario4_wind_profile_v1.py`, class
`Scenario4SubmissionAgent`. Use
`BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_wind_profile_launch42.yaml`
for its reviewed settings. Place the archive contents at the root of the
official Balloon Popping Challenge v0.2.2 repository, preserving the paths.

To reproduce a local evaluation without credentials, run from that root:

```powershell
python scripts/run_submission.py BalloonPoppingGymEnv/evaluation/configs/submission_scenario4_wind_profile_launch42.yaml --dry-run
```

`scripts/build_scenario4_submission_agent.py` shows how the standalone agent
was composed from the included time-allocation and observed-wind sources. The
build script refuses to overwrite the included standalone agent. The working
notes in `doc/scenario4_qualification_work.md` record the paired seed tests.
Their scores are measurements, not a promise for independently drawn live
qualification seeds.
