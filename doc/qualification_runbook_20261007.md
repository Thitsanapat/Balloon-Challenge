# Scenario 4 qualification: operator runbook

This is for the announced 10 October 2026 qualification round. The organizer
specifies Balloon Popping Challenge **v0.2.2**, a newly drawn seed immediately
before each of two evaluations, and screen sharing while the agent runs. The
organizer must redraw a seed previously used for training, tuning, or testing.

## Before the event

- On the Bangkok clock, the optional screen-share test is **7 October,
  07:00–16:00**; the qualification event is **10 October, 18:00–21:00**.
- Use this same computer and `.venv`. Do not update the simulator, installed
  RocketPy/ActiveRocketPy, or default agent between rehearsal and live runs.
- Connect power and a stable network; turn off pop-up notifications before
  sharing your screen. A live evaluation and JSON packaging can take several
  minutes. Allow at least 1 GB of free disk space.
- Double-click `run_qualification.cmd` for a rehearsal with a **fresh testing
  seed**, or launch it with `--seed <number> --mode rehearsal`. Never rehearse
  with a seed that the organizer may later assign for a live run.
- The launcher checks the official v0.2.2 core files and byte-for-byte parity
  between the pinned ActiveRocketPy source and installed `rocketpy` Python
  files. If this check fails, stop and fix the environment first.
- Final-night default: standalone `submission_scenario4_pure_joint_v2.py`
  with `submission_scenario4_pure_joint_launch42.yaml`. It scored
  19 pops in total versus 12 for the guarded default over four
  paired test seeds (3 wins, 1 tie). This is a small test, not a promise for
  the live seed, and not 19 pops in a single flight. Its source-review package is
  `review/scenario4_finalnight_joint_source_20261010.zip`. The original guarded
  source/config are unchanged and can be selected before a run with
  `run_qualification_guarded_backup.cmd`. Do not switch agents
  by editing the configuration on event day. Experimental MPC+PPO is not
  selected; it scored 3.5 versus 4.1 pops per flight over ten unseen pairs.

## Each live evaluation

1. Receive the organizer's seed. Run `run_qualification.cmd --seed <number>
   --preflight` if you want a read-only check. If it says **USED/RESERVED**, tell
   the organizer and ask for a new seed *before* evaluating. The local registry
   cannot prove every seed in old PPO jobs: see the limitation below.
2. Double-click `run_qualification.cmd`; enter the exact seed. Type `LIVE` at
   the mode prompt. Check the displayed seed, agent source/hash, and team name.
   The secret is deliberately hidden. Type `RUN` only when those details are
   correct. This records the seed as used before the simulation starts.
3. Keep sharing the evaluation screen. Wait for `QUALIFICATION RESULT`, the
   score, and `UPLOAD THIS JSON`. The official Scenario 4 YAML is then
   restored to its original bytes. The run creates **one submission JSON object**
   under `BalloonPoppingGymEnv/evaluation/results/`.
4. Upload **only** the exact file following `UPLOAD THIS JSON`. Do not upload a
   trajectory, metrics file, ZIP, or the clean receipt. The JSON includes the
   team secret; do not open it on the shared screen or put it in a public repo.
   Save the upload/leaderboard confirmation separately, with seed, score, file
   name, and displayed SHA-256. The local receipt says `uploaded: false`
   because this launcher does not upload on your behalf.
5. For the second evaluation, use its **new** organizer seed and repeat. The
   launcher refuses known reused seeds, including seeds from failed attempts.

If the run or computer is interrupted while Scenario 4's seed is temporarily
changed, run `run_qualification.cmd --recover` before another evaluation. The
recovery uses the preserved backup and refuses to overwrite an unrelated
manual edit. The interrupted seed remains recorded as used. Coordinate with
the organizer about whether that evaluation needs a redraw.

## Records and limitations

- `qualification/used_seeds.json` collects seeds visible in local results,
  experiment plans, code and archives. `qualification/used_seed_manual.jsonl`
  is append-only for live/rehearsal attempts. Do not delete or reset either.
- Some older PPO jobs did not record an exact seed for every episode. The
  registry reserves a conservative prefix reconstructed from their RNG and
  monitor data, but **cannot certify all historical seeds**. If a drawn seed is
  suspect despite a `NOT FOUND` result, disclose the uncertainty and request
  a redraw rather than asserting it was never used.
- The clean run receipts are in `qualification/receipts/` and contain no team
  secret. The submission JSON does contain the secret and stays ignored by Git.
- Local rehearsal on seed `314159265` scored **4 pops**, restored the official
  YAML byte-for-byte, and created no submission JSON. A separate local packaging
  test on seed `271828182` scored **6 pops**, made a valid JSON object, and
  restored the YAML. Neither file was uploaded; neither score forecasts an
  independently drawn live seed. Both test seeds are now recorded as used.
- The official packer may warn that it could not fetch an online copy of
  `evaluate.py` when network access is restricted. The launcher has already
  compared local core files, the weather file and installed `rocketpy` against
  the local organizer v0.2.2 tag; investigate any *local* parity failure before
  running. The online-check warning alone did not prevent JSON packaging in
  the local test.
- The guarded standalone packaging test on already-used seed `0` scored
  **5 pops** and produced a valid Scenario 4 JSON object. It was not uploaded.
  It checks the output format, not performance on an independent seed.
- The guarded launcher rehearsal on seed `830471237` scored **2 pops**, made
  no submission JSON, and restored the official Scenario 4 YAML. A separate
  run of the previous default also scored **2 pops** on that seed. This seed
  is now used and must not be accepted as a newly drawn live seed.
- The final-night joint standalone local packaging rehearsal on seed
  `741093826` scored **4 pops**, created a validated submission JSON object,
  and restored the Scenario 4 YAML byte-for-byte. Nothing was uploaded. That
  JSON is a rehearsal artifact, not a substitute for the organizer's live seed.
  See `qualification/finalnight_joint_readiness.json` and
  `doc/scenario4_finalnight_20261010.md`. Its agent SHA256 begins `9ff7f93d21a9`.

Official reference: the competition's v0.2.2 release, README and Discussion
#165. The operator should still follow any newer direct instruction from the
organizers.
