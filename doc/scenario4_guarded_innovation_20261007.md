# Scenario 4 safety-gated endpoint innovation, 7 October 2026

This is an observation-only agent selected as the qualification default after
the paired test and packaging checks below. The candidate
starts from the earlier active-endpoint innovation controller, but refuses an
extra correction if its force allocation exceeds public thrust/tilt limits or
if observed spline tracking error is greater than twice the public balloon
radius. It does not change target selection, the official simulator, or the
baseline spline/attitude commands when the gate rejects a correction.

Earlier unguarded innovation scored 5/4/2 on development seeds 0/1/2 versus
the reviewed default's 5/2/4. Seed 2 had 531 correction saturations and a
very large spline-tracking error; that motivated this physically defined gate.

Full fresh official Scenario-4 development flights of the frozen guarded source
scored **5/4/3** on seeds 0/1/2. Seed 2 improved from 2 under unguarded
innovation to 3, with 283 infeasible corrections skipped and spline-tracking
error sum falling from 17,320 to 1,594 m across the flight. It still scored
below the reviewed default's 4 on that seed. The candidate kept 5/4 on seeds
0/1 and skipped 0/58 saturated corrections respectively. These are known
development seeds, not evidence about newly drawn qualification seeds.

Reports: ignored `s4_guarded_innovation_dev_seed2_20261007.json` and
`s4_guarded_innovation_dev_seed01_20261007.json` in evaluation/results.

## Frozen fresh-seed test plan, before seeing results

The following seeds were drawn from `numpy.random.default_rng(2026100707)`
and checked absent from the local used-seed registry before this plan. Stage 1
uses `1507667004` and `1458234498`; stage 2, only if stage 1's paired total
does not regress, uses `721771533` and `651490381`. On each seed, run the
reviewed default and the guarded candidate with fresh official resets, no
cached balloon field or submission JSON. Record complete scores, termination,
source hashes, and wall time. A negative stage 1 ends this candidate; a
nonnegative stage 1 merely justifies stage 2, not promotion. Promotion would
also require a convincing four-seed gain, no severe individual regression,
and a reproducible reviewed standalone agent/source package. None of these
four seeds may be reused for tuning after evaluation.

## Fresh paired results

Both agents were evaluated with fresh official Scenario-4 resets on each seed.
Each of the eight flights terminated normally without truncation, and the
recorded agent source dependencies did not change during a flight. The
baseline and guarded controller used the same local Python environment.

| Predeclared seed | Reviewed default | Guarded innovation | Difference |
| ---: | ---: | ---: | ---: |
| 1507667004 | 0 | 3 | +3 |
| 1458234498 | 3 | 2 | -1 |
| 721771533 | 6 | 7 | +1 |
| 651490381 | 4 | 5 | +1 |
| **Total across four flights** | **13** | **17** | **+4** |

The candidate won three pairs and lost one. Its mean was 4.25 pops per flight
versus 3.25 for the incumbent on this small sample; this is not a guaranteed
live-seed result or a 17-pop single flight. The second seed is a real one-pop
regression. The candidate's post-reset flight wall times were 126.4, 119.3,
121.3 and 106.3 s; baseline times were 130.2, 122.4, 110.4 and 102.8 s.
These comparisons are sensitive to system load and are not deterministic
runtime guarantees. Full reports are ignored local result files named
`s4_guarded_holdout_stage{1,2}_{baseline,candidate}_20261007.json`.

## Standalone source parity and operational status

`scripts/build_scenario4_guarded_submission_agent.py` reproducibly generated
`submission_scenario4_guarded_innovation_v1.py` (SHA-256
`dab27623c797c70c6fc5c13b48e4faa2e146a694c7de737f6cae8138485fb333`).
Its only agent import is the official `BaseAgent`; the derived implementation
does not import development agent modules. The official dry-run path scored 5
on development seed 0, matching the original guarded implementation. A second
fresh official-environment parity run on already-used holdout seed 721771533
scored 7 with identical pop IDs and times to the multi-file candidate. This
reuse checks packaging parity only; it is not an additional independent seed.

The official packaging path on already-used seed 0 scored 5 and produced a
valid Scenario 4 JSON object with the standalone agent. This is a packaging
check, not another independent performance test. The source-review archive
is `review/scenario4_guarded_innovation_source_v2.zip`; it contains only
allowlisted source, tests and documentation, not the credential-bearing JSON.
The qualification launcher now selects this standalone agent by default. A
full launcher rehearsal on newly used seed 830471237 scored 2, produced no
submission JSON, and restored the official Scenario 4 YAML byte-for-byte.
The previous default independently scored 2 on the same seed, a tie. Across
these five observed pairs the totals are therefore 19 versus 15; the fifth
pair was added for operational validation after the four-seed promotion plan,
not predeclared as part of the holdout. The live evaluation will draw new
seeds independently, so do not promise any particular score.
