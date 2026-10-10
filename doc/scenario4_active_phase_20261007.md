# Scenario 4: active-leg capture-phase experiment (7 October 2026)

This is an exploratory continuation of the selected observation-only guarded
innovation controller. The selected qualification launcher and standalone
source were not changed. The experiment is motivated by the *principle* in
Ahn et al., "Optimal terminal-time determination for the ZEM/ZEV feedback
guidance law with generalized performance index" (Astrodynamics 3, 127–136,
2019; DOI 10.1007/s42064-019-0039-x): choosing terminal time while balancing
control energy. The code is **not** that orbital guidance law. It merely ranks
up to three earlier crossing points on the already selected spline using a
short observed-balloon forecast, vertical alignment, estimated horizontal
correction effort, time, and phase-change cost. Search is confined to the
last three seconds and at most 0.75 seconds earlier than the planned endpoint.
The correction still passes the previous thrust/tilt allocation and tracking
gate. It reads only `observation` and `given_parameters` at runtime.

Seven targeted unit tests passed, including cases that select an earlier
crossing, retain an aligned endpoint, and leave longer-horizon plans alone.
Three complete fresh official Scenario-4 episodes were run on already-used
development seeds, with the experimental source/config unchanged:

| Seed | Selected guarded | Active phase | Accepted phase advances | Flight wall time after reset, selected → experiment |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5 | 5 | 0 | 133.3 → 145.0 s |
| 1 | 4 | 4 | 0 | 122.0 → 139.0 s |
| 2 | 3 | 3 | 0 | 131.7 → 144.8 s |

All flights terminated normally without truncation. The same balloon IDs
were popped on each paired seed. The phase advance was never accepted, so
the equality in score says **nothing about the effect of an actual shifted
intercept**. The ranking overhead cost roughly 12–17 more wall seconds per
flight locally; elapsed times are load-sensitive, not deterministic limits.
Metrics are in the ignored local report
`evaluation/results/s4_active_phase_dev_20261007.json`. This did not consume
any new untouched seed and is not an independent qualification estimate.

**Decision: reject for live use.** A relaxed phase threshold chosen merely
to force activation would be tuning on the known fields, not evidence that
the shifted crossing is safer. The current model's trajectory already aligns
its endpoint to the forecast well enough that moving the crossing earlier
does not beat the control-effort and phase penalties.

The separate observation-only telemetry for selected seed 2 found two
officially unpopped near approaches (targets 9 and 42) with true closest
distances about 1.95 and 1.74 m versus a 1.5 m capture radius. At the sampled
closest *observed* approaches, spline tracking error was 4.35 and 4.98 m.
These are different measurement/time conventions, so the tracking error is
not by itself a causal attribution; forecast and alignment errors can cancel.
It does suggest that the next investigation should estimate whether an
active leg is trackable within the observed actuator reserve, then prefer a
less aggressive corridor or replan early. Do not simply raise controller
gains or bypass the tracking gate: earlier tests of both kinds regressed or
increased saturation. Validate any new corridor model offline against later
observations and official outcomes before allowing it to change commands.

The selected standalone agent remains
`submission_scenario4_guarded_innovation_v1.py`, SHA-256
`dab27623c797c70c6fc5c13b48e4faa2e146a694c7de737f6cae8138485fb333`.
