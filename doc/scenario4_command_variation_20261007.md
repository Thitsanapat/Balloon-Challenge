# Scenario 4: bounded command-variation experiment (7 October 2026)

## Score checkpoint

The best completed *single* Scenario-4 flight among the local top-level
`s4*.json` and `*scenario4*.json` metrics is **7 pops on seed 721771533**
with the selected guarded-innovation agent. A standalone packaging-parity run
on the same seed also scored 7, but is not an independent seed. The earlier
17-pops figure is the sum of four different flights, not one flight and not
a live qualification score. The selected live launcher still points to the
standalone guarded agent, not the experiment below.

## Literature principle and implementation

Weiss and Shima, *Optimal Linear-Quadratic Missile Guidance Laws with Penalty
on Command Variability* (Journal of Guidance, Control, and Dynamics, 2015,
DOI 10.2514/1.G000738) formulate guidance that penalizes rapidly changing
commands as well as miss distance and effort. NASA's *The Synthesis of Optimum
Homing Missile Guidance Systems with Statistical Inputs* (1959, NTRS
19980228243) likewise emphasizes that unbounded commands are not physically
realizable and evaluates miss distance with acceleration limits.

`scenario4_slew_guarded_agent.py` is a **small heuristic adaptation**, not
an implementation or performance reproduction of either paper's optimal
controller. It retains the selected route, spline, observed-wind forecast,
terminal tracking gate, and thrust/tilt allocation. Only the accepted
horizontal endpoint correction is projected onto a ball centered at the
previous correction with radius `8 m/s^3 * dt`. A rejected correction returns
to the nominal command immediately and resets the stored correction; that
safety fallback can still create a command discontinuity. This variant uses
only observations and `given_parameters` at runtime and contains no
seed-specific route, hidden state, or simulator truth.

## Complete official-simulator development results

The candidate source and config were unchanged across three complete fresh
Scenario-4 episodes. These are **already used development seeds**, not a new
held-out estimate. The prior guarded scores were saved complete official
episodes on the same local environment, not simultaneous reruns.

| Seed | Selected guarded | Slew-limited experiment | Limited / applied correction steps |
| ---: | ---: | ---: | ---: |
| 0 | 5 | 5 | 74 / 2063 |
| 1 | 4 | 4 | 129 / 2302 |
| 2 | 3 | 3 | 135 / 1685 |
| **Three-flight total** | **12** | **12** | |

All candidate flights terminated normally, without truncation. Seed 0 popped
the same five balloon IDs. Seeds 1 and 2 popped different subsets but the
same number of balloons. The new limiter was active; the score tie is not
because the branch was never exercised. Six focused and guarded-controller
unit tests passed with `python -m unittest`. The local Python environment
still differs from the official lockfile, so a separate pinned-environment
parity check remains relevant before the event.

Candidate metrics:
`evaluation/results/s4_slew_guarded_dev_20261007.json` (ignored local report).
Selected prior metrics:
`s4_guarded_innovation_dev_seed01_20261007.json` and
`s4_guarded_innovation_dev_seed2_20261007.json` in the same results folder.

**Decision:** do not promote. The selected qualification source remains
`submission_scenario4_guarded_innovation_v1.py`, SHA-256
`dab27623c797c70c6fc5c13b48e4faa2e146a694c7de737f6cae8138485fb333`.

## Next model hypothesis

Command smoothness alone did not increase pops. The next bounded experiment
should preserve the successful long route, then optimize only the arrival
*time and target point* of the active leg over a short observed-forecast
horizon. Ahn et al., *Optimal terminal-time determination for the ZEM/ZEV
feedback guidance law with generalized performance index* (Astrodynamics,
2019, DOI 10.1007/s42064-019-0039-x), supplies the principle of balancing
terminal time with control energy; it does **not** establish that its orbital
equations transfer directly to this low-altitude rocket. Any implementation
must retain public thrust/tilt/rate checks, use only currently available
observations, and beat the selected agent on newly reserved paired seeds
before the live launcher changes. The earlier long-route timing optimizer
rarely adopted a proposal, so merely rerunning it with more iterations is
not a justified next step.
