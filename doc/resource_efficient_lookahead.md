# Resource-efficient route lookahead experiments

These are deterministic cached-field development benchmarks for scenario 1,
seed 0, ending at 60 seconds. They are not official leaderboard scores. The
cache supplies only the simulator's balloon field; each agent still receives
the ordinary observation at every step.

The useful result is `CommittedSplineAgent`. The ordinary spline planner
already solves two intercepts but throws away the second choice after a pop.
The committed variant keeps that choice, promotes it immediately, and refreshes
the following choice while flying. If the commitment becomes invalid or
infeasible, it falls back to the bounded ordinary search.

| Method | Search bounds | Score | Pair candidates | Result |
|---|---:|---:|---:|---|
| Spline baseline | first 8, next 3 | 7 | 4,805 | Reference |
| Committed | first 8, next 3 | 7 | 2,465 | Same score, 49% fewer candidates |
| Committed | first 4, next 3 | 7 | 1,889 | Same score, 61% fewer candidates |
| **Committed** | **first 4, next 2** | **7** | **1,278** | **Selected: 73% fewer candidates** |
| Committed | first 3, next 2 | 6 | 1,156 | Candidate set is too narrow |
| Committed | first 2, next 2 | 5 | 1,047 | Candidate set is too narrow |
| Committed | first 8, next 1 | 5 | 884 | One continuation is too brittle |
| Greedy-tail lookahead | weight 0.02 or 0.1 | 6 | 12,870 | Changes the route too early |
| Fly-through velocity | 2 m/s | 6 | 4,959 | Harder tracking outweighs momentum |
| Fly-through velocity | 5 or 8 m/s | 5 | 6,452 / 4,905 | Too aggressive |
| Force nearby #80 after/before #8 | scripted | 6 | 1,424 / 1,369 | Proximity alone ignores turn cost |

The selected configuration is
`evaluation/configs/committed_spline_launch24.yaml`. Its verified pop sequence
on this development field is `58, 27, 37, 35, 56, 8, 12`.

The main lesson is that nearest-neighbour distance is not a sufficient route
cost. A useful continuation must also be dynamically reachable with the current
velocity, thrust reserve, gimbal/tilt constraints, and remaining burn time.
Keeping one feasible continuation is valuable; forcing a geometrically nearby
but dynamically poor waypoint is not.
