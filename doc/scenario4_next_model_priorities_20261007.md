# Scenario 4 next model priorities after 7 October diagnostics

The selected qualification agent remains the standalone guarded-innovation
controller. Three isolated experiments today did not pass their predeclared
promotion gates:

| Experiment | Development result | Decision |
| --- | --- | --- |
| Shorter/weaker endpoint correction | Seeds 0/1: 6/2 versus 5/4 | Reject: two-pop regression on seed 1 |
| Longer GNSS position filter | Seed 0: 3 versus 5; navigation RMS 0.471 versus 0.438 m | Reject: score and state estimate worsened |
| Faster spline tracking | Seed 2: 4 versus 3; mean tracking error 0.709 versus 0.564 m and saturated steps 1,346 versus 1,155 | Reject: mechanism worsened despite one extra pop |

A read-only seed-2 telemetry replay reproduced the selected agent's three
pops. Two unpopped targets came within 1.95 and 1.74 m in official truth,
against a 1.5 m capture radius. At sampled closest observed approaches,
reference-tracking errors were about 4–5 m. These are *different time and
measurement conventions*: the noisy observation distance can be smaller than
official truth. The vectors can also cancel, so no single error magnitude is
a causal explanation by itself. The evidence points to short-lived terminal
misses, not a simple global gain adjustment.

The next substantive model should be an actuator-aware **reachable corridor**
for the currently committed target. In an offline diagnostic, start from each
live observation and current actuator state, roll forward a small set of
bounded control alternatives over 0.5–2 seconds using a locally calibrated
rocket response, and compare their predicted closest approach to the
balloon's observed-state forecast. Measure forecast error and false-positive
capture claims against later observations/official outcomes over many seeds.
Only then let the agent prefer a safer corridor or incidental group of
balloons; if the local model is uncalibrated, retain the existing spline
command. This is distinct from the earlier full-route chance score and the
preview MPC + PPO variant, both of which lost independent tests.

Any new controller must read only `observation` and `given_parameters` at
runtime; offline simulator truth may be used solely for diagnostic validation,
never fed to an agent or stored as seed-specific commands. Freeze a candidate
before drawing new paired test seeds, publish all wins and losses, and change
the live launcher only after a repeatable gain and standalone-source check.
