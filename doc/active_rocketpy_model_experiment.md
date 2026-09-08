# ActiveRocketPy model experiment

## Purpose

This experiment audited the simulator equations before changing guidance.  It
tests three hypotheses: sensor/model feedback, non-upright initial conditions,
and multi-target route lookahead.  Results are reported even when they are worse
than the seven-point reference.

Primary implementations consulted:

- RocketPy: <https://github.com/RocketPy-Team/RocketPy>
- ActiveRocketPy: <https://github.com/ARRC-Rocket/ActiveRocketPy>
- Balloon Popping Challenge: <https://github.com/ARRC-Rocket/BalloonPoppingChallenge>

The local challenge checkout pins ActiveRocketPy at
`3b3d6c02feafab8e0ef29908c75a8190a6cdc268`.  Scenario 1 uses the generalized
nonlinear 6-DOF equations.  The implementation includes time-varying mass and
inertia, atmosphere-dependent aerodynamic forces and moments, gravity,
buoyancy, Earth rotation, TVC, roll torque, and throttle.  The ActiveRocketPy
throttle limitation is important: throttle scales thrust but does not change
the precomputed propellant mass history or the 30-second burnout time.

## Reduced online force model

`model_aware_spline_agent.py` transforms accelerometer specific force from body
coordinates to the world frame, subtracts the applied TVC/throttle thrust, uses
the median horizontal balloon velocity as a wind observation, and fits

\[
  a_d = b-k\|v-w\|(v-w).
\]

This is deliberately an observer over allowed measurements, not access to the
simulator's hidden rocket state.  The scalar fit cannot separate body drag from
fin lift and other aerodynamic terms, so it is validated by ablation rather
than assumed correct.

| Seed-0 cached-field variant | Score | Main observation |
|---|---:|---|
| Seven-point spline reference | 7 | Reference retained |
| Full force fit | 6 | Fitted `k` about 0.002; absorbed fin forces |
| No quadratic drag | 6 | Bias term still mixed aerodynamic effects |
| Faster filters | 6 | Lower residual, no score gain |
| Accelerometer only | 6 | Sensor-state change itself did not improve routing |

The force observer is therefore experimental and is not promoted into the best
submission.

## Initial attitude experiment

`inclined_spline_agent.py` treats launch inclination and heading as part of the
6-DOF initial condition.  Heading is selected from the observed, drift-predicted
first target.  Inclinations 55, 65, 70, and 75 degrees all scored 5 when the
automatic target was #58.  Forcing the nearby #85 cluster scored 6 at 55, 65,
and 75 degrees.  The latter produced the physically distinct route
#85, #7, #96, #89, #36, #28, but did not beat the reference.

## Route objective and launch-time experiments

`lookahead_spline_agent.py` ranks feasible two-intercept prefixes using their
time plus an optimistic nearest-neighbour tail through four more observed
balloons.  It selected a different cluster (#77, #33, #32, #1, #44, #64), but
weights 0.5, 1, 2, and 4 all scored 6 on seed 0.  A fresh, unmodified Scenario-1
run at launch time 45 seconds also scored 6, improving the ordinary 45-second
controller's 5 but not the overall reference.

The full seed-0 field cache is a development harness only.  The benchmark
injects it into the environment, validates scenario/seed/release steps, and
never passes the cache or future release schedule to the agent.  It exactly
reproduced the fresh 24-second reference: score 7 with identical pop times.

Launch-time sweep on that full field:

| Launch time (s) | Score |
|---:|---:|
| 22 | 5 |
| 23 | 7 |
| 23.25 | 6 |
| 23.5 | 6 |
| 23.75 | 7 |
| 24 | 7 |
| 24.25 | 7 |
| 25 | 6 |
| 26 | 5 |

Later lookahead launches at 30, 36, and 42 seconds scored 6, 5, and 6.  Waiting
for the cloud to fill did not compensate for lost powered-flight opportunity.

## Conclusion

The experiments reject three plausible but incomplete explanations: a wider
spiral, an estimated scalar drag correction, and initial-axis tilt alone.  The
limiting behavior is route-level: the current optimizer minimizes the next two
arrival times and then reaches burnout with a large downward velocity.  A
credible next model should optimize a powered prefix and a passive post-burn
fly-through together, including terminal velocity at burnout.  Until that is
implemented and verified over several fresh random seeds, the 24-second
two-intercept controller remains the best evidence-backed configuration at
7 points on seed 0 and 6 on seed 1; 14 points has not been demonstrated.
