# Physics-based guidance experiment

## Research and what is actually implemented

1. [Russ Tedrake, MIT Underactuated Robotics, Trajectory Optimization](https://underactuated.csail.mit.edu/trajopt.html#section4).
   The differential-flatness discussion connects position derivatives to thrust
   direction and attitude, explaining why a smooth geometric path alone is not
   sufficient without control limits and feedback. Here, acceleration determines
   the requested thrust axis; jerk supplies an angular-rate feedforward term.
   This does **not** prove that the aerodynamic six-DOF rocket is a flat system.
2. [Burke, Chapman and Shames, Generating Minimum-Snap Quadrotor Trajectories Really Fast (2020)](https://arxiv.org/abs/2008.00595).
   This motivates smooth polynomial paths and attention to numerical conditioning.
   Our code uses normalized-time **minimum-jerk quintics**, derived below, rather
   than implementing their minimum-snap spline solver. It does not claim their
   computational or optimality guarantees.
3. [Choi and Kim, Powered Descent Guidance via First-Order Optimization with Expansive Projection (2024 version)](https://arxiv.org/html/2310.00397v2).
   Its powered-flight formulation explicitly constrains thrust magnitude and
   pointing direction while accounting for changing mass. We adopt those physical
   constraints but use sampled candidate rejection, not their ExProj solver or
   lossless convexification. This is a local search, not a global optimal route.
4. [NASA Goddard, Motion in a Circle](https://pwg.gsfc.nasa.gov/stargaze/Scircul.htm).
   The centripetal relation a=v^2/R explains the cost of circling. A spiral can be
   useful only when its radius and speed fit the acceleration budget and the
   balloon distribution. Merely adding circles does not increase intercept count.

## Equations matched to this simulator

Use ENU coordinates in metres, seconds, kilograms, and radians internally:

    p_dot = v
    v_dot = g + u*b3 + d
    u = throttle * T_max / m(t)
    g = [0, 0, -9.80665]

`d` represents unmodelled aerodynamic acceleration and model error, estimated
from GNSS velocity differences and previous thrust commands with a low-pass
filter. `b3` is estimated from the initial launch attitude and integrated gyro.
No hidden rocket state, future balloon cache, seed, or balloon ID is read by the
agent. The benchmark, outside the agent, may reuse recorded balloon paths.

Scenario 1 exposes thrust=1080 N and body mass=70 kg. Including the supplied
oxidizer, gas and annular solid grain gives approximate wet mass=90.8913 kg,
so initial maximum thrust acceleration is 11.8823 m/s^2. Ignoring aerodynamic
forces, maintaining altitude leaves at most

    a_horizontal = sqrt((T/m)^2 - g^2) = 6.7096 m/s^2.

Thus a constant-altitude turn at 30 m/s needs radius at least 134.14 m and takes
28.09 seconds per circle. At 10 m/s the corresponding radius is 14.90 m.
These are instantaneous point-mass estimates, not exact six-DOF bounds: mass,
wind, fins, attitude lag and height change affect the real flight.

The old controller divides requested thrust acceleration by a fixed 12 and
normalizes the direction even when the magnitude cannot be produced. That can
discard needed vertical support. `allocate_acceleration` preserves the requested
vertical component within available thrust, then limits horizontal acceleration:

    u_z = clip(requested_z, 0, T/m)
    u_xy_max = min(sqrt((T/m)^2-u_z^2), u_z*tan(max_tilt)).

Saturation is logged as a model/tracking limitation, not treated as successful
tracking of the originally requested trajectory.

## Interception rather than pursuit of a moving point

For each currently released candidate and candidate arrival time H, predict
balloon position with the **currently observed** velocity:

    p_target(H) = p_balloon + v_balloon * H.

A just-released balloon's near-zero velocity is replaced with the median velocity
of already moving balloons. This remains an approximation; wind shear and balloon
acceleration make long predictions less accurate.

Choose endpoint velocity equal to balloon velocity by default. This deliberately
reduces relative speed at the intercept and sacrifices throughput for precision.
An explicit `arrival_speed` experiment can add a through-target component.
With `lookahead_velocity=true`, its direction points to the nearest predicted
released neighbour. Its speed is bounded using the level-flight lateral reserve
and `v^2 <= a_horizontal * neighbour_distance`, allowing half that distance for
braking. The full sampled force/rate checks still decide whether to accept it;
this one-neighbour heuristic does not optimize the entire route.

For fixed endpoint position, velocity and acceleration, minimizing
`integral ||p'''(t)||^2 dt` gives `p''''''(t)=0` via the Euler-Lagrange equation:
the solution is a quintic. Use normalized time s=t/H and coefficients c0...c5.
With c0=p0, c1=H*v0, c2=H^2*a0/2, define:

    dp = p_target - c0 - c1 - c2
    dv = H*v_target - c1 - 2*c2
    da = -2*c2                 # terminal acceleration zero
    c3 = 10*dp - 4*dv + da/2
    c4 = -15*dp + 7*dv - da
    c5 = 6*dp - 3*dv + da/2.

At 33 samples along each candidate, reject excessive thrust, excessive tilt,
excessive thrust-axis rate, excessive throttle rate, ground crossings and plans
that require powered flight after burnout. This is **sampled feasibility**, not
a continuous-time guarantee. The earliest feasible candidate among up to twelve
nearby released balloons is selected. Replanning is every 0.4 s; a target is held
while a feasible intercept plan remains current. This is a receding-horizon
heuristic, not a multi-target trajectory optimizer.

## Feedback and actuation

Track the reference using feedforward acceleration and critically damped PD:

    a_command = a_reference + wn^2*(p_reference-p) + 2*wn*(v_reference-v).

For b3=thrust_vector/||thrust_vector||, use
`omega_ff = b3 cross jerk / ||thrust_vector||` as the nominal angular-rate
feedforward (neglecting the disturbance derivative). Geometric attitude error is
the cross product of actual and desired axes. Convert angular acceleration to
torque using an approximate inertia, then to TVC using `torque=T*lever*sin(delta)`.
TVC outputs are in **degrees**, with the simulator's angle and rate limits.
The inertia estimate includes a parallel-axis tank contribution; it is not a
complete aerodynamic attitude model. Feedback and the six-DOF tests remain vital.

## Why chemistry is not the optimization variable here

The official scenario fixes propellant densities, grain geometry, thrust source,
mass-flow schedules and burn duration. `balloon_world.py` passes fixed tank flow
and burn windows to HybridMotor. In the pinned `flight.py`, throttle multiplies
thrust while mass-flow and burnout remain time-scheduled. Reducing throttle does
not extend that 30-second window in this environment. Therefore modifying fuel
chemistry is not an agent action, and a real-engine fuel-saving argument cannot
be assumed to apply. The new controller estimates the scheduled mass decrease;
it does not change the official vehicle or scoring parameters.

## Verification and reproducibility

`tests/test_physics_guidance.py` checks polynomial endpoint conditions, preservation
of vertical support under lateral saturation, rejection of impossible thrust and
post-burnout plans, circular-motion demand, finite prelaunch actions, and actuator
rate limits. All six checks passed on 2026-09-07 using standard-library unittest.

Run from the repository root with the project environment:

```powershell
$env:MPLCONFIGDIR = Join-Path $env:TEMP 'bpc-mpl-cache'
$env:PYTHONWARNINGS = 'ignore'
.\.venv\Scripts\python.exe -m unittest tests\test_physics_guidance.py -v
.\.venv\Scripts\python.exe scripts\benchmark_cached_field.py BalloonPoppingGymEnv\evaluation\configs\physics_scenario_1.yaml BalloonPoppingGymEnv\evaluation\results\20260907T144737Z_trajectory.json --report BalloonPoppingGymEnv\evaluation\results\physics_full_seed0_metrics.json
.\.venv\Scripts\python.exe BalloonPoppingGymEnv\evaluation\evaluate.py BalloonPoppingGymEnv\evaluation\configs\physics_scenario_1.yaml
.\.venv\Scripts\python.exe scripts\evaluate_guidance.py BalloonPoppingGymEnv\evaluation\configs\physics_scenario_1.yaml --seeds 0 1 --report BalloonPoppingGymEnv\evaluation\results\physics_fresh_seed0_seed1_metrics.json
```

The cached benchmark is a development shortcut, **not a submission generator**.
Its cached source must match scenario and seed, and it cannot run beyond the
recorded field horizon. Final score claims must distinguish cached-field results
from a fresh official-environment evaluation. The empirical five-point comparator
is preserved in `multi_target_5point_reference.yaml`; it is explicitly seed-specific.

The former constant-speed geometric route search is not evidence of 20-point
physical feasibility. Neither the cited papers nor the new controller establish
that 20 points are reachable in this specific field and burn window.

## Controlled experiment results (2026-09-07)

Same Scenario 1 / seed 0 recorded field, unmodified rocket and scoring dynamics:

| Controller/config | Score | Pop times (seconds) |
| --- | ---: | --- |
| Empirical five-point reference | 5 | 9.48, 10.34, 15.89, 21.11, 27.57 |
| Physics quintic, matched arrival velocity | 4 | 13.49, 17.88, 23.60, 31.15 |
| Physics quintic, neighbour-directed arrival velocity | 3 | 17.84, 24.86, 31.48 |

The new method has **not** beaten the tuned reference. The exit-velocity ablation
changed the first feasible target and did not improve throughput in this field;
retain matched arrival velocity as the default research baseline. Scores alone
cannot establish robustness. Fresh evaluations with explicit seeds are saved by
`scripts/evaluate_guidance.py` separately from cached-field metrics.
