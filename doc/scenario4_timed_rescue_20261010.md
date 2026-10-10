# Scenario 4: bounded later-intercept rescue (10 October 2026)

The selected agent replans a multi-balloon spline from each observation. In
development flights on seeds 0 and 2 it attempted a same-deadline joint
repair 43 and 39 times but accepted 4 times in each flight. Those counters
show an opportunity to check an alternative intercept time; they do not prove
that the rejected cases were capturable.

`Scenario4TimedRescueAgent` tries 0.4, 0.8 and 1.2 seconds of extra time on
the first leg **only after** the same-deadline repair failed. It retains the
observed target order, shifts later deadlines by the same amount, rebuilds
the entire spline from the current estimated rocket state and current observed
balloon states, and accepts only the first candidate passing the inherited
thrust, tilt and rate feasibility checks at 65 samples per leg. It does not
extend past public burn time. This is a small receding-horizon timing search,
not a continuous optimal-control solution or a guarantee of collision.

The agent does not read seeds, simulator hidden state, future trajectory,
diagnostic truth or stored actions. The official Scenario 4 evaluator and
physics are untouched. Rule interpretation follows the organizer's
[README](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/tree/v0.2.2)
and [Discussion #165](https://github.com/ARRC-Rocket/BalloonPoppingChallenge/discussions/165).

Two focused tests check that a feasible later intercept preserves target
order and that an unreleased/popped active target cannot trigger it. The
reproducible development script `scripts.validate_timed_rescue` uses fresh
official simulator processes on already-used seeds 0 and 2, at most two CPU
workers and no GPU. It compares with completed, hash-verified pure-joint
flights (6 and 4 pops) and writes a frozen source hash and criterion before
flying. Predeclared screen: at least +2 pops combined, no seed worse by more
than one pop, and the new rescue accepted at least once. Passing requires
separate independent paired seeds, source packaging and submission parity
before considering a live switch.

The local qualification launcher still selects the previously verified
standalone `submission_scenario4_pure_joint_v2.py`. A development score by
itself does not change this choice.

## Result

The complete official flights scored 5 and 4 versus 6 and 4 for the current
pure-joint agent. A later-intercept rescue was accepted once on seed 0 and
twice on seed 2. This fails the predeclared score gate. Keep the current
qualification agent; do not retune the delay list on these development seeds.
Detailed credential-free reports are in
`evaluation/results/timed_rescue_dev_20261010/`.
