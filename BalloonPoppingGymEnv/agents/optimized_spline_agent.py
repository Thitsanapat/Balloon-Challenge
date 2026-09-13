"""Refine the reference pair route's times and endpoint derivatives with SQP."""

import numpy as np

from BalloonPoppingGymEnv.agents.beam_intercept_agent import batch_samples
from BalloonPoppingGymEnv.agents.joint_beam_agent import JointBeamAgent
from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent, two_intercept_spline


class OptimizedSplineAgent(SplineRouteAgent):
    # Shared numerical optimizer; no beam search or committed-route policy here.
    _curves = JointBeamAgent._curves
    _margins = JointBeamAgent._margins
    _solve = JointBeamAgent._solve

    def __init__(self, given_parameters, solver_iterations=40, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.solver_iterations = int(solver_iterations)
        self.optimize_derivatives = True
        self.endpoint_velocities = self.endpoint_accelerations = None
        self.diagnostics.update(joint_calls=0, joint_accepted=0)

    def _make_plan(self, observation, position, velocity, now):
        old_target = self.target_index
        previous = len(self.route_events)
        super()._make_plan(observation, position, velocity, now)
        if len(self.route_events) == previous:
            return
        _, first, second, t1, total = self.route_events[-1]
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1)==1)
        drift = states[:, 3:6].copy()
        moving = released[np.linalg.norm(drift[released], axis=1)>.5]
        if moving.size:
            drift[np.linalg.norm(drift, axis=1)<=.5] = np.median(drift[moving], axis=0)
        times = np.array([t1, total-t1])
        curves = np.asarray(two_intercept_spline(position, velocity, self.acceleration,
            states[first, :3]+drift[first]*t1, states[second, :3]+drift[second]*total,
            drift[second], *times))
        _, vs, acs, _ = batch_samples(curves, times)
        self.endpoint_velocities, self.endpoint_accelerations = vs[:, -1], acs[:, -1]
        solved = self._solve(position, velocity, states, drift, [first, second], times,
                              now, old_target == first)
        if solved is not None:
            curves, times = solved
            self.plan, self.plan_start, self.plan_duration = curves[0], now, float(times[0])
