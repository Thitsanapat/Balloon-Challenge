"""Numerical planner tests independent of a cached balloon field."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.beam_intercept_agent import batch_samples
from BalloonPoppingGymEnv.agents.joint_beam_agent import JointBeamAgent
from BalloonPoppingGymEnv.agents.optimized_spline_agent import OptimizedSplineAgent
from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent, two_intercept_spline
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class JointBeamTests(unittest.TestCase):
    def setUp(self):
        _, given = load_scenario_parameters(1)
        self.agent = JointBeamAgent(given, launch_time=0., solver_iterations=40)
        self.agent.acceleration = np.zeros(3)
        self.agent.endpoint_velocities = np.array([[0., 0., 2.], [0., 0., 2.]])
        self.agent.endpoint_accelerations = np.zeros((2, 3))
        self.states = np.zeros((2, 6))
        self.states[:, 2] = [106., 112.]
        self.position = np.array([0., 0., 100.])
        self.velocity = np.array([0., 0., 2.])

    def test_curve_chain_has_continuous_derivatives(self):
        curves = self.agent._curves(self.position, self.velocity, self.states,
                                    self.states[:, 3:], [0, 1], np.array([3., 3.]))
        samples = batch_samples(curves, [3., 3.], 33)
        for values in samples[:3]:
            np.testing.assert_allclose(values[0, -1], values[1, 0], atol=1e-9)
        np.testing.assert_allclose(samples[0][:, -1], self.states[:, :3], atol=1e-9)
        self.assertGreaterEqual(np.min(self.agent._margins(curves, [3., 3.], 10.)), 0.)

    def test_optimizer_keeps_feasible_seed_and_does_not_increase_time(self):
        result = self.agent._solve(self.position, self.velocity, self.states,
                                   self.states[:, 3:], [0, 1], np.array([3., 3.]), 10., False)
        self.assertIsNotNone(result)
        curves, durations = result
        self.assertLessEqual(float(sum(durations)), 6.+1e-8)
        self.assertLess(float(sum(durations)), 5.)
        self.assertGreaterEqual(np.min(self.agent._margins(curves, durations, 10.)), -1e-5)
        samples = batch_samples(curves, durations, 65)
        np.testing.assert_allclose(samples[0][:, -1], self.states[:, :3], atol=1e-8)
        for values in samples[:3]:
            np.testing.assert_allclose(values[0, -1], values[1, 0], atol=1e-8)

    def test_burnout_constraint_rejects_late_route(self):
        curves = self.agent._curves(self.position, self.velocity, self.states,
                                    self.states[:, 3:], [0, 1], np.array([3., 3.]))
        self.assertLess(np.min(self.agent._margins(curves, [3., 3.], 29.)), 0.)

    def test_reference_policy_refinement_preserves_observation(self):
        _, given = load_scenario_parameters(1)
        agent = OptimizedSplineAgent(given, launch_time=0.)
        agent.acceleration = np.zeros(3)
        observation = {'balloon_states':self.states.copy(), 'balloon_status':np.ones(2)}
        original = observation['balloon_states'].copy()
        def reference(instance, obs, p, v, now):
            instance.plan = two_intercept_spline(p, v, instance.acceleration,
                self.states[0, :3], self.states[1, :3], np.zeros(3), 3., 3.)[0]
            instance.target_index = 0
            instance.plan_start, instance.plan_duration = now, 3.
            instance.route_events.append([now, 0, 1, 3., 6.])
        with patch.object(SplineRouteAgent, '_make_plan', reference):
            agent._make_plan(observation, self.position, self.velocity, 10.)
        self.assertEqual(agent.diagnostics['joint_calls'], 1)
        self.assertEqual(agent.diagnostics['joint_accepted'], 1)
        np.testing.assert_array_equal(observation['balloon_states'], original)
        endpoint = batch_samples(agent.plan[None], [agent.plan_duration])[0][0, -1]
        np.testing.assert_allclose(endpoint, self.states[0, :3], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
