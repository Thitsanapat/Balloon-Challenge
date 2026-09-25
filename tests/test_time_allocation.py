import unittest
from unittest.mock import patch
import numpy as np
from tests.test_constrained_chain import model
from BalloonPoppingGymEnv.agents.time_allocation_agent import optimize_times, TimeAllocationAgent
from BalloonPoppingGymEnv.agents.final_approach_agent import FinalApproachAgent
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import batch_samples


class TimeAllocationTests(unittest.TestCase):
    def test_shortens_and_preserves_moving_intercepts_and_continuity(self):
        agent = model()
        agent.acceleration = np.zeros(3)
        agent.terminal_weight = 0.
        agent.leg_horizon = 10.
        states = np.array([[10., 0., 20., 1., 0., 0.], [20., 0., 20., 1., 0., 0.]])
        saved = states.copy()
        initial = np.array([4., 4.])
        result = optimize_times(agent, (0, 1), initial, states,
                                np.array([0., 0., 20.]), np.array([5., 0., 0.]), 0., iterations=35)
        self.assertIsNotNone(result)
        times, curves, _, _ = result
        self.assertLess(sum(times), 7.)
        p, v, a, _ = batch_samples(curves, times, 65)
        np.testing.assert_allclose(p[:, -1], states[:, :3]+states[:, 3:]*np.cumsum(times)[:, None], atol=1e-8)
        for values in (p, v, a):
            np.testing.assert_allclose(values[0, -1], values[1, 0], atol=1e-8)
        self.assertTrue(np.all(agent._valid(curves, times, np.array([[0.], [times[0]]]), samples=65)[0]))
        np.testing.assert_array_equal(states, saved)
        np.testing.assert_array_equal(initial, [4., 4.])

    def test_invalid_horizon_rejected(self):
        agent = model()
        for ts in ([31.], [np.nan], [.1]):
            self.assertIsNone(optimize_times(agent, (0,), ts, np.zeros((1, 6)), np.zeros(3), np.zeros(3), 0.))

    def test_disabled_is_exact_parent_passthrough(self):
        agent = TimeAllocationAgent.__new__(TimeAllocationAgent)
        agent.route_events = []
        agent.timing_candidates = 0
        with patch.object(FinalApproachAgent, '_make_plan') as parent:
            obs, p, v = {}, np.zeros(3), np.zeros(3)
            agent._make_plan(obs, p, v, 0.)
            parent.assert_called_once_with(obs, p, v, 0.)

    def test_failed_optimization_preserves_executable_plan(self):
        agent = TimeAllocationAgent.__new__(TimeAllocationAgent)
        agent.route_events = []
        agent.route, agent.deadlines = [0], [4.]
        agent.ends_v, agent.ends_a = [np.ones(3)], [np.zeros(3)]
        agent.plan = np.array([42.])
        agent.timing_candidates, agent.timing_iterations, agent.timing_safety = 4, 20, 0.
        agent.diagnostics = dict(timing_solves=0, timing_adopted=0, timing_extra_planned_hits=0)
        states = np.array([[10., 0., 20., 0., 0., 0.]])
        with patch.object(FinalApproachAgent, '_make_plan', side_effect=lambda *args:
                agent.route_events.append([0., [0], 4.])), patch(
                'BalloonPoppingGymEnv.agents.time_allocation_agent.optimize_times', return_value=None):
            agent._make_plan(dict(balloon_states=states, balloon_status=np.ones(1)),
                             np.array([0., 0., 20.]), np.zeros(3), 0.)
        self.assertEqual(agent.route, [0])
        self.assertEqual(agent.deadlines, [4.])
        np.testing.assert_array_equal(agent.plan, [42.])
        self.assertEqual(agent.diagnostics['timing_adopted'], 0)

    def test_nonfinite_solver_result_falls_back_to_feasible_nominal(self):
        from types import SimpleNamespace
        agent = model()
        agent.acceleration, agent.terminal_weight, agent.leg_horizon = np.zeros(3), 0., 10.
        states = np.array([[10., 0., 20., 0., 0., 0.]])
        with patch('BalloonPoppingGymEnv.agents.time_allocation_agent.minimize',
                   return_value=SimpleNamespace(x=np.array([np.nan]))):
            result = optimize_times(agent, (0,), [4.], states,
                                    np.array([0., 0., 20.]), np.zeros(3), 0.)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result[0], [4.])


if __name__ == '__main__':
    unittest.main()
