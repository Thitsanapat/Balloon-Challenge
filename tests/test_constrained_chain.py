import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from BalloonPoppingGymEnv.agents.constrained_chain_agent import (
    ConstrainedChainAgent, constraint_margins, repair_derivatives, repair_margins,
)
from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain, free_chain_matrices
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent
from BalloonPoppingGymEnv.agents.momentum_beam_agent import MomentumBeamAgent
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import batch_samples


def model():
    agent = SimpleNamespace(thrust=200., dry_mass=10., initial_mass=10., mass_flow=0.,
        disturbance=np.zeros(3), launch_time=0., burn_time=30., max_tilt=np.radians(75.),
        max_axis_rate=1.2, reserve=0., elevation=0., control={'throttle_rate_limit':2.})
    agent._valid = lambda *args, **kwargs: MomentumBeamAgent._valid(agent, *args, **kwargs)
    return agent


class ConstrainedChainTests(unittest.TestCase):
    def test_safety_margin_only_tightens_and_preserves_initial_constraints(self):
        agent = model()
        durations = np.array([2., 2.])
        curve = free_chain(np.array([0., 0., 20.]), np.array([5., 0., 0.]), np.zeros(3),
                           np.array([[10., 0., 20.], [20., 0., 20.]]), durations, np.zeros(3))
        original = constraint_margins(agent, curve, durations, 0., 17)
        robust = repair_margins(agent, curve, durations, 0., 17, .03)
        self.assertTrue(np.all(robust <= original))
        for block in (0, 2, 3):
            self.assertEqual(robust[block*34], original[block*34])
            self.assertAlmostEqual(original[(block+1)*34-1]-robust[(block+1)*34-1], .03)
        np.testing.assert_array_equal(original[4*34:5*34], robust[4*34:5*34])

    def test_repair_preserves_positions_and_continuity_and_makes_invalid_curve_feasible(self):
        agent = model()
        durations = np.array([2., 2.])
        targets = np.array([[10., 0., 20.], [20., 0., 20.]])
        curve = free_chain(np.array([0., 0., 20.]), np.array([5., 0., 0.]),
                           np.zeros(3), targets, durations, np.zeros(3))
        designs, _, _ = free_chain_matrices(tuple(durations), 0.)
        curve[:, :, 0] += designs[:, :, 0]*8.
        before = curve.copy()
        starts = np.array([[0.], [2.]])
        self.assertFalse(np.all(agent._valid(curve, durations, starts, samples=65)[0]))
        repaired = repair_derivatives(agent, curve, durations, 0., iterations=45)
        self.assertIsNotNone(repaired)
        self.assertTrue(np.all(agent._valid(repaired, durations, starts, samples=65)[0]))
        p, v, a, _ = batch_samples(repaired, durations)
        np.testing.assert_allclose(p[:, -1], targets, atol=1e-8)
        np.testing.assert_allclose(v[0, 0], [5., 0., 0.], atol=1e-8)
        np.testing.assert_allclose(a[0, 0], 0., atol=1e-8)
        for values in (p, v, a):
            np.testing.assert_allclose(values[:-1, -1], values[1:, 0], atol=1e-8)
        np.testing.assert_array_equal(curve, before)

    def test_margin_sign_matches_physical_validator(self):
        agent = model()
        durations = np.array([2., 2.])
        curve = free_chain(np.array([0., 0., 20.]), np.array([5., 0., 0.]), np.zeros(3),
                           np.array([[10., 0., 20.], [20., 0., 20.]]), durations, np.zeros(3))
        designs, _, _ = free_chain_matrices(tuple(durations), 0.)
        for offset in (0., 1., 5., 10.):
            trial = curve.copy()
            trial[:, :, 0] += designs[:, :, 0]*offset
            margins = constraint_margins(agent, trial, durations, 0., samples=65)
            valid = agent._valid(trial, durations, np.array([[0.], [2.]]), samples=65)[0]
            self.assertEqual(bool(np.all(margins >= 0.)), bool(np.all(valid)))

    def test_burn_deadline_and_nonfinite_inputs_are_rejected(self):
        agent = model()
        curve = np.zeros((1, 6, 3))
        curve[0, 0, 2] = 20.
        self.assertIsNone(repair_derivatives(agent, curve, [31.], 0.))
        curve[0, 1, 0] = np.nan
        self.assertIsNone(repair_derivatives(agent, curve, [2.], 0.))

    def test_failed_repair_keeps_original_plan_and_observations(self):
        agent = ConstrainedChainAgent.__new__(ConstrainedChainAgent)
        agent.route_events = []
        agent.route, agent.deadlines = [0], [10.]
        agent.ends_v, agent.ends_a = [np.array([1., 0., 0.])], [np.zeros(3)]
        agent.acceleration, agent.terminal_weight = np.zeros(3), 0.
        agent.repair_candidates, agent.repair_radius, agent.repair_iterations = 4, 18., 5
        agent.repair_interval, agent.next_repair = 0., 0.
        agent.failed_until = {}
        agent.plan = np.array([42.])
        agent.diagnostics = dict(constraint_repairs=0, constraint_repair_accepted=0, constraint_inserted_hits=0)
        states = np.array([[10., 0., 20., 0., 0., 0.], [5., 1., 20., 0., 0., 0.]])
        original = states.copy()
        with patch.object(ChainLaunchAgent, '_make_plan', side_effect=lambda *args:
                agent.route_events.append([0., [0], 10.])), patch(
                'BalloonPoppingGymEnv.agents.constrained_chain_agent.repair_derivatives', return_value=None):
            agent._make_plan(dict(balloon_states=states, balloon_status=np.ones(2)),
                             np.array([0., 0., 20.]), np.array([1., 0., 0.]), 0.)
        self.assertEqual(agent.diagnostics['constraint_repairs'], 1)
        self.assertEqual(agent.route, [0])
        self.assertEqual(agent.deadlines, [10.])
        np.testing.assert_array_equal(agent.plan, [42.])
        np.testing.assert_array_equal(states, original)

    def test_successful_insertion_stores_solved_derivatives_without_extending_finish(self):
        agent = ConstrainedChainAgent.__new__(ConstrainedChainAgent)
        agent.__dict__.update(model().__dict__)
        agent._valid = lambda *args, **kwargs: MomentumBeamAgent._valid(agent, *args, **kwargs)
        agent.route_events = []
        agent.route, agent.deadlines = [0], [10.]
        agent.acceleration, agent.terminal_weight = np.zeros(3), 0.
        agent.repair_candidates, agent.repair_radius, agent.repair_iterations = 4, 18., 10
        agent.repair_interval, agent.next_repair, agent.failed_until = 0., 0., {}
        agent.diagnostics = dict(constraint_repairs=0, constraint_repair_accepted=0, constraint_inserted_hits=0)
        selected = []
        agent._select = lambda *args: selected.append(args)
        states = np.array([[10., 0., 20., 0., 0., 0.], [5., 1., 20., 0., 0., 0.]])
        with patch.object(ChainLaunchAgent, '_make_plan', side_effect=lambda *args:
                agent.route_events.append([0., [0], 10.])):
            agent._make_plan(dict(balloon_states=states, balloon_status=np.ones(2)),
                             np.array([0., 0., 20.]), np.array([1., 0., 0.]), 0.)
        self.assertEqual(agent.route, [1, 0])
        self.assertAlmostEqual(agent.deadlines[-1], 10.)
        self.assertEqual(agent.diagnostics['constraint_repair_accepted'], 1)
        self.assertEqual(len(agent.ends_v), 2)
        _, curve, duration, _ = selected[0]
        p, v, a, _ = batch_samples(curve[None], [duration])
        np.testing.assert_allclose(p[0, -1], states[1, :3], atol=1e-8)
        np.testing.assert_allclose(v[0, -1], agent.ends_v[0], atol=1e-8)
        np.testing.assert_allclose(a[0, -1], agent.ends_a[0], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
