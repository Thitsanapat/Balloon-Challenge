import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.bridge_chain_agent import BridgeChainAgent
from BalloonPoppingGymEnv.agents.submission_time_allocation_v1 import (
    ChainSubmissionAgent, MomentumBeamAgent, free_chain, batch_samples,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from tests.test_constrained_chain import model


def physical_agent(**kwargs):
    _, given = load_scenario_parameters(1)
    agent = BridgeChainAgent(given, **kwargs)
    agent.__dict__.update(model().__dict__)
    # The fixture's lambda is bound to its SimpleNamespace; bind this instance.
    agent._valid = lambda *args, **kw: MomentumBeamAgent._valid(agent, *args, **kw)
    agent.acceleration, agent.terminal_weight = np.zeros(3), 0.
    agent.leg_horizon, agent.time_grid = 3., 1.
    return agent


class BridgeChainTests(unittest.TestCase):
    def test_disabled_is_exact_parent_passthrough(self):
        agent = BridgeChainAgent.__new__(BridgeChainAgent)
        agent.bridge_budget = 0
        observation, position, velocity = {}, np.zeros(3), np.zeros(3)
        with patch.object(ChainSubmissionAgent, '_make_plan') as parent:
            agent._make_plan(observation, position, velocity, 0.)
        parent.assert_called_once_with(observation, position, velocity, 0.)

    def test_rejected_two_waypoint_prefix_can_have_feasible_third_waypoint(self):
        agent = physical_agent(bridge_budget=128, bridge_width=3)
        position, velocity = np.array([0., 0., 20.]), np.array([5., 0., 1.])
        targets = np.array([
            [8.550103400180685, 2.3385759372317465, 32.46948684425898],
            [26.180261248466735, -2.9559548889404725, 25.202986002465984],
            [44.61355126718051, 3.159801945861926, 25.60979435710162],
        ])
        states = np.c_[targets, np.zeros((3, 3))]
        saved = states.copy()
        one = free_chain(position, velocity, np.zeros(3), targets[:1], [3.], np.zeros(3), 0.)
        valid, _, vs, acs = agent._valid(one, np.array([3.]), np.array([[0.]]), samples=65)
        self.assertTrue(np.all(valid))
        two = free_chain(position, velocity, np.zeros(3), targets[:2], [3., 2.], np.zeros(3), 0.)
        self.assertFalse(np.all(agent._valid(two, np.array([3., 2.]), np.array([[0.], [3.]]), samples=65)[0]))
        agent.route, agent.deadlines = [0], [3.]
        agent.ends_v, agent.ends_a = list(vs), list(acs)
        # Fixed proposals isolate prefix pruning, not the ranking heuristic.
        agent._candidate_ids = lambda states, released, route, *args: [len(route)] if len(route) < 3 else []
        agent._bridge_times = lambda prefix, remaining, now: [2.] if prefix else [3.]
        result = agent._bridge_search(states, np.arange(3), position, velocity, 0.)
        self.assertIsNotNone(result)
        ids, durations, curves, _, _ = result
        self.assertEqual(ids, (0, 1, 2))
        self.assertEqual(durations, (3., 2., 2.))
        self.assertGreater(agent.diagnostics['bridge_recovered'], 0)
        starts = np.array([[0.], [3.], [5.]])
        self.assertTrue(np.all(agent._valid(curves, np.array(durations), starts, samples=65)[0]))
        p, v, a, _ = batch_samples(curves, durations, 65)
        np.testing.assert_allclose(p[:, -1], targets, atol=1e-8)
        for values in (p, v, a):
            np.testing.assert_allclose(values[:-1, -1], values[1:, 0], atol=1e-8)
        np.testing.assert_array_equal(states, saved)
        self.assertEqual(agent.route, [0])  # search itself cannot execute its proposals

    def test_only_released_targets_budget_and_burn_bound_are_respected(self):
        agent = physical_agent(bridge_budget=7, bridge_width=2)
        agent.burn_time = 4.1
        agent.failed_until[2] = 100.
        states = np.array([[8., 0., 22., 0., 0., 0.], [1., 0., 20., 0., 0., 0.],
                           [2., 0., 20., 0., 0., 0.], [18., 0., 24., 0., 0., 0.]])
        real = agent._chain_curves
        calls = []
        def check(p, v, targets, durations, drift):
            calls.append(tuple(durations))
            self.assertLessEqual(sum(durations), agent.burn_time-.0001)
            self.assertTrue(all(any(np.array_equal(target, states[i, :3]) for i in (0, 3))
                                for target in targets))
            return real(p, v, targets, durations, drift)
        agent._chain_curves = check
        agent._bridge_search(states, np.array([0, 2, 3]), np.array([0., 0., 20.]), np.zeros(3), 0.)
        self.assertGreater(len(calls), 0)
        self.assertLessEqual(len(calls), 7)
        self.assertEqual(agent.diagnostics['bridge_candidates'], len(calls))

    def test_dense_failure_preserves_incumbent(self):
        agent = physical_agent(bridge_budget=16)
        original = agent._valid
        def reject_dense(*args, samples=17):
            valid, p, v, a = original(*args, samples=samples)
            if samples == 65:
                valid[:] = False
            return valid, p, v, a
        agent._valid = reject_dense
        states = np.array([[10., 0., 20., 0., 0., 0.]])
        result = agent._bridge_search(states, np.array([0]), np.array([0., 0., 20.]), np.array([5., 0., 0.]), 0.)
        self.assertIsNone(result)
        self.assertGreater(agent.diagnostics['bridge_dense_rejected'], 0)
        self.assertEqual(agent.route, [])

    def test_root_can_continue_to_three_targets_and_short_first_leg_matters(self):
        states = np.array([[5., 0., 20., 0., 0., 0.], [10., 0., 20., 0., 0., 0.],
                           [15., 0., 20., 0., 0., 0.]])
        counts = []
        for first_min in (2., 1.):
            agent = physical_agent(bridge_budget=256, bridge_width=3,
                                   bridge_first_leg_min=first_min, search_depth=3)
            agent.launched = True
            agent.burn_time = 4.1  # At now=1, three 1-second legs still fit.
            result = agent._bridge_search(states, np.arange(3), np.array([0., 0., 20.]),
                                          np.array([5., 0., 0.]), 1.)
            counts.append(len(result[0]) if result else 0)
            self.assertLessEqual(agent.diagnostics['bridge_candidates'], 256)
            if first_min == 1.:
                self.assertEqual(result[0], (0, 1, 2))
                self.assertEqual(result[1], (1., 1., 1.))
        self.assertLess(counts[0], 3)
        self.assertEqual(counts[1], 3)

    def test_invalid_chain_cannot_receive_two_recovery_successors(self):
        agent = physical_agent(bridge_budget=128, search_depth=6)
        lengths = []
        original = agent._valid
        def always_invalid(curves, *args, **kwargs):
            lengths.append(len(curves))
            valid, p, v, a = original(curves, *args, **kwargs)
            valid[:] = False
            return valid, p, v, a
        agent._valid = always_invalid
        states = np.array([[5.*i, 0., 20., 0., 0., 0.] for i in range(1, 7)])
        result = agent._bridge_search(states, np.arange(6), np.array([0., 0., 20.]),
                                      np.array([5., 0., 0.]), 0.)
        self.assertIsNone(result)
        self.assertTrue(lengths)
        self.assertLessEqual(max(lengths), 2)

    def test_commitment_does_not_trigger_extra_search(self):
        agent = physical_agent()
        with patch.object(ChainSubmissionAgent, '_make_plan'), patch.object(agent, '_bridge_search') as search:
            agent._make_plan({}, np.zeros(3), np.zeros(3), 0.)
        search.assert_not_called()

    def test_launch_axis_probes_do_not_trigger_extra_search(self):
        agent = physical_agent()
        agent.launch_selected = False
        def parent(*args):
            agent.diagnostics['beam_searches'] += 1
        with patch.object(ChainSubmissionAgent, '_make_plan', side_effect=parent), \
                patch.object(agent, '_bridge_search') as search:
            agent._make_plan({}, np.zeros(3), np.zeros(3), 0.)
        search.assert_not_called()

    def test_first_leg_grid_only_changes_after_launch(self):
        agent = physical_agent(bridge_first_leg_min=.5)
        self.assertEqual(agent._bridge_times((), 10., 0.)[0], 2.)
        self.assertEqual(agent._bridge_times((), 10., 10.)[0], 2.)
        agent.launched = True
        self.assertEqual(agent._bridge_times((), 10., .5)[0], 2.)
        self.assertEqual(agent._bridge_times((), 10., .51)[0], .5)
        self.assertEqual(agent._bridge_times((0,), 10., .51)[0], 1.)

    def test_tight_budget_probes_late_grid_and_incumbent_first_time(self):
        agent = physical_agent(bridge_budget=12, search_depth=3)
        agent.leg_horizon, agent.time_grid = 10., .5
        states = np.array([[5.*i, 0., 20., 0., 0., 0.] for i in range(1, 4)])
        first_durations = []
        original_chain, original_valid = agent._chain_curves, agent._valid
        def record(p, v, targets, durations, drift):
            if len(durations) == 1:
                first_durations.append(durations[0])
            return original_chain(p, v, targets, durations, drift)
        def reject(*args, **kwargs):
            valid, p, v, a = original_valid(*args, **kwargs)
            valid[:] = False
            return valid, p, v, a
        agent._chain_curves, agent._valid = record, reject
        agent._bridge_search(states, np.arange(3), np.array([0., 0., 20.]),
                             np.array([5., 0., 0.]), 0.)
        self.assertIn(2., first_durations)
        self.assertIn(10., first_durations)
        self.assertLessEqual(agent.diagnostics['bridge_candidates'], 12)
        agent.deadlines = [8.]
        probes = agent._bridge_probes((), 20., 0., 9, 3)
        np.testing.assert_array_equal(probes, [2., 8., 10.])

    def test_grid_never_exceeds_remaining_burn_due_to_tolerance(self):
        agent = physical_agent()
        self.assertEqual(len(agent._bridge_times((), 1.9999, 0.)), 0)
        np.testing.assert_array_equal(agent._bridge_times((0,), 1.9999, 0.), [1.])

    def test_invalid_limits_rejected(self):
        _, given = load_scenario_parameters(1)
        for settings in [dict(bridge_budget=-1), dict(bridge_budget=4097),
                         dict(bridge_width=1.5), dict(bridge_prefixes=0),
                         dict(bridge_branch_targets=9), dict(bridge_first_leg_min=.49),
                         dict(bridge_first_leg_min=np.nan)]:
            with self.assertRaises(ValueError):
                BridgeChainAgent(given, **settings)


if __name__ == '__main__':
    unittest.main()
