"""The optional deadline recovery must preserve the old plan on failure."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.chain_beam_agent import ChainBeamAgent
from BalloonPoppingGymEnv.agents.scenario4_deadline_rescue_agent import (
    Scenario4DeadlineRescueAgent,
)


def active_leg(agent_type, old_endpoint=100.0):
    agent = agent_type.__new__(agent_type)
    agent.route = [0, 1]
    agent.route_events = []
    agent.deadlines = [1.0, 5.0]
    agent.ends_v = [np.zeros(3), np.zeros(3)]
    agent.ends_a = [np.zeros(3), np.zeros(3)]
    agent.commitment = True
    agent.target_index = 0
    agent.plan = np.zeros((6, 3))
    agent.plan[0] = [old_endpoint, 0., 20.]
    agent.plan_start = 0.0
    agent.plan_duration = 1.0
    agent.acceleration = np.zeros(3)
    agent.disturbance = np.zeros(3)
    agent.radius = 1.5
    agent.launch_time = 0.0
    agent.burn_time = 30.0
    agent.leg_horizon = 10.0
    agent.thrust = 200.0
    agent.dry_mass = 10.0
    agent.initial_mass = 10.0
    agent.mass_flow = 0.0
    agent.max_tilt = np.radians(75.)
    agent.max_axis_rate = 1.2
    agent.reserve = 0.8
    agent.elevation = 0.0
    agent.control = {'throttle_rate_limit': 2.0}
    agent.target_events = []
    agent.diagnostics = dict(plans=0, committed_refresh_failed=0,
                             deadline_rescue_attempts=0,
                             deadline_rescue_accepted=0)
    if agent_type is Scenario4DeadlineRescueAgent:
        agent.profile_max_range = 0.0  # The wind-profile mixin needs no peers here.
        agent.timing_candidates = 0
        agent.final_approach_guard = 0.0
        agent.rescue_max_remaining = 6.0
        agent.rescue_trigger_radius = 1.0
        agent.rescue_offsets = np.array([0.0, 1.0, 2.0])
    return agent


def observation():
    return {'balloon_states': np.array([
        [10., 0., 20., 0., 0., 0.],
        [20., 0., 20., 0., 0., 0.],
    ]), 'balloon_status': np.ones(2, dtype=int)}


class Scenario4DeadlineRescueTests(unittest.TestCase):
    def test_default_chain_keeps_previous_plan_when_refresh_fails(self):
        agent = active_leg(ChainBeamAgent)
        old_plan = agent.plan.copy()
        with patch.object(agent, '_feasible', return_value=False):
            agent._make_plan(observation(), np.array([0., 0., 20.]),
                             np.array([5., 0., 0.]), 0.)
        self.assertEqual(agent.route, [0, 1])
        self.assertEqual(agent.deadlines, [1., 5.])
        np.testing.assert_array_equal(agent.plan, old_plan)

    def test_rescue_accepts_dense_checked_later_intercept_and_drops_stale_tail(self):
        agent = active_leg(Scenario4DeadlineRescueAgent)
        observed = observation()
        saved = observed['balloon_states'].copy()
        with patch.object(agent, '_feasible', return_value=False):
            agent._make_plan(observed, np.array([0., 0., 20.]),
                             np.array([5., 0., 0.]), 0.)
        self.assertEqual(agent.diagnostics['committed_refresh_failed'], 1)
        self.assertEqual(agent.diagnostics['deadline_rescue_attempts'], 1)
        self.assertEqual(agent.diagnostics['deadline_rescue_accepted'], 1)
        self.assertEqual(agent.route, [0])
        self.assertEqual(agent.deadlines, [3.])
        self.assertEqual(agent.plan_duration, 3.)
        np.testing.assert_array_equal(observed['balloon_states'], saved)
        np.testing.assert_allclose(agent.plan.sum(axis=0), [10., 0., 20.], atol=1e-8)
        self.assertTrue(agent._valid(agent.plan[None, :, :], np.array([3.]),
                                     np.array([[0.]]), samples=65)[0][0])

    def test_near_target_and_failed_candidates_keep_previous_plan(self):
        for near, reject_dense in ((True, False), (False, True)):
            with self.subTest(near=near, reject_dense=reject_dense):
                agent = active_leg(Scenario4DeadlineRescueAgent,
                                   old_endpoint=10. if near else 100.)
                old_plan = agent.plan.copy()
                with patch.object(agent, '_feasible', return_value=False):
                    if reject_dense:
                        with patch.object(agent, '_valid', return_value=(
                                np.array([False]), None, None, None)):
                            agent._make_plan(observation(), np.array([0., 0., 20.]),
                                             np.array([5., 0., 0.]), 0.)
                    else:
                        agent._make_plan(observation(), np.array([0., 0., 20.]),
                                         np.array([5., 0., 0.]), 0.)
                self.assertEqual(agent.route, [0, 1])
                self.assertEqual(agent.deadlines, [1., 5.])
                self.assertEqual(agent.diagnostics['deadline_rescue_accepted'], 0)
                np.testing.assert_array_equal(agent.plan, old_plan)

    def test_reserve_margin_can_reject_otherwise_valid_candidate(self):
        agent = active_leg(Scenario4DeadlineRescueAgent)
        with patch.object(agent, '_feasible', return_value=False), patch(
                'BalloonPoppingGymEnv.agents.scenario4_deadline_rescue_agent.repair_margins',
                return_value=np.array([-0.01])):
            agent._make_plan(observation(), np.array([0., 0., 20.]),
                             np.array([5., 0., 0.]), 0.)
        self.assertEqual(agent.diagnostics['deadline_rescue_accepted'], 0)
        self.assertEqual(agent.route, [0, 1])


if __name__ == '__main__':
    unittest.main()
