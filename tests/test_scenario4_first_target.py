"""Bounded first-target search and state isolation for Scenario 4."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_first_target_agent import (
    Scenario4FirstTargetAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_recoverable_launch_agent import (
    Scenario4RecoverableLaunchAgent,
)


def bare_agent():
    agent = Scenario4FirstTargetAgent.__new__(Scenario4FirstTargetAgent)
    agent.first_target_trials = 2
    agent.first_target_depth = 3
    agent.first_target_beam_width = 8
    agent.first_target_branch_targets = 6
    agent.continuation_credit = 0.12
    agent.robustness_floor = 0.05
    agent.first_time_cost = 0.08
    agent._forced_first_index = None
    agent.first_target_comparisons = []
    agent.first_target_selected = None
    agent.target_events = []
    agent.route = []
    agent.deadlines = []
    agent.plan = None
    agent.search_depth = 14
    agent.beam_width = 12
    agent.branch_targets = 8
    agent.timing_candidates = 4
    agent.failed_until = {}
    return agent


def observation():
    states = np.zeros((11, 6))
    states[:, 0] = 5. + 5. * np.arange(len(states))
    return {'balloon_states': states,
            'balloon_status': np.ones(len(states), dtype=int)}


class FirstTargetTests(unittest.TestCase):
    def test_force_applies_only_at_depth_zero_and_respects_cooldown(self):
        agent = bare_agent()
        obs = observation()
        states = obs['balloon_states']
        released = np.arange(len(states))
        agent._forced_first_index = 2
        self.assertEqual(agent._candidate_ids(
            states, released, (), 0., np.zeros(3), np.zeros(3), 0.), [2])
        self.assertEqual(agent._candidate_ids(
            states, released, (2,), 2., np.zeros(3), np.zeros(3), 0.)[0], 0)
        agent.failed_until[2] = 1.
        self.assertEqual(agent._candidate_ids(
            states, released, (), 0., np.zeros(3), np.zeros(3), 0.), [])

    def test_two_alternatives_can_beat_long_nominal_route(self):
        agent = bare_agent()
        obs = observation()
        calls = []

        def fake_parent_plan(observation, position, velocity, now):
            forced = agent._forced_first_index
            calls.append((forced, agent.search_depth, agent.beam_width,
                          agent.branch_targets, agent.timing_candidates))
            if forced is None:
                agent.route = list(range(9))
            else:
                agent.route = [forced, 9, 10]
            agent.deadlines = [now + 5. + i for i in range(len(agent.route))]
            agent.plan = np.full((6, 3), -1. if forced is None else forced)

        agent._correction_fraction = lambda duration, now: {
            0: 0.1, 1: 0.9, 2: 0.4,
        }[agent.route[0]]
        with patch.object(Scenario4RecoverableLaunchAgent, '_make_plan',
                          side_effect=fake_parent_plan):
            agent._make_plan(obs, np.zeros(3), np.zeros(3), 0.)

        self.assertEqual([call[0] for call in calls], [None, 1, 2])
        self.assertEqual(calls[0][1:], (14, 12, 8, 4))
        self.assertEqual(calls[1][1:], (3, 8, 6, 0))
        self.assertEqual(calls[2][1:], (3, 8, 6, 0))
        self.assertEqual(agent.route[0], 1)
        self.assertEqual(agent.first_target_selected, 1)
        self.assertEqual(len(agent.first_target_comparisons), 3)
        self.assertEqual(agent._forced_first_index, None)
        self.assertEqual((agent.search_depth, agent.beam_width,
                          agent.branch_targets, agent.timing_candidates),
                         (14, 12, 8, 4))
        np.testing.assert_array_equal(agent.plan, np.ones((6, 3)))

    def test_later_replans_do_not_repeat_first_target_search(self):
        agent = bare_agent()
        agent.route = [4]
        agent.target_events = [(42., 4)]
        with patch.object(Scenario4RecoverableLaunchAgent, '_make_plan') as parent:
            agent._make_plan(observation(), np.zeros(3), np.zeros(3), 43.)
        parent.assert_called_once()

    def test_fallback_to_another_target_cannot_win_forced_trial(self):
        agent = bare_agent()
        agent.first_target_trials = 1
        agent._correction_fraction = lambda duration, now: 0.5

        def fake_parent_plan(observation, position, velocity, now):
            # Imitate a joint-beam miss followed by the parent's fallback.
            agent.route = [0]
            agent.deadlines = [now + 5.]
            agent.plan = np.zeros((6, 3))

        with patch.object(Scenario4RecoverableLaunchAgent, '_make_plan',
                          side_effect=fake_parent_plan):
            agent._make_plan(observation(), np.zeros(3), np.zeros(3), 0.)
        self.assertEqual(agent.route, [0])
        self.assertEqual(agent.first_target_comparisons[1], (1, None))
        self.assertEqual(agent._forced_first_index, None)

    def test_state_restored_if_forced_trial_fails(self):
        agent = bare_agent()
        agent.first_target_trials = 1
        agent._correction_fraction = lambda duration, now: 0.5

        def fake_parent_plan(observation, position, velocity, now):
            if agent._forced_first_index is not None:
                raise RuntimeError('unexpected planner error')
            agent.route = [0]
            agent.deadlines = [now + 5.]
            agent.plan = np.zeros((6, 3))

        with patch.object(Scenario4RecoverableLaunchAgent, '_make_plan',
                          side_effect=fake_parent_plan):
            with self.assertRaisesRegex(RuntimeError, 'unexpected'):
                agent._make_plan(observation(), np.zeros(3), np.zeros(3), 0.)
        self.assertEqual(agent.route, [0])
        self.assertEqual(agent._forced_first_index, None)
        self.assertEqual(agent.search_depth, 14)


if __name__ == '__main__':
    unittest.main()
