"""Reachability checks for observation-only Scenario 4 launch ranking."""

import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_recoverable_launch_agent import (
    Scenario4RecoverableLaunchAgent,
)


def bare_agent():
    agent = Scenario4RecoverableLaunchAgent.__new__(
        Scenario4RecoverableLaunchAgent)
    agent.correction_radii = np.array((1.5, 3.0, 5.0))
    agent.correction_fractions = np.array((0.5, 0.75))
    agent.continuation_credit = 0.12
    agent.robustness_floor = 0.05
    agent.first_time_cost = 0.08
    agent.route = [0]
    agent.deadlines = [6.0]
    agent.plan = np.zeros((6, 3))
    agent.plan[2, 2] = 18.0  # Constant +1 m/s^2 for a six-second leg.
    agent.launch_time = 0.0
    agent.burn_time = 20.0
    agent.thrust = 20.0
    agent.dry_mass = 1.0
    agent.initial_mass = 1.0
    agent.mass_flow = 0.0
    agent.disturbance = np.zeros(3)
    agent.reserve = 0.0
    agent.max_tilt = np.radians(75.0)
    agent.max_axis_rate = 1.2
    agent.elevation = 0.0
    agent.control = {'throttle_rate_limit': 10.0}
    return agent


class RecoverableLaunchTests(unittest.TestCase):
    def test_counterfactuals_use_dense_physics_checks(self):
        agent = bare_agent()
        fraction = agent._correction_fraction(6.0, 0.0)
        self.assertGreaterEqual(fraction, 0.0)
        self.assertLessEqual(fraction, 1.0)
        self.assertAlmostEqual(fraction * 48, round(fraction * 48))
        self.assertGreater(fraction, 0.0)
        # The score must react to the same public thrust and turn-rate bounds
        # that limit the executable trajectory, not merely target distance.
        tight_thrust = bare_agent()
        tight_thrust.thrust = 11.0
        self.assertLess(tight_thrust._correction_fraction(6.0, 0.0),
                        fraction)
        slow_turn = bare_agent()
        slow_turn.max_axis_rate = 0.2
        self.assertLess(slow_turn._correction_fraction(6.0, 0.0),
                        fraction)

    def test_low_recoverability_can_outweigh_one_nominal_hit(self):
        agent = bare_agent()
        observation = {'balloon_states': np.zeros((1, 6)),
                       'balloon_status': np.ones(1, dtype=int)}
        agent._correction_fraction = lambda duration, now: 0.0
        uncertain = agent._launch_candidate_key(None, 8, 20.0,
                                                observation, 0.0)
        agent._correction_fraction = lambda duration, now: 1.0
        recoverable = agent._launch_candidate_key(None, 7, 20.0,
                                                  observation, 0.0)
        self.assertLess(recoverable, uncertain)

    def test_equal_recoverability_retains_more_route_credit(self):
        agent = bare_agent()
        agent._correction_fraction = lambda duration, now: 0.5
        short = agent._launch_candidate_key(None, 4, 20.0, {}, 0.0)
        long = agent._launch_candidate_key(None, 5, 20.0, {}, 0.0)
        self.assertLess(long, short)

    def test_empty_plan_and_invalid_duration_rank_last(self):
        agent = bare_agent()
        agent.route = []
        agent.plan = None
        self.assertEqual(agent._launch_candidate_key(
            None, 0, float('inf'), {}, 0.0)[0], 1)
        agent.route = [0]
        agent.plan = np.zeros((6, 3))
        agent.deadlines = [0.0]
        self.assertEqual(agent._launch_candidate_key(
            None, 1, 10.0, {}, 0.0)[0], 1)


if __name__ == '__main__':
    unittest.main()
