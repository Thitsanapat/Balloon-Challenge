import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_trackability_agent import (
    Scenario4TrackabilityAgent,
    first_leg_tracking_risk,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class SimpleVehicle:
    disturbance = np.zeros(3)
    launch_time = 0.0
    dry_mass = initial_mass = 1.0
    mass_flow = 0.0
    thrust = 15.0
    max_tilt = np.radians(75.0)
    max_axis_rate = 1.2
    control = {"throttle_rate_limit": 1.0}


class TrackabilityTests(unittest.TestCase):
    def test_more_lateral_acceleration_costs_more(self):
        # Constant accelerations with identical duration and vertical motion.
        easy = np.zeros((6, 3))
        hard = np.zeros((6, 3))
        hard[2, 0] = 16.0  # a_x = 8 m/s^2 over the 2-second leg
        vehicle = SimpleVehicle()
        easy_risk = first_leg_tracking_risk(easy, 2.0, 0.0, vehicle)
        hard_risk = first_leg_tracking_risk(hard, 2.0, 0.0, vehicle)
        self.assertTrue(0 <= easy_risk < hard_risk <= 1)

    def test_bad_curve_is_rejected(self):
        with self.assertRaises(ValueError):
            first_leg_tracking_risk(np.zeros((6, 3)), 0, 0, SimpleVehicle())

    def test_zero_weight_preserves_original_node_priority(self):
        _, given = load_scenario_parameters(4)
        agent = Scenario4TrackabilityAgent(
            given, launch_candidates=0, timing_candidates=0,
            trackability_time_weight=0)
        curve = np.zeros((6, 3))
        node = ((1,), (2.0,), np.array([curve]), np.zeros(3), np.zeros(3))
        self.assertEqual(agent._node_priority(node, None, None, 0), 2.0)
        self.assertEqual(agent.diagnostics["trackability_ranked_nodes"], 0)


if __name__ == "__main__":
    unittest.main()
