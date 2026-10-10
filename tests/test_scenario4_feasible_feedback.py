"""Focused behavior checks for the isolated Scenario 4 feedback controller."""

import numpy as np
import unittest

from BalloonPoppingGymEnv.agents.scenario4_feasible_feedback_agent import (
    Scenario4FeasibleFeedbackAgent,
    feasible_feedback_gain,
)
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    G,
    allocate_acceleration,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def check_feasible_gain_keeps_feedback_inside_published_force_envelope():
    gains = np.array([0.7, 1.0, 1.4])
    reference_a = np.zeros(3)
    position_error = np.array([3.0, 0.0, 0.0])
    gain, residual, feasible = feasible_feedback_gain(
        reference_a, position_error, np.zeros(3), np.zeros(3),
        available=10.0, max_tilt=np.radians(75.0), gains=gains,
    )
    assert feasible
    assert gain == 0.7
    assert residual <= 0.1
    requested = reference_a + gain**2 * position_error - G
    np.testing.assert_allclose(
        allocate_acceleration(requested, 10.0, np.radians(75.0)),
        requested,
    )


def check_feasible_gain_uses_fastest_candidate_when_headroom_exists():
    gain, _, feasible = feasible_feedback_gain(
        np.zeros(3), np.array([0.1, 0.0, 0.0]), np.zeros(3),
        np.zeros(3), available=15.0, max_tilt=np.radians(75.0),
        gains=np.array([0.7, 1.0, 1.4]),
    )
    assert feasible
    assert gain == 1.4


def check_no_feasible_candidate_selects_least_lost_force():
    gains = np.array([0.7, 1.0, 1.4])
    gain, residual, feasible = feasible_feedback_gain(
        np.zeros(3), np.array([8.0, 0.0, 0.0]), np.zeros(3),
        np.zeros(3), available=10.0, max_tilt=np.radians(75.0),
        gains=gains,
    )
    assert not feasible
    assert gain == 0.7
    assert residual > 0.1


def check_live_reference_update_reduces_gain_without_modifying_plan():
    _, given = load_scenario_parameters(4)
    agent = Scenario4FeasibleFeedbackAgent(given)
    position = np.array([0.0, 0.0, 100.0])
    agent.plan = np.zeros((6, 3))
    agent.plan[0] = position
    original = agent.plan.copy()
    agent.plan_duration = 2.0
    agent.plan_start = 42.0
    agent.launched = True
    agent.launch_time = 42.0
    agent._update_feedback_gain(42.0, position - [8.0, 0.0, 0.0], np.zeros(3))
    assert agent.feedback_min_frequency <= agent.tracking_frequency < 1.3
    assert agent.diagnostics["feedback_updates"] == 1
    np.testing.assert_array_equal(agent.plan, original)


def check_invalid_frequency_range_is_rejected():
    _, given = load_scenario_parameters(4)
    with unittest.TestCase().assertRaises(ValueError):
        Scenario4FeasibleFeedbackAgent(
            given, feedback_min_frequency=1.4, feedback_max_frequency=0.7,
        )


class FeasibleFeedbackTests(unittest.TestCase):
    def test_force_envelope(self):
        check_feasible_gain_keeps_feedback_inside_published_force_envelope()

    def test_fastest_feasible_gain(self):
        check_feasible_gain_uses_fastest_candidate_when_headroom_exists()

    def test_least_saturated_fallback(self):
        check_no_feasible_candidate_selects_least_lost_force()

    def test_live_reference_update(self):
        check_live_reference_update_reduces_gain_without_modifying_plan()

    def test_invalid_settings(self):
        check_invalid_frequency_range_is_rejected()


if __name__ == "__main__":
    unittest.main()
