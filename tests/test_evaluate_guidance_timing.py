"""Development timing separates environment reset from the later episode."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import evaluate_guidance


class _OneStepEnv:
    def __init__(self, *, render_mode, parameters):
        self.parameters = parameters
        self.closed = False

    def reset(self, *, seed):
        assert self.parameters["scenario"]["random_seed"] == seed
        return _observation(0.0), {}

    def step(self, action):
        return _observation(1.0), 0, True, False, {"popped_count": 0}

    def close(self):
        self.closed = True


def _observation(now):
    return {
        "simulation_time": now,
        "balloon_states": np.zeros((1, 6)),
        "balloon_status": np.zeros(1, dtype=int),
        "rocket_sensors": np.zeros(12),
    }


class _OneStepAgent:
    def __init__(self, given, **kwargs):
        self.given = given

    def get_action(self, observation):
        return {"throttle": 0.0}


class GuidanceTimingTests(unittest.TestCase):
    def test_reset_and_post_reset_time_partition_total_wall_time(self):
        config = {
            "scenario_number": 4,
            "agent_module_path": str(Path(__file__)),
            "agent_class_name": "_OneStepAgent",
        }
        with (patch.object(evaluate_guidance, "BalloonPoppingEnv", _OneStepEnv),
              patch.object(evaluate_guidance, "load_scenario_parameters",
                           return_value=({"scenario": {}}, {"environment": {"elevation": 0.0}})),
              patch.object(evaluate_guidance, "_load_agent_class", return_value=_OneStepAgent),
              patch.object(evaluate_guidance, "agent_source_fingerprints", return_value={}),
              patch.object(evaluate_guidance.time, "monotonic", side_effect=[100.0, 104.0, 113.0])):
            result = evaluate_guidance.evaluate(config, seed=7)

        self.assertEqual(result["reset_wall_seconds"], 4.0)
        self.assertEqual(result["post_reset_wall_seconds"], 9.0)
        self.assertEqual(result["wall_seconds"], 13.0)
        self.assertEqual(result["score"], 0)
        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])


if __name__ == "__main__":
    unittest.main()
