import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import yaml
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.scenario4_joint_innovation_agent import Scenario4JointInnovationAgent
from BalloonPoppingGymEnv.agents.scenario4_joint_refresh_agent import Scenario4JointRefreshAgent
from BalloonPoppingGymEnv.agents.scenario4_forecast_innovation_agent import Scenario4ForecastInnovationAgent
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import Scenario4WindProfileAgent


class JointInnovationTests(unittest.TestCase):
    def setUp(self):
        self.given = load_scenario_parameters(4)[1]
        self.config = yaml.safe_load(Path('BalloonPoppingGymEnv/evaluation/configs/'
                                         'scenario4_joint_innovation_launch42.yaml').read_text())

    def test_both_components_are_initialized_and_correct_hooks_selected(self):
        a = Scenario4JointInnovationAgent(self.given, **self.config['agent_kwargs'])
        self.assertFalse(a.joint_refresh_allow_shortening)
        self.assertIn('joint_refresh_attempts', a.diagnostics)
        self.assertIn('innovation_applied_steps', a.diagnostics)
        self.assertIs(a._rescue_committed_leg.__func__, Scenario4JointRefreshAgent._rescue_committed_leg)
        self.assertIs(a._attitude_action.__func__, Scenario4ForecastInnovationAgent._attitude_action)

    def test_zero_strength_preserves_full_route_controller_commands(self):
        config = yaml.safe_load(Path('BalloonPoppingGymEnv/evaluation/configs/'
                                    'scenario4_joint_refresh_full_route_launch42.yaml').read_text())
        plain = Scenario4JointRefreshAgent(self.given, **config['agent_kwargs'])
        combined = Scenario4JointInnovationAgent(self.given, innovation_strength=0,
                                                **config['agent_kwargs'])
        curve = np.zeros((6, 3))
        curve[0] = [1, 0, 20]
        for a in [plain, combined]:
            a.plan, a.plan_start, a.plan_duration = curve.copy(), 42., 5.
            a.filtered_position = np.array([0., 0., 10.])
            a.target_index = 0
        combined._innovation_snapshot = (44., np.array([1]),
                                         np.array([[9., 0, 20, 1, 0, 0]]))
        args = (np.array([0., 0., 1.]), np.zeros(3), 11., np.zeros(3), 13.)
        for _ in range(10):
            for left, right in zip(plain._attitude_action(*args), combined._attitude_action(*args)):
                np.testing.assert_array_equal(left, right)

    def test_observation_snapshot_is_cleared_even_on_exception(self):
        a = Scenario4JointInnovationAgent(self.given)
        obs = dict(simulation_time=44., balloon_states=np.zeros((1, 6)),
                   balloon_status=np.ones(1))
        with patch.object(Scenario4WindProfileAgent, 'get_action', side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                a.get_action(obs)
        self.assertIsNone(a._innovation_snapshot)


if __name__ == '__main__':
    unittest.main()
