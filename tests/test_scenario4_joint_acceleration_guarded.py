import unittest
from pathlib import Path

import yaml

from BalloonPoppingGymEnv.agents.scenario4_acceleration_wind_agent import (
    Scenario4AccelerationWindAgent, _AugmentedProfile,
)
from BalloonPoppingGymEnv.agents.scenario4_guarded_innovation_agent import (
    Scenario4GuardedInnovationAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_joint_acceleration_guarded_agent import (
    Scenario4JointAccelerationGuardedAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_reachable_refresh_agent import (
    Scenario4ReachableRefreshAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class JointAccelerationGuardedTests(unittest.TestCase):
    def test_both_observation_hooks_and_route_rescue_dispatch(self):
        parameters = load_scenario_parameters(4)[1]
        config = yaml.safe_load(Path(
            'BalloonPoppingGymEnv/evaluation/configs/'
            'scenario4_joint_acceleration_guarded_launch42.yaml').read_text())
        agent = Scenario4JointAccelerationGuardedAgent(
            parameters, **config['agent_kwargs'])
        self.assertIsInstance(agent.wind_profile, _AugmentedProfile)
        self.assertIn('acceleration_augmented_calls', agent.diagnostics)
        self.assertIn('guarded_applied_steps', agent.diagnostics)
        self.assertIn('reach_infeasible_joint_refresh', agent.diagnostics)
        self.assertIs(agent.get_action.__func__,
                      Scenario4AccelerationWindAgent.get_action)
        self.assertIs(agent._attitude_action.__func__,
                      Scenario4GuardedInnovationAgent._attitude_action)
        self.assertIs(agent._rescue_committed_leg.__func__,
                      Scenario4ReachableRefreshAgent._rescue_committed_leg)


if __name__ == '__main__':
    unittest.main()
