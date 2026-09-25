import unittest
from pathlib import Path
import numpy as np
from BalloonPoppingGymEnv.agents import submission_final_approach_v1 as bundle
from BalloonPoppingGymEnv.agents.final_approach_agent import FinalApproachAgent
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from scripts.audit_agent_submission import source_review


class FinalBundleTests(unittest.TestCase):
    def test_standalone_source_review_and_parent(self):
        source = Path(bundle.__file__).read_text(encoding='utf-8')
        _, flags = source_review(source)
        self.assertEqual(flags, [])
        self.assertTrue(issubclass(bundle.ChainSubmissionAgent, bundle.FinalApproachAgent))

    def test_initial_actions_and_final_hold_match_development(self):
        _, given = load_scenario_parameters(1)
        kwargs = dict(launch_time=24., max_tilt=75., max_axis_rate=1.2, terminal_weight=0.)
        frozen, development = bundle.ChainSubmissionAgent(given, **kwargs), FinalApproachAgent(given, **kwargs)
        observation = dict(simulation_time=0., rocket_sensors=np.full(12, np.nan),
                           balloon_states=np.zeros((1, 6)), balloon_status=np.ones(1))
        left, right = frozen.get_action(observation), development.get_action(observation)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        for agent in (frozen, development):
            agent.plan, agent.target_index = np.zeros((6, 3)), 0
            agent.plan_start, agent.plan_duration = 0., 10.
            agent._make_plan(observation, np.zeros(3), np.zeros(3), 9.88)
            self.assertEqual(agent.diagnostics['final_approach_holds'], 1)
            self.assertEqual(agent.plan_duration, 10.)


if __name__ == '__main__':
    unittest.main()
