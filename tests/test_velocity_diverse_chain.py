import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.submission_time_allocation_v1 import ChainBeamAgent
from BalloonPoppingGymEnv.agents.velocity_diverse_chain_agent import VelocityDiverseAgent
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class VelocityDiverseTests(unittest.TestCase):
    def setUp(self):
        _, given = load_scenario_parameters(1)
        self.agent = VelocityDiverseAgent(given)

    def test_distinct_exit_velocity_survives_pair_quota(self):
        self.agent.beam_width = 3
        def node(duration, velocity):
            return ((1, 2), (1., duration), None, np.zeros(3), np.array([velocity, 0., 0.]))
        children = [node(1., 0.), node(2., 1.), node(3., 8.)]
        kept = self.agent._retain_diverse(children)
        self.assertEqual([n[4][0] for n in kept], [0., 8.])

    def test_disabled_and_launch_probe_use_original_joint_chain(self):
        args = ({}, np.zeros(3), np.zeros(3), 24.)
        for bin_size, selected in ((0., True), (4., False)):
            self.agent.exit_velocity_bin = bin_size
            self.agent.launch_selected = selected
            with patch.object(ChainBeamAgent, '_make_plan') as parent:
                self.agent._make_plan(*args)
            parent.assert_called_once_with(self.agent, *args)

    def test_short_first_leg_only_after_launch_settles(self):
        self.agent.exit_velocity_bin = 0.
        self.agent.first_leg_min = 1.
        self.agent.launched = False
        launch = self.agent.launch_time
        self.assertEqual(self.agent._first_leg_start(launch+1.), 2.)
        self.agent.launched = True
        self.assertEqual(self.agent._first_leg_start(launch+.5), 2.)
        self.assertEqual(self.agent._first_leg_start(launch+.51), 1.)

    def test_zero_velocity_bin_retains_original_beam_pruning(self):
        self.agent.exit_velocity_bin = 0.
        self.agent.beam_width = 3
        def node(duration, velocity):
            return ((1, 2), (1., duration), None, np.zeros(3), np.array([velocity, 0., 0.]))
        kept = self.agent._retain_diverse([node(1., 0.), node(2., 1.), node(3., 8.)])
        self.assertEqual([n[4][0] for n in kept], [0., 1.])

    def test_invalid_settings(self):
        _, given = load_scenario_parameters(1)
        for kwargs in (dict(exit_velocity_bin=-1), dict(exit_velocity_bin=np.nan),
                       dict(pair_quota=0), dict(pair_quota=1.5),
                       dict(first_leg_min=.4)):
            with self.assertRaises(ValueError):
                VelocityDiverseAgent(given, **kwargs)


if __name__ == '__main__':
    unittest.main()
