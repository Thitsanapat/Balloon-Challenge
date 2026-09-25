import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.agents.constrained_beam_agent import ConstrainedBeamAgent
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent


class ConstrainedBeamTests(unittest.TestCase):
    def test_repair_budget_and_failure_fallback(self):
        agent = ConstrainedBeamAgent.__new__(ConstrainedBeamAgent)
        agent.beam_repairs_left, agent.beam_repair_depth = 1, 1
        agent.beam_repair_now, agent.beam_repair_violation = 0., .15
        agent.beam_repair_iterations = 2
        agent.beam_repair_safety = 0.
        agent.diagnostics = dict(beam_constraint_repairs=0, beam_constraint_repaired=0)
        original = np.zeros((1, 6, 3))
        with patch.object(ChainLaunchAgent, '_chain_curves', return_value=original), patch(
            'BalloonPoppingGymEnv.agents.constrained_beam_agent.constraint_margins', return_value=np.array([-.1])), patch(
            'BalloonPoppingGymEnv.agents.constrained_beam_agent.repair_derivatives', return_value=None) as solver:
            for _ in range(3):
                result = agent._chain_curves(np.zeros(3), np.zeros(3), np.zeros((1, 3)), [2.], np.zeros(3))
                self.assertIs(result, original)
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(agent.beam_repairs_left, 0)
        self.assertEqual(agent.diagnostics['beam_constraint_repaired'], 0)

    def test_far_infeasible_and_already_feasible_routes_do_not_spend_budget(self):
        for margin in (-.2, .1):
            agent = ConstrainedBeamAgent.__new__(ConstrainedBeamAgent)
            agent.beam_repairs_left, agent.beam_repair_depth = 1, 1
            agent.beam_repair_now, agent.beam_repair_violation = 0., .15
            original = np.zeros((1, 6, 3))
            with patch.object(ChainLaunchAgent, '_chain_curves', return_value=original), patch(
                'BalloonPoppingGymEnv.agents.constrained_beam_agent.constraint_margins', return_value=np.array([margin])):
                self.assertIs(agent._chain_curves(np.zeros(3), np.zeros(3), np.zeros((1, 3)), [2.], np.zeros(3)), original)
            self.assertEqual(agent.beam_repairs_left, 1)


if __name__ == '__main__':
    unittest.main()
