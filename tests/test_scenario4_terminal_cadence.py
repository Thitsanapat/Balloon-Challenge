"""Terminal cadence uses observation timing and restores ordinary settings."""

import unittest
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_terminal_cadence_agent import (
    Scenario4TerminalCadenceAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)


class TerminalCadenceTests(unittest.TestCase):
    def make_agent(self):
        agent = Scenario4TerminalCadenceAgent.__new__(Scenario4TerminalCadenceAgent)
        agent.replan_interval = .4
        agent.terminal_replan_interval = .2
        agent.terminal_replan_start = 3.
        agent.terminal_replan_stop = .6
        agent.plan = object()
        agent.target_index = 1
        agent.plan_start = 10.
        agent.plan_duration = 5.
        agent.next_plan = 13.4
        agent.diagnostics = {"terminal_cadence_steps": 0}
        return agent

    def test_shortens_only_live_terminal_window(self):
        agent = self.make_agent()
        seen = []

        def parent(_, observation):
            seen.append((agent.replan_interval, agent.next_plan))
            return "action"

        with patch.object(Scenario4PureJointRefreshAgent,
                          "get_action", parent):
            result = agent.get_action({"simulation_time": 12.5,
                                       "balloon_status": np.array([0, 1])})
        self.assertEqual(result, "action")
        self.assertEqual(seen, [(.2, 12.7)])
        self.assertEqual(agent.replan_interval, .4)
        self.assertEqual(agent.diagnostics["terminal_cadence_steps"], 1)

    def test_no_change_if_target_is_gone(self):
        agent = self.make_agent()
        with patch.object(Scenario4PureJointRefreshAgent,
                          "get_action", return_value="action"):
            agent.get_action({"simulation_time": 12.5,
                              "balloon_status": np.array([0, 0])})
        self.assertEqual(agent.next_plan, 13.4)
        self.assertEqual(agent.replan_interval, .4)
        self.assertEqual(agent.diagnostics["terminal_cadence_steps"], 0)


if __name__ == "__main__":
    unittest.main()
