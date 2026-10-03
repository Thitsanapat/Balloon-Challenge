"""Checks for the optional reserve-aware active-leg validator."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from BalloonPoppingGymEnv.agents.chain_beam_agent import ChainBeamAgent
from BalloonPoppingGymEnv.agents.scenario4_reserve_refresh_agent import (
    Scenario4ReserveRefreshAgent,
)


def model():
    agent = Scenario4ReserveRefreshAgent.__new__(Scenario4ReserveRefreshAgent)
    agent.thrust = 200.
    agent.dry_mass = 10.
    agent.initial_mass = 10.
    agent.mass_flow = 0.
    agent.launch_time = 0.
    agent.burn_time = 30.
    agent.disturbance = np.zeros(3)
    agent.max_tilt = np.radians(75.)
    agent.max_axis_rate = 1.2
    agent.reserve = 0.8
    agent.elevation = 0.
    agent.control = {'throttle_rate_limit': 2.}
    agent.diagnostics = dict(reserve_refresh_checks=0,
                             reserve_refresh_rejected=0,
                             dense_refresh_rejected=0)
    return agent


def constant_vertical_acceleration(value):
    curve = np.zeros((6, 3))
    curve[0, 2] = 20.
    curve[2, 2] = 2. * value  # duration=2: a=2*c[2]/duration**2
    return curve


class Scenario4ReserveRefreshTests(unittest.TestCase):
    def test_default_hook_uses_existing_feasibility_result(self):
        agent = ChainBeamAgent.__new__(ChainBeamAgent)
        curve = np.zeros((6, 3))
        with patch.object(agent, '_feasible', return_value=True) as feasible:
            self.assertTrue(agent._accept_committed_curve(curve, 2., 0.))
            feasible.assert_called_once_with(curve, 2., 0.)

    def test_rejects_curve_that_uses_reserved_thrust(self):
        agent = model()
        curve = constant_vertical_acceleration(10.)
        self.assertTrue(agent._feasible(curve, 2., 0.))
        self.assertFalse(agent._accept_committed_curve(curve, 2., 0.))
        self.assertEqual(agent.diagnostics['reserve_refresh_checks'], 1)
        self.assertEqual(agent.diagnostics['reserve_refresh_rejected'], 1)

    def test_accepts_curve_with_margin_and_skips_invalid_nominal(self):
        agent = model()
        self.assertTrue(agent._accept_committed_curve(
            constant_vertical_acceleration(9.), 2., 0.))
        self.assertEqual(agent.diagnostics['reserve_refresh_checks'], 1)
        self.assertEqual(agent.diagnostics['dense_refresh_rejected'], 0)
        self.assertFalse(agent._accept_committed_curve(
            constant_vertical_acceleration(12.), 2., 0.))
        self.assertEqual(agent.diagnostics['reserve_refresh_checks'], 1)

    def test_config_changes_only_agent_identity(self):
        configs = Path(__file__).resolve().parents[1] / 'BalloonPoppingGymEnv/evaluation/configs'
        original = yaml.safe_load((configs / 'scenario4_wind_profile_reserve08.yaml').read_text())
        variant = yaml.safe_load((configs / 'scenario4_reserve_refresh_reserve08.yaml').read_text())
        self.assertEqual(variant['scenario_number'], original['scenario_number'])
        self.assertEqual(variant['agent_kwargs'], original['agent_kwargs'])
        self.assertEqual(variant['agent_class_name'], 'Scenario4ReserveRefreshAgent')


if __name__ == '__main__':
    unittest.main()
