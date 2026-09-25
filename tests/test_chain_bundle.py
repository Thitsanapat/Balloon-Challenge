"""Standalone submission must preserve development planner calculations."""

import ast
import unittest
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents import chain_beam_agent as development
from BalloonPoppingGymEnv.agents import chain_launch_agent as launch
from BalloonPoppingGymEnv.agents import submission_chain_launch_v1 as bundled


class ChainBundleTests(unittest.TestCase):
    def test_numerical_chain_matches_development_source(self):
        args = (np.array([0., 0., 20.]), np.array([1., 0., 2.]),
                np.array([.1, 0., 1.]),
                np.array([[5., 2., 40.], [12., -3., 55.]]),
                [3., 2.], np.array([.5, 0., 2.]))
        np.testing.assert_array_equal(development.free_chain(*args), bundled.free_chain(*args))

    def test_launch_candidates_match_development_source(self):
        args = (np.array([[20., 0., 45., 1., 0., 2.],
                          [-30., 0., 30., 0., 0., 1.]]),
                [0, 1], np.array([0., 0., 20.]), 20.)
        np.testing.assert_array_equal(launch.launch_axes(*args), bundled.launch_axes(*args))

    def test_bundle_has_no_local_agent_dependencies(self):
        tree = ast.parse(Path(bundled.__file__).read_text(encoding='utf-8'))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        agent_imports = [name for name in imports if name.startswith('BalloonPoppingGymEnv')]
        self.assertEqual(agent_imports, ['BalloonPoppingGymEnv.agents.base_agent'])


if __name__ == '__main__':
    unittest.main()
