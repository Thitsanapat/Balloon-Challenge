import ast
import unittest

from scripts.build_scenario4_adaptive_entry_submission_agent import (
    build_source,
)


class AdaptiveEntryBuildTests(unittest.TestCase):
    def test_frozen_sources_form_self_contained_agent(self):
        source = build_source()
        tree = ast.parse(source)
        names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn('Scenario4PureJointRefreshAgent', names)
        self.assertIn('Scenario4EntryHorizonAgent', names)
        self.assertIn('Scenario4AdaptiveEntryAgent', names)
        imported_agents = [node.module for node in ast.walk(tree)
                           if isinstance(node, ast.ImportFrom) and node.module and
                           node.module.startswith('BalloonPoppingGymEnv.agents.')]
        self.assertEqual(imported_agents, ['BalloonPoppingGymEnv.agents.base_agent'])
        compile(source, '<test-standalone>', 'exec')


if __name__ == '__main__':
    unittest.main()
