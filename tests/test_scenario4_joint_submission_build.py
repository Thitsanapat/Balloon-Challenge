import ast
import inspect
from pathlib import Path
import textwrap
import unittest

import numpy as np
import yaml

from scripts.build_scenario4_joint_submission_agent import build_source
from BalloonPoppingGymEnv.agents.scenario4_pure_joint_refresh_agent import (
    Scenario4PureJointRefreshAgent,
)
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class JointSubmissionBuildTests(unittest.TestCase):
    def test_live_path_standalone_matches_reproducible_source(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'BalloonPoppingGymEnv/agents/submission_scenario4_pure_joint_v2.py'
        self.assertEqual(path.read_text(encoding='utf-8'), build_source())

    def test_live_path_configuration_preserves_reviewed_policy_parameters(self):
        root = Path(__file__).resolve().parents[1]
        live = yaml.safe_load((root / 'BalloonPoppingGymEnv/evaluation/configs/'
                              'submission_scenario4_pure_joint_launch42.yaml').read_text())
        reviewed = yaml.safe_load((root / 'review/scenario4_pure_joint_candidate_v2.yaml').read_text())
        for key in ('scenario_number', 'agent_class_name', 'agent_kwargs'):
            self.assertEqual(live[key], reviewed[key])
        self.assertTrue((root / live['agent_module_path']).is_file())

    def test_all_inherited_method_bodies_match_current_modular_source(self):
        class RenameProfile(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == 'Scenario4WindProfileAgent':
                    node.id = 'Scenario4SubmissionAgent'
                return node

        classes = {node.name: node for node in ast.parse(build_source()).body
                   if isinstance(node, ast.ClassDef)}
        for cls in Scenario4PureJointRefreshAgent.__mro__:
            if not cls.__module__.startswith('BalloonPoppingGymEnv.agents.'):
                continue
            name = ('Scenario4SubmissionAgent' if cls.__name__ ==
                    'Scenario4WindProfileAgent' else cls.__name__)
            if name == 'BaseAgent':
                continue  # The official base is imported, not bundled.
            packaged = {node.name: node for node in classes[name].body
                        if isinstance(node, ast.FunctionDef)}
            for method_name, function in vars(cls).items():
                if not inspect.isfunction(function):
                    continue
                expected = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
                expected = RenameProfile().visit(expected)
                with self.subTest(cls=cls.__name__, method=method_name):
                    self.assertIn(method_name, packaged)
                    # Dedenting a method changes multiline docstring spacing;
                    # compare executable bodies rather than documentation layout.
                    for node in (expected, packaged[method_name]):
                        if ast.get_docstring(node) is not None:
                            node.body = node.body[1:]
                    self.assertEqual(ast.dump(expected), ast.dump(packaged[method_name]))

    def test_standalone_imports_only_official_base_and_math_libraries(self):
        tree = ast.parse(build_source())
        modules = [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
        self.assertEqual(modules, ['functools', 'scipy.optimize',
                                   'BalloonPoppingGymEnv.agents.base_agent'])

    def test_rescue_produces_same_spline_deadlines_and_derivatives(self):
        namespace = {}
        exec(compile(build_source(), '<joint-build-test>', 'exec'), namespace)
        _, given = load_scenario_parameters(4)
        reference = Scenario4PureJointRefreshAgent(given, launch_candidates=0,
                                                    timing_candidates=0)
        standalone = namespace['Scenario4PureJointRefreshAgent'](
            given, launch_candidates=0, timing_candidates=0)
        observation = dict(balloon_status=np.ones(2, dtype=int),
                           balloon_states=np.array([[0., 0., 30., 0., 0., 1.],
                                                    [3., 0., 40., 0., 0., 1.]]))
        for agent in (reference, standalone):
            agent.route = [0, 1]
            agent.deadlines = [10., 13.]
            agent.launch_time = 4.
            agent.acceleration = np.array([0., 0., 1.])
        outcomes = [a._rescue_committed_leg(observation, np.array([0., 0., 20.]),
                                          np.array([0., 0., 2.]), 7., 3.)
                    for a in (reference, standalone)]
        self.assertEqual(outcomes, [True, True])
        for name in ('plan', 'ends_v', 'ends_a', 'deadlines'):
            np.testing.assert_array_equal(getattr(reference, name), getattr(standalone, name))

    def test_infeasible_committed_leg_dispatches_to_packaged_rescue(self):
        namespace = {}
        exec(compile(build_source(), '<joint-dispatch-test>', 'exec'), namespace)
        _, given = load_scenario_parameters(4)
        agent = namespace['Scenario4PureJointRefreshAgent'](
            given, launch_candidates=0, timing_candidates=0)
        agent.route, agent.deadlines = [0, 1], [10., 13.]
        agent.ends_v = [np.zeros(3), np.zeros(3)]
        agent.ends_a = [np.zeros(3), np.zeros(3)]
        agent.launch_time = 4.
        agent.commitment = True
        agent.acceleration = np.array([0., 0., 1.])
        agent.plan, agent.plan_start, agent.plan_duration = np.zeros((6, 3)), 7., 3.
        agent._feasible = lambda *args: False
        observation = dict(balloon_status=np.ones(2, dtype=int),
                           balloon_states=np.array([[0., 0., 30., 0., 0., 1.],
                                                    [3., 0., 40., 0., 0., 1.]]))
        # The stale bundle used to keep its old plan here without ever calling
        # the subclass's recovery hook, despite passing a direct rescue test.
        namespace['ChainBeamAgent']._make_plan(
            agent, observation, np.array([0., 0., 20.]),
            np.array([0., 0., 2.]), 7.)
        self.assertEqual(agent.diagnostics['pure_joint_accepted'], 1)
        self.assertEqual(agent.deadlines, [10., 13.])


if __name__ == '__main__':
    unittest.main()
