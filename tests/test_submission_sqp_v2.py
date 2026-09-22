"""Submission bundle preserves behavior and the observation-only boundary."""
import ast
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.optimized_spline_agent import OptimizedSplineAgent
from BalloonPoppingGymEnv.agents.capture_sqp_agent import CaptureSQPAgent
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import SubmissionAgent, get_initial_attitude
from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude as official_attitude
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def test_bundle_imports_only_math_and_official_base_class():
    path = Path('BalloonPoppingGymEnv/agents/submission_sqp_v2.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    allowed = {'functools', 'numpy', 'scipy.optimize',
               'BalloonPoppingGymEnv.agents.base_agent'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name in allowed for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module in allowed
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {'open', 'exec', 'eval', '__import__'}


def test_commanded_attitude_convention_matches_official():
    for inclination in (0, 35, 67, 90):
        for heading in (0, 29, 90, 180, 270, 359):
            np.testing.assert_allclose(get_initial_attitude(inclination, heading),
                                       official_attitude(inclination, heading), atol=1e-15)


def test_bundle_matches_controller_without_mutating_inputs():
    _, given = load_scenario_parameters(1)
    original_given = copy.deepcopy(given)
    kwargs = dict(launch_time=24, replan_interval=.8, first_candidates=3,
                  next_candidates=2, solver_iterations=4)
    baseline = OptimizedSplineAgent(copy.deepcopy(given), **kwargs)
    bundle = SubmissionAgent(given, **kwargs)
    states = np.zeros((100, 6))
    states[:, 2] = 20
    states[:3] = [[2, 0, 30, 0, 0, 3], [5, 1, 36, 0, 0, 3],
                  [-4, 2, 40, 0, 0, 3]]
    status = np.zeros((100, 1), dtype=int)
    status[:3] = 1
    states.setflags(write=False)
    status.setflags(write=False)
    for now in (0, 24, 24.01, 24.02, 24.82, 25.62):
        sensors = (np.full(12, np.nan) if now <= 24 else
                   np.array([0, 0, 0, 0, 0, 12, 0, 0, 21, 0, 0, 2.]))
        sensors.setflags(write=False)
        obs = dict(simulation_time=now, balloon_states=states,
                   balloon_status=status, rocket_sensors=sensors)
        expected = baseline.get_action(obs)
        actual = bundle.get_action(obs)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)
            assert np.all(np.isfinite(actual[key]))
    assert given == original_given


def test_capture_sqp_aim_stays_inside_real_radius_without_changing_observation():
    _, given = load_scenario_parameters(1)
    agent = CaptureSQPAgent(given, capture_fraction=.5)
    states = np.array([[10., 0., 30., 0., 0., 2.], [0., 0., 20., 0., 0., 0.]])
    before = states.copy()
    states.setflags(write=False)
    observation = {'balloon_states': states, 'balloon_status': np.ones((2, 1))}
    with patch.object(OptimizedSplineAgent, '_make_plan') as planner:
        agent._make_plan(observation, np.array([0., 0., 20.]), np.zeros(3), 24.)
    planned = planner.call_args.args[0]['balloon_states']
    np.testing.assert_array_equal(states, before)
    np.testing.assert_array_equal(planned[:, 3:], states[:, 3:])
    assert np.all(np.linalg.norm(planned[:, :3] - states[:, :3], axis=1) < agent.radius)
