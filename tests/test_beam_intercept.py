import numpy as np
import unittest

from BalloonPoppingGymEnv.agents.beam_intercept_agent import flythrough_curves, batch_samples
from BalloonPoppingGymEnv.agents.physics_guidance_agent import sample_curve


def test_position_only_natural_endpoint_and_continuity():
    p, v, a = np.array([2., 4., 30.]), np.array([3., 0., 2.]), np.array([.1, .2, .3])
    targets = np.array([[15., -4., 40.], [1., 2., 30.]])
    t = np.array([3., 5.])
    curves = flythrough_curves(p, v, a, targets, np.zeros((2, 3)), t, np.zeros(2))
    for i, c in enumerate(curves):
        start = sample_curve(c, t[i], 0.)
        end = sample_curve(c, t[i], t[i])
        for got, expected in zip(start[:3], (p, v, a)):
            np.testing.assert_allclose(got, expected, atol=1e-10)
        np.testing.assert_allclose(end[0], targets[i], atol=1e-10)
        np.testing.assert_allclose(end[3], 0., atol=1e-10)
        np.testing.assert_allclose(24*c[4]+120*c[5], 0., atol=1e-10)


def test_braked_endpoint_and_batch_derivatives_match_scalar():
    t = np.array([2., 4.])
    targets = np.array([[3., 4., 30.], [8., 10., 50.]])
    tv = np.array([[1., 2., 3.], [4., 3., 2.]])
    curves = flythrough_curves([0., 0., 20.], [0., 1., 2.], [0., 0., 2.],
                               targets, tv, t, np.ones(2))
    batch = batch_samples(curves, t, 25)
    for i, c in enumerate(curves):
        scalar = sample_curve(c, t[i], np.linspace(0., t[i], 25))
        for got, expected in zip(batch, scalar):
            np.testing.assert_allclose(got[i], expected, atol=1e-9)
        np.testing.assert_allclose(scalar[1][-1], tv[i], atol=1e-9)
        np.testing.assert_allclose(scalar[2][-1], 0., atol=1e-9)


def load_tests(loader, tests, pattern):
    tests.addTests(unittest.FunctionTestCase(test) for test in (
        test_position_only_natural_endpoint_and_continuity,
        test_braked_endpoint_and_batch_derivatives_match_scalar,
    ))
    return tests
