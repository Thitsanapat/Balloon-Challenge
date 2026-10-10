"""A selected target forecast must not chatter with rolling evidence."""

import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_latched_capture_agent import (
    _LatchedConfidenceProfile,
)
from BalloonPoppingGymEnv.agents.scenario4_capture_confidence_agent import (
    CaptureEvidence,
)


class _Baseline:
    def correction(self, states, status, index, horizon):
        return np.zeros(2)


class _Acceleration:
    def correction(self, states, status, index, horizon, baseline):
        return np.array([0.5, 0.0])


class LatchedCaptureTests(unittest.TestCase):
    def test_model_choice_stays_fixed_until_target_changes(self):
        evidence = CaptureEvidence(min_evidence=1, minimum_gain=0)
        states = np.array([[0., 0., 100., 0., 0., 0.],
                           [0., 0., 100., 0., 0., 0.]])
        status = np.array([1, 1])
        profile = _LatchedConfidenceProfile(
            _Baseline(), _Acceleration(), evidence, 0.5, 3.0)
        profile.set_target(0, states, status, 1.0)
        self.assertFalse(profile.use_acceleration)
        evidence.history.append((100., 4.0, 0.0))
        profile.set_target(0, states, status, 2.0)
        self.assertFalse(profile.use_acceleration)
        np.testing.assert_allclose(profile.correction(
            states, status, 0, 2.0), [0., 0.])
        profile.set_target(1, states, status, 3.0)
        self.assertTrue(profile.use_acceleration)
        np.testing.assert_allclose(profile.correction(
            states, status, 1, 2.0), [1., 0.])
        evidence.history.append((100., 0.0, 4.0))
        profile.set_target(1, states, status, 4.0)
        self.assertTrue(profile.use_acceleration)
        self.assertEqual(len(profile.target_choices), 2)


if __name__ == '__main__':
    unittest.main()
