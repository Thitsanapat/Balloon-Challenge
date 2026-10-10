"""Causal forecast-selection checks; no official environment state is used."""

import unittest

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_capture_confidence_agent import (
    CaptureEvidence, _ConfidenceProfile,
)


class _Profile:
    def correction(self, states, status, index, horizon):
        return np.array([0.0, 0.0])


class _Forecaster:
    def correction(self, states, status, index, horizon, baseline):
        return np.array([1.0, 0.0])


class CaptureEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.states = np.array([[0., 0., 100., 1., 0., 0.]])
        self.status = np.array([1])

    def test_future_observation_only_updates_when_due(self):
        evidence = CaptureEvidence(horizon=2, sample_period=10,
                                   min_evidence=1, minimum_gain=0)
        evidence.observe(0, self.states, self.status, _Profile(),
                         _Forecaster(), 0.75, active_target=0)
        self.assertEqual(len(evidence.history), 0)
        self.states[0, 0] = 3.0
        evidence.observe(1, self.states, self.status, _Profile(),
                         _Forecaster(), 0.75, active_target=0)
        self.assertEqual(len(evidence.history), 0)
        evidence.observe(2, self.states, self.status, _Profile(),
                         _Forecaster(), 0.75, active_target=0)
        self.assertEqual(len(evidence.history), 1)
        self.assertEqual(evidence.history[0][1:], (1.0, 0.0))
        self.assertTrue(evidence.choose_acceleration(100.0))

    def test_reject_when_new_forecast_worse_or_insufficient(self):
        evidence = CaptureEvidence(min_evidence=3)
        evidence.history.append((100., 0.2, 2.0))
        self.assertFalse(evidence.choose_acceleration(100.))
        evidence.history.extend([(100., 0.2, 2.0)] * 2)
        self.assertFalse(evidence.choose_acceleration(100.))
        self.assertIsNone(evidence.capture_gain(200.))

    def test_popped_balloon_not_used_as_future_truth(self):
        evidence = CaptureEvidence(horizon=1, sample_period=10,
                                   min_evidence=1)
        evidence.observe(0, self.states, self.status, _Profile(),
                         _Forecaster(), 0.75)
        evidence.observe(1, self.states, np.array([2]), _Profile(),
                         _Forecaster(), 0.75)
        self.assertEqual(len(evidence.history), 0)

    def test_only_active_target_and_positive_gain_are_augmented(self):
        evidence = CaptureEvidence(min_evidence=1, minimum_gain=0)
        evidence.history.append((100., 4.0, 0.0))
        profile = _ConfidenceProfile(_Profile(), _Forecaster(), evidence,
                                     blend=0.5, max_horizon=3.)
        profile.active_target = 0
        np.testing.assert_allclose(profile.correction(
            self.states, self.status, 0, 2.), [2., 0.])
        np.testing.assert_allclose(profile.correction(
            self.states, self.status, 0, 4.), [0., 0.])
        profile.active_target = None
        np.testing.assert_allclose(profile.correction(
            self.states, self.status, 0, 2.), [0., 0.])


if __name__ == '__main__':
    unittest.main()
