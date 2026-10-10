import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.agents.scenario4_joint_refresh_agent import Scenario4JointRefreshAgent
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import sample_curve


class JointRefreshTests(unittest.TestCase):
    def make_agent(self):
        a = Scenario4JointRefreshAgent.__new__(Scenario4JointRefreshAgent)
        a.route = [0, 1]
        a.deadlines = [4., 6.]
        a.launch_time, a.burn_time = 0., 30.
        a.joint_refresh_min_duration = .3
        a.joint_refresh_allow_shortening = True
        a.acceleration = np.zeros(3)
        a.terminal_weight = 0.
        a.diagnostics = dict(joint_refresh_attempts=0, joint_refresh_accepted=0,
                             joint_refresh_dropped_legs=0)
        return a

    def observation(self):
        return dict(balloon_states=np.array([[0., 0, 10, 1, 0, 1],
                                             [5, 0, 12, 1, 0, 1]]),
                    balloon_status=np.ones(2))

    def test_success_keeps_deadlines_and_selects_validated_curve(self):
        a = self.make_agent()
        obs = self.observation()
        saved = obs['balloon_states'].copy()
        with patch.object(a, '_valid', return_value=(np.ones(2, bool), None,
                          np.zeros((2, 3)), np.zeros((2, 3)))) as valid, \
                patch.object(a, '_select') as select:
            self.assertTrue(a._rescue_committed_leg(obs, np.zeros(3), np.zeros(3), 1., 3.))
            self.assertEqual(valid.call_args.kwargs['samples'], 65)
            self.assertEqual(select.call_args.args[0], 0)
            self.assertEqual(a.route, [0, 1])
            self.assertEqual(a.deadlines, [4., 6.])
            curves, times = valid.call_args.args[:2]
            for index, (curve, duration) in enumerate(zip(curves, times)):
                endpoint = sample_curve(curve, duration, duration)[0]
                expected = saved[index, :3] + saved[index, 3:] * sum(times[:index+1])
                np.testing.assert_allclose(endpoint, expected, atol=1e-10)
        np.testing.assert_array_equal(saved, obs['balloon_states'])

    def test_failed_check_preserves_old_route(self):
        a = self.make_agent()
        with patch.object(a, '_valid', return_value=(np.array([False]), None, None, None)), \
                patch.object(a, '_select') as select:
            self.assertFalse(a._rescue_committed_leg(self.observation(), np.zeros(3), np.zeros(3), 1., 3.))
            select.assert_not_called()
            self.assertEqual(a.route, [0, 1])
            self.assertEqual(a.deadlines, [4., 6.])

    def test_shortening_is_explicit_and_can_be_disabled(self):
        a = self.make_agent()
        def valid(curves, times, starts, samples):
            count = len(times)
            return np.full(count, count == 1), None, np.zeros((count, 3)), np.zeros((count, 3))
        with patch.object(a, '_valid', side_effect=valid), patch.object(a, '_select'):
            self.assertTrue(a._rescue_committed_leg(self.observation(), np.zeros(3), np.zeros(3), 1., 3.))
            self.assertEqual(a.route, [0])
            self.assertEqual(a.diagnostics['joint_refresh_dropped_legs'], 1)
        a = self.make_agent()
        a.joint_refresh_allow_shortening = False
        with patch.object(a, '_valid', side_effect=valid), patch.object(a, '_select'):
            self.assertFalse(a._rescue_committed_leg(self.observation(), np.zeros(3), np.zeros(3), 1., 3.))
            self.assertEqual(a.route, [0, 1])

    def test_unreleased_or_expired_cannot_be_selected(self):
        a = self.make_agent()
        obs = self.observation()
        obs['balloon_status'][0] = 0
        with patch.object(a, '_select') as select:
            self.assertFalse(a._rescue_committed_leg(obs, np.zeros(3), np.zeros(3), 1., 3.))
            self.assertFalse(a._rescue_committed_leg(self.observation(), np.zeros(3), np.zeros(3), 4., 0.))
            select.assert_not_called()


if __name__ == '__main__':
    unittest.main()
