import copy
import unittest
from types import SimpleNamespace
import numpy as np
from scripts.guidance_telemetry import GuidanceTelemetry


class GuidanceTelemetryTests(unittest.TestCase):
    def test_common_time_error_decomposition_and_no_input_mutation(self):
        curve = np.zeros((6, 3))
        curve[0], curve[1] = [0., 0., 20.], [10., 0., 0.]
        agent = SimpleNamespace(target_index=0, plan=curve, plan_start=0., plan_duration=10.,
                                available_acceleration=lambda now: 15.)
        obs = dict(simulation_time=0., balloon_states=np.array([[10., 0., 20., 0., 0., 0.]]),
                   balloon_status=np.array([1]), rocket_sensors=np.zeros(12))
        action = dict(throttle=.8, tvc=np.zeros(2))
        saved = copy.deepcopy(obs)
        observer = GuidanceTelemetry()
        observer.before_step(agent, obs, action)
        np.testing.assert_array_equal(obs['balloon_states'], saved['balloon_states'])
        np.testing.assert_array_equal(agent.plan, curve)
        after = copy.deepcopy(obs)
        after['simulation_time'] = 10.
        after['rocket_sensors'][6:9] = [8., 1., 20.]
        after['balloon_states'][0, :3] = [11., 0., 20.]
        observer.after_step(after)
        result = observer.finish(10.)
        closest = result['target_episodes'][0]['closest']
        total = sum(np.asarray(closest[name]) for name in
                    ('tracking_vector', 'alignment_vector', 'prediction_error_vector'))
        np.testing.assert_allclose(total, [-3., 1., 0.])
        self.assertAlmostEqual(closest['tracking_error'], np.sqrt(5.))
        self.assertEqual(closest['prediction_error'], 1.)
        self.assertEqual(result['unpopped_forecast_checks'][0]['endpoint_error'], 1.)
        self.assertEqual(len(observer.forecasts), 0)

    def test_popped_forecasts_are_not_counted_as_prediction_misses(self):
        observer = GuidanceTelemetry()
        curve = np.zeros((6, 3))
        agent = SimpleNamespace(target_index=0, plan=curve, plan_start=0., plan_duration=1.,
                                available_acceleration=lambda now: 15.)
        obs = dict(simulation_time=0., balloon_states=np.zeros((1, 6)),
                   balloon_status=np.array([1]), rocket_sensors=np.zeros(12))
        observer.before_step(agent, obs, dict(throttle=.5, tvc=np.zeros(2)))
        obs['simulation_time'], obs['balloon_status'][0] = .5, 2
        observer.after_step(obs)
        self.assertEqual(observer.episodes[0]['end_reason'], 'popped')
        self.assertEqual(observer.finished_forecasts, [])
        self.assertEqual(observer.forecasts, [])


if __name__ == '__main__':
    unittest.main()
