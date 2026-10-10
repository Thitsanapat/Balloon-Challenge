import copy
import hashlib
import json
from types import SimpleNamespace
import unittest

import numpy as np

from BalloonPoppingGymEnv.training.route_shadow import RouteShadowCollector
from scripts.collect_route_shadow import paired_gate, update_hash


def observation(now=1., status=(1, 1, 0)):
    return dict(simulation_time=now, balloon_status=np.array(status),
                balloon_states=np.array([[1., 0., 10., 0., 0., 1.],
                                         [2., 0., 12., 0., 0., 1.], [np.nan]*6]))


def controller():
    return SimpleNamespace(launched=True, target_index=0, plan=np.zeros((6, 3)),
        plan_start=1., plan_duration=2., route=[0, 1], deadlines=[3., 5.],
        launch_time=0., burn_time=30., thrust=100., dry_mass=2., initial_mass=4.,
        mass_flow=.01, failed_until={}, filtered_position=np.array([0., 0., 1.]),
        filtered_velocity=np.zeros(3), acceleration=np.zeros(3),
        ends_v=[np.ones(3), np.ones(3)], ends_a=[np.zeros(3), np.zeros(3)])


class RouteShadowTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.collector = RouteShadowCollector(self.rows.append)
        self.agent = controller()

    def test_snapshot_nonmutation_no_rng_and_no_unreleased_data(self):
        obs = observation()
        before_agent, before_obs = copy.deepcopy(vars(self.agent)), copy.deepcopy(obs)
        rng = np.random.get_state()
        self.collector.after_action(self.agent, obs)
        self.collector.after_action(self.agent, obs)
        self.assertEqual(len(self.rows), 2)  # start + decision, no duplicate
        for k, v in before_agent.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(getattr(self.agent, k), v)
            elif k in ('ends_v', 'ends_a'):
                np.testing.assert_array_equal(getattr(self.agent, k), v)
            else:
                self.assertEqual(getattr(self.agent, k), v)
        for k in obs:
            np.testing.assert_array_equal(obs[k], before_obs[k])
        np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
        self.assertEqual(rng[2:], np.random.get_state()[2:])
        row = self.rows[1]
        self.assertEqual(row['visible_states'][2], [0.] * 6)
        self.assertNotIn('seed', row)
        self.assertEqual(len(row['route_prefixes']), 2)
        self.assertTrue(all(p['actual_hits'] is None for p in row['route_prefixes']))
        json.dumps(row, allow_nan=False)
        self.agent.ends_v[0][:] = 999
        self.assertEqual(row['ends_v'][0], [1., 1., 1.])

    def test_actuator_snapshot_is_observed_and_serializable(self):
        obs = observation()
        obs['rocket_sensors'] = np.r_[[.1, .2, .3], np.zeros(9)]
        self.agent.quaternion = np.array([1., 0., 0., 0.])
        self.agent.previous_tvc = np.array([2., -1.])
        self.agent.last_desired_axis = np.array([0., 0., 1.])
        self.agent.control = {'max_gimbal_angle': 7.}
        self.agent.diagnostics = {'saturated_steps': 4}
        action = {'tvc': np.array([3., 4.]), 'throttle': .8}
        self.collector.after_action(self.agent, obs, action)
        snapshot = self.rows[-1]['actuator_snapshot']
        self.assertEqual(snapshot['gyro'], [.1, .2, .3])
        self.assertEqual(snapshot['estimated_gimbal_output'], [2., -1.])
        self.assertEqual(snapshot['tvc_command'], [3., 4.])
        self.assertEqual(snapshot['allocator_saturated_total'], 4)
        self.assertEqual(snapshot['max_gimbal_angle'], 7.)
        json.dumps(self.rows[-1], allow_nan=False)

    def test_hit_at_deadline_wins_and_future_leg_unlabelled(self):
        self.collector.after_action(self.agent, observation())
        self.collector.observe(observation(3., (2, 1, 0)))
        counts = self.collector.finish(observation(4., (2, 1, 0)))
        self.assertEqual(counts['hit'], 1)
        outcomes = [r for r in self.rows if r['kind'] == 'outcome']
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]['target'], 0)

    def test_refresh_keeps_attempt_and_extends_real_deadline(self):
        self.collector.after_action(self.agent, observation())
        self.agent.plan_start, self.agent.plan_duration = 2., 2.
        self.agent.deadlines = [4., 6.]
        self.collector.after_action(self.agent, observation(2.))
        self.collector.observe(observation(3.1))
        self.assertIsNotNone(self.collector.active)
        self.assertEqual(self.rows[1]['attempt_id'], self.rows[2]['attempt_id'])
        self.collector.observe(observation(4.))
        self.assertEqual(self.rows[-1]['outcome'], 'miss')
        self.collector.after_action(self.agent, observation(4.1))
        self.assertEqual(self.collector.next_attempt, 1)

    def test_switch_is_cancelled_and_episode_end_is_censored(self):
        self.collector.after_action(self.agent, observation())
        self.agent.target_index = 1
        self.agent.plan_start = 2.
        self.agent.route = [1]
        self.agent.deadlines = [4.]
        self.collector.after_action(self.agent, observation(2.))
        counts = self.collector.finish(observation(2.1))
        self.assertEqual(counts['cancelled'], 1)
        self.assertEqual(counts['censored'], 1)
        self.assertEqual(counts['miss'], 0)

    def test_stale_route_not_presented_as_alternative(self):
        self.agent.route = [1]
        self.agent.deadlines = [3.]
        self.collector.after_action(self.agent, observation())
        self.assertFalse(self.rows[1]['route_metadata_aligned'])
        self.assertEqual(self.rows[1]['route_prefixes'], [])

    def test_blocked_active_target_not_valid_imitation_label(self):
        self.agent.failed_until = {0: 3.}
        self.collector.after_action(self.agent, observation())
        self.assertFalse(self.rows[1]['imitation_eligible'])

    def test_simultaneous_incidental_and_target_pops_not_double_counted(self):
        self.collector.after_action(self.agent, observation())
        self.collector.observe(observation(2., (2, 2, 0)))
        self.collector.observe(observation(2., (2, 2, 0)))
        result = self.collector.finish(observation(2.1, (2, 2, 0)))
        self.assertEqual(result['observed_pops'], 2)
        self.assertEqual(result['active_target_pops'], 1)
        self.assertEqual(result['incidental_pops'], 1)
        self.assertEqual(result['hit'], 1)
        events = [r for r in self.rows if r['kind'] == 'pop']
        self.assertEqual([r['active_attempt_id'] for r in events], [0, 0])
        self.assertEqual(self.collector.finish(observation(2.1, (2, 2, 0))), result)

    def test_late_pop_after_deadline_has_no_false_active_credit(self):
        self.collector.after_action(self.agent, observation())
        self.collector.observe(observation(3.))
        self.collector.observe(observation(3.1, (2, 1, 0)))
        result = self.collector.finish(observation(4., (2, 1, 0)))
        self.assertEqual(result['miss'], 1)
        self.assertEqual(result['hit'], 0)
        self.assertEqual(result['unassigned_pops'], 1)

    def test_initial_pops_not_rewards_and_status_reset_rejected(self):
        self.collector.observe(observation(0., (2, 1, 0)))
        self.collector.observe(observation(1., (2, 2, 0)))
        self.assertEqual(self.collector.counts['initial_pops'], 1)
        self.assertEqual(self.collector.counts['observed_pops'], 1)
        with self.assertRaises(ValueError):
            self.collector.observe(observation(2., (1, 2, 0)))

    def test_parity_gate_requires_actions_observations_and_normal_termination(self):
        result = dict(action_sha256='a', observation_sha256='b', steps=50,
                      score=3, final_time=5., terminated=True, truncated=False)
        self.assertTrue(paired_gate(result, result.copy()))
        for key, value in [('action_sha256', 'c'), ('observation_sha256', 'd'),
                           ('truncated', True), ('score', 4)]:
            self.assertFalse(paired_gate(result, dict(result, **{key: value})))
        failed = dict(result, truncated=True)
        self.assertFalse(paired_gate(failed, failed))
        a, b = hashlib.sha256(), hashlib.sha256()
        update_hash(a, {'x': np.array([np.nan, 1.])})
        update_hash(b, {'x': [float('nan'), 1.]})
        self.assertEqual(a.digest(), b.digest())


if __name__ == '__main__':
    unittest.main()
