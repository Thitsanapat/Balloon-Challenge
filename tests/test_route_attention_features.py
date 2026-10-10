import unittest
import numpy as np
from BalloonPoppingGymEnv.agents.route_attention_features import routing_features


class RoutingFeatureTests(unittest.TestCase):
    def features(self, obs, **kw):
        return routing_features(obs,np.zeros(3),np.zeros(3),np.zeros(3),20.,12.,**kw)

    def test_only_released_rows_and_no_input_mutation(self):
        states=np.array([[1.,2.,30.,0.,0.,2.],[3.,2.,30.,0.,0.,2.],
                         [1e8,2.,0.,0.,0.,0.],[np.nan]*6])
        obs=dict(balloon_states=states,balloon_status=np.array([1,1,0,1]))
        saved=states.copy()
        result=self.features(obs,visited=[0],feasible=np.array([True,True,True,True]))
        np.testing.assert_array_equal(result['candidate_mask'],[False,True,False,False])
        self.assertEqual(result['nodes'].shape,(4,12))
        self.assertEqual(result['context'].shape,(9,))
        self.assertTrue(np.isfinite(result['nodes']).all())
        np.testing.assert_array_equal(states,saved)
        states[2]=[-1e20]*6
        np.testing.assert_array_equal(self.features(obs,visited=[0])['nodes'],result['nodes'])

    def test_permutation_and_exhausted_time(self):
        obs=dict(balloon_states=np.array([[1.,2.,3.,0.,0.,1.],[3.,4.,5.,1.,0.,2.]]),balloon_status=[1,1])
        a=self.features(obs)
        b=self.features(dict(balloon_states=obs['balloon_states'][::-1],balloon_status=[1,1]))
        np.testing.assert_allclose(a['nodes'][::-1],b['nodes'])
        self.assertFalse(self.features(obs,elapsed=20.)['candidate_mask'].any())
        with self.assertRaises(ValueError):
            self.features(obs,feasible=np.array([1,1]))

    def test_empty_observation(self):
        result=self.features(dict(balloon_states=np.empty((0,6)),balloon_status=[]))
        self.assertEqual(result['nodes'].shape,(0,12))


if __name__ == '__main__':
    unittest.main()
