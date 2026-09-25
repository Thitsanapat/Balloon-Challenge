import unittest
import numpy as np
from BalloonPoppingGymEnv.agents.cluster_chain_agent import cluster_affinity, cluster_order


class ClusterTests(unittest.TestCase):
    def test_only_released_balloons_contribute_without_mutating_observation(self):
        states = np.zeros((4,6))
        before = states.copy()
        affinity = cluster_affinity(states,[0,2],20.)
        self.assertEqual(affinity[0,2],1.)
        self.assertEqual(affinity[0,0],0.)
        self.assertEqual(affinity[1].sum(),0.)
        np.testing.assert_array_equal(states,before)

    def test_near_group_is_preferred_but_far_group_does_not_justify_detour(self):
        states = np.zeros((5,6))
        states[:,0] = [10.,15.,16.,17.,18.]
        ids = list(range(5))
        affinity = cluster_affinity(states,ids,3.)
        order = cluster_order(states,ids,np.zeros(3),np.zeros(3),0.,affinity,5.,3.,8.)
        self.assertNotEqual(order[0],0)
        states[1:,0] += 500.
        affinity = cluster_affinity(states,ids,3.)
        order = cluster_order(states,ids,np.zeros(3),np.zeros(3),0.,affinity,5.,3.,8.)
        self.assertEqual(order[0],0)

    def test_velocity_divergence_reduces_group_value(self):
        states = np.zeros((2,6))
        joined = cluster_affinity(states,[0,1],20.)[0,1]
        states[1,3] = 20.
        self.assertLess(cluster_affinity(states,[0,1],20.)[0,1],joined*.01)


if __name__=='__main__':
    unittest.main()
