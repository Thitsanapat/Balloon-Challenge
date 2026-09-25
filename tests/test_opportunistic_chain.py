import unittest
from unittest.mock import patch
import numpy as np
from BalloonPoppingGymEnv.agents.opportunistic_chain_agent import OpportunisticChainAgent,closest_approaches
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent


class ClosestApproachTests(unittest.TestCase):
    def test_moving_balloon_crossing_between_samples(self):
        curve = np.zeros((1,6,3))
        curve[0,0] = [0.,0.,20.]
        curve[0,1] = [10.,0.,0.]
        states = np.array([[5.,5.,20.,0.,-1.,0.],[5.,0.,24.,0.,0.,0.]])
        original = states.copy()
        distance,times = closest_approaches(curve,[10.],states,samples=2)
        np.testing.assert_allclose(distance,[0.,4.],atol=1e-10)
        np.testing.assert_allclose(times,[5.,5.],atol=1e-10)
        np.testing.assert_array_equal(states,original)

    def test_periodic_insertion_revisits_a_committed_route_without_extending_finish(self):
        for interval,expected in ((0.,0),(1.,1)):
            agent = OpportunisticChainAgent.__new__(OpportunisticChainAgent)
            agent.route_events = [[0.,[0],10.]]
            agent.route,agent.deadlines = [0],[10.]
            agent.hit_offsets = {}
            agent.insertion_interval,agent.next_insertion_check = interval,0.
            agent.max_insertions,agent.insertion_candidates = 1,8
            agent.insertion_radius,agent.insertion_offset_fraction,agent.radius = 4.,.5,1.5
            agent.acceleration,agent.terminal_weight = np.zeros(3),0.
            agent.diagnostics = {'insertion_attempts':0,'inserted_waypoints':0}
            # This test covers scheduling/state transitions, not feasibility.
            agent._valid = lambda c,t,s,samples: (np.ones(len(t),dtype=bool),*[np.zeros((len(t),3)) for _ in range(3)])
            agent._select = lambda *args: None
            states = np.array([[10.,0.,20.,0.,0.,0.],[5.,.5,20.,0.,0.,0.]])
            before = states.copy()
            observation = {'balloon_states':states,'balloon_status':np.ones(2)}
            with patch.object(ChainLaunchAgent,'_make_plan',return_value=None):
                agent._make_plan(observation,np.array([0.,0.,20.]),np.array([1.,0.,0.]),0.)
            self.assertEqual(agent.diagnostics['inserted_waypoints'],expected)
            self.assertAlmostEqual(agent.deadlines[-1],10.)
            if expected:
                self.assertEqual(agent.route,[1,0])
                self.assertEqual(len(agent.route_events),2)
            np.testing.assert_array_equal(states,before)


if __name__=='__main__':
    unittest.main()
