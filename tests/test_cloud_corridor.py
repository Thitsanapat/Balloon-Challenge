import unittest
import numpy as np

from BalloonPoppingGymEnv.agents.cloud_corridor_agent import cloud_geometry,CloudCorridorAgent
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class CloudTests(unittest.TestCase):
    def test_recovers_inclined_axis(self):
        heights = np.linspace(0.,100.,11)
        points = np.column_stack((.5*heights,-.2*heights,heights))
        _,axis,_,radial,_,_ = cloud_geometry(points)
        expected = np.array([.5,-.2,1.])
        np.testing.assert_allclose(axis,expected/np.linalg.norm(expected),atol=1e-12)
        np.testing.assert_allclose(radial,0.,atol=1e-12)

    def test_geometry_rotates_with_horizontal_wind_direction(self):
        points = np.random.default_rng(2).normal(size=(20,3))*50
        rotation = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        original = cloud_geometry(points)
        rotated = cloud_geometry(points@rotation.T+np.array([40.,-30.,100.]))
        np.testing.assert_allclose(rotated[1],rotation@original[1],atol=1e-12)
        for a,b in zip(original[2:],rotated[2:]):
            np.testing.assert_allclose(a,b,atol=1e-10)

    def test_flat_cloud_finite_and_density_prefers_cluster(self):
        points = np.array([[0.,0.,20.],[1.,0.,20.],[0.,1.,20.],[1000.,1000.,20.]])
        result = cloud_geometry(points)
        self.assertTrue(all(np.all(np.isfinite(v)) for v in result))
        np.testing.assert_allclose(result[1],[0,0,1])
        self.assertGreater(result[-1][0],result[-1][-1])

    def test_planning_does_not_mutate_observation(self):
        _,given = load_scenario_parameters(1)
        agent = CloudCorridorAgent(given,first_candidates=2,next_candidates=2)
        states = np.array([[5.,0.,100.,1.,0.,1.],[10.,2.,110.,1.,0.,1.],[30.,20.,120.,1.,0.,1.]])
        status = np.array([1,1,1])
        obs = {"balloon_states":states.copy(),"balloon_status":status.copy()}
        agent._make_plan(obs,np.array([0.,0.,100.]),np.zeros(3),5.)
        np.testing.assert_array_equal(obs["balloon_states"],states)
        np.testing.assert_array_equal(obs["balloon_status"],status)
        self.assertIsNotNone(agent.plan)
        self.assertTrue(agent._feasible(agent.plan,agent.plan_duration,5.))


if __name__=="__main__":
    unittest.main()
