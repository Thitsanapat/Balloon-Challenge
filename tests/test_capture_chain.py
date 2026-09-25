import unittest
import numpy as np
from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain
from BalloonPoppingGymEnv.agents.capture_chain_agent import CaptureChainAgent,optimize_offsets,jerk_energy
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import batch_samples


class CaptureChainTests(unittest.TestCase):
    def setUp(self):
        self.p,self.v,self.a = np.array([0.,0.,20.]),np.array([3.,0.,2.]),np.zeros(3)
        self.targets = np.array([[12.,4.,35.],[20.,-3.,42.],[24.,9.,50.]])
        self.durations = [3.,2.,2.5]
        self.curves = free_chain(self.p,self.v,self.a,self.targets,self.durations,np.zeros(3),0.)

    def test_offsets_reduce_jerk_and_stay_inside_capture_spheres(self):
        offsets,curves = optimize_offsets(self.curves,self.durations,.75)
        self.assertLess(jerk_energy(curves,self.durations),jerk_energy(self.curves,self.durations))
        self.assertTrue(np.all(np.linalg.norm(offsets,axis=1)<=.75+1e-10))
        ps,vs,acs,_ = batch_samples(curves,self.durations)
        np.testing.assert_allclose(ps[:,-1],self.targets+offsets,atol=1e-9)
        for values in (ps,vs,acs):
            np.testing.assert_allclose(values[:-1,-1],values[1:,0],atol=1e-9)
        np.testing.assert_allclose(ps[0,0],self.p,atol=1e-9)
        np.testing.assert_allclose(vs[0,0],self.v,atol=1e-9)
        np.testing.assert_allclose(acs[0,0],self.a,atol=1e-9)

    def test_zero_offset_is_exact_control(self):
        offsets,curves = optimize_offsets(self.curves,self.durations,0.)
        np.testing.assert_array_equal(offsets,0.)
        np.testing.assert_array_equal(curves,self.curves)

    def test_affine_update_matches_resolving_the_full_chain(self):
        offsets,curves = optimize_offsets(self.curves,self.durations,.5)
        expected = free_chain(self.p,self.v,self.a,self.targets+offsets,self.durations,np.zeros(3),0.)
        np.testing.assert_allclose(curves,expected,atol=1e-8)

    def test_committed_reference_keeps_offset_without_mutating_sensor_data(self):
        agent = CaptureChainAgent.__new__(CaptureChainAgent)
        agent.hit_offsets = {0:np.array([.2,-.3,.1])}
        states = np.array([[10.,20.,30.,1.,2.,3.]])
        before = states.copy()
        target = agent._intercept_target(states,0,2.)
        np.testing.assert_allclose(target,[12.2,23.7,36.1])
        np.testing.assert_array_equal(states,before)


if __name__=='__main__':
    unittest.main()
