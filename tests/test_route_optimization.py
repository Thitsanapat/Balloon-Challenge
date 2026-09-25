import unittest
import numpy as np

from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain
from BalloonPoppingGymEnv.agents.coverage_mpc_agent import integration_basis
from BalloonPoppingGymEnv.agents.momentum_beam_agent import arrival_curves
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import batch_samples
from BalloonPoppingGymEnv.agents.ramp_beam_agent import ramp_solution, ramp_reference


class RouteOptimizationTests(unittest.TestCase):
    def test_mixed_short_and_long_legs_preserve_straight_constant_velocity(self):
        p = np.array([0.,0.,20.])
        v = np.array([4.,2.,3.])
        durations = [.25,.5,3.,8.]
        targets = p+np.cumsum(durations)[:,None]*v
        curves = free_chain(p,v,np.zeros(3),targets,durations,v,0.)
        ps,vs,acs,_ = batch_samples(curves,durations)
        np.testing.assert_allclose(ps[:,-1],targets,atol=1e-7)
        np.testing.assert_allclose(vs,np.broadcast_to(v,vs.shape),atol=1e-7)
        np.testing.assert_allclose(acs,0.,atol=1e-7)

    def test_ramped_intercept_position_and_derivative_continuity(self):
        p,v,a = np.array([0.,0.,20.]),np.array([3.,2.,4.]),np.array([.1,.2,1.])
        target = np.array([[20.,10.,40.]])
        duration,ramp = 4.,.7
        ad,ev = ramp_solution(p,v,a,target,np.array([duration]),np.array([ramp]))
        first,second = ramp_reference(p,v,a,ad[0],duration,ramp)
        ps,vs,acs,_ = batch_samples(np.asarray([first,second]),[ramp,duration-ramp])
        np.testing.assert_allclose(ps[1,-1],target[0],atol=1e-10)
        np.testing.assert_allclose(vs[1,-1],ev[0],atol=1e-10)
        for values in (ps,vs,acs):
            np.testing.assert_allclose(values[0,-1],values[1,0],atol=1e-10)

    def test_acceleration_integration_constant_and_linear(self):
        knots = np.array([0.,1.,2.,4.])
        times = np.array([0.,.3,1.,1.7,3.,4.])
        p,v = integration_basis(knots,times)
        np.testing.assert_allclose(p@np.ones(4),times**2/2,atol=1e-12)
        np.testing.assert_allclose(v@np.ones(4),times,atol=1e-12)
        np.testing.assert_allclose(p@knots,times**3/6,atol=1e-12)
        np.testing.assert_allclose(v@knots,times**2/2,atol=1e-12)

    def test_single_free_chain_matches_natural_endpoint_solution(self):
        p,v,a = np.array([1.,2.,3.]),np.array([2.,-1.,3.]),np.array([.2,0.,1.])
        target = np.array([[10.,5.,30.]])
        actual = free_chain(p,v,a,target,[3.],np.zeros(3),0.)
        expected = arrival_curves(p,v,a,target,np.zeros((1,3)),np.array([3.]),[0.])
        np.testing.assert_allclose(actual,expected,atol=1e-9)

    def test_chain_hits_all_targets_with_continuous_derivatives(self):
        targets = np.array([[10.,2.,40.],[20.,10.,55.],[10.,20.,75.]])
        durations = [3.,2.,2.5]
        curves = free_chain(np.array([0.,0.,20.]),np.zeros(3),np.zeros(3),
                            targets,durations,np.array([0.,0.,3.]),.1)
        ps,vs,acs,_ = batch_samples(curves,durations)
        np.testing.assert_allclose(ps[:,-1],targets,atol=1e-8)
        for values in (ps,vs,acs):
            np.testing.assert_allclose(values[:-1,-1],values[1:,0],atol=1e-8)


if __name__ == '__main__':
    unittest.main()
