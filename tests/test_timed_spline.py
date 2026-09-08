import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from BalloonPoppingGymEnv.agents.timed_spline_agent import TimedSplineAgent
from BalloonPoppingGymEnv.agents.spline_route_agent import SplineRouteAgent
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


class TimingTests(unittest.TestCase):
    def test_infeasible_solver_result_retains_valid_pair(self):
        _,given = load_scenario_parameters(1)
        obs = {"balloon_states":np.array([[8.,0.,100.,1.,0.,1.],[16.,2.,105.,1.,0.,1.]]),
               "balloon_status":np.array([1,1])}
        p,v = np.array([0.,0.,100.]),np.zeros(3)
        baseline,agent = SplineRouteAgent(given),TimedSplineAgent(given)
        baseline._make_plan(obs,p,v,5.)
        self.assertGreater(baseline.diagnostics["pair_plans"],0)
        with patch("BalloonPoppingGymEnv.agents.timed_spline_agent.minimize",
                   return_value=SimpleNamespace(x=np.array([.01,.01]),success=True)):
            agent._make_plan(obs,p,v,5.)
        self.assertEqual(agent.diagnostics["timing_improvements"],0)
        np.testing.assert_allclose(agent.plan,baseline.plan)


if __name__=="__main__":
    unittest.main()
