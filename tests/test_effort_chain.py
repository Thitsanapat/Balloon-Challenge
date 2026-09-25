import unittest
import numpy as np
from BalloonPoppingGymEnv.agents.effort_chain_agent import effort_chain
from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain
from BalloonPoppingGymEnv.agents.capture_chain_agent import jerk_energy
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import batch_samples, G


def effort_energy(curves, durations, bias):
    factors = np.array([2., 6., 12., 20.])
    gram = np.outer(factors, factors)/(np.arange(4)[:, None]+np.arange(4)[None, :]+1)
    shifted = curves[:, 2:].copy()
    shifted[:, 0] -= bias*np.asarray(durations)[:, None]**2/2
    return sum(np.einsum('ij,ik,kj->', c, gram, c)/t**3 for c, t in zip(shifted, durations))


class EffortChainTests(unittest.TestCase):
    def test_zero_weight_is_exact_original_and_inputs_are_unchanged(self):
        targets = np.array([[10., 2., 40.], [20., 10., 55.], [10., 20., 75.]])
        before = targets.copy()
        args = (np.array([0., 0., 20.]), np.zeros(3), np.zeros(3), targets, [3., 2., 2.5], np.zeros(3))
        np.testing.assert_array_equal(effort_chain(*args, G, 0.), free_chain(*args))
        np.testing.assert_array_equal(targets, before)

    def test_effort_reduced_and_joint_cost_optimal_against_feasible_perturbations(self):
        p, v, a = np.array([0., 0., 20.]), np.array([2., 1., 3.]), np.array([0., 0., 1.])
        targets = np.array([[10., 2., 40.], [20., 10., 55.], [10., 20., 75.]])
        times, weight = np.array([3., 2., 2.5]), .3
        original = free_chain(p, v, a, targets, times, np.zeros(3))
        optimized = effort_chain(p, v, a, targets, times, np.zeros(3), G, weight)
        cost = lambda c: jerk_energy(c, times)+weight*effort_energy(c, times, G)
        self.assertLess(effort_energy(optimized, times, G), effort_energy(original, times, G))
        self.assertLess(cost(optimized), cost(original))
        ps, vs, acs, _ = batch_samples(optimized, times)
        np.testing.assert_allclose(ps[:, -1], targets, atol=1e-8)
        np.testing.assert_allclose(ps[0, 0], p, atol=1e-8)
        np.testing.assert_allclose(vs[0, 0], v, atol=1e-8)
        np.testing.assert_allclose(acs[0, 0], a, atol=1e-8)
        for values in (ps, vs, acs):
            np.testing.assert_allclose(values[:-1, -1], values[1:, 0], atol=1e-8)
        # Perturb every solved velocity/acceleration degree of freedom in both
        # directions; preserve waypoint positions and shared derivatives.
        from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain_matrices
        design, _, _ = free_chain_matrices(tuple(times), 0.)
        for variable in range(design.shape[2]):
            for axis in range(3):
                perturb = np.zeros_like(optimized)
                perturb[:, :, axis] = design[:, :, variable]*.01
                for sign in (-1, 1):
                    self.assertGreater(cost(optimized+sign*perturb), cost(optimized))


if __name__ == '__main__':
    unittest.main()
