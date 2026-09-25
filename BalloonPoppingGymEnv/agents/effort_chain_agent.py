"""Joint waypoint derivatives with jerk and specific-thrust quadratic costs.

The extra cost is integral ||a - gravity - estimated_disturbance||^2 dt.
It is a thrust-to-mass effort proxy, not fuel consumption. Burn time and mass
flow still follow the provided motor model and are not extended by throttling.
"""

from functools import lru_cache
import numpy as np
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent
from BalloonPoppingGymEnv.agents.chain_beam_agent import free_chain, free_chain_matrices
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import boundary_curve, G


@lru_cache(maxsize=2048)
def effort_matrices(durations, terminal_weight, effort_weight):
    designs, jerk_projections, jerk_inverse = free_chain_matrices(durations, terminal_weight)
    hessian = np.linalg.inv(jerk_inverse)
    factors = np.array([2., 6., 12., 20.])
    gram = np.outer(factors, factors)/(np.arange(4)[:, None]+np.arange(4)[None, :]+1)
    projections = []
    for t, design in zip(durations, designs):
        projection = design[2:].T@(gram/t**3)
        hessian += effort_weight*projection@design[2:]
        projections.append(projection)
    return designs, jerk_projections, np.asarray(projections), np.linalg.inv(hessian)


def effort_chain(p, v, a, targets, durations, drift, bias, effort_weight, terminal_weight=0.):
    if effort_weight == 0:
        return free_chain(p, v, a, targets, durations, drift, terminal_weight)
    durations = tuple(float(t) for t in durations)
    zero = np.zeros(3)
    origins = [p]+list(targets[:-1])
    bases = np.array([boundary_curve(origins[i], v if i == 0 else zero,
        a if i == 0 else zero, targets[i], zero, zero, t) for i, t in enumerate(durations)])
    designs, jerk, effort, inverse = effort_matrices(durations, float(terminal_weight), float(effort_weight))
    shifted = bases[:, 2:].copy()
    shifted[:, 0] -= np.asarray(bias)*np.asarray(durations)[:, None]**2/2
    rhs = np.einsum('nik,nkj->ij', jerk, bases[:, 3:])
    rhs += effort_weight*np.einsum('nik,nkj->ij', effort, shifted)
    rhs[-2] -= terminal_weight*drift
    derivatives = -inverse@rhs
    return bases+np.einsum('nki,ij->nkj', designs, derivatives)


class EffortChainAgent(ChainLaunchAgent):
    def __init__(self, given_parameters, effort_weight=.3, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.effort_weight = float(effort_weight)
        if not np.isfinite(self.effort_weight) or self.effort_weight < 0:
            raise ValueError('Nonnegative finite effort weight required')

    def _chain_curves(self, position, velocity, targets, durations, drift):
        return effort_chain(position, velocity, self.acceleration, targets, durations,
                            drift, G+self.disturbance, self.effort_weight, self.terminal_weight)
