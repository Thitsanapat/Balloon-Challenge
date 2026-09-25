"""Choose the rail direction by comparing online, feasible multi-hit routes."""

import copy
import numpy as np

from BalloonPoppingGymEnv.agents.chain_beam_agent import ChainBeamAgent
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import G, get_initial_attitude


def launch_axes(states, released, position, available, count=3):
    """Constant-acceleration intercept estimates from observed released targets.

    These generate directions, not executable plans. Full chain feasibility is
    checked separately, including changing thrust-to-mass ratio and turn rate.
    """
    candidates = []
    for index in released:
        for duration in np.arange(2., 10.01, .5):
            target = states[index, :3] + states[index, 3:6]*duration
            accel = 2*(target-position)/duration**2
            force = accel-G
            norm = np.linalg.norm(force)
            if (norm <= .98*available and accel[2] > .1
                    and force[2] >= norm*np.cos(np.radians(35.))):
                candidates.append((duration, force/norm))
                break
    axes = [np.array([0., 0., 1.])]
    if count <= 0:
        return axes
    for _, axis in sorted(candidates, key=lambda item: item[0]):
        if all(np.dot(axis, other) < np.cos(np.radians(4.)) for other in axes):
            axes.append(axis)
        if len(axes) >= count+1:
            break
    return axes


class ChainLaunchAgent(ChainBeamAgent):
    def __init__(self, given_parameters, launch_candidates=3, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.launch_candidates = int(launch_candidates)
        if self.launch_candidates < 0:
            raise ValueError('launch_candidates must be nonnegative')
        self.launch_comparisons = []
        self.launch_selected = False

    def get_action(self, observation):
        now = float(observation['simulation_time'])
        if not self.launched and not self.launch_selected and now >= self.launch_time:
            states = np.asarray(observation['balloon_states'], dtype=float)
            released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1)==1)
            position = np.array([0., 0., self.elevation])
            available = self.available_acceleration(now)
            # Independent copies of our own state: candidate planners cannot
            # contaminate one another, and never receive environment internals.
            original = copy.deepcopy(self.__dict__)
            best = None
            comparisons = []
            for axis in launch_axes(states, released, position, available, self.launch_candidates):
                self.__dict__ = copy.deepcopy(original)
                self.acceleration = G+available*axis
                self._make_plan(observation, position, np.zeros(3), now)
                count = len(self.route)
                finish = self.deadlines[-1]-now if count else float('inf')
                key = (-count, finish)
                comparisons.append([axis.tolist(), count, finish if count else None])
                if best is None or key < best[0]:
                    best = key, axis.copy()
            self.__dict__ = original
            axis = best[1]
            inclination = np.degrees(np.arcsin(np.clip(axis[2], -1., 1.)))
            heading = np.degrees(np.arctan2(axis[0], axis[1])) % 360.
            self.launch_attitude = np.array([inclination, heading])
            self.quaternion = get_initial_attitude(inclination, heading)
            self.launch_selected = True
            self.launch_comparisons = comparisons
        return super().get_action(observation)
