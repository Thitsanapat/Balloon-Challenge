"""Observation-only first-target selector above the verified chain controller.

The selector chooses one candidate for the first leg of a newly built route.
The inherited joint trajectory search then chooses all following legs and still
performs the same physical checks. Learned models only receive this module's
feature matrix; they do not receive IDs, seeds, stored fields, or future data.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.submission_time_allocation_v1 import ChainSubmissionAgent


SELECTOR_FEATURES = 12


def ranked_candidate_ids(agent, observation, count):
    """Return nearby released candidates using only the current observation."""
    states = np.asarray(observation['balloon_states'], dtype=float)
    status = np.asarray(observation['balloon_status']).reshape(-1)
    sensors = np.asarray(observation['rocket_sensors'], dtype=float)
    now = float(observation['simulation_time'])
    position = np.asarray(sensors[6:9], dtype=float)
    velocity = np.asarray(sensors[9:12], dtype=float)
    if not np.all(np.isfinite(position)):
        position = np.array([0., 0., agent.elevation])
    if not np.all(np.isfinite(velocity)):
        velocity = np.zeros(3)
    ids = [int(i) for i in np.flatnonzero(status == 1)
           if agent.failed_until.get(int(i), 0.) <= now]
    ids.sort(key=lambda i: np.linalg.norm(
        states[i, :3] + states[i, 3:6]*1.5 - position - velocity*1.5))
    return ids[:count], position, velocity


def selector_features(agent, observation, count):
    """Fixed-size public target matrix and mask for all selector models."""
    ids, position, velocity = ranked_candidate_ids(agent, observation, count)
    states = np.asarray(observation['balloon_states'], dtype=float)
    now = float(observation['simulation_time'])
    rows = np.zeros((count, SELECTOR_FEATURES), dtype=np.float32)
    for slot, index in enumerate(ids):
        relative = states[index, :3] - position
        relative_velocity = states[index, 3:6] - velocity
        distance = np.linalg.norm(relative)
        closing = -np.dot(relative, relative_velocity)/max(distance, 1e-6)
        rows[slot] = np.array([
            *(relative/100.), *(relative_velocity/25.), distance/150.,
            closing/25., relative[2]/100.,
            agent.available_acceleration(now)/20.,
            np.clip((agent.launch_time+agent.burn_time-now)/agent.burn_time, 0., 1.),
            1.,
        ], dtype=np.float32)
    sensors = np.asarray(observation['rocket_sensors'], dtype=float)
    global_features = np.array([
        now/100., np.nan_to_num(sensors[8], nan=agent.elevation)/300.,
        np.linalg.norm(np.nan_to_num(sensors[9:12], nan=0.))/40., len(ids)/count,
    ], dtype=np.float32)
    return np.r_[rows.ravel(), global_features], np.arange(count) < len(ids), ids


class SelectorChainAgent(ChainSubmissionAgent):
    """Chain controller whose first candidate is an externally supplied rank."""
    def __init__(self, given_parameters, selector_candidates=8,
                 selector_enabled=True, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.selector_candidates = int(selector_candidates)
        self.selector_enabled = bool(selector_enabled)
        if not 1 <= self.selector_candidates <= 16:
            raise ValueError('selector_candidates must be an integer in [1, 16]')
        self.selector_action = 0
        self.selector_events = []
        self.diagnostics.update(selector_queries=0, selector_forced_first_legs=0,
                                selector_invalid_actions=0)

    def set_selector_action(self, action):
        if not np.isfinite(action) or int(action) != action:
            raise ValueError('selector action must be a finite integer')
        action = int(action)
        if not 0 <= action < self.selector_candidates:
            raise ValueError('selector action is outside the candidate range')
        self.selector_action = action

    def current_selector_features(self, observation):
        return selector_features(self, observation, self.selector_candidates)

    def _candidate_ids(self, states, released, route, elapsed, position, velocity, now):
        # Keep all parent behavior except for the root after the launch direction
        # is committed. Launch-axis trials operate on temporary controller copies.
        if route or not self.selector_enabled or not self.launch_selected:
            return super()._candidate_ids(states, released, route, elapsed, position, velocity, now)
        allowed = [int(i) for i in released
                   if self.failed_until.get(int(i), 0.) <= now]
        allowed.sort(key=lambda i: np.linalg.norm(
            states[i, :3] + states[i, 3:6]*1.5 - position - velocity*1.5))
        pool = allowed[:self.selector_candidates]
        self.diagnostics['selector_queries'] += 1
        if self.selector_action >= len(pool):
            self.diagnostics['selector_invalid_actions'] += 1
            return super()._candidate_ids(states, released, route, elapsed, position, velocity, now)
        chosen = pool[self.selector_action]
        self.selector_events.append([float(now), int(chosen), int(self.selector_action)])
        self.diagnostics['selector_forced_first_legs'] += 1
        return [chosen]
