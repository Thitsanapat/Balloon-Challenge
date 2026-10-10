"""Observation-only route dataset collector, OUTSIDE the frozen flight agent.

Call after get_action and after env.step. Never calls a planning/control method,
uses an RNG, or retains an agent/observation reference. Future route legs are
unexecuted proposals, not negative examples. This is not a learned controller.
"""
import numpy as np

from BalloonPoppingGymEnv.agents.route_attention_features import (
    ROUTE_FEATURE_SCHEMA, routing_features,
)

EVENT_SCHEMA = 'route_shadow_events_v2'


def _finite_vector(value, length):
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return vector.tolist() if vector.shape == (length,) and np.all(np.isfinite(vector)) else None


class RouteShadowCollector:
    def __init__(self, emit):
        self.emit = emit  # writer belongs to runner, never to deepcopy-able agent
        self.active = None
        self.last_signature = None
        self.next_attempt = 0
        self.next_decision = 0
        self.previous_status = None
        self.finished = False
        self.counts = dict(decisions=0, hit=0, miss=0, cancelled=0, censored=0,
                           initial_pops=0, observed_pops=0, active_target_pops=0,
                           incidental_pops=0, unassigned_pops=0)

    def _close(self, outcome, now, reason):
        if self.active is None:
            return
        self.emit(dict(kind='outcome', attempt_id=self.active['id'],
                       target=self.active['target'], time=now, outcome=outcome,
                       deadline=self.active['deadline'], reason=reason))
        self.counts[outcome] += 1
        self.active = None

    def observe(self, observation):
        """Record all pop transitions, then resolve the active first-leg attempt.

        Association with an attempt is temporal, NOT counterfactual route credit.
        Initial popped rows are state, not newly earned rewards. Repeated calls
        on the same observation cannot count another pop.
        """
        if self.finished:
            raise ValueError('Collector already finished; start a new collector per episode')
        now = float(observation['simulation_time'])
        status = np.asarray(observation['balloon_status']).reshape(-1)
        if not np.all(np.isin(status, [0, 1, 2])):
            raise ValueError('Invalid observed balloon status')
        if self.previous_status is None:
            initial = np.flatnonzero(status == 2).tolist()
            self.counts['initial_pops'] = len(initial)
            self.emit(dict(kind='episode_start', time=now, event_schema=EVENT_SCHEMA,
                           balloon_count=len(status), initial_popped_ids=initial))
        else:
            if status.shape != self.previous_status.shape or np.any(status < self.previous_status):
                raise ValueError('Balloon status regressed or count changed')
            for index in np.flatnonzero((status == 2) & (self.previous_status != 2)):
                target = int(index)
                active = self.active
                relation = ('unassigned' if active is None else
                            'active_target' if target == active['target'] else 'incidental')
                self.emit(dict(kind='pop', time=now, balloon_id=target,
                               event_id=self.counts['observed_pops'], relation=relation,
                               active_attempt_id=active['id'] if active else None,
                               active_target=active['target'] if active else None))
                self.counts['observed_pops'] += 1
                self.counts[relation + '_pops'] += 1
        self.previous_status = status.copy()
        if self.active is None:
            return
        target = self.active['target']
        if status[target] == 2:
            self._close('hit', now, 'observed_popped')
        elif now >= self.active['deadline'] - 1e-9:
            self._close('miss', now, 'latest_committed_deadline_passed_without_pop')
        elif status[target] != 1:
            self._close('cancelled', now, 'target_no_longer_released')

    def after_action(self, agent, observation, action=None):
        """Read final committed state, after hypothetical launch searches ended."""
        self.observe(observation)
        now = float(observation['simulation_time'])
        target = getattr(agent, 'target_index', None)
        if (not agent.launched or agent.plan is None or target is None):
            self._close('cancelled', now, 'controller_has_no_active_plan')
            self.last_signature = None
            return
        target = int(target)
        status = np.asarray(observation['balloon_status']).reshape(-1)
        deadline = float(agent.plan_start + agent.plan_duration)
        if not 0 <= target < len(status) or status[target] != 1 or deadline <= now:
            self._close('cancelled', now, 'active_plan_not_executable')
            return
        if self.active is not None and self.active['target'] != target:
            self._close('cancelled', now, 'controller_switched_target')
        route = tuple(int(i) for i in agent.route)
        deadlines = tuple(float(t) for t in agent.deadlines)
        signature = (target, float(agent.plan_start), deadline, route, deadlines)
        if self.active is not None:
            self.active['deadline'] = deadline  # a real controller refresh
        if signature == self.last_signature:
            return
        if agent.filtered_position is None or agent.filtered_velocity is None:
            return
        remaining = max(0., float(agent.launch_time + agent.burn_time - now))
        elapsed = max(0., now - agent.launch_time)
        available = (0. if elapsed >= agent.burn_time else
                     agent.thrust / max(agent.dry_mass, agent.initial_mass - agent.mass_flow*elapsed))
        blocked = [int(i) for i, until in agent.failed_until.items() if until > now]
        features = routing_features(
            observation, agent.filtered_position, agent.filtered_velocity,
            agent.acceleration, remaining, available, blocked=blocked)
        # Preserve the existing planner's executed choice even if its temporary
        # candidate-block mask excludes it. Do not turn it into a masked CE label.
        imitation_eligible = bool(features['candidate_mask'][target])
        if self.active is None:
            self.active = dict(id=self.next_attempt, target=target, deadline=deadline)
            self.next_attempt += 1
        # Fallback controllers may retain stale chain metadata: never mistake
        # that for the active trajectory. Prefixes are descriptions, NOT freshly
        # solved/densely validated alternatives and have no factual outcomes.
        aligned = (bool(route) and route[0] == target and len(route) == len(deadlines)
                   and abs(deadlines[0] - deadline) < 1e-6
                   and all(0 <= i < len(status) for i in route)
                   and all(b > a for a, b in zip((now,) + deadlines, deadlines)))
        prefixes = []
        if aligned:
            durations = np.diff(np.r_[now, deadlines]).tolist()
            for size in range(1, len(route) + 1):
                prefixes.append(dict(targets=list(route[:size]), durations=durations[:size],
                                     actual_hits=None, feasibility='inherited_plan_not_revalidated'))
        states = np.asarray(observation['balloon_states'], dtype=float)
        visible_states = np.zeros_like(states)
        visible_states[features['visible_mask']] = states[features['visible_mask']]
        record = dict(kind='decision', decision_id=self.next_decision,
                      attempt_id=self.active['id'], time=now, target=target,
                      deadline=deadline, feature_schema=ROUTE_FEATURE_SCHEMA,
                      features={k: v.tolist() for k, v in features.items()},
                      visible_states=visible_states.tolist(),
                      position=np.asarray(agent.filtered_position).tolist(),
                      velocity=np.asarray(agent.filtered_velocity).tolist(),
                      acceleration=np.asarray(agent.acceleration).tolist(),
                      imitation_eligible=imitation_eligible,
                      route_metadata_aligned=aligned, route_prefixes=prefixes,
                      forecast='raw_observed_velocity; controller may use wind-profile correction',
                      ends_v=[np.asarray(v).tolist() for v in agent.ends_v] if aligned else [],
                      ends_a=[np.asarray(a).tolist() for a in agent.ends_a] if aligned else [],
                      actuator_snapshot=dict(
                          gyro=_finite_vector(np.asarray(observation.get('rocket_sensors', ()))[:3], 3),
                          estimated_quaternion=_finite_vector(getattr(agent, 'quaternion', None), 4),
                          estimated_gimbal_output=_finite_vector(getattr(agent, 'previous_tvc', None), 2),
                          desired_axis=_finite_vector(getattr(agent, 'last_desired_axis', None), 3),
                          tvc_command=_finite_vector(action.get('tvc') if action else None, 2),
                          throttle_command=float(action['throttle']) if action else None,
                          allocator_saturated_total=int(getattr(agent, 'diagnostics', {}).get('saturated_steps', 0)),
                          max_gimbal_angle=float(agent.control['max_gimbal_angle'])
                          if hasattr(agent, 'control') else None))
        self.emit(record)
        self.counts['decisions'] += 1
        self.next_decision += 1
        self.last_signature = signature

    def finish(self, observation):
        if self.finished:
            return dict(self.counts)
        self.observe(observation)
        self._close('censored', float(observation['simulation_time']),
                    'episode_ended_before_committed_deadline')
        self.emit(dict(kind='episode_end', time=float(observation['simulation_time']),
                       final_popped_ids=np.flatnonzero(self.previous_status == 2).tolist(),
                       counts=dict(self.counts)))
        self.finished = True
        return dict(self.counts)
