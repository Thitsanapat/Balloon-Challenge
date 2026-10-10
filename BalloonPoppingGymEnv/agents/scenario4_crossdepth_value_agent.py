"""Experimental across-depth route-value selection, with the frozen GNC intact.

Collect a bounded set of nodes through the existing priority hook WITHOUT
changing its search order. Short and long densely feasible routes then compete.
The six-feature model was calibrated on executed guarded-policy first legs;
later reference-state predictions and their discounted sum are a SURROGATE,
not independently calibrated multi-leg hit probabilities. No hidden simulator
state, seed, replay, or outcome is available to this agent.
"""
import json
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.submission_scenario4_pure_joint_v2 import (
    Scenario4PureJointRefreshAgent, free_chain, sample_curve,
)
from BalloonPoppingGymEnv.agents.scenario4_learned_route_agent import (
    FrozenCaptureModel, capture_features,
)


def discounted_capture_value(probabilities, durations, discount_seconds,
                             failed_entry_cost_per_second):
    p, times = np.asarray(probabilities, float), np.asarray(durations, float)
    if (p.ndim != 1 or times.shape != p.shape or not len(p) or
            not np.all(np.isfinite(p)) or not np.all(np.isfinite(times)) or
            np.any((p < 0) | (p > 1)) or np.any(times <= 0) or
            not np.isfinite(discount_seconds) or discount_seconds <= 0 or
            not np.isfinite(failed_entry_cost_per_second) or
            failed_entry_cost_per_second < 0):
        raise ValueError('Finite probabilities, positive durations/discount required')
    # Marginal probabilities are summed, not multiplied: a miss need not end
    # the flight or make every subsequent target unreachable.
    return float(np.dot(p, np.exp(-np.cumsum(times)/discount_seconds)) -
                 failed_entry_cost_per_second*times[0]*(1.-p[0]))


class Scenario4CrossDepthValueAgent(Scenario4PureJointRefreshAgent):
    def __init__(self, given_parameters, model_path, value_discount_seconds=20.,
                 failed_entry_cost_per_second=.08, proposals_per_depth=3,
                 value_adoption_margin=.05, max_first_feature_z=3., **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.capture_model = FrozenCaptureModel(json.loads(
            Path(model_path).read_text(encoding='utf-8')))
        self.value_discount_seconds = float(value_discount_seconds)
        self.failed_entry_cost_per_second = float(failed_entry_cost_per_second)
        self.proposals_per_depth = int(proposals_per_depth)
        self.value_adoption_margin = float(value_adoption_margin)
        self.max_first_feature_z = float(max_first_feature_z)
        discounted_capture_value([.5], [1.], self.value_discount_seconds,
                                 self.failed_entry_cost_per_second)
        if (self.proposals_per_depth < 1 or self.proposals_per_depth > 8 or
                not np.isfinite(self.value_adoption_margin) or
                self.value_adoption_margin < 0 or
                not np.isfinite(self.max_first_feature_z) or
                self.max_first_feature_z <= 0):
            raise ValueError('Invalid bounded route-value settings')
        self._value_context = None
        self._value_pool = None
        self._value_states = None
        self._last_route_value = None
        self.value_events = []
        self.diagnostics.update(value_nodes_scored=0, value_pool_peak=0,
                                value_routes_adopted=0, value_shorter_adopted=0,
                                value_unsupported_first_legs=0)

    def _route_value(self, node, states, now):
        route, times, curves = node[:3]
        if not route or curves is None or self._value_context is None:
            return None
        position, velocity = self._value_context
        probabilities = []
        elapsed = 0.
        for leg, (target, duration) in enumerate(zip(route, times)):
            if leg:
                position, velocity, _, _ = sample_curve(curves[leg], duration, 0.)
            state = np.asarray(states[target], float).copy()
            state[:3] += state[3:6]*elapsed
            available = self.available_acceleration(now+elapsed)
            if available <= 0:
                return None
            features = capture_features(state, position, velocity, duration, available)
            z = (features-self.capture_model.center)/self.capture_model.scale
            if leg == 0 and np.max(np.abs(z)) > self.max_first_feature_z:
                self.diagnostics['value_unsupported_first_legs'] += 1
                return None
            probabilities.append(self.capture_model.probability(features))
            elapsed += duration
        return discounted_capture_value(probabilities, times,
                                        self.value_discount_seconds,
                                        self.failed_entry_cost_per_second)

    def _node_priority(self, node, states, released, now):
        original = super()._node_priority(node, states, released, now)
        if self._value_pool is None:
            return original
        value = self._route_value(node, states, now)
        self.diagnostics['value_nodes_scored'] += 1
        if value is None or not np.isfinite(value):
            return original
        self._value_states = states
        bucket = self._value_pool.setdefault(len(node[0]), [])
        bucket.append((value, node))
        bucket.sort(key=lambda item: (-item[0], sum(item[1][1])))
        del bucket[self.proposals_per_depth:]
        self.diagnostics['value_pool_peak'] = max(
            self.diagnostics['value_pool_peak'], sum(map(len, self._value_pool.values())))
        return original

    def _make_plan(self, observation, position, velocity, now):
        if not self.launched:
            # Isolate route selection after launch. The launch-axis comparisons
            # remain byte-for-byte incumbent behavior in this first ablation.
            return super()._make_plan(observation, position, velocity, now)
        self._value_context = (np.asarray(position, float).copy(),
                               np.asarray(velocity, float).copy())
        self._value_pool, self._value_states = {}, None
        self._last_route_value = None
        try:
            event_count = len(self.route_events)
            super()._make_plan(observation, position, velocity, now)
            if len(self.route_events) == event_count or not self._value_pool:
                return
            states = self._value_states
            original_route = tuple(self.route)
            times = tuple(np.diff(np.r_[now, self.deadlines]))
            targets = states[list(original_route), :3] + (
                states[list(original_route), 3:6]*np.cumsum(times)[:, None])
            curves = free_chain(position, velocity, self.acceleration, targets,
                                times, states[original_route[-1], 3:6],
                                self.terminal_weight)
            incumbent = (original_route, times, curves, None, None)
            original_value = self._route_value(incumbent, states, now)
            if original_value is None:
                return  # Out-of-support incumbent is not overridden.
            self._last_route_value = original_value
            candidates = [item for bucket in self._value_pool.values() for item in bucket]
            candidates.sort(key=lambda item: (-item[0], sum(item[1][1])))
            for value, node in candidates:
                if value <= original_value+self.value_adoption_margin:
                    break
                route, durations, reference = node[:3]
                # Recheck every leg densely; learned value never bypasses physics.
                starts = now+np.r_[0., np.cumsum(durations)[:-1]]
                valid, _, ends_v, ends_a = self._valid(
                    reference, np.asarray(durations), starts[:, None], samples=65)
                if not np.all(valid):
                    continue
                self.route, self.deadlines = list(route), list(now+np.cumsum(durations))
                self.ends_v, self.ends_a = list(ends_v), list(ends_a)
                self.route_events[-1] = [now, list(route), float(sum(durations))]
                self._select(route[0], reference[0], durations[0], now)
                self._last_route_value = value
                self.diagnostics['value_routes_adopted'] += 1
                self.diagnostics['value_shorter_adopted'] += int(len(route) < len(original_route))
                self.value_events.append(dict(time=float(now),
                    original_count=len(original_route), selected_count=len(route),
                    original_value=original_value, selected_value=value))
                break
        finally:
            self._value_context = self._value_pool = self._value_states = None
