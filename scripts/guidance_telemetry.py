"""Read-only development observer; its measurements are never fed to agents.

Use observations and copies of the agent's current reference, not simulator
internals. At a common sample, rocket-balloon error decomposes exactly into
tracking + reference/prediction alignment + balloon-prediction error.
"""

import numpy as np


def reference_position(curve, duration, elapsed):
    return (float(elapsed)/duration)**np.arange(len(curve)) @ curve


class GuidanceTelemetry:
    def __init__(self):
        self.episodes = []
        self.current = None
        self.reference = None
        self.forecasts = []
        self.finished_forecasts = []
        self.command = None

    def before_step(self, agent, observation, action):
        target = getattr(agent, 'target_index', None)
        plan = getattr(agent, 'plan', None)
        now = float(observation['simulation_time'])
        if self.current is not None and target != self.current['target']:
            self.current['end_time'] = now
            self.current['end_reason'] = 'target_changed'
            self.current = None
        if target is None or plan is None:
            self.reference = None
            return
        target = int(target)
        if self.current is None:
            self.current = dict(target=target, start_time=now, closest=None,
                                max_tracking_error=0., samples=0)
            self.episodes.append(self.current)
        key = (target, float(agent.plan_start), float(agent.plan_duration))
        if self.reference is None or self.reference['key'] != key:
            state = np.asarray(observation['balloon_states'][target], dtype=float).copy()
            curve = np.asarray(plan, dtype=float).copy()
            deadline = key[1]+key[2]
            endpoint = reference_position(curve, key[2], key[2])
            self.reference = dict(key=key, curve=curve, origin_time=now, balloon=state,
                                  deadline=deadline)
            self.forecasts.append(dict(target=target, origin_time=now, deadline=deadline,
                                       endpoint=endpoint.copy(), balloon=state.copy()))
        self.command = dict(throttle=float(action['throttle']),
                            tvc=np.asarray(action['tvc'], dtype=float).tolist(),
                            available_acceleration=float(agent.available_acceleration(now)))

    def after_step(self, observation):
        now = float(observation['simulation_time'])
        states = np.asarray(observation['balloon_states'], dtype=float)
        status = np.asarray(observation['balloon_status']).reshape(-1)
        pending = []
        for forecast in self.forecasts:
            target = forecast['target']
            if status[target] == 2:
                continue  # Popped before forecast horizon: not a missed intercept.
            if now < forecast['deadline']:
                pending.append(forecast)
                continue
            if status[target] == 1:
                dt = now-forecast['origin_time']
                prediction = forecast['balloon'][:3]+forecast['balloon'][3:6]*dt
                # Compare the endpoint at its deadline to a short interpolation
                # back from this sample; the crossing offset is <= one step.
                observed_at_deadline = states[target, :3]-states[target, 3:6]*(now-forecast['deadline'])
                self.finished_forecasts.append(dict(target=target,
                    origin_time=forecast['origin_time'], deadline=forecast['deadline'],
                    horizon=forecast['deadline']-forecast['origin_time'],
                    sample_offset=now-forecast['deadline'],
                    velocity_prediction_error=float(np.linalg.norm(prediction-states[target, :3])),
                    endpoint_error=float(np.linalg.norm(forecast['endpoint']-observed_at_deadline))))
        self.forecasts = pending
        if self.current is None or self.reference is None:
            return
        target = self.current['target']
        sensors = np.asarray(observation['rocket_sensors'], dtype=float)
        if not np.all(np.isfinite(sensors[6:12])):
            return
        ref = self.reference
        _, start, duration = ref['key']
        reference = reference_position(ref['curve'], duration, np.clip(now-start, 0., duration))
        prediction = ref['balloon'][:3]+ref['balloon'][3:6]*(now-ref['origin_time'])
        tracking = sensors[6:9]-reference
        alignment = reference-prediction
        prediction_error = prediction-states[target, :3]
        distance = float(np.linalg.norm(sensors[6:9]-states[target, :3]))
        self.current['samples'] += 1
        self.current['max_tracking_error'] = max(self.current['max_tracking_error'], float(np.linalg.norm(tracking)))
        if self.current['closest'] is None or distance < self.current['closest']['distance']:
            self.current['closest'] = dict(time=now, distance=distance,
                time_to_deadline=ref['deadline']-now, tracking_vector=tracking.tolist(),
                alignment_vector=alignment.tolist(), prediction_error_vector=prediction_error.tolist(),
                tracking_error=float(np.linalg.norm(tracking)),
                alignment_error=float(np.linalg.norm(alignment)),
                prediction_error=float(np.linalg.norm(prediction_error)), command=self.command)
        if status[target] == 2:
            self.current['end_time'], self.current['end_reason'] = now, 'popped'
            self.current = None
            self.reference = None

    def finish(self, now):
        if self.current is not None:
            self.current['end_time'], self.current['end_reason'] = float(now), 'episode_ended'
        return dict(scope='Observation/reference-only diagnostics, never provided to the agent; distances are sampled, not official collision checks.',
                    target_episodes=self.episodes, unpopped_forecast_checks=self.finished_forecasts)
