"""Observation-only beam search over dynamically continuous fly-throughs.

Every node carries time, velocity and acceleration. Terminal derivatives are
free for minimum-jerk interception, with optional braking alternatives. This
is a sampled reduced-model planner, not a full six-DOF optimality guarantee.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.physics_guidance_agent import (
    G, PhysicsGuidanceAgent, sample_curve,
)
from BalloonPoppingGymEnv.agents.spline_route_agent import boundary_curve


def flythrough_curves(p, v, a, targets, target_velocities, durations, braking):
    """Batch minimum-jerk position-only curves blended with stopped arrivals."""
    t = np.asarray(durations, dtype=float)[:, None]
    c = np.zeros((len(t), 6, 3))
    c[:, 0] = p
    c[:, 1] = np.asarray(v) * t
    c[:, 2] = np.asarray(a) * t**2 / 2
    dp = targets - c[:, 0] - c[:, 1] - c[:, 2]
    # Euler-Lagrange natural endpoint conditions give jerk(T)=snap(T)=0.
    c[:, 3] = 5 * dp / 3
    c[:, 4] = -5 * dp / 6
    c[:, 5] = dp / 6
    stopped = c.copy()
    dv = target_velocities * t - c[:, 1] - 2 * c[:, 2]
    da = -2 * c[:, 2]
    stopped[:, 3] = 10 * dp - 4 * dv + da / 2
    stopped[:, 4] = -15 * dp + 7 * dv - da
    stopped[:, 5] = 6 * dp - 3 * dv + da / 2
    mix = np.asarray(braking)[:, None, None]
    return c * (1 - mix) + stopped * mix


def batch_samples(curves, durations, samples=25):
    s = np.linspace(0., 1., samples)
    c = curves
    result = []
    for order in range(4):
        basis = s[:, None] ** np.arange(c.shape[1])
        result.append(np.einsum('sk,nkj->nsj', basis, c)
                      / np.asarray(durations)[:, None, None]**order)
        c = c[:, 1:] * np.arange(1, c.shape[1])[None, :, None]
    return result


class BeamInterceptAgent(PhysicsGuidanceAgent):
    def __init__(self, given_parameters, beam_width=20, search_depth=5,
                 branch_targets=10, time_grid=.5, leg_horizon=10.,
                 braking_options=(0., .5, 1.), reserve=0., **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.beam_width = int(beam_width)
        self.search_depth = int(search_depth)
        self.branch_targets = int(branch_targets)
        self.time_grid = float(time_grid)
        self.leg_horizon = float(leg_horizon)
        self.braking_options = np.asarray(braking_options, dtype=float)
        self.reserve = float(reserve)
        if min(self.beam_width, self.search_depth, self.branch_targets,
               self.time_grid, self.leg_horizon) <= 0:
            raise ValueError('Search limits must be positive')
        if np.any((self.braking_options < 0) | (self.braking_options > 1)):
            raise ValueError('Braking fractions must lie in [0,1]')
        self.end_velocity = self.end_acceleration = None
        self.route_events = []
        self.diagnostics.update(beam_searches=0, beam_candidates=0,
                                beam_feasible=0, maximum_depth=0)

    def _valid_batch(self, curves, durations, start):
        p, v, a, j = batch_samples(curves, durations)
        force = a - G - self.disturbance
        mag = np.linalg.norm(force, axis=2)
        t = start + durations[:, None] * np.linspace(0., 1., mag.shape[1])
        age = np.maximum(0., t - self.launch_time)
        mass = np.maximum(self.dry_mass, self.initial_mass-self.mass_flow*age)
        limit = np.where(age < self.burn_time, self.thrust/mass, 0.)
        axes = force / np.maximum(mag[:, :, None], 1e-9)
        rate = np.linalg.norm(j-axes*np.sum(axes*j, axis=2)[:, :, None], axis=2)
        rate /= np.maximum(mag, 1e-9)
        throttle = mag / np.maximum(limit, 1e-9)
        valid = np.all((mag <= limit-self.reserve) & (mag >= .5)
                       & (force[:, :, 2] >= mag*np.cos(self.max_tilt))
                       & (rate <= self.max_axis_rate)
                       & (p[:, :, 2] >= self.elevation-.05), axis=1)
        change = np.abs(np.diff(throttle, axis=1)) / np.diff(t, axis=1)
        valid &= np.all(change <= self.control['throttle_rate_limit'], axis=1)
        return valid, p[:, -1], v[:, -1], a[:, -1]

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1)==1)
        drift = states[:, 3:6].copy()
        moving = released[np.linalg.norm(drift[released], axis=1) > .5]
        if moving.size:
            drift[np.linalg.norm(drift, axis=1) <= .5] = np.median(drift[moving], axis=0)

        # Keep the arrival deadline and derivatives until the target is hit.
        if (self.plan is not None and self.target_index in released
                and self.end_velocity is not None):
            remaining = self.plan_start+self.plan_duration-now
            if remaining > .12:
                c = boundary_curve(position, velocity, self.acceleration,
                                   states[self.target_index, :3]+drift[self.target_index]*remaining,
                                   self.end_velocity, self.end_acceleration, remaining)
                if self._feasible(c, remaining, now):
                    self.plan, self.plan_start, self.plan_duration = c, now, remaining
                return
            self.failed_until[self.target_index] = now+2.
            self.target_index = None
            self.plan = None

        cutoff = self.launch_time+self.burn_time-1e-4
        # node: absolute time, p, v, a, route, first curve, first duration.
        beam = [(now, position, velocity, self.acceleration, (), None, None, (), ())]
        best = None
        self.diagnostics['beam_searches'] += 1
        for depth in range(self.search_depth):
            children = []
            for start, p, v, a, route, first_curve, first_duration, leg_times, leg_curves in beam:
                ids = [int(i) for i in released if i not in route
                       and self.failed_until.get(int(i), 0.) <= now]
                if not ids:
                    continue
                predicted = states[:, :3]+drift*(start-now)
                ids.sort(key=lambda i: np.linalg.norm(predicted[i]-p-v*1.5))
                ids = ids[:self.branch_targets]
                ts = np.arange(.5, min(self.leg_horizon, cutoff-start)+.001, self.time_grid)
                if not len(ts):
                    continue
                ii, tt, bb = np.meshgrid(ids, ts, self.braking_options, indexing='ij')
                ii, tt, bb = ii.ravel(), tt.ravel(), bb.ravel()
                targets = states[ii, :3]+drift[ii]*(start-now+tt[:, None])
                curves = flythrough_curves(p, v, a, targets, drift[ii], tt, bb)
                valid, ends, ev, ea = self._valid_batch(curves, tt, start)
                self.diagnostics['beam_candidates'] += len(tt)
                self.diagnostics['beam_feasible'] += int(np.sum(valid))
                for k in np.flatnonzero(valid):
                    children.append((start+tt[k], ends[k], ev[k], ea[k],
                                     route+(int(ii[k]),),
                                     curves[k] if depth==0 else first_curve,
                                     tt[k] if depth==0 else first_duration,
                                     leg_times+(float(tt[k]),), leg_curves+(curves[k],)))
            if not children:
                break
            # Keep competing route prefixes and terminal states, not only one ID.
            children.sort(key=lambda n: n[0])
            beam, quotas = [], {}
            for node in children:
                key = (node[4][0], node[4][-1])
                if quotas.get(key, 0) >= 2:
                    continue
                quotas[key] = quotas.get(key, 0)+1
                beam.append(node)
                if len(beam) >= self.beam_width:
                    break
            best = beam[0]
            self.diagnostics['maximum_depth'] = max(self.diagnostics['maximum_depth'], depth+1)
        if best is None:
            self.end_velocity = self.end_acceleration = None
            super()._make_plan(observation, position, velocity, now)
            return
        finish, _, _, _, route, c, duration, leg_times, leg_curves = best
        self.selected_leg_times = leg_times
        self.selected_leg_curves = np.asarray(leg_curves)
        target = route[0]
        if target != self.target_index:
            self.target_events.append((now, target))
        self.target_index = target
        self.plan, self.plan_start, self.plan_duration = c, now, float(duration)
        _, self.end_velocity, self.end_acceleration, _ = sample_curve(c, duration, duration)
        self.diagnostics['plans'] += 1
        self.route_events.append([float(now), list(route), float(finish-now)])
