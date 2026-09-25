"""Online beam search with free arrival velocity and state-diverse pruning.

All target predictions are extrapolated from the current observation. No field,
seed, release schedule, or trajectory is loaded. Reduced-model plans are checked
against thrust, attitude-rate and throttle-rate limits, then tracked with GNC.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    PhysicsGuidanceAgent, G, boundary_curve, sample_curve, batch_samples,
)


def arrival_curves(p, v, a, targets, drift, durations, braking):
    """Minimum jerk with free final derivatives, blended with drift arrivals."""
    t = np.asarray(durations)[:, None]
    c = np.zeros((len(t), 6, 3))
    c[:, 0], c[:, 1], c[:, 2] = p, v*t, a*t*t/2
    dp = targets-c[:, 0]-c[:, 1]-c[:, 2]
    dv = drift*t-c[:, 1]-2*c[:, 2]
    da = -2*c[:, 2]
    natural = np.stack((5*dp/3, -5*dp/6, dp/6), axis=1)
    stopped = np.stack((10*dp-4*dv+da/2, -15*dp+7*dv-da,
                        6*dp-3*dv+da/2), axis=1)
    mix = np.asarray(braking)[:, None, None]
    c[:, 3:] = natural*(1-mix)+stopped*mix
    return c


class MomentumBeamAgent(PhysicsGuidanceAgent):
    def __init__(self, given_parameters, beam_width=36, search_depth=8,
                 branch_targets=12, time_grid=.5, leg_horizon=10.,
                 braking_options=(0., .25, .5, .75, 1.), capture_fraction=.25,
                 reserve=0., commitment=True, velocity_bin=5.,
                 arrival_model='natural', first_quota=0, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.beam_width = int(beam_width)
        self.search_depth = int(search_depth)
        self.branch_targets = int(branch_targets)
        self.time_grid = float(time_grid)
        self.leg_horizon = float(leg_horizon)
        self.braking_options = np.asarray(braking_options)
        self.capture_fraction = float(capture_fraction)
        self.reserve = float(reserve)
        self.commitment = bool(commitment)
        self.velocity_bin = float(velocity_bin)
        self.arrival_model = str(arrival_model)
        self.first_quota = int(first_quota)
        if min(self.beam_width, self.search_depth, self.branch_targets,
               self.time_grid, self.leg_horizon, self.velocity_bin) <= 0:
            raise ValueError('Positive search limits required')
        if not 0 <= self.capture_fraction < 1:
            raise ValueError('Capture fraction must be in [0,1)')
        self.route = []
        self.deadlines = []
        self.ends_v = []
        self.ends_a = []
        self.route_events = []
        self.diagnostics.update(beam_searches=0, beam_candidates=0,
                                beam_feasible=0, maximum_depth=0)

    def _valid(self, curves, durations, start, samples=17):
        p, v, a, j = batch_samples(curves, durations, samples)
        force = a-G-self.disturbance
        mag = np.linalg.norm(force, axis=2)
        times = start+durations[:, None]*np.linspace(0, 1, samples)
        age = np.maximum(times-self.launch_time, 0.)
        limit = self.thrust/np.maximum(self.dry_mass, self.initial_mass-self.mass_flow*age)
        axes = force/np.maximum(mag[:, :, None], 1e-9)
        rate = np.linalg.norm(j-axes*np.sum(axes*j, axis=2)[:, :, None], axis=2)/np.maximum(mag, 1e-9)
        throttle = mag/limit
        # Reserve is tapered at the initial condition, which cannot be changed.
        reserve = self.reserve*np.linspace(0, 1, samples)
        valid = np.all((mag <= limit-reserve+1e-6) & (mag >= .5)
                       & (axes[:, :, 2] >= np.cos(self.max_tilt))
                       & (rate <= self.max_axis_rate)
                       & (p[:, :, 2] >= self.elevation-.05), axis=1)
        valid &= np.all(np.abs(np.diff(throttle, axis=1))/np.diff(times, axis=1)
                        <= self.control['throttle_rate_limit'], axis=1)
        return valid, p[:, -1], v[:, -1], a[:, -1]

    def _make_plan(self, observation, position, velocity, now):
        states = np.asarray(observation['balloon_states'], dtype=float)
        status = np.asarray(observation['balloon_status']).reshape(-1)
        released = np.flatnonzero(status == 1)
        drift = states[:, 3:6]
        while self.route and status[self.route[0]] != 1:
            self.route.pop(0)
            self.deadlines.pop(0)
            self.ends_v.pop(0)
            self.ends_a.pop(0)
        # Preserve an intercept deadline rather than repeatedly postponing it.
        if self.route and (self.commitment or self.target_index == self.route[0]):
            duration = self.deadlines[0]-now
            if duration > .12:
                target = states[self.route[0], :3]+drift[self.route[0]]*duration
                c = boundary_curve(position, velocity, self.acceleration, target,
                                   self.ends_v[0], self.ends_a[0], duration)
                if self._feasible(c, duration, now):
                    self._select(self.route[0], c, duration, now)
                    return
                if self.plan is not None and self.target_index == self.route[0] and now < self.plan_start+self.plan_duration:
                    return
            else:
                self.failed_until[self.route[0]] = now+1.
        self.route, self.deadlines, self.ends_v, self.ends_a = [], [], [], []
        cutoff = self.launch_time+self.burn_time-1e-4
        # time, position, velocity, acceleration, IDs, curves, leg durations, ends
        beam = [(now, position, velocity, self.acceleration, (), (), (), (), ())]
        best = None
        self.diagnostics['beam_searches'] += 1
        for depth in range(self.search_depth):
            children = []
            for start, p, v, a, route, cs, ts, evs, eas in beam:
                ids = [int(i) for i in released if i not in route
                       and self.failed_until.get(int(i), 0) <= now]
                if not ids:
                    continue
                projected = states[:, :3]+drift*(start-now)
                # Rank by required change of velocity at several intercept times.
                estimates = []
                for look in (1., 2., 4., 7.):
                    delta = projected[ids]+drift[ids]*look-p-v*look
                    estimates.append(np.linalg.norm(delta, axis=1)/look**2+look*.12)
                order = np.argsort(np.min(estimates, axis=0))[:self.branch_targets]
                ids = np.asarray(ids)[order]
                durations = np.arange(.5, min(self.leg_horizon, cutoff-start)+.001, self.time_grid)
                if not len(durations):
                    continue
                ii, tt, bb = np.meshgrid(ids, durations, self.braking_options, indexing='ij')
                ii, tt, bb = ii.ravel(), tt.ravel(), bb.ravel()
                targets = projected[ii]+drift[ii]*tt[:, None]
                ballistic = p+v*tt[:, None]+a*tt[:, None]**2/2
                delta = ballistic-targets
                lengths = np.linalg.norm(delta, axis=1)
                targets += delta*(np.minimum(self.radius*self.capture_fraction, lengths)/np.maximum(lengths, 1e-9))[:, None]
                curves = arrival_curves(p, v, a, targets, drift[ii], tt, bb)
                if self.arrival_model != 'natural':
                    # Preserve the constant-acceleration intercept velocity,
                    # independently allowing the endpoint acceleration to relax.
                    end_v = 2*(targets-p)/tt[:, None]-v
                    end_v = end_v*(1-bb[:, None])+drift[ii]*bb[:, None]
                    end_a = np.zeros_like(end_v)
                    if self.arrival_model == 'climb':
                        end_a[:, 2] = 1.5
                    c0, c1, c2 = p, v*tt[:, None], a*tt[:, None]**2/2
                    dp = targets-c0-c1-c2
                    dv = end_v*tt[:, None]-c1-2*c2
                    da = end_a*tt[:, None]**2-2*c2
                    curves[:, 3] = 10*dp-4*dv+da/2
                    curves[:, 4] = -15*dp+7*dv-da
                    curves[:, 5] = 6*dp-3*dv+da/2
                valid, ends, ev, ea = self._valid(curves, tt, start)
                self.diagnostics['beam_candidates'] += len(tt)
                self.diagnostics['beam_feasible'] += int(valid.sum())
                for k in np.flatnonzero(valid):
                    children.append((start+tt[k], ends[k], ev[k], ea[k],
                                     route+(int(ii[k]),), cs+(curves[k],),
                                     ts+(float(tt[k]),), evs+(ev[k],), eas+(ea[k],)))
            if not children:
                break
            children.sort(key=lambda n: n[0])
            beam, bins, quotas = [], set(), {}
            for node in children:
                # Keep physically different arrival states at the same balloon.
                key = (node[4][0], node[4][-1],
                       tuple(np.round(node[2]/self.velocity_bin).astype(int)))
                if key in bins:
                    continue
                if self.first_quota and quotas.get(node[4][0], 0) >= self.first_quota:
                    continue
                bins.add(key)
                quotas[node[4][0]] = quotas.get(node[4][0], 0)+1
                beam.append(node)
                if len(beam) >= self.beam_width:
                    break
            best = beam[0]
            self.diagnostics['maximum_depth'] = max(self.diagnostics['maximum_depth'], depth+1)
        if best is None:
            return super()._make_plan(observation, position, velocity, now)
        finish, _, _, _, route, curves, durations, evs, eas = best
        if not self._feasible(curves[0], durations[0], now):
            return super()._make_plan(observation, position, velocity, now)
        self.route = list(route)
        self.deadlines = list(now+np.cumsum(durations))
        self.ends_v, self.ends_a = list(evs), list(eas)
        self.route_events.append([now, list(route), float(finish-now)])
        self._select(route[0], curves[0], durations[0], now)

    def _select(self, index, curve, duration, now):
        if self.target_index != index:
            self.target_events.append((now, int(index)))
        self.target_index = int(index)
        self.plan, self.plan_start, self.plan_duration = curve, now, float(duration)
        self.diagnostics['plans'] += 1
