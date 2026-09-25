"""Full-power interception by online shooting from observed balloon states.

For each lateral minimum-jerk intercept, integrate vertical acceleration from
the thrust sphere. Solve intercept time so the vertical miss vanishes. Beam
search retains terminal momentum. All predictions originate in current sensors.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.momentum_beam_agent import MomentumBeamAgent, arrival_curves
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import G, batch_samples, boundary_curve


class PoweredAscentAgent(MomentumBeamAgent):
    def __init__(self, given_parameters, power_options=(.98, .90),
                 minimum_climb=-1., **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.power_options = np.asarray(power_options, dtype=float)
        self.minimum_climb = float(minimum_climb)
        if np.any((self.power_options <= 0) | (self.power_options > 1)):
            raise ValueError('Power fractions must be in (0,1]')

    def _shoot(self, p, v, a, targets, drift, durations, powers, start):
        n = len(durations)
        curves = arrival_curves(p, v, a, targets, drift, durations, np.zeros(n))
        if self.arrival_model != 'natural':
            # Relax lateral acceleration after each hit while retaining speed.
            tt = durations[:, None]
            end_v = 2*(targets-p)/tt-v
            dp = targets-curves[:, 0]-curves[:, 1]-curves[:, 2]
            dv = end_v*tt-curves[:, 1]-2*curves[:, 2]
            da = -2*curves[:, 2]
            curves[:, 3] = 10*dp-4*dv+da/2
            curves[:, 4] = -15*dp+7*dv-da
            curves[:, 5] = 6*dp-3*dv+da/2
        _, vs, acs, _ = batch_samples(curves, durations, 21)
        s = np.linspace(0., 1., 21)
        age = np.maximum(0., start-self.launch_time+durations[:, None]*s)
        thrust = self.thrust/np.maximum(self.dry_mass, self.initial_mass-self.mass_flow*age)
        lateral = acs[:, :, :2]-self.disturbance[:2]
        remaining = (thrust*powers[:, None])**2-np.sum(lateral*lateral, axis=2)
        az = np.sqrt(np.maximum(remaining, 0))+G[2]+self.disturbance[2]
        weights = np.ones(21)
        weights[1:-1:2] = 4
        weights[2:-1:2] = 2
        weights /= 60.
        zend = p[2]+v[2]*durations+durations**2*(az @ (weights*(1-s)))
        vzend = v[2]+durations*(az @ weights)
        valid = np.all(remaining > .25, axis=1)
        valid &= np.all(az >= self.minimum_climb, axis=1)
        return zend-targets[:, 2], valid, curves, vs[:, -1], acs[:, -1], vzend, az[:, -1]

    def _branches(self, p, v, a, states, drift, ids, start, now):
        cutoff = self.launch_time+self.burn_time-1e-4
        ts = np.arange(.5, min(self.leg_horizon, cutoff-start)+.001, self.time_grid)
        if len(ts) < 2:
            return []
        ii, pp, tt = np.meshgrid(ids, self.power_options, ts, indexing='ij')
        ii, pp, tt = ii.ravel(), pp.ravel(), tt.ravel()
        targets = states[ii, :3]+drift[ii]*(start-now+tt[:, None])
        error, valid, *_ = self._shoot(p, v, a, targets, drift[ii], tt, pp, start)
        errors = error.reshape(-1, len(ts))
        validity = valid.reshape(-1, len(ts))
        row, col = np.where((errors[:, :-1]*errors[:, 1:] <= 0)
                            & validity[:, :-1] & validity[:, 1:])
        if not len(row):
            return []
        flat = row*len(ts)+col
        ids2, powers = ii[flat], pp[flat]
        lo, hi = tt[flat].copy(), tt[flat+1].copy()
        elo, ehi = error[flat].copy(), error[flat+1].copy()
        for _ in range(5):
            frac = np.clip(-elo/np.where(np.abs(ehi-elo)>1e-10, ehi-elo, 1e-10), .05, .95)
            durations = lo+(hi-lo)*frac
            targets = states[ids2, :3]+drift[ids2]*(start-now+durations[:, None])
            err, ok, curves, ev, ea, vz, az = self._shoot(p, v, a, targets, drift[ids2], durations, powers, start)
            upper = elo*err <= 0
            hi, ehi = np.where(upper, durations, hi), np.where(upper, err, ehi)
            lo, elo = np.where(upper, lo, durations), np.where(upper, elo, err)
        # The vertical polynomial matches the shooting integral's endpoint state.
        ev[:, 2], ea[:, 2] = vz, az
        for k in range(len(durations)):
            zcurve = boundary_curve(p, v, a, targets[k], ev[k], ea[k], durations[k])
            curves[k, :, 2] = zcurve[:, 2]
        feasible, ends, ev, ea = self._valid(curves, durations, start, samples=33)
        feasible &= ok & (np.abs(err)<.05)
        self.diagnostics['beam_candidates'] += len(ii)
        self.diagnostics['beam_feasible'] += int(feasible.sum())
        return [(int(ids2[k]), float(durations[k]), curves[k], ends[k], ev[k], ea[k])
                for k in np.flatnonzero(feasible)]

    def _make_plan(self, observation, position, velocity, now):
        # The sensor-derived initial acceleration and the reduced thrust model
        # can disagree slightly near saturation. Project this reference into
        # the model's reachable set before expanding future trajectories.
        initial_force = self.acceleration-G-self.disturbance
        force_norm = np.linalg.norm(initial_force)
        force_limit = .97*self.available_acceleration(now)
        if force_norm > force_limit and force_limit > 0:
            self.acceleration = G+self.disturbance+initial_force*force_limit/force_norm
        states = np.asarray(observation['balloon_states'], dtype=float)
        status = np.asarray(observation['balloon_status']).reshape(-1)
        released = np.flatnonzero(status == 1)
        drift = states[:, 3:6]
        while self.route and status[self.route[0]] != 1:
            self.route.pop(0)
            self.deadlines.pop(0)
            self.ends_v.pop(0)
            self.ends_a.pop(0)
        if self.route and (self.commitment or self.target_index == self.route[0]):
            duration = self.deadlines[0]-now
            if duration > .15:
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
                estimates = []
                for look in (1., 2., 4., 7.):
                    delta = projected[ids]+drift[ids]*look-p-v*look
                    estimates.append(np.linalg.norm(delta, axis=1)/look**2+look*.12)
                ids = np.asarray(ids)[np.argsort(np.min(estimates, axis=0))[:self.branch_targets]]
                for i, t, c, ep, ev, ea in self._branches(p, v, a, states, drift, ids, start, now):
                    children.append((start+t, ep, ev, ea, route+(i,), cs+(c,),
                                     ts+(t,), evs+(ev,), eas+(ea,)))
            if not children:
                break
            children.sort(key=lambda n:n[0])
            beam, bins = [], set()
            for node in children:
                key = (node[4][0], node[4][-1], tuple(np.round(node[2]/self.velocity_bin).astype(int)))
                if key in bins:
                    continue
                bins.add(key)
                beam.append(node)
                if len(beam) >= self.beam_width:
                    break
            best = beam[0]
            self.diagnostics['maximum_depth'] = max(self.diagnostics['maximum_depth'], depth+1)
        if best is None:
            # Continue a modest ascent while new observations expose opportunities.
            if self.plan is not None and now < self.plan_start+self.plan_duration:
                return
            accel = np.array([0., 0., min(1., max(0., self.available_acceleration(now)-9.80665))])
            target = position+velocity*.5+accel*.5**2/2
            self.plan = boundary_curve(position, velocity, self.acceleration, target,
                                       velocity+accel*.5, accel, .5)
            self.plan_start, self.plan_duration, self.target_index = now, .5, None
            self.diagnostics['no_feasible_plan'] += 1
            return
        finish, _, _, _, route, curves, durations, evs, eas = best
        self.route, self.deadlines = list(route), list(now+np.cumsum(durations))
        self.ends_v, self.ends_a = list(evs), list(eas)
        self.route_events.append([now, list(route), float(finish-now)])
        self._select(route[0], curves[0], durations[0], now)
