"""Short-horizon optimal tracking correction followed by frozen residual PPO.

All references come from the agent's current observation-derived plan. The
MPC uses public mass/thrust parameters and filtered sensors; it never reads
the environment object, future balloon trajectories, truth info or seed.
Parent controller/source and frozen policy weights remain unchanged.
"""
import numpy as np

from BalloonPoppingGymEnv.agents.multiscenario_ppo_policy_agent import MultiScenarioPPOPolicyAgent
from BalloonPoppingGymEnv.agents.preview_mpc import preview_correction
from BalloonPoppingGymEnv.agents.submission_sqp_v2 import allocate_acceleration, sample_curve, G


class PreviewMPCPPOAgent(MultiScenarioPPOPolicyAgent):
    def __init__(self, given_parameters, checkpoint, *, mpc_gain=1.,
                 mpc_bound=1.5, mpc_interval=0.12, mpc_steps=8,
                 mpc_refresh=0.2, use_ppo=True):
        super().__init__(given_parameters, checkpoint=checkpoint)
        self.mpc_gain = float(mpc_gain)
        self.mpc_bound = float(mpc_bound)
        self.mpc_interval = float(mpc_interval)
        self.mpc_steps = int(mpc_steps)
        self.mpc_refresh = float(mpc_refresh)
        self.use_ppo = bool(use_ppo)
        if (not np.isfinite(self.mpc_gain) or not 0 <= self.mpc_gain <= 1
                or not np.isfinite(self.mpc_bound) or not 0 < self.mpc_bound <= 3
                or not np.isfinite(self.mpc_interval) or not .05 <= self.mpc_interval <= .25
                or not 2 <= self.mpc_steps <= 16
                or not np.isfinite(self.mpc_refresh) or not .05 <= self.mpc_refresh <= .5):
            raise ValueError('Invalid bounded preview MPC settings')
        self._mpc_now = 0.
        self._mpc_next = 0.
        self._mpc_target = None
        self._mpc_delta = np.zeros(3)
        self.mpc_diagnostics = dict(solves=0, nonzero=0, skipped_short_plan=0)

    def set_residual(self, action):
        return super().set_residual(action if self.use_ppo else np.zeros(3))

    def get_action(self, observation):
        self._mpc_now = float(observation['simulation_time'])
        return super().get_action(observation)

    def _attitude_action(self, axis, jerk, magnitude, gyro, available):
        now = self._mpc_now
        if self._mpc_target != self.target_index:
            self._mpc_target = self.target_index
            self._mpc_next = 0.
            self._mpc_delta = np.zeros(3)
        if (self.mpc_gain > 0 and self.plan is not None and self.target_index is not None
                and self.filtered_position is not None and self.filtered_velocity is not None
                and available > 0 and now + 1e-8 >= self._mpc_next):
            elapsed = max(0., now-self.plan_start)
            count = min(self.mpc_steps, int((self.plan_duration-elapsed+1e-9)/self.mpc_interval))
            if count >= 2:
                future = elapsed + self.mpc_interval*np.arange(1, count+1)
                reference_p, reference_v, _, _ = sample_curve(
                    self.plan, self.plan_duration, future)
                force = np.asarray(axis, dtype=float)*float(magnitude)
                nominal_a = force + G + self.disturbance
                self._mpc_delta = preview_correction(
                    self.filtered_position, self.filtered_velocity, nominal_a,
                    reference_p, reference_v, self.mpc_interval, self.mpc_bound,
                    previous=self._mpc_delta)
                self.mpc_diagnostics['solves'] += 1
                self.mpc_diagnostics['nonzero'] += int(np.linalg.norm(self._mpc_delta) > 1e-6)
            else:
                self._mpc_delta = np.zeros(3)
                self.mpc_diagnostics['skipped_short_plan'] += 1
            self._mpc_next = now + self.mpc_refresh
        if self.mpc_gain > 0 and np.any(self._mpc_delta) and available > 0:
            force = allocate_acceleration(
                np.asarray(axis, dtype=float)*float(magnitude)
                + self.mpc_gain*self._mpc_delta, available, self.max_tilt)
            magnitude = float(np.linalg.norm(force))
            axis = force/max(magnitude, 1e-9) if magnitude > 1e-9 else np.array([0., 0., 1.])
        # Parent adds the frozen PPO residual after this constrained preview.
        return super()._attitude_action(axis, jerk, magnitude, gyro, available)
