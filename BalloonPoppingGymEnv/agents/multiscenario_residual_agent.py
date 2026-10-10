"""Shared residual controller with public physical context, not a scenario ID.

Schema intentionally differs from the old 96-feature policy. Old checkpoints
must not be resumed or deployed with this encoder. No seed or hidden weather
is encoded. Zero residual preserves the full-route refresh controller.
"""
import numpy as np
from BalloonPoppingGymEnv.agents.scenario4_residual_agent import (
    Scenario4ResidualController, scenario4_residual_features,
)
from BalloonPoppingGymEnv.agents.scenario4_joint_refresh_agent import Scenario4JointRefreshAgent

FEATURE_SCHEMA = 'multiscenario_residual_v1_112'
FEATURE_SIZE = 112


class MultiScenarioResidualController(Scenario4ResidualController, Scenario4JointRefreshAgent):
    def __init__(self, given_parameters, **kwargs):
        kwargs.setdefault('joint_refresh_allow_shortening', False)
        super().__init__(given_parameters, **kwargs)
        sensors = given_parameters['rocket']['sensors']
        c = self.control
        self.public_context = np.array([
            sensors['gnss_position_accuracy']/5., sensors['gnss_altitude_accuracy']/5.,
            sensors['gnss_velocity_accuracy']/.1, sensors['gyro_noise_density']/.01,
            (c['gimbal_time_constant'] or 0)/.1,
            (c['roll_torque_time_constant'] or 0)/.1,
            (c['throttle_time_constant'] or 0)/.1,
            c['max_gimbal_angle']/20., c['gimbal_rate_limit']/100.,
            self.thrust/self.initial_mass/20., self.burn_time/60.,
            float(given_parameters['balloon']['release_interval'] or 0),
        ])

    def get_action(self, observation):
        status = np.asarray(observation['balloon_status']).reshape(-1)
        target = self.target_index
        if (self.plan is None or target is None or not 0 <= target < len(status)
                or status[target] != 1):
            self.set_residual(np.zeros(3))
        return super().get_action(observation)


def multiscenario_features(agent, observation):
    now = float(observation['simulation_time'])
    remaining = agent.plan_start + agent.plan_duration - now if agent.plan is not None else 0.
    extra = np.r_[agent.public_context, remaining/10., len(agent.route)/20.,
                  agent.radius/3., float(agent.plan is not None)]
    return np.clip(np.nan_to_num(np.r_[scenario4_residual_features(agent, observation),
                                     extra]), -10., 10.).astype(np.float32)
