"""Combine observation-only wind-profile forecasting and sensor fusion.

This is an experimental composition, not a selected competition policy.
Both parents ultimately inherit ``TimeAllocationAgent`` and use only the
current observation, retained observation history, and given parameters.
"""

from BalloonPoppingGymEnv.agents.scenario4_fusion_agent import Scenario4FusionAgent
from BalloonPoppingGymEnv.agents.scenario4_wind_profile_agent import Scenario4WindProfileAgent


class Scenario4WindFusionAgent(Scenario4WindProfileAgent, Scenario4FusionAgent):
    """Use altitude-aware balloon drift and gravity-corrected IMU/GNSS fusion."""
