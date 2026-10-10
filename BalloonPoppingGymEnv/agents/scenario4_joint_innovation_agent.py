"""Controlled composition of full-route refresh and observed forecast feedback.

Each component keeps its existing public-parameter/observation-only behavior.
This experiment tests their interaction; benefits are not assumed additive.
"""
from BalloonPoppingGymEnv.agents.scenario4_joint_refresh_agent import Scenario4JointRefreshAgent
from BalloonPoppingGymEnv.agents.scenario4_forecast_innovation_agent import Scenario4ForecastInnovationAgent


class Scenario4JointInnovationAgent(Scenario4JointRefreshAgent, Scenario4ForecastInnovationAgent):
    def __init__(self, given_parameters, **kwargs):
        kwargs.setdefault('joint_refresh_allow_shortening', False)
        super().__init__(given_parameters, **kwargs)
