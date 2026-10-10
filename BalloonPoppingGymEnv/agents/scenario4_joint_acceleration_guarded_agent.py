"""Experimental composition of causal target acceleration and route rescue.

The first route search remains the selected guarded agent's wind-profile
search.  Only an already committed target can receive the observation-derived
short-horizon acceleration residual.  A failed committed refresh may release
its arrival derivatives without dropping any planned targets.  This module
does not read seed, simulator internals, truth, or prerecorded trajectories.
"""

from BalloonPoppingGymEnv.agents.scenario4_acceleration_wind_agent import (
    Scenario4AccelerationWindAgent,
)
from BalloonPoppingGymEnv.agents.scenario4_reachable_refresh_agent import (
    Scenario4ReachableRefreshAgent,
)


class Scenario4JointAccelerationGuardedAgent(
        Scenario4AccelerationWindAgent, Scenario4ReachableRefreshAgent):
    """Cooperative MRO retains both observation hooks and joint refresh."""
