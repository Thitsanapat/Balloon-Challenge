"""Choose launch time and inclination from currently observed opportunities."""

import numpy as np

from BalloonPoppingGymEnv.agents.submission_sqp_v2 import (
    SubmissionAgent, G, get_initial_attitude,
)


class AdaptiveLaunchAgent(SubmissionAgent):
    def __init__(self,given_parameters,earliest_launch=15.,latest_launch=45.,
                 intercept_threshold=4.5,decision_interval=.5,**kwargs):
        kwargs['launch_time'] = float(latest_launch)
        super().__init__(given_parameters,**kwargs)
        self.earliest_launch = float(earliest_launch)
        self.latest_launch = float(latest_launch)
        self.intercept_threshold = float(intercept_threshold)
        self.decision_interval = float(decision_interval)
        self.next_decision = self.earliest_launch
        self.launch_decision = None

    def get_action(self,observation):
        now = float(observation['simulation_time'])
        if not self.launched and now>=self.next_decision:
            self.next_decision = now+self.decision_interval
            states = np.asarray(observation['balloon_states'],dtype=float)
            ids = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1)==1)
            position = np.array([0.,0.,self.elevation])
            best = None
            limit = .98*self.thrust/self.initial_mass
            for duration in np.arange(2.,10.01,.25):
                target = states[ids,:3]+states[ids,3:6]*duration
                acceleration = 2*(target-position)/duration**2
                forces = acceleration-G
                magnitude = np.linalg.norm(forces,axis=1)
                feasible = ((magnitude<=limit) & (acceleration[:,2]>.1)
                            & (forces[:,2]>=magnitude*np.cos(np.radians(30.))))
                if np.any(feasible):
                    choices = np.flatnonzero(feasible)
                    # Prefer a first intercept with nearby follow-on targets.
                    predicted = states[ids,:3]+states[ids,3:6]*(duration+3.)
                    counts = [np.sum(np.linalg.norm(predicted-target[k],axis=1)<45.) for k in choices]
                    k = choices[int(np.argmax(counts))]
                    best = duration,int(ids[k]),forces[k]
                    break
            if best is not None and (best[0]<=self.intercept_threshold or now>=self.latest_launch):
                duration,index,force = best
                axis = force/np.linalg.norm(force)
                inclination = np.degrees(np.arcsin(axis[2]))
                heading = np.degrees(np.arctan2(axis[0],axis[1])) % 360.
                self.launch_attitude = np.array([inclination,heading])
                self.quaternion = get_initial_attitude(inclination,heading)
                self.launch_time = now
                self.launch_decision = [now,index,duration,float(inclination),float(heading)]
            elif now>=self.latest_launch:
                self.launch_time = now
        return super().get_action(observation)
