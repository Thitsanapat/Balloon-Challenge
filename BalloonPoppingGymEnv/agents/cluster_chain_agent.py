"""Online cluster-aware candidate selection, with nearby-single fallback.

Grouping is a search heuristic, never a substitute for the trajectory's thrust,
attitude, slew and ground-clearance checks. Uses released observations only.
"""

import numpy as np
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent


def cluster_affinity(states, released, radius, lookahead=4., velocity_scale=5.):
    states = np.asarray(states, dtype=float)
    weights = np.zeros((len(states), len(states)))
    ids = np.asarray(released, dtype=int)
    if not len(ids):
        return weights
    predicted = states[ids,:3]+lookahead*states[ids,3:6]
    dp = predicted[:,None,:]-predicted[None,:,:]
    dv = states[ids,None,3:6]-states[None,ids,3:6]
    local = np.exp(-.5*np.sum(dp*dp,axis=2)/radius**2
                   -.5*np.sum(dv*dv,axis=2)/velocity_scale**2)
    np.fill_diagonal(local,0.)
    weights[np.ix_(ids,ids)] = local
    return weights


def cluster_order(states, ids, position, velocity, elapsed, affinity,
                  acceleration, bonus, horizon):
    """Travel-time proxy minus reachable-neighborhood reward, both in seconds."""
    ids = np.asarray(ids,dtype=int)
    if not len(ids):
        return []
    probe = 1.5
    delta = states[ids,:3]+states[ids,3:6]*(elapsed+probe)-position-probe*velocity
    eta = np.sqrt(2*np.linalg.norm(delta,axis=1)/max(acceleration,1.))
    density = affinity[np.ix_(ids,ids)].sum(axis=1)
    utility = eta-bonus*np.log1p(density)*np.exp(-eta/max(horizon,.1))
    return ids[np.argsort(utility,kind='stable')].tolist()


class ClusterChainAgent(ChainLaunchAgent):
    def __init__(self,given_parameters,cluster_radius=20.,cluster_bonus=3.,
                 cluster_beam_bonus=1.,nearest_fraction=.5,**kwargs):
        super().__init__(given_parameters,**kwargs)
        self.cluster_radius = float(cluster_radius)
        self.cluster_bonus = float(cluster_bonus)
        self.cluster_beam_bonus = float(cluster_beam_bonus)
        self.nearest_fraction = float(nearest_fraction)
        if not np.all(np.isfinite([self.cluster_radius,self.cluster_bonus,self.cluster_beam_bonus,self.nearest_fraction])):
            raise ValueError('Cluster settings must be finite')
        if self.cluster_radius<=0 or min(self.cluster_bonus,self.cluster_beam_bonus)<0:
            raise ValueError('Positive radius and nonnegative cluster weights required')
        if not 0<=self.nearest_fraction<=1:
            raise ValueError('nearest_fraction must lie in [0,1]')

    def _make_plan(self,observation,position,velocity,now):
        states = np.asarray(observation['balloon_states'],dtype=float)
        released = np.flatnonzero(np.asarray(observation['balloon_status']).reshape(-1)==1)
        self.affinity = cluster_affinity(states,released,self.cluster_radius)
        return super()._make_plan(observation,position,velocity,now)

    def _candidate_ids(self,states,released,route,elapsed,position,velocity,now):
        nearest = super()._candidate_ids(states,released,route,elapsed,position,velocity,now)
        if self.cluster_bonus==0:
            return nearest
        ids = [int(i) for i in released if i not in route and self.failed_until.get(int(i),0)<=now]
        remaining = self.launch_time+self.burn_time-now-elapsed
        ranked = cluster_order(states,ids,position,velocity,elapsed,self.affinity,
                               self.available_acceleration(now+elapsed)*.5,
                               self.cluster_bonus,min(self.leg_horizon,remaining))
        # Reserve slots for locally cheap singleton hits, even with a dense
        # cluster elsewhere. Subsequent physical checks may reject either set.
        keep = max(1,int(self.branch_targets*self.nearest_fraction))
        selected = nearest[:keep]
        for index in ranked:
            if index not in selected:
                selected.append(index)
            if len(selected)>=self.branch_targets:
                break
        return selected

    def _node_priority(self,node,states,released,now):
        route,times,_,_,_ = node
        elapsed = sum(times)
        remaining_ids = [int(i) for i in released if i not in route]
        density = float(self.affinity[route[-1],remaining_ids].sum())
        budget = max(self.launch_time+self.burn_time-now,.1)
        return elapsed-self.cluster_beam_bonus*np.log1p(density)*max(0.,1-elapsed/budget)
