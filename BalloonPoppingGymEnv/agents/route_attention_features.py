"""Observation-only set features for a NOT-YET-TRAINED routing prototype.

No scenario/seed, environment object, stored field or future samples accepted.
Positions at a partial route's elapsed time are linear predictions from the
current observed velocities, not ground truth. Feasibility is supplied by the
agent's reduced physical model, never claimed from geometric distance alone.
"""
import numpy as np

ROUTE_FEATURE_SCHEMA = 'observed_route_set_v1_12_9'
NODE_FEATURES = 12
CONTEXT_FEATURES = 9


def routing_features(observation, position, velocity, acceleration, remaining,
                     available_acceleration, *, elapsed=0., visited=(), blocked=(),
                     feasible=None):
    states = np.asarray(observation['balloon_states'], dtype=float)
    status = np.asarray(observation['balloon_status']).reshape(-1)
    vectors = [np.asarray(v, dtype=float) for v in (position, velocity, acceleration)]
    if (states.ndim != 2 or states.shape[1] != 6 or status.shape != (len(states),)
            or any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in vectors)):
        raise ValueError('Expected balloon states/status and finite controller vectors')
    scalars = np.array([remaining, available_acceleration, elapsed], dtype=float)
    if not np.all(np.isfinite(scalars)) or min(scalars) < 0:
        raise ValueError('Finite nonnegative timing and available acceleration required')
    position, velocity, acceleration = vectors
    valid = (status == 1) & np.all(np.isfinite(states), axis=1)
    candidate = valid.copy()
    for indices in (visited, blocked):
        for index in indices:
            if not isinstance(index, (int, np.integer)) or not 0 <= index < len(states):
                raise ValueError('Invalid observed balloon index')
            candidate[index] = False
    if feasible is not None:
        feasible = np.asarray(feasible)
        if feasible.shape != candidate.shape or feasible.dtype != np.bool_:
            raise ValueError('Feasibility must be a boolean mask matching balloons')
        candidate &= feasible
    if elapsed >= remaining:
        candidate[:] = False
    nodes = np.zeros((len(states), NODE_FEATURES), np.float32)
    ids = np.flatnonzero(valid)
    if len(ids):
        current = states[ids]
        projected = current[:, :3] + elapsed*current[:, 3:]
        relative = projected-position
        relative_v = current[:, 3:]-velocity
        distance = np.linalg.norm(relative, axis=1)
        closing = -np.sum(relative*relative_v, axis=1)/np.maximum(distance,1e-6)
        # Group context uses released finite balloons only; never unreleased rows.
        pair_dist = np.linalg.norm(projected[:, None]-projected[None, :], axis=2)
        np.fill_diagonal(pair_dist, np.inf)
        density = np.stack([(pair_dist <= r).sum(axis=1)/max(len(ids)-1,1)
                            for r in (10.,25.)], axis=1)
        nodes[ids] = np.clip(np.c_[relative/150.,relative_v/30.,distance/150.,
            closing/30.,np.linalg.norm(current[:,3:],axis=1)/30.,density,
            candidate[ids].astype(float)],-10.,10.)
    context = np.clip(np.r_[velocity/40., acceleration/20.,
        max(remaining-elapsed,0.)/30.,elapsed/30.,available_acceleration/20.],-10.,10.).astype(np.float32)
    return dict(nodes=nodes, context=context, visible_mask=valid, candidate_mask=candidate)
