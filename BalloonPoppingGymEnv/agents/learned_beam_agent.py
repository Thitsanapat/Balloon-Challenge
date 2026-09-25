"""Observation-only linear ranking used to *retain* a joint-chain beam.

This is deliberately not a target-forcing selector.  Every parent candidate is
still generated and checked by the original joint trajectory search.  The
learned score only resolves otherwise near-equal root-node pruning decisions.
All model values are supplied as small numeric agent parameters, never loaded
from a file at flight time.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.route_selector_agent import (
    SELECTOR_FEATURES,
    selector_features,
)
from BalloonPoppingGymEnv.agents.submission_time_allocation_v1 import (
    ChainSubmissionAgent,
)


RANK_FEATURES = SELECTOR_FEATURES + 4


class LearnedBeamAgent(ChainSubmissionAgent):
    """Blend a fixed observation-only ranker into root beam tie breaking.

    ``rank_bonus`` is measured in seconds and bounded below one duration-grid
    interval.  A zero bonus is an exact parent-order control.  The score cache
    lives only for one current-observation planning call; it has no balloon IDs,
    seed, replay, or future state after that call returns.
    """

    def __init__(self, given_parameters, rank_coefficients=None,
                 rank_feature_mean=None, rank_feature_scale=None,
                 rank_bonus=0., **kwargs):
        super().__init__(given_parameters, **kwargs)
        if rank_coefficients is None:
            rank_coefficients = np.zeros(RANK_FEATURES)
        if rank_feature_mean is None:
            rank_feature_mean = np.zeros(RANK_FEATURES)
        if rank_feature_scale is None:
            rank_feature_scale = np.ones(RANK_FEATURES)
        self.rank_coefficients = np.asarray(rank_coefficients, dtype=float)
        self.rank_feature_mean = np.asarray(rank_feature_mean, dtype=float)
        self.rank_feature_scale = np.asarray(rank_feature_scale, dtype=float)
        self.rank_bonus = float(rank_bonus)
        if (self.rank_coefficients.shape != (RANK_FEATURES,)
                or self.rank_feature_mean.shape != (RANK_FEATURES,)
                or self.rank_feature_scale.shape != (RANK_FEATURES,)
                or not np.all(np.isfinite(self.rank_coefficients))
                or not np.all(np.isfinite(self.rank_feature_mean))
                or not np.all(np.isfinite(self.rank_feature_scale))
                or np.any(self.rank_feature_scale <= 0)
                or not np.isfinite(self.rank_bonus)
                or not 0. <= self.rank_bonus <= .49):
            raise ValueError('Invalid bounded learned-beam ranker parameters')
        self._root_ranks = {}
        self.diagnostics.update(learned_beam_ranked_searches=0,
                                learned_beam_ranked_roots=0)

    def _root_scores(self, observation):
        """Map current public candidate rows to bounded centered rank values."""
        values, mask, ids = selector_features(self, observation,
                                              self.branch_targets)
        rows = values[:self.branch_targets*SELECTOR_FEATURES].reshape(
            self.branch_targets, SELECTOR_FEATURES)
        global_features = values[self.branch_targets*SELECTOR_FEATURES:]
        scores = {}
        for row, valid, index in zip(rows, mask, ids):
            if not valid:
                continue
            feature = np.r_[row, global_features]
            standardized = (feature-self.rank_feature_mean)/self.rank_feature_scale
            scores[int(index)] = float(np.tanh(np.dot(standardized,
                                                       self.rank_coefficients)))
        return scores

    def _node_priority(self, node, states, released, now):
        base = super()._node_priority(node, states, released, now)
        if not self._root_ranks or not node[0]:
            return base
        return base-self.rank_bonus*self._root_ranks.get(int(node[0][0]), 0.)

    def _make_plan(self, observation, position, velocity, now):
        # Keep parent behavior during temporary launch-axis trials.  They are
        # genuine planning calls but only choose the launch attitude.
        can_rank = (self.launch_selected and self.rank_bonus > 0.)
        self._root_ranks = self._root_scores(observation) if can_rank else {}
        if can_rank:
            self.diagnostics['learned_beam_ranked_searches'] += 1
            self.diagnostics['learned_beam_ranked_roots'] += len(self._root_ranks)
        try:
            return super()._make_plan(observation, position, velocity, now)
        finally:
            self._root_ranks = {}
