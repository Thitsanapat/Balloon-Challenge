"""Try a few observable first targets before committing a Scenario 4 route.

The ordinary launch search compares rail directions, but each direction's beam
may prune away a target that has a more recoverable first leg.  This variant
keeps the ordinary search as a candidate and runs at most two short, joint
route searches with a different released first target.  All target states and
physical limits come from current observations and ``given_parameters``.
"""

import copy

import numpy as np

from BalloonPoppingGymEnv.agents.scenario4_recoverable_launch_agent import (
    Scenario4RecoverableLaunchAgent,
)


class Scenario4FirstTargetAgent(Scenario4RecoverableLaunchAgent):
    """Choose the first target by correction reachability and route value.

    Forced candidates use a shallow joint beam for a bounded launch decision.
    The normal beam search is always considered, and only the first planning
    call makes these extra comparisons.  Later replans follow the inherited
    committed-route logic.
    """

    def __init__(self, given_parameters, first_target_trials=2,
                 first_target_depth=3, first_target_beam_width=8,
                 first_target_branch_targets=6, **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.first_target_trials = int(first_target_trials)
        self.first_target_depth = int(first_target_depth)
        self.first_target_beam_width = int(first_target_beam_width)
        self.first_target_branch_targets = int(first_target_branch_targets)
        if (self.first_target_trials < 0 or self.first_target_depth < 1 or
                self.first_target_beam_width < 1 or
                self.first_target_branch_targets < 1):
            raise ValueError("Invalid first-target search limits")
        self._forced_first_index = None
        self.first_target_comparisons = []
        self.first_target_selected = None

    def _candidate_ids(self, states, released, route, elapsed, position,
                       velocity, now):
        forced = self._forced_first_index
        if forced is not None and not route:
            if (forced in released and
                    self.failed_until.get(forced, 0.) <= now and
                    np.all(np.isfinite(states[forced]))):
                return [forced]
            return []
        return super()._candidate_ids(states, released, route, elapsed,
                                      position, velocity, now)

    def _launch_candidate_key(self, axis, count, finish, observation, now):
        # A speculative eighth leg earns no more launch credit than a third:
        # the target choice is driven by the first executable intercept.
        return super()._launch_candidate_key(
            axis, min(count, self.first_target_depth), finish,
            observation, now)

    def _first_plan_key(self, observation, now):
        count = len(self.route)
        finish = self.deadlines[-1] - now if count else float('inf')
        return self._launch_candidate_key(None, count, finish,
                                          observation, now)

    def _make_plan(self, observation, position, velocity, now):
        # The first-choice comparison is repeated once on the actual launch
        # axis after the rail-direction comparison restores its copied state.
        # Once a target has been selected, regular committed replanning runs.
        if (not self.first_target_trials or self.target_events or self.route or
                self._forced_first_index is not None):
            return super()._make_plan(observation, position, velocity, now)

        states = np.asarray(observation['balloon_states'], dtype=float)
        released = np.flatnonzero(
            np.asarray(observation['balloon_status']).reshape(-1) == 1)
        candidates = super()._candidate_ids(
            states, released, (), 0., position, velocity, now)
        original = copy.deepcopy(self.__dict__)

        # The untouched parent search is a fallback candidate, including its
        # ordinary joint-route depth and time allocation.
        super()._make_plan(observation, position, velocity, now)
        best_state = copy.deepcopy(self.__dict__)
        best_key = self._first_plan_key(observation, now)
        baseline_first = self.route[0] if self.route else None
        comparisons = [(baseline_first, best_key)]
        selected = baseline_first

        alternatives = [int(i) for i in candidates if i != baseline_first]
        try:
            for index in alternatives[:self.first_target_trials]:
                self.__dict__ = copy.deepcopy(original)
                self._forced_first_index = index
                self.search_depth = min(self.search_depth, self.first_target_depth)
                self.beam_width = min(self.beam_width,
                                      self.first_target_beam_width)
                self.branch_targets = min(self.branch_targets,
                                          self.first_target_branch_targets)
                self.timing_candidates = 0
                failed = False
                try:
                    super()._make_plan(observation, position, velocity, now)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    failed = True
                finally:
                    self._forced_first_index = None
                    self.search_depth = original['search_depth']
                    self.beam_width = original['beam_width']
                    self.branch_targets = original['branch_targets']
                    self.timing_candidates = original['timing_candidates']

                # If the joint beam found nothing, its fallback planner can
                # select another target.  That is not a forced-first result.
                if failed or not self.route or self.route[0] != index:
                    comparisons.append((index, None))
                    continue
                key = self._first_plan_key(observation, now)
                comparisons.append((index, key))
                if key < best_key:
                    best_key = key
                    best_state = copy.deepcopy(self.__dict__)
                    selected = index
        finally:
            self.__dict__ = best_state
            self.first_target_comparisons = comparisons
            self.first_target_selected = selected
