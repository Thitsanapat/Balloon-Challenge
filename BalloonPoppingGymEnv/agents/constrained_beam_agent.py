"""Repair near-feasible target sequences inside beam expansion itself."""

import numpy as np
from BalloonPoppingGymEnv.agents.chain_launch_agent import ChainLaunchAgent
from BalloonPoppingGymEnv.agents.constrained_chain_agent import constraint_margins, repair_derivatives


class ConstrainedBeamAgent(ChainLaunchAgent):
    def __init__(self, given_parameters, beam_repair_budget=16,
                 beam_repair_depth=6, beam_repair_violation=.15,
                 beam_repair_iterations=25, beam_repair_safety=0., **kwargs):
        super().__init__(given_parameters, **kwargs)
        self.beam_repair_budget = int(beam_repair_budget)
        self.beam_repair_depth = int(beam_repair_depth)
        self.beam_repair_violation = float(beam_repair_violation)
        self.beam_repair_iterations = int(beam_repair_iterations)
        self.beam_repair_safety = float(beam_repair_safety)
        if (self.beam_repair_budget < 0 or min(self.beam_repair_depth, self.beam_repair_iterations) < 1
                or not np.isfinite(self.beam_repair_violation) or self.beam_repair_violation <= 0
                or not np.isfinite(self.beam_repair_safety) or self.beam_repair_safety < 0):
            raise ValueError('Invalid beam repair limits')
        self.beam_repairs_left = 0
        self.beam_repair_now = 0.
        self.diagnostics.update(beam_constraint_repairs=0, beam_constraint_repaired=0)

    def _make_plan(self, observation, position, velocity, now):
        self.beam_repairs_left = self.beam_repair_budget
        self.beam_repair_now = now
        return super()._make_plan(observation, position, velocity, now)

    def _chain_curves(self, position, velocity, targets, durations, drift):
        original = super()._chain_curves(position, velocity, targets, durations, drift)
        if self.beam_repairs_left <= 0 or len(durations) < self.beam_repair_depth:
            return original
        margins = constraint_margins(self, original, durations, self.beam_repair_now, samples=17)
        worst = float(np.min(margins))
        # Avoid spending optimization effort on clearly unreachable orders.
        if worst >= 0. or worst < -self.beam_repair_violation:
            return original
        self.beam_repairs_left -= 1
        self.diagnostics['beam_constraint_repairs'] += 1
        repaired = repair_derivatives(self, original, durations, self.beam_repair_now,
                                      self.beam_repair_iterations, self.beam_repair_safety)
        if repaired is None:
            return original
        self.diagnostics['beam_constraint_repaired'] += 1
        return repaired
