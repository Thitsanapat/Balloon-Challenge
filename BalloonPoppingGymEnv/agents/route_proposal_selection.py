"""Pure selection boundary for a future learned route-value policy.

Does not run an environment or infer success probabilities. A trained scorer
must supply expected-hit values and the caller must densely check physics.
Unlike deepest-first selection, feasible short and long routes compete here.
"""
from dataclasses import dataclass
import math
from typing import Callable, Iterable


@dataclass(frozen=True)
class RouteProposal:
    targets: tuple[int, ...]
    durations: tuple[float, ...]
    expected_hits: float


def choose_route(proposals: Iterable[RouteProposal], remaining: float,
                 densely_feasible: Callable[[RouteProposal], bool]):
    """Return a valid highest-value proposal, or None for existing recovery.

Expected-hit values must come from permitted observations/model predictions,
not simulator outcomes for counterfactual routes. Actual feasibility and curve
installation stay with the controller; None is never a flight action.
"""
    if not math.isfinite(remaining) or remaining < 0:
        raise ValueError('Finite nonnegative remaining powered time required')
    valid = []
    for p in proposals:
        if (not p.targets or len(p.targets) != len(p.durations)
                or len(set(p.targets)) != len(p.targets)
                or any(type(i) is not int or i < 0 for i in p.targets)
                or not math.isfinite(p.expected_hits) or p.expected_hits < 0
                or any(not math.isfinite(t) or t <= 0 for t in p.durations)
                or sum(p.durations) > remaining):
            continue
        valid.append(p)
    # Do not rank primarily by planned route length. Time breaks equal values.
    valid.sort(key=lambda p: (-p.expected_hits, sum(p.durations)))
    for p in valid:
        if densely_feasible(p):
            return p
    return None
