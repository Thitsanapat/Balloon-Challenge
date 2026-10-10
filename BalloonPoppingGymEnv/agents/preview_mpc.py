"""Small observation-based, bounded translational preview MPC.

The base force is held constant only inside this short predictive model. The
optimization returns the first acceleration correction; the real controller
replans from fresh sensors and applies its existing thrust/tilt allocation.
This is not a simulator rollout or access to hidden state.
"""
import numpy as np
from scipy.optimize import lsq_linear


def preview_correction(position, velocity, nominal_acceleration, reference_p,
                       reference_v, interval, bound, previous=None,
                       *, position_weight=1.0, velocity_weight=0.25,
                       control_weight=0.12, slew_weight=0.15):
    """Solve a linear-quadratic finite horizon problem with box acceleration bounds."""
    p, v, a = (np.asarray(x, dtype=float) for x in
               (position, velocity, nominal_acceleration))
    ref_p, ref_v = (np.asarray(x, dtype=float) for x in (reference_p, reference_v))
    n = len(ref_p)
    prev = np.zeros(3) if previous is None else np.asarray(previous, dtype=float)
    if (p.shape != (3,) or v.shape != (3,) or a.shape != (3,) or prev.shape != (3,)
            or ref_p.shape != (n, 3) or ref_v.shape != (n, 3) or not 1 <= n <= 32
            or not np.all(np.isfinite(np.r_[p, v, a, prev, ref_p.ravel(), ref_v.ravel()]))
            or not np.isfinite(interval) or interval <= 0
            or not np.isfinite(bound) or bound <= 0
            or any(not np.isfinite(w) or w < 0 for w in
                   (position_weight, velocity_weight, control_weight, slew_weight))
            or position_weight + velocity_weight <= 0 or control_weight + slew_weight <= 0):
        raise ValueError('Finite state, ordered reference, horizon and positive bound required')

    k = np.arange(1, n + 1, dtype=float)
    j = np.arange(n, dtype=float)
    time = k * interval
    # A[k,j] is the effect on position at the end of interval k from a
    # correction held over interval j; j >= k has no effect yet.
    p_matrix = np.maximum(k[:, None] - j[None, :] - 0.5, 0.) * interval**2
    v_matrix = (j[None, :] < k[:, None]).astype(float) * interval
    p_nominal = p[None, :] + time[:, None]*v + 0.5*time[:, None]**2*a
    v_nominal = v[None, :] + time[:, None]*a
    slew_matrix = np.eye(n) - np.eye(n, k=-1)
    matrix = np.vstack((np.sqrt(position_weight)*p_matrix,
                        np.sqrt(velocity_weight)*v_matrix,
                        np.sqrt(control_weight)*np.eye(n),
                        np.sqrt(slew_weight)*slew_matrix))
    correction = np.empty(3)
    for axis in range(3):
        slew_target = np.zeros(n)
        slew_target[0] = prev[axis]
        target = np.r_[np.sqrt(position_weight)*(ref_p[:, axis]-p_nominal[:, axis]),
                       np.sqrt(velocity_weight)*(ref_v[:, axis]-v_nominal[:, axis]),
                       np.zeros(n), np.sqrt(slew_weight)*slew_target]
        fit = lsq_linear(matrix, target, bounds=(-bound, bound),
                         max_iter=30, tol=1e-5)
        if not fit.success or not np.all(np.isfinite(fit.x)):
            return np.zeros(3)
        correction[axis] = fit.x[0]
    return np.clip(correction, -bound, bound)
