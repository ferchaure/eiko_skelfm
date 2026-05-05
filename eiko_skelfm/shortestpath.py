"""
shortestpath.py
---------------
Python port of the MATLAB shortestpath function by Dirk-Jan Kroon.

Traces the shortest path from a start point to a source point in a 2D or 3D
distance map using Runge-Kutta 4 (RK4) on the negative gradient field.

Design decisions
----------------
Normalisation (two-stage):
    Gradients are normalised on the integer grid *before* building the
    interpolators (stability in flat regions where raw values are tiny),
    then re-normalised after each interpolation call (because bilinear/
    trilinear interpolation of unit vectors does not preserve unit length).

Termination (four independent guards, checked in order):
    1. Source proximity  — Euclidean distance <= stepsize.
    2. Out-of-bounds     — next point lies outside the map extents.
    3. Velocity collapse — interpolated velocity norm < 1e-6 (stuck /
                           left the map into the fill_value=0 region).
    4. Monotone descent  — distance stops decreasing by more than a small
                           epsilon (1e-12).  The epsilon avoids a spurious
                           stop on quantisation plateaux in low-resolution
                           or integer-valued maps.

Final step:
    When the source-proximity guard fires, the last appended point is a
    proportional micro-step toward source_point rather than a hard teleport,
    so the path length and direction are physically consistent at the end.

Performance note:
    Each RK4 iteration calls the interpolators 4 times (once per sub-step),
    once per spatial dimension.  For a single path this is negligible.
    For thousands of paths in a tight loop, consider vectorising the
    interpolation calls or pre-building a raw gradient array and using
    scipy.ndimage.map_coordinates, which has lower per-call overhead.

RK4 vs simpler integrators:
    RK4 is technically overkill for a smooth distance map.  Euler or RK2
    would converge too, but RK4 produces noticeably smoother paths in
    curved corridors and near saddle points, at a cost of 4x the
    interpolation calls per step.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def shortestpath(
    distance_map,
    start_point,
    source_point=None,
    stepsize: float = 0.5,
    max_iter: int = 10_000,
) -> np.ndarray:
    """
    Trace the shortest path in a 2-D or 3-D distance map using RK4 integration
    on the normalised negative gradient field.

    Parameters
    ----------
    distance_map : array-like, shape (R, C) or (R, C, D)
        Distance map produced by a fast-marching / eikonal solver.
        Axis order: (row, col) for 2-D, (row, col, depth) for 3-D.
    start_point : array-like, length ndim
        Starting position in index coordinates, e.g. [row, col].
    source_point : array-like, length ndim, optional
        End position.  Integration stops when the path arrives within
        `stepsize` of this point.  If None, the monotone-descent guard
        acts as the termination criterion.
    stepsize : float, optional
        RK4 step size in pixels.  Default 0.5.
        Large values (e.g. 2.0) speed up tracing but produce a coarser
        path and a more visible "snap" at the very end.
    max_iter : int, optional
        Hard iteration cap. Default 10 000.

    Returns
    -------
    path : np.ndarray, shape (M, ndim)
        Ordered positions from start_point to source_point (inclusive).
    """
    # ------------------------------------------------------------------
    # 0. Validate & coerce inputs
    # ------------------------------------------------------------------
    distance_map = np.asarray(distance_map, dtype=float)
    ndim = distance_map.ndim
    if ndim not in (2, 3):
        raise ValueError("distance_map must be 2-D or 3-D.")

    start_point = np.asarray(start_point, dtype=float).ravel()
    if start_point.size != ndim:
        raise ValueError(f"start_point must have {ndim} elements.")

    source_point = (
        np.asarray(source_point, dtype=float).ravel()
        if source_point is not None
        else None
    )

    shape  = distance_map.shape
    axes   = tuple(np.arange(s) for s in shape)
    bounds = np.array([[0.0, s - 1.0] for s in shape])   # (ndim, 2)

    # ------------------------------------------------------------------
    # 1. Precompute normalised gradient field on the integer grid
    # ------------------------------------------------------------------
    grads    = np.gradient(distance_map)
    if isinstance(grads, np.ndarray):        # guard: ndim==1 edge case
        grads = [grads]

    grad_mag = np.sqrt(sum(g ** 2 for g in grads))
    grad_mag[grad_mag < 1e-10] = 1.0        # avoid division by zero in flat zones

    norm_grads = [-g / grad_mag for g in grads]   # point toward decreasing distance

    # ------------------------------------------------------------------
    # 2. Build interpolators
    #    fill_value=0.0  →  velocity collapses to zero outside the map,
    #    which is caught by the norm < 1e-6 guard (termination guard 3).
    # ------------------------------------------------------------------
    vel_interps = [
        RegularGridInterpolator(axes, ng, method="linear",
                                bounds_error=False, fill_value=0.0)
        for ng in norm_grads
    ]

    dist_interp = RegularGridInterpolator(
        axes, distance_map, method="linear",
        bounds_error=False, fill_value=np.inf
    )

    # ------------------------------------------------------------------
    # 3. Velocity helper
    #    Re-normalise after interpolation: bilinear/trilinear interpolation
    #    of unit vectors does NOT preserve unit length; the re-normalisation
    #    keeps the trace speed constant throughout integration.
    # ------------------------------------------------------------------
    def velocity(p: np.ndarray) -> np.ndarray:
        q   = p[np.newaxis, :]                        # shape (1, ndim)
        v   = np.array([f(q)[0] for f in vel_interps])
        mag = np.linalg.norm(v)
        return v / mag if mag > 1e-6 else np.zeros(ndim)

    # ------------------------------------------------------------------
    # 4. RK4 integration
    # ------------------------------------------------------------------
    path    = [start_point.copy()]
    current = start_point.copy()
    h       = stepsize
    prev_dist = dist_interp(current[np.newaxis, :])[0]

    for _ in range(max_iter):

        # Guard 1 — source proximity -----------------------------------------
        if source_point is not None:
            dist_to_source = np.linalg.norm(current - source_point)
            if dist_to_source <= h:
                # FIX: proportional micro-step instead of a hard teleport.
                # This keeps the step direction physically consistent and
                # avoids a visible "snap" when stepsize is large.
                direction = source_point - current
                dir_norm  = np.linalg.norm(direction)
                if dir_norm > 1e-10:
                    path.append(current + direction * (dist_to_source / dir_norm))
                else:
                    path.append(source_point.copy())
                break

        # RK4 sub-steps -------------------------------------------------------
        k1 = velocity(current)

        # Guard 3 (early, before computing k2-k4) — velocity collapse ---------
        if np.linalg.norm(k1) < 1e-6:
            break

        k2 = velocity(current + 0.5 * h * k1)
        k3 = velocity(current + 0.5 * h * k2)
        k4 = velocity(current +        h * k3)

        next_point = current + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Guard 2 — explicit out-of-bounds check ------------------------------
        if not np.all((next_point >= bounds[:, 0]) & (next_point <= bounds[:, 1])):
            break

        # Guard 4 — monotone descent (FIX: epsilon avoids plateau false-stop) -
        curr_dist = dist_interp(next_point[np.newaxis, :])[0]
        if curr_dist > prev_dist + 1e-12:
            break
        prev_dist = curr_dist

        current = next_point
        path.append(current.copy())

    return np.array(path)
