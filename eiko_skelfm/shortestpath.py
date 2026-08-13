"""
Design decisions
----------------
Gradient field (pointmin):
    pointmin builds a **discrete descent-direction field**:
    for every voxel it finds the first strictly-lower neighbour among the 8 (2D)
    / 26 (3D) connected neighbours (padded with the image maximum at the border)
    and stores the unit vector toward it.  Voxels with no lower neighbour get
    zero.


Path composition:
    The output array starts empty; the start point is NOT included.  Points are
    appended after each successful RK4 step, and the nearest source is hard-
    appended as the final point when the proximity guard fires.
"""

import numpy as np


# ---------------------------------------------------------------------------
# pointmin — discrete descent-direction field (port of pointmin.m)
# ---------------------------------------------------------------------------

# 8-connected neighbour offsets for 2-D
_NE2 = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [ 0, -1],          [ 0, 1],
    [ 1, -1], [ 1, 0], [ 1, 1],
], dtype=float)

# 26-connected neighbour offsets for 3-D
_NE3 = np.array([
    [di, dj, dk]
    for di in (-1, 0, 1)
    for dj in (-1, 0, 1)
    for dk in (-1, 0, 1)
    if not (di == 0 and dj == 0 and dk == 0)
], dtype=float)


def _pointmin(I: np.ndarray):
    """
    Build the discrete descent-direction field from a 2-D or 3-D array I.

    For every element, iterates over neighbours in connectivity order.  If a
    neighbour is strictly smaller than the current minimum seen so far, the
    direction toward that neighbour (unit vector) is recorded and the local
    minimum is updated.  Elements on the border are padded with ``max(I)`` so
    that boundary voxels never point outward.

    Returns
    -------
    grads : list of ndarray, each shape == I.shape
        One array per spatial axis.  For 2-D: [Fx, Fy].
        For 3-D: [Fx, Fy, Fz].
        The vectors are unit-length and point toward the strictly lower
        neighbour (same convention as the rk4.c gradient array which is
        already negated in shortestpath.m: GradientVolume = -Fx, -Fy).
    """
    ndim = I.ndim
    I = np.asarray(I, dtype=float)

    pad_val = float(I.max())
    J = np.full(tuple(s + 2 for s in I.shape), pad_val)

    if ndim == 2:
        J[1:-1, 1:-1] = I
        Ne = _NE2
    else:
        J[1:-1, 1:-1, 1:-1] = I
        Ne = _NE3

    # Working copy of I that tracks the current minimum seen for each voxel
    I_min = I.copy()

    # Output direction components, initialised to zero (no lower neighbour found)
    components = [np.zeros(I.shape) for _ in range(ndim)]

    for row in Ne:
        # Slicing into the padded array with the neighbour offset
        slices_J = tuple(
            slice(1 + int(row[ax]), 1 + int(row[ax]) + I.shape[ax])
            for ax in range(ndim)
        )
        In = J[slices_J]          # values at this neighbour for every voxel

        # Unit direction toward this neighbour
        norm = float(np.linalg.norm(row))
        D = row / norm            # unit vector, length ndim

        check = In < I_min        # strictly lower than best seen so far
        I_min[check] = In[check]  # update running minimum

        for ax in range(ndim):
            components[ax][check] = D[ax]

    return components   # [Fx, Fy] or [Fx, Fy, Fz]


# ---------------------------------------------------------------------------
# Bilinear / trilinear interpolation (port of interpgrad2d / interpgrad3d)
# ---------------------------------------------------------------------------

def _interp2d(gradient_array: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation of a 2-component vector field at sub-pixel ``point``.

    Parameters
    ----------
    gradient_array : ndarray, shape (nx, ny, 2)
        Stacked gradient components along the last axis.
    point : ndarray, shape (2,)
        Coordinates (x, y) in 0-based index space.

    Returns
    -------
    ndarray, shape (2,)
    """
    nx, ny = gradient_array.shape[:2]
    x, y = point[0], point[1]

    x0 = int(np.floor(x)); x1 = x0 + 1
    y0 = int(np.floor(y)); y1 = y0 + 1

    xc = x - x0; yc = y - y0
    xci = 1.0 - xc; yci = 1.0 - yc

    perc = np.array([xci * yci, xci * yc, xc * yci, xc * yc])

    # Clamp to boundary (stick to boundary, like rk4.c)
    x0 = max(0, min(x0, nx - 1)); x1 = max(0, min(x1, nx - 1))
    y0 = max(0, min(y0, ny - 1)); y1 = max(0, min(y1, ny - 1))

    # Interpolate both components simultaneously
    vals = (
        gradient_array[x0, y0] * perc[0]
        + gradient_array[x0, y1] * perc[1]
        + gradient_array[x1, y0] * perc[2]
        + gradient_array[x1, y1] * perc[3]
    )
    return vals  # shape (2,)


def _interp3d(gradient_array: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Trilinear interpolation of a 3-component vector field at sub-voxel ``point``.

    Parameters
    ----------
    gradient_array : ndarray, shape (nx, ny, nz, 3)
    point : ndarray, shape (3,)

    Returns
    -------
    ndarray, shape (3,)
    """
    nx, ny, nz = gradient_array.shape[:3]
    x, y, z = point[0], point[1], point[2]

    x0 = int(np.floor(x)); x1 = x0 + 1
    y0 = int(np.floor(y)); y1 = y0 + 1
    z0 = int(np.floor(z)); z1 = z0 + 1

    xc = x - x0; yc = y - y0; zc = z - z0
    xci = 1.0 - xc; yci = 1.0 - yc; zci = 1.0 - zc

    p = np.array([
        xci * yci * zci, xci * yci * zc,
        xci * yc  * zci, xci * yc  * zc,
        xc  * yci * zci, xc  * yci * zc,
        xc  * yc  * zci, xc  * yc  * zc,
    ])

    # Clamp to boundary
    x0 = max(0, min(x0, nx - 1)); x1 = max(0, min(x1, nx - 1))
    y0 = max(0, min(y0, ny - 1)); y1 = max(0, min(y1, ny - 1))
    z0 = max(0, min(z0, nz - 1)); z1 = max(0, min(z1, nz - 1))

    vals = (
        gradient_array[x0, y0, z0] * p[0]
        + gradient_array[x0, y0, z1] * p[1]
        + gradient_array[x0, y1, z0] * p[2]
        + gradient_array[x0, y1, z1] * p[3]
        + gradient_array[x1, y0, z0] * p[4]
        + gradient_array[x1, y0, z1] * p[5]
        + gradient_array[x1, y1, z0] * p[6]
        + gradient_array[x1, y1, z1] * p[7]
    )
    return vals  # shape (3,)


# ---------------------------------------------------------------------------
# Bounds check helpers
# ---------------------------------------------------------------------------

def _in_bounds_2d(point: np.ndarray, size: tuple) -> bool:
    return (
        0 <= point[0] <= size[0] - 1
        and 0 <= point[1] <= size[1] - 1
    )


def _in_bounds_3d(point: np.ndarray, size: tuple) -> bool:
    return (
        0 <= point[0] <= size[0] - 1
        and 0 <= point[1] <= size[1] - 1
        and 0 <= point[2] <= size[2] - 1
    )


# ---------------------------------------------------------------------------
# RK4 step (port of RK4STEP_2D / RK4STEP_3D from rk4.c)
# ---------------------------------------------------------------------------

def _rk4_step_2d(
    grad_array: np.ndarray,
    start: np.ndarray,
    step_size: float,
) -> np.ndarray | None:
    """
    One RK4 step in 2-D.  Returns the new point, or None if any intermediate
    point goes out of bounds (matching rk4.c returning ``[0, 0]``).

    Parameters
    ----------
    grad_array : ndarray, shape (nx, ny, 2)
        Stacked negated unit-direction field (descent directions).
    start : ndarray, shape (2,)
    step_size : float

    Returns
    -------
    next_point : ndarray, shape (2,) or None
    """
    size = grad_array.shape[:2]

    def _kstep(pt):
        v = _interp2d(grad_array, pt)
        n = np.sqrt(v[0] ** 2 + v[1] ** 2)
        if n == 0.0:
            # Zero gradient (flat region / no lower neighbour in pointmin).
            # rk4.c would produce NaN here, which fails bounds check → abort.
            return None
        return v * (step_size / n)

    k1 = _kstep(start)
    if k1 is None:
        return None
    tp = start - k1 * 0.5
    if not _in_bounds_2d(tp, size):
        return None

    k2 = _kstep(tp)
    if k2 is None:
        return None
    tp = start - k2 * 0.5
    if not _in_bounds_2d(tp, size):
        return None

    k3 = _kstep(tp)
    if k3 is None:
        return None
    tp = start - k3
    if not _in_bounds_2d(tp, size):
        return None

    k4 = _kstep(tp)
    if k4 is None:
        return None

    nxt = start - (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if not _in_bounds_2d(nxt, size):
        return None

    return nxt


def _rk4_step_3d(
    grad_array: np.ndarray,
    start: np.ndarray,
    step_size: float,
) -> np.ndarray | None:
    """
    One RK4 step in 3-D.  Returns the new point, or None if out of bounds.

    Parameters
    ----------
    grad_array : ndarray, shape (nx, ny, nz, 3)
    start : ndarray, shape (3,)
    step_size : float

    Returns
    -------
    next_point : ndarray, shape (3,) or None
    """
    size = grad_array.shape[:3]

    def _kstep(pt):
        v = _interp3d(grad_array, pt)
        n = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if n == 0.0:
            return None
        return v * (step_size / n)

    k1 = _kstep(start)
    if k1 is None:
        return None
    tp = start - k1 * 0.5
    if not _in_bounds_3d(tp, size):
        return None

    k2 = _kstep(tp)
    if k2 is None:
        return None
    tp = start - k2 * 0.5
    if not _in_bounds_3d(tp, size):
        return None

    k3 = _kstep(tp)
    if k3 is None:
        return None
    tp = start - k3
    if not _in_bounds_3d(tp, size):
        return None

    k4 = _kstep(tp)
    if k4 is None:
        return None

    nxt = start - (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if not _in_bounds_3d(nxt, size):
        return None

    return nxt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shortestpath(
    distance_map,
    start_point,
    source_point=None,
    stepsize: float = 0.5,
) -> np.ndarray:
    """
    Trace the shortest path in a 2-D or 3-D distance map using RK4 integration
    on the discrete descent-direction field (pointmin).

    Faithful port of:
      ``FastMarching_version3b/shortestpath.m`` (outer loop & termination)
      ``FastMarching_version3b/shortestpath/rk4.c`` (RK4 step)
      ``FastMarching_version3b/functions/pointmin.m`` (gradient field)

    Parameters
    ----------
    distance_map : array-like, shape (R, C) or (R, C, D)
        Distance map produced by a fast-marching / eikonal solver.
        Axis order: (row, col) for 2-D, (row, col, depth) for 3-D.
    start_point : array-like, length ndim
        Starting position in index coordinates, e.g. [row, col].
    source_point : array-like or None, optional
        End position(s).  Shape (ndim,) for a single source, or (ndim, K)
        for K source points.  Integration stops when the path arrives within
        ``stepsize`` of the nearest source.  If None, only the stall guard
        fires.
    stepsize : float, optional
        RK4 step size in pixels.  Default 0.5.

    Returns
    -------
    path : np.ndarray, shape (M, ndim)
        Ordered positions from just after start_point to source_point.
        The start point is **not** included (matching shortestpath.m).
        The nearest source is hard-appended as the last point when the
        proximity guard fires (matching shortestpath.m line 100).
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

    # source_point: normalise to (K, ndim)
    if source_point is not None:
        sp = np.asarray(source_point, dtype=float)
        if sp.ndim == 1:
            sp = sp.reshape(1, -1)       # (1, ndim)
        elif sp.shape[0] == ndim and sp.ndim == 2:
            sp = sp.T                    # (dim, K) → (K, dim)
        # else already (K, ndim)
        source_point = sp

    # ------------------------------------------------------------------
    # 1. Build discrete descent-direction field (pointmin)
    # ------------------------------------------------------------------
    components = _pointmin(distance_map)   # list of ndim arrays, each shape == dm.shape

    # Negate so that the gradient array contains *descent* directions,
    # matching shortestpath.m:  GradientVolume(:,:,1) = -Fx
    # (pointmin returns Fx pointing toward the lower neighbour, so
    #  negating gives the direction the marcher should move)
    neg_comps = [-c for c in components]

    # Stack into a single array for efficient indexing
    # Shape: (nx, ny, 2) for 2-D or (nx, ny, nz, 3) for 3-D
    grad_array = np.stack(neg_comps, axis=-1)

    # ------------------------------------------------------------------
    # 2. Choose the appropriate RK4 step function
    # ------------------------------------------------------------------
    rk4_step = _rk4_step_2d if ndim == 2 else _rk4_step_3d

    # ------------------------------------------------------------------
    # 3. Integration loop (matching shortestpath.m)
    # ------------------------------------------------------------------
    path: list = []          # start point NOT included, matching MATLAB
    current = start_point.copy()
    i = 0                    # iteration counter (1-based in MATLAB, but used for Movement)

    # History ring for stall detection (movement check vs. 10 iters back)
    history: list = []

    while True:
        end_point = rk4_step(grad_array, current, stepsize)

        # Guard 1 — out of bounds (EndPoint(1)==0 in MATLAB)
        if end_point is None:
            break

        # Guard 2 — stalled path: movement vs. point 10 iterations back
        if i > 10:
            movement = float(np.linalg.norm(end_point - history[0]))
            if movement < stepsize:
                break
        # keep a sliding window of 11 points
        history.append(end_point.copy())
        if len(history) > 11:
            history.pop(0)

        i += 1
        path.append(end_point.copy())

        # Guard 3 — source proximity
        if source_point is not None:
            dists = np.linalg.norm(source_point - end_point[np.newaxis, :], axis=1)
            idx = int(np.argmin(dists))
            dist_to_end = dists[idx]

            if dist_to_end < stepsize:
                # Hard-append the nearest source (matching shortestpath.m:100)
                path.append(source_point[idx].copy())
                break

        current = end_point

    return np.array(path) if path else np.empty((0, ndim))
