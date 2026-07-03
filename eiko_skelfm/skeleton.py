"""
skeleton.py
-----------
Python/NumPy/SciPy port of skeleton.m

Computes accurate centerline skeletons of binary 2-D or 3-D images using
fast-marching distance transforms.


"""

import numpy as np
from scipy.ndimage import binary_dilation

from .msfm import msfm                    # fast-marching travel-time solver
from .shortestpath import shortestpath    # gradient-descent / RK-4 tracer


def skeleton(I: np.ndarray, verbose: bool = True) -> list:
    """
    Compute the skeleton (centerlines) of a binary image or volume.

    Parameters
    ----------
    I : ndarray, bool
        2-D or 3-D binary image (True = foreground).
    verbose : bool, optional
        Print progress messages (default True).

    Returns
    -------
    S : list of ndarray, shape (N_i, 2) or (N_i, 3)
        Each element is an ordered array of (row, col[, slice]) coordinates
        describing one skeleton branch.
    """
    I = np.asarray(I, dtype=bool)
    IS3D = I.ndim == 3

    # --- Distance to vessel boundary ----------------------------------------
    boundary_distance = _get_boundary_distance(I, IS3D)
    if verbose:
        print("Distance Map Constructed")

    # --- Starting point: voxel farthest from the boundary -------------------
    source_point, max_d = _max_distance_point(boundary_distance, I)

    # --- Speed image: bias the marcher toward the medial axis ---------------
    speed_image = (boundary_distance / max_d) ** 4
    speed_image[speed_image == 0] = 1e-10

    skeleton_segments: list = []
    itt = 0

    while True:
        if verbose:
            print(f"Find Branches Iterations : {itt}")

        # Fast-marching from all previously found branch points
        T, Y = msfm(speed_image, source_point, use_second=False, use_cross=False, return_y=True)

        # Farthest unvisited point becomes the new branch start
        start_point, _ = _max_distance_point(Y, I)

        # Trace the geodesic back to the nearest source point
        shortest_line = shortestpath(T, start_point, source_point, 1)
        line_length = _get_line_length(shortest_line, IS3D)

        # Stop when the new branch is shorter than the largest vessel diameter
        if line_length < max_d * 2:
            break

        itt += 1
        skeleton_segments.append(shortest_line)

        # Append the new branch to the set of marching source points
        # source_point shape: (dim, N); shortest_line shape: (N, dim)
        source_point = np.hstack([source_point, shortest_line.T])

    S = _organize_skeleton(skeleton_segments, IS3D)
    if verbose:
        print(f"Skeleton Branches Found : {len(S)}")

    return S


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_line_length(L: np.ndarray, IS3D: bool) -> float:
    """Euclidean arc-length of a polyline.

    Parameters
    ----------
    L : ndarray, shape (N, 2) or (N, 3)
        Ordered sequence of coordinates.
    IS3D : bool

    Returns
    -------
    float
    """
    diffs = L[1:] - L[:-1]          # (N-1, dim)
    dist = np.sqrt((diffs ** 2).sum(axis=1))
    return float(dist.sum())


def _organize_skeleton(skeleton_segments: list, IS3D: bool) -> list:
    """
    Merge and split raw skeleton segments at junction points.

    The MATLAB original uses sparse indexing; here we use a dense boolean
    array for clarity (segments are typically short).

    Parameters
    ----------
    skeleton_segments : list of ndarray, shape (N_i, 2|3)
    IS3D : bool

    Returns
    -------
    list of ndarray
    """
    n = len(skeleton_segments)
    if n == 0:
        return []

    dim = 3 if IS3D else 2

    # Collect both endpoints of every segment  --  shape (2n, dim)
    endpoints = np.zeros((n * 2, dim))
    max_seg_len = 1
    for w, ss in enumerate(skeleton_segments):
        max_seg_len = max(max_seg_len, len(ss))
        endpoints[w * 2]     = ss[0]
        endpoints[w * 2 + 1] = ss[-1]

    # cut_skel[w, k] == True  →  segment w should be cut at index k
    cut_skel = np.zeros((n, max_seg_len), dtype=bool)
    connect_distance_sq = 4.0   # 2^2

    for w, ss in enumerate(skeleton_segments):
        m = len(ss)
        # Pairwise squared distances: endpoints (2n,) vs segment points (m,)
        # Broadcasting: (2n, 1) - (1, m) → (2n, m)
        D = np.zeros((len(endpoints), m))
        for d in range(dim):
            D += (endpoints[:, d:d+1] - ss[:, d]) ** 2

        # Which endpoints are "close" to any point on this segment?
        close = D.min(axis=1) < connect_distance_sq   # (2n,)
        close[w * 2]     = False   # ignore self-endpoints
        close[w * 2 + 1] = False

        if close.any():
            for ep_idx in np.where(close)[0]:
                k = int(D[ep_idx].argmin())
                # Only cut in the interior (not near the segment's own tips)
                if 2 < k < (m - 2):
                    cut_skel[w, k] = True

    # Split each segment at its cut-points and collect the sub-segments
    S: list = []
    for w, ss in enumerate(skeleton_segments):
        cut_indices = np.where(cut_skel[w, : len(ss)])[0].tolist()
        boundaries = [0] + cut_indices + [len(ss)]
        for i in range(len(boundaries) - 1):
            S.append(ss[boundaries[i]: boundaries[i + 1]])

    return S


def _get_boundary_distance(I: np.ndarray, IS3D: bool) -> np.ndarray:
    """
    Fast-marching distance from every foreground voxel to the nearest
    foreground/background boundary.

    Parameters
    ----------
    I : ndarray, bool
    IS3D : bool

    Returns
    -------
    ndarray, float  (same shape as I; zero outside the foreground mask)
    """
    struct = np.ones((3,) * I.ndim, dtype=bool)
    dilated = binary_dilation(I, structure=struct)
    boundary = np.logical_xor(I, dilated)

    # Source points: all boundary voxels  →  shape (dim, N_boundary)
    ind = np.argwhere(boundary)        # (N, dim)
    source_point = ind.T               # (dim, N)

    speed_image = np.ones(I.shape, dtype=float)
    boundary_distance = msfm(speed_image, source_point, False, True)

    boundary_distance[~I] = 0.0
    return boundary_distance


def _max_distance_point(
    boundary_distance: np.ndarray,
    I: np.ndarray
) -> tuple:
    """
    Return the foreground voxel with the highest distance value.

    Parameters
    ----------
    boundary_distance : ndarray
    I : ndarray, bool

    Returns
    -------
    pos : ndarray, shape (dim, 1)
        Zero-based index coordinates of the maximum-distance voxel.
    max_d : float
        The maximum distance value.
    """
    bd = boundary_distance.copy()
    bd[~I] = 0.0

    max_d = float(bd.max())
    if not np.isfinite(max_d):
        raise ValueError("_max_distance_point: maximum distance is not finite.")

    flat_idx = int(bd.argmax())
    coords = np.array(np.unravel_index(flat_idx, I.shape), dtype=float)
    pos = coords.reshape(-1, 1)   # column vector  (dim, 1)

    return pos, max_d