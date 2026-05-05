import numpy as np
from . import _msfm

def msfm(speed, source_points, use_second=True, use_cross=True, return_y=False):
    """
    Fast Marching Method solver.

    Parameters
    ----------
    speed : ndarray (2D or 3D), float64
        Speed image. Must be Fortran-contiguous.
    source_points : ndarray, shape (ndim, N), float64
        Source points (0-indexed). Must be Fortran-contiguous with
        2 rows for 2D or 3 rows for 3D.
    use_second : bool
        Use second-order derivatives.
    use_cross : bool
        Use cross/diagonal stencils.
    return_y : bool
        Also return Euclidean distance (for skeleton extraction).

    Returns
    -------
    T : ndarray, same shape as speed (distance field, Fortran order)
    Y : ndarray or None (Euclidean distance, Fortran order)
    """
    speed = np.asfortranarray(speed, dtype=np.float64)
    speed = np.clip(speed, 1e-8, None)
    source_points = np.asfortranarray(source_points, dtype=np.float64)

    if speed.ndim == 3:
        if source_points.shape[0] != 3:
            raise ValueError("source_points must be 3xN for 3D speed")
        result = _msfm.msfm3d(speed, source_points,
                               int(use_second), int(use_cross),
                               int(return_y))
    elif speed.ndim == 2:
        if source_points.shape[0] != 2:
            raise ValueError("source_points must be 2xN for 2D speed")
        result = _msfm.msfm2d(speed, source_points,
                               int(use_second), int(use_cross),
                               int(return_y))
    else:
        raise ValueError("speed must be 2D or 3D")

    return result
