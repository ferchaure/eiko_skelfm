#!/usr/bin/env python3
"""
Integration test for skeleton extraction on a synthetic 3D volume.

Strategy
--------
1. Build a ground-truth skeleton in a 40 × 100 × 100 binary volume:
   a dot at the center with 4 lines radiating outward in different
   directions (two roughly along axis-1, two roughly along axis-2,
   with slight offsets so they are not axis-aligned).

2. Convolve that thin skeleton with a 3-D Gaussian kernel to produce
   a smooth, tubular "vessel" volume, then threshold to get a binary
   mask.

3. Run `skeleton()` on the binary mask.

4. Assert that every extracted skeleton branch lies close to the
   original ground-truth skeleton (Hausdorff-like proximity check).

Run
---
    pip install -e .          # build the C extension first
    python test_skeleton_3d.py
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from eiko_skelfm.skeleton import skeleton


# ---------------------------------------------------------------------------
# 1. Build the ground-truth skeleton
# ---------------------------------------------------------------------------

def make_ground_truth(shape=(40, 100, 100)):
    """
    Create a binary volume containing a thin skeleton:
    a center dot with 4 lines going in different directions.

    Returns
    -------
    skel : ndarray, bool, shape `shape`
        True on the skeleton voxels.
    gt_coords : ndarray, shape (N, 3)
        Row/col/depth coordinates of all skeleton voxels.
    """
    nz, ny, nx = shape
    skel = np.zeros(shape, dtype=bool)

    cz, cy, cx = nz // 2, ny // 2, nx // 2  # center

    # Center dot
    skel[cz, cy, cx] = True

    # Line 1 — toward +y (cosine wave in x)
    for t in range(1, 40):
        y = cy + t
        x = cx + int(5 * (1 - np.cos(t / 4.0)))
        if 0 <= y < ny and 0 <= x < nx:
            skel[cz, y, x] = True

    # Line 2 — toward −y (broken into 2 segments in 3D)
    for t in range(1, 30):
        y = cy - t
        x = cx - t // 4
        if 0 <= y < ny and 0 <= x < nx:
            skel[cz, y, x] = True
    end_y, end_x = cy - 29, cx - 29 // 4
    for t in range(1, 30):
        y = end_y - t // 2
        x = end_x - t // 4
        z = cz + t // 2
        if 0 <= y < ny and 0 <= x < nx and 0 <= z < nz:
            skel[z, y, x] = True

    # Line 3 — toward +x (helix in y, z)
    for t in range(1, 40):
        x = cx + t
        y = cy + int(4 * (np.cos(t / 3.0) - 1))
        z = cz + int(4 * np.sin(t / 3.0))
        if 0 <= y < ny and 0 <= x < nx and 0 <= z < nz:
            skel[z, y, x] = True

    # Line 4 — toward −x  (with slight −z drift)
    for t in range(1, 40):
        x = cx - t
        z = cz - t // 4
        if 0 <= x < nx and 0 <= z < nz:
            skel[z, cy, x] = True

    # --- Add branches to each arm ---
    # Branch 1 (off the cosine wave, +y arm)
    for t in range(1, 15):
        y = cy + 20 + t // 2
        x = cx + int(5 * (1 - np.cos(5.0))) + t
        z = cz + t // 2
        if 0 <= y < ny and 0 <= x < nx and 0 <= z < nz: skel[z, y, x] = True

    # Branch 2 (off the bent arm's knee, -y arm)
    for t in range(1, 15):
        y = end_y - t // 3
        x = end_x - t
        z = cz - t // 2
        if 0 <= y < ny and 0 <= x < nx and 0 <= z < nz: skel[z, y, x] = True

    # Branch 3 (off the helix, +x arm)
    for t in range(1, 15):
        x = cx + 20 + t // 2
        y = cy + int(4 * (np.cos(20/3.0) - 1)) + t
        z = cz + int(4 * np.sin(20/3.0)) + t // 2
        if 0 <= y < ny and 0 <= x < nx and 0 <= z < nz: skel[z, y, x] = True

    # Branch 4 (off the -x arm)
    for t in range(1, 15):
        x = cx - 20 - t // 2
        y = cy - t
        z = cz - 5 + t
        if 0 <= y < ny and 0 <= x < nx and 0 <= z < nz: skel[z, y, x] = True

    gt_coords = np.argwhere(skel)  # (N, 3)
    return skel, gt_coords


# ---------------------------------------------------------------------------
# 2. Convolve with Gaussian → binary mask
# ---------------------------------------------------------------------------

def skel_to_binary_volume(skel, sigma=3.0, threshold_frac=0.15):
    """
    Convolve the thin skeleton with a Gaussian kernel and threshold
    to produce a realistic-looking binary tubular volume.

    Parameters
    ----------
    skel : ndarray, bool
    sigma : float
        Standard deviation of the Gaussian kernel.
    threshold_frac : float
        Threshold as a fraction of the maximum after convolution.

    Returns
    -------
    binary_vol : ndarray, bool
    """
    vol = gaussian_filter(skel.astype(np.float64), sigma=sigma)
    binary_vol = vol > (vol.max() * threshold_frac)
    return binary_vol


# ---------------------------------------------------------------------------
# 3. Proximity check helpers
# ---------------------------------------------------------------------------

def hausdorff_one_way(A, B):
    """
    Max distance from each point in A to the nearest point in B.
    """
    tree = cKDTree(B)
    dists, _ = tree.query(A)
    return float(dists.max()), float(dists.mean())


def skeleton_segments_to_coords(segments):
    """
    Concatenate all skeleton segments into a single (N, 3) array,
    rounding to ints for the tree query.
    """
    if not segments:
        return np.empty((0, 3))
    return np.vstack(segments)


# ---------------------------------------------------------------------------
# 4. Main test
# ---------------------------------------------------------------------------

def test_skeleton_recovers_ground_truth():
    """
    End-to-end test: ground-truth skeleton → Gaussian blur → threshold
    → skeleton() → compare with ground truth.
    """
    print("=" * 65)
    print("Test: skeleton extraction on synthetic 3D volume")
    print("=" * 65)

    # --- Build phantom ---
    shape = (40, 100, 100)
    skel_thin, gt_coords = make_ground_truth(shape)
    print(f"Ground-truth skeleton voxels : {len(gt_coords)}")

    binary_vol = skel_to_binary_volume(skel_thin, sigma=3.0, threshold_frac=0.15)
    fg_count = int(binary_vol.sum())
    print(f"Binary volume foreground     : {fg_count}  "
          f"({100 * fg_count / binary_vol.size:.1f}% of total)")

    # --- Run skeleton extraction ---
    print("\nRunning skeleton()  (this may take a minute)...")
    segments = skeleton(binary_vol, verbose=True)
    print(f"\nExtracted segments           : {len(segments)}")

    assert len(segments) > 0, "skeleton() returned no segments!"

    extracted = skeleton_segments_to_coords(segments)
    print(f"Extracted skeleton points    : {len(extracted)}")

    # --- Proximity check ---
    #   For every ground-truth point, find the closest extracted point.
    #   The max (Hausdorff) distance should be small relative to sigma.
    max_gt_to_ext, mean_gt_to_ext = hausdorff_one_way(gt_coords, extracted)
    max_ext_to_gt, mean_ext_to_gt = hausdorff_one_way(extracted, gt_coords)

    print(f"\nGT → Extracted  :  max={max_gt_to_ext:.2f}  mean={mean_gt_to_ext:.2f}")
    print(f"Extracted → GT  :  max={max_ext_to_gt:.2f}  mean={mean_ext_to_gt:.2f}")

    # Tolerances — the skeleton should stay within ~sigma of the GT lines
    TOLERANCE_MEAN = 5.0   # average distance (voxels)
    TOLERANCE_MAX  = 12.0  # worst-case distance (voxels)

    ok_mean = mean_gt_to_ext < TOLERANCE_MEAN
    ok_max  = max_gt_to_ext < TOLERANCE_MAX

    print(f"\nMean GT→Ext < {TOLERANCE_MEAN}  : {'PASS' if ok_mean else 'FAIL'}")
    print(f"Max  GT→Ext < {TOLERANCE_MAX} : {'PASS' if ok_max else 'FAIL'}")

    # Also check the extracted skeleton doesn't wildly deviate from GT
    ok_ext_mean = mean_ext_to_gt < TOLERANCE_MEAN
    ok_ext_max  = max_ext_to_gt < TOLERANCE_MAX

    print(f"Mean Ext→GT < {TOLERANCE_MEAN}  : {'PASS' if ok_ext_mean else 'FAIL'}")
    print(f"Max  Ext→GT < {TOLERANCE_MAX} : {'PASS' if ok_ext_max else 'FAIL'}")

    all_pass = ok_mean and ok_max and ok_ext_mean and ok_ext_max

    # --- Branch count check ---
    # We expect roughly 5-25 branches due to complex topology
    n_branches = len(segments)
    ok_branches = 5 <= n_branches <= 25
    print(f"\nBranch count {n_branches} in [5, 25] : {'PASS' if ok_branches else 'FAIL'}")

    all_pass = all_pass and ok_branches

    print("\n" + "=" * 65)
    if all_pass:
        print("ALL CHECKS PASSED ✓")
    else:
        print("SOME CHECKS FAILED ✗")
    print("=" * 65)

    assert all_pass, "Skeleton extraction test failed — see details above."


if __name__ == "__main__":
    test_skeleton_recovers_ground_truth()
