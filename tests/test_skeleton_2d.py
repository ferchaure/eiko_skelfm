#!/usr/bin/env python3
"""
Integration test for skeleton extraction on a synthetic 2D image.

Strategy
--------
1. Build a ground-truth skeleton in a 100 × 100 binary image:
   a dot at the center with 4 lines radiating outward in different
   directions (with slight off-axis drift so they aren't trivially
   axis-aligned).

2. Convolve that thin skeleton with a 2-D Gaussian kernel to produce
   a smooth, tubular "vessel" image, then threshold to get a binary
   mask.

3. Run `skeleton()` on the binary mask.

4. Assert that every extracted skeleton branch lies close to the
   original ground-truth skeleton (Hausdorff-like proximity check).

Run
---
    pip install -e .          # build the C extension first
    python test_skeleton_2d.py
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from eiko_skelfm.skeleton import skeleton


# ---------------------------------------------------------------------------
# 1. Build the ground-truth skeleton
# ---------------------------------------------------------------------------

def make_ground_truth(shape=(100, 100)):
    """
    Create a binary image containing a thin skeleton:
    a center dot with 4 lines going in different directions.

    Returns
    -------
    skel : ndarray, bool, shape `shape`
        True on the skeleton pixels.
    gt_coords : ndarray, shape (N, 2)
        Row/col coordinates of all skeleton pixels.
    """
    ny, nx = shape
    skel = np.zeros(shape, dtype=bool)

    cy, cx = ny // 2, nx // 2  # center

    # Center dot
    skel[cy, cx] = True

    # Line 1 — toward +x (cosine wave)
    for t in range(1, 40):
        x, y = cx + t, cy + int(4 * (1 - np.cos(t / 4.0)))
        if 0 <= y < ny and 0 <= x < nx:
            skel[y, x] = True

    # Line 2 — toward −x (broken into 2 segments)
    for t in range(1, 30):
        x, y = cx - t, cy - t // 4
        if 0 <= y < ny and 0 <= x < nx:
            skel[y, x] = True
    for t in range(1, 30):
        x, y = (cx - 29) - t // 3, (cy - 7) - t
        if 0 <= y < ny and 0 <= x < nx:
            skel[y, x] = True

    # Line 3 — toward +y (slight −x drift)
    for t in range(1, 40):
        y, x = cy + t, cx - t // 5
        if 0 <= y < ny and 0 <= x < nx:
            skel[y, x] = True

    # Line 4 — toward −y (slight +x drift)
    for t in range(1, 40):
        y, x = cy - t, cx + t // 5
        if 0 <= y < ny and 0 <= x < nx:
            skel[y, x] = True

    # --- Add branches to each arm ---
    # Branch 1 (off the cosine wave)
    for t in range(1, 20):
        x, y = cx + 20 + t // 2, cy + int(4 * (1 - np.cos(5.0))) + t
        if 0 <= y < ny and 0 <= x < nx: skel[y, x] = True

    # Branch 2 (off the bent arm's knee)
    for t in range(1, 20):
        x, y = cx - 29 - t, cy - 7 + t // 3
        if 0 <= y < ny and 0 <= x < nx: skel[y, x] = True

    # Branch 3 (off the +y arm)
    for t in range(1, 20):
        x, y = cx - 4 + t, cy + 20 + t // 3
        if 0 <= y < ny and 0 <= x < nx: skel[y, x] = True

    # Branch 4 (off the -y arm)
    for t in range(1, 20):
        x, y = cx + 4 - t, cy - 20 - t // 3
        if 0 <= y < ny and 0 <= x < nx: skel[y, x] = True

    gt_coords = np.argwhere(skel)  # (N, 2)
    return skel, gt_coords


# ---------------------------------------------------------------------------
# 2. Convolve with Gaussian → binary mask
# ---------------------------------------------------------------------------

def skel_to_binary_image(skel, sigma=3.0, threshold_frac=0.15):
    """
    Convolve the thin skeleton with a Gaussian kernel and threshold
    to produce a realistic-looking binary tubular image.

    Parameters
    ----------
    skel : ndarray, bool
    sigma : float
        Standard deviation of the Gaussian kernel.
    threshold_frac : float
        Threshold as a fraction of the maximum after convolution.

    Returns
    -------
    binary_img : ndarray, bool
    """
    img = gaussian_filter(skel.astype(np.float64), sigma=sigma)
    binary_img = img > (img.max() * threshold_frac)
    return binary_img


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
    Concatenate all skeleton segments into a single (N, 2) array.
    """
    if not segments:
        return np.empty((0, 2))
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
    print("Test: skeleton extraction on synthetic 2D image")
    print("=" * 65)

    # --- Build phantom ---
    shape = (100, 100)
    skel_thin, gt_coords = make_ground_truth(shape)
    print(f"Ground-truth skeleton pixels : {len(gt_coords)}")

    binary_img = skel_to_binary_image(skel_thin, sigma=3.0, threshold_frac=0.15)
    fg_count = int(binary_img.sum())
    print(f"Binary image foreground      : {fg_count}  "
          f"({100 * fg_count / binary_img.size:.1f}% of total)")

    # --- Run skeleton extraction ---
    print("\nRunning skeleton()...")
    segments = skeleton(binary_img, verbose=True)
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
    TOLERANCE_MEAN = 5.0   # average distance (pixels)
    TOLERANCE_MAX  = 12.0  # worst-case distance (pixels)

    print(f"\nMean GT→Ext < {TOLERANCE_MEAN}  : {mean_gt_to_ext:.2f}")
    assert mean_gt_to_ext < TOLERANCE_MEAN, \
        f"Mean distance GT→Ext {mean_gt_to_ext:.2f} exceeds {TOLERANCE_MEAN}"

    print(f"Max  GT→Ext < {TOLERANCE_MAX} : {max_gt_to_ext:.2f}")
    assert max_gt_to_ext < TOLERANCE_MAX, \
        f"Max distance GT→Ext {max_gt_to_ext:.2f} exceeds {TOLERANCE_MAX}"

    print(f"Mean Ext→GT < {TOLERANCE_MEAN}  : {mean_ext_to_gt:.2f}")
    assert mean_ext_to_gt < TOLERANCE_MEAN, \
        f"Mean distance Ext→GT {mean_ext_to_gt:.2f} exceeds {TOLERANCE_MEAN}"

    print(f"Max  Ext→GT < {TOLERANCE_MAX} : {max_ext_to_gt:.2f}")
    assert max_ext_to_gt < TOLERANCE_MAX, \
        f"Max distance Ext→GT {max_ext_to_gt:.2f} exceeds {TOLERANCE_MAX}"

    # --- Branch count check ---
    n_branches = len(segments)
    print(f"\nBranch count {n_branches} in [5, 25]")
    assert 5 <= n_branches <= 25, \
        f"Branch count {n_branches} outside expected range [5, 25]"


if __name__ == "__main__":
    test_skeleton_recovers_ground_truth()
