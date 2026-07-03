#!/usr/bin/env python3
"""
example_skeleton_2d.py
======================
Demonstrates 2D skeleton extraction with eiko_skelfm.

Creates a synthetic tubular structure by drawing a thin skeleton
(center dot + 4 radiating lines), convolving it with a Gaussian
kernel, and thresholding. Then runs the skeleton() pipeline and
plots each stage.

Figures are saved to docs/figures/.

Usage
-----
    pip install -e ".[dev]"
    python docs/example_skeleton_2d.py
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from eiko_skelfm.skeleton import skeleton
from eiko_skelfm.msfm import msfm


# ── 1. Build a thin ground-truth skeleton ────────────────────────────────

shape = (100, 100)
ny, nx = shape
skel = np.zeros(shape, dtype=bool)

cy, cx = ny // 2, nx // 2

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

print(f"Ground-truth skeleton pixels: {skel.sum()}")


# ── 2. Convolve with a Gaussian kernel ──────────────────────────────────

sigma = 3.0
convolved = gaussian_filter(skel.astype(np.float64), sigma=sigma)
binary_mask = convolved > (convolved.max() * 0.15)

print(f"Binary mask foreground: {binary_mask.sum()} pixels "
      f"({100 * binary_mask.sum() / binary_mask.size:.1f}%)")


# ── 3. Extract the skeleton ─────────────────────────────────────────────

print("Running skeleton extraction...")
segments = skeleton(binary_mask, verbose=True)
print(f"Extracted {len(segments)} branch segments")


# ── 4. Plot everything ──────────────────────────────────────────────────

import os
fig_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(fig_dir, exist_ok=True)

# ─── Color palette ───
BG       = "#0f0f12"
ACCENT   = "#00e5ff"
GOLD     = "#ffd740"
PURPLE   = "#b388ff"
PINK     = "#ff80ab"

branch_colors = [ACCENT, GOLD, PURPLE, PINK, "#69f0ae", "#ff6e40",
                 "#40c4ff", "#eeff41", "#ea80fc", "#ff5252"]


# ─── Figure 1: four-panel overview ───
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), facecolor=BG)

for ax in axes:
    ax.set_facecolor(BG)
    ax.tick_params(colors="#888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333")

# Panel A – thin skeleton
axes[0].imshow(skel.astype(float), cmap="gray", origin="lower",
               interpolation="nearest")
axes[0].set_title("A · Ground-truth skeleton", color="white", fontsize=11)

# Panel B – Gaussian-convolved
axes[1].imshow(convolved, cmap="inferno", origin="lower")
axes[1].set_title(f"B · After Gaussian blur (σ={sigma})", color="white",
                  fontsize=11)

# Panel C – binary mask
axes[2].imshow(binary_mask.astype(float), cmap="gray", origin="lower",
               interpolation="nearest")
axes[2].set_title("C · Thresholded binary mask", color="white", fontsize=11)

# Panel D – extracted skeleton overlaid on mask
axes[3].imshow(binary_mask.astype(float), cmap="gray", origin="lower",
               interpolation="nearest", alpha=0.35)
for i, seg in enumerate(segments):
    color = branch_colors[i % len(branch_colors)]
    axes[3].plot(seg[:, 1], seg[:, 0], linewidth=2.0, color=color,
                 label=f"branch {i+1}")
axes[3].legend(fontsize=7, loc="lower right", framealpha=0.6,
               facecolor="#222", labelcolor="white")
axes[3].set_title("D · Extracted skeleton branches", color="white",
                  fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(fig_dir, "skeleton_2d_overview.png"),
            dpi=180, facecolor=BG, bbox_inches="tight")
print(f"Saved → {fig_dir}/skeleton_2d_overview.png")
plt.close(fig)


# ─── Figure 2: skeleton on binary mask (standalone, for README) ───
fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor=BG)
ax2.set_facecolor(BG)
ax2.imshow(binary_mask.astype(float), cmap="gray", origin="lower",
           interpolation="nearest", alpha=0.35)
for i, seg in enumerate(segments):
    color = branch_colors[i % len(branch_colors)]
    ax2.plot(seg[:, 1], seg[:, 0], linewidth=2.5, color=color)
ax2.set_title("Skeleton extraction — eiko_skelfm", color="white",
              fontsize=13, pad=10)
ax2.tick_params(colors="#888", labelsize=7)
for spine in ax2.spines.values():
    spine.set_color("#333")

fig2.savefig(os.path.join(fig_dir, "skeleton_2d_result.png"),
             dpi=180, facecolor=BG, bbox_inches="tight")
print(f"Saved → {fig_dir}/skeleton_2d_result.png")
plt.close(fig2)

print("\nDone!")
