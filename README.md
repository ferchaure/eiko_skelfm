# eiko_skelfm

[![PyPI version](https://img.shields.io/pypi/v/eiko-skelfm.svg)](https://pypi.org/project/eiko-skelfm/)

Fast-marching skeleton extraction for 2D and 3D binary images.  
Port of Kroon's MATLAB/MEX FastMarching toolbox to standalone C + Python/NumPy.

![Skeleton extraction result](https://raw.githubusercontent.com/ferchaure/eiko_skelfm/main/docs/figures/skeleton_2d_result.png)

## Installation

```bash
pip install -e .

# with dev dependencies (pytest, matplotlib)
pip install -e ".[dev]"
```

## Quick Example — 2D skeleton

```python
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from eiko_skelfm.skeleton import skeleton

# 1. Create a thin skeleton: center dot + 4 radiating lines
shape = (100, 100)
skel = np.zeros(shape, dtype=bool)
cy, cx = 50, 50
skel[cy, cx] = True
for t in range(1, 40):
    skel[cy + int(4 * (1 - np.cos(t / 4.0))), cx + t] = True  # line → +x (cosine)
    skel[cy + t, cx - t // 5] = True   # line → +y
    skel[cy - t, cx + t // 5] = True   # line → −y
for t in range(1, 30):
    skel[cy - t // 4, cx - t] = True                 # line → −x (seg 1)
    skel[cy - 7 - t, cx - 29 - t // 3] = True        # line → −x (seg 2)
for t in range(1, 20):
    skel[cy + int(4 * (1 - np.cos(5.0))) + t, cx + 20 + t // 2] = True # branch 1
    skel[cy - 7 + t // 3, cx - 29 - t] = True                          # branch 2
    skel[cy + 20 + t // 3, cx - 4 + t] = True                          # branch 3
    skel[cy - 20 - t // 3, cx + 4 - t] = True                          # branch 4

# 2. Convolve with a Gaussian to create a tubular binary mask
convolved = gaussian_filter(skel.astype(float), sigma=3.0)
binary_mask = convolved > convolved.max() * 0.15

# 3. Extract the skeleton
segments = skeleton(binary_mask, verbose=True)

# 4. Plot the result
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
axes[0].imshow(skel, cmap="gray", origin="lower")
axes[0].set_title("Ground-truth skeleton")

axes[1].imshow(convolved, cmap="inferno", origin="lower")
axes[1].set_title("After Gaussian blur (σ=3)")

axes[2].imshow(binary_mask, cmap="gray", origin="lower")
axes[2].set_title("Binary mask")

axes[3].imshow(binary_mask, cmap="gray", origin="lower", alpha=0.35)
for seg in segments:
    axes[3].plot(seg[:, 1], seg[:, 0], linewidth=2)
axes[3].set_title("Extracted skeleton")

plt.tight_layout()
plt.savefig("skeleton_2d.png", dpi=150)
plt.show()
```

![Four-panel overview](https://raw.githubusercontent.com/ferchaure/eiko_skelfm/main/docs/figures/skeleton_2d_overview.png)

See [`docs/example_skeleton_2d.py`](docs/example_skeleton_2d.py) for the full runnable script.

## Build Standalone C Library

### Linux (GCC)
```bash
gcc -O2 -std=c99 -c src/common.c src/msfm2d.c src/msfm3d.c -lm
ar rcs libeiko_skelfm.a common.o msfm2d.o msfm3d.o
```

### macOS (Clang)
```bash
clang -O2 -std=c99 -c src/common.c src/msfm2d.c src/msfm3d.c
ar rcs libeiko_skelfm.a common.o msfm2d.o msfm3d.o
```

### Windows (MSVC)
```cmd
cl /O2 /c src\common.c src\msfm2d.c src\msfm3d.c
lib /OUT:fastmarching.lib common.obj msfm2d.obj msfm3d.obj
```

### Windows (MinGW)
```bash
gcc -O2 -std=c99 -c src/common.c src/msfm2d.c src/msfm3d.c 
ar rcs libeiko_skelfm.a common.o msfm2d.o msfm3d.o
```

### CMake
```bash
cmake -B build -S .
cmake --build build
```

