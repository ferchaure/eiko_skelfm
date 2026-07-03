#pragma once
#include <stdbool.h>

int msfm3d(
    const double *F,        // speed volume [nx × ny × nz], column-major
    double *T,              // output distance, caller-allocated
    double *Y,              // output euclidian dist (NULL if not needed)
    const double *src_pts,  // source points [3 × N], 0-indexed
    int n_src,
    int nx, int ny, int nz,
    bool usesecond,
    bool usecross
);
