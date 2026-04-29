#pragma once
#include "stdbool.h"

int msfm2d(
    const double *F,        // speed image [nx × ny], column-major
    double *T,              // output distance [nx × ny], caller-allocated
    double *Y,              // output euclidian dist (NULL if not needed)
    const double *src_pts,  // source points [2 × N], 0-indexed
    int n_src,              // number of source points
    int nx, int ny,         // image dimensions
    bool use_second,         // use 2nd-order derivatives
    bool use_cross           // use cross/diagonal stencils
);
