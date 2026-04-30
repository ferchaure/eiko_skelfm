#pragma once
int rk4_step_2d(
    const double *grad, int *grad_size,
    const double *start, double *next, double step_size
);
int rk4_step_3d(
    const double *grad, int *grad_size,
    const double *start, double *next, double step_size
);
