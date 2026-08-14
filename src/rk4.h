#ifndef RK4_H
#define RK4_H

int RK4STEP_2D(double *gradientArray, int *gradientArraySize, double *startPoint, double *nextPoint, double stepSize);
int RK4STEP_3D(double *gradientArray, int *gradientArraySize, double *startPoint, double *nextPoint, double stepSize);

#endif // RK4_H
