#pragma once

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define eps 2.2204460492503131e-16
#define doublemax 1e50
#define INF 2e50
#define listINF 2.345e50
#ifndef min
#define min(a,b)        ((a) < (b) ? (a): (b))
#endif
#ifndef max
#define max(a,b)        ((a) > (b) ? (a): (b))
#endif

int minarray(double *A, int l);
double pow2(double val);
int iszero(double a);
int isnotzero(double a);
void roots(double* Coeff, double* ans);
int maxarray(double *A, int l);
static inline int mindex3(int x, int y, int z, int sizx, int sizy);
static inline bool IsFinite(double x);
static inline bool IsInf(double x);
static inline bool IsListInf(double x);
static inline bool isntfrozen3d(int i, int j, int k, int *dims, bool *Frozen);
static inline bool isfrozen3d(int i, int j, int k, int *dims, bool *Frozen);
int p2x(int x);
void show_list(double **listval, int *listprop);
void initialize_list(double ** listval, int *listprop);
void destroy_list(double ** listval, int *listprop);
void list_add(double ** listval, int *listprop, double val);
int list_minimum(double ** listval, int *listprop);
void list_remove(double ** listval, int *listprop, int index);
void list_remove_replace(double ** listval, int *listprop, int index);
void listupdate(double **listval, int *listprop, int index, double val);
static inline int mindex2(int x, int y, int sizx);
static inline bool isntfrozen2d(int i, int j, int *dims, bool *Frozen);
static inline bool isfrozen2d(int i, int j, int *dims, bool *Frozen);

static inline int mindex3(int x, int y, int z, int sizx, int sizy) { return x+y*sizx+z*sizx*sizy; }

static inline bool IsFinite(double x) { return (x <= doublemax  && x >= -doublemax ); }
static inline bool IsInf(double x)    { return (x >= doublemax ); }

static inline bool IsListInf(double x){ return (x == listINF ); }

static inline bool isntfrozen3d(int i, int j, int k, int *dims, bool *Frozen) {
    return (i>=0)&&(j>=0)&&(k>=0)&&(i<dims[0])&&(j<dims[1])&&(k<dims[2])&&(Frozen[mindex3(i, j, k, dims[0], dims[1])]==0);
}
static inline bool isfrozen3d(int i, int j, int k, int *dims, bool *Frozen) {
    return (i>=0)&&(j>=0)&&(k>=0)&&(i<dims[0])&&(j<dims[1])&&(k<dims[2])&&(Frozen[mindex3(i, j, k, dims[0], dims[1])]==1);
}

static inline int mindex2(int x, int y, int sizx) { return x+y*sizx; }

static inline bool isntfrozen2d(int i, int j, int *dims, bool *Frozen)
{
    return (i>=0)&&(j>=0)&&(i<dims[0])&&(j<dims[1])&&(Frozen[i+j*dims[0]]==0);
}
static inline bool isfrozen2d(int i, int j, int *dims, bool *Frozen)
{
    return (i>=0)&&(j>=0)&&(i<dims[0])&&(j<dims[1])&&(Frozen[i+j*dims[0]]==1);
}
