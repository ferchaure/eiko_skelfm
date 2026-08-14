#include <math.h>

#ifndef false
#define false 0
#endif
#ifndef true
#define true 1
#endif

static inline int mindex2(int x, int y, int sizx)  { return y*sizx+x;}
static inline int mindex3(int x, int y, int z, int sizx, int sizy)  { return z*sizy*sizx+y*sizx+x;}

static inline int checkBounds2d( double *point, int *Isize) {
    if((point[0]<0)||(point[1]<0)||(point[0]>(Isize[0]-1))||(point[1]>(Isize[1]-1))) { return false; }
    return true;
}

static inline int checkBounds3d( double *point, int *Isize) {
    if((point[0]<0)||(point[1]<0)||(point[2]<0)||(point[0]>(Isize[0]-1))||(point[1]>(Isize[1]-1))||(point[2]>(Isize[2]-1))) { return false; }
    return true;
}

static inline double norm2(double *a) { return sqrt(a[0]*a[0]+a[1]*a[1]); }
static inline double norm3(double *a) { return sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]); }

static inline void interpgrad2d(double *Ireturn, double *I, int *Isize, double *point) {
    int xBas0, xBas1, yBas0, yBas1;
    double perc[4]={0, 0, 0, 0};
    double xCom, yCom, xComi, yComi;
    double fTlocalx, fTlocaly;
    int f;
    int index[4];
    
    fTlocalx = floor(point[0]); fTlocaly = floor(point[1]);
    xBas0=(int) fTlocalx; yBas0=(int) fTlocaly;
    xBas1=xBas0+1; yBas1=yBas0+1;
    
    xCom=point[0]-fTlocalx; yCom=point[1]-fTlocaly;
    xComi=(1-xCom); yComi=(1-yCom);
    perc[0]=xComi * yComi;
    perc[1]=xComi * yCom;
    perc[2]=xCom * yComi;
    perc[3]=xCom * yCom;
    
    if(xBas0<0) { xBas0=0; if(xBas1<0) { xBas1=0; }}
    if(yBas0<0) { yBas0=0; if(yBas1<0) { yBas1=0; }}
    if(xBas1>(Isize[0]-1)) { xBas1=Isize[0]-1; if(xBas0>(Isize[0]-1)) { xBas0=Isize[0]-1; }}
    if(yBas1>(Isize[1]-1)) { yBas1=Isize[1]-1; if(yBas0>(Isize[1]-1)) { yBas0=Isize[1]-1; }}
    
    index[0]=mindex2(xBas0, yBas0, Isize[0]);
    index[1]=mindex2(xBas0, yBas1, Isize[0]);
    index[2]=mindex2(xBas1, yBas0, Isize[0]);
    index[3]=mindex2(xBas1, yBas1, Isize[0]);
    f=Isize[0]*Isize[1];
    
    Ireturn[0]=I[index[0]]*perc[0]+I[index[1]]*perc[1]+I[index[2]]*perc[2]+I[index[3]]*perc[3];
    Ireturn[1]=I[index[0]+f]*perc[0]+I[index[1]+f]*perc[1]+I[index[2]+f]*perc[2]+I[index[3]+f]*perc[3];
}

static inline void interpgrad3d(double *Ireturn, double *I, int *Isize, double *point) {
    int xBas0, xBas1, yBas0, yBas1, zBas0, zBas1;
    double perc[8];
    double xCom, yCom, zCom;
    double xComi, yComi, zComi;
    double fTlocalx, fTlocaly, fTlocalz;
    int f0, f1;
    int index[8];
    double temp;
    
    fTlocalx = floor(point[0]); fTlocaly = floor(point[1]); fTlocalz = floor(point[2]);
    xBas0=(int) fTlocalx; yBas0=(int) fTlocaly; zBas0=(int) fTlocalz;
    xBas1=xBas0+1; yBas1=yBas0+1; zBas1=zBas0+1;
    
    xCom=point[0]-fTlocalx;  yCom=point[1]-fTlocaly;   zCom=point[2]-fTlocalz;
    xComi=(1-xCom); yComi=(1-yCom); zComi=(1-zCom);
    perc[0]=xComi * yComi; perc[1]=perc[0] * zCom; perc[0]=perc[0] * zComi;
    perc[2]=xComi * yCom;  perc[3]=perc[2] * zCom; perc[2]=perc[2] * zComi;
    perc[4]=xCom * yComi;  perc[5]=perc[4] * zCom; perc[4]=perc[4] * zComi;
    perc[6]=xCom * yCom;   perc[7]=perc[6] * zCom; perc[6]=perc[6] * zComi;
    
    if(xBas0<0) { xBas0=0; if(xBas1<0) { xBas1=0; }}
    if(yBas0<0) { yBas0=0; if(yBas1<0) { yBas1=0; }}
    if(zBas0<0) { zBas0=0; if(zBas1<0) { zBas1=0; }}
    if(xBas1>(Isize[0]-1)) { xBas1=Isize[0]-1; if(xBas0>(Isize[0]-1)) { xBas0=Isize[0]-1; }}
    if(yBas1>(Isize[1]-1)) { yBas1=Isize[1]-1; if(yBas0>(Isize[1]-1)) { yBas0=Isize[1]-1; }}
    if(zBas1>(Isize[2]-1)) { zBas1=Isize[2]-1; if(zBas0>(Isize[2]-1)) { zBas0=Isize[2]-1; }}
    
    index[0]=mindex3(xBas0, yBas0, zBas0, Isize[0], Isize[1]);
    index[1]=mindex3(xBas0, yBas0, zBas1, Isize[0], Isize[1]);
    index[2]=mindex3(xBas0, yBas1, zBas0, Isize[0], Isize[1]);
    index[3]=mindex3(xBas0, yBas1, zBas1, Isize[0], Isize[1]);
    index[4]=mindex3(xBas1, yBas0, zBas0, Isize[0], Isize[1]);
    index[5]=mindex3(xBas1, yBas0, zBas1, Isize[0], Isize[1]);
    index[6]=mindex3(xBas1, yBas1, zBas0, Isize[0], Isize[1]);
    index[7]=mindex3(xBas1, yBas1, zBas1, Isize[0], Isize[1]);
    f0=Isize[0]*Isize[1]*Isize[2];
    f1=f0+f0;
    
    temp=I[index[0]]*perc[0]+I[index[1]]*perc[1]+I[index[2]]*perc[2]+I[index[3]]*perc[3];
    Ireturn[0]=temp+I[index[4]]*perc[4]+I[index[5]]*perc[5]+I[index[6]]*perc[6]+I[index[7]]*perc[7];
    temp=I[index[0]+f0]*perc[0]+I[index[1]+f0]*perc[1]+I[index[2]+f0]*perc[2]+I[index[3]+f0]*perc[3];
    Ireturn[1]=temp+I[index[4]+f0]*perc[4]+I[index[5]+f0]*perc[5]+I[index[6]+f0]*perc[6]+I[index[7]+f0]*perc[7];
    temp=I[index[0]+f1]*perc[0]+I[index[1]+f1]*perc[1]+I[index[2]+f1]*perc[2]+I[index[3]+f1]*perc[3];
    Ireturn[2]=temp+I[index[4]+f1]*perc[4]+I[index[5]+f1]*perc[5]+I[index[6]+f1]*perc[6]+I[index[7]+f1]*perc[7];
}
 
int RK4STEP_2D(double *gradientArray, int *gradientArraySize, double *startPoint, double *nextPoint, double stepSize) {
    double k1[2], k2[2], k3[2], k4[2];
    double tempPoint[2];
    double tempnorm;
    
    interpgrad2d(k1, gradientArray, gradientArraySize, startPoint);
    tempnorm=norm2(k1);
    if(tempnorm == 0.0) return false;
    k1[0] = k1[0]*stepSize/tempnorm;
    k1[1] = k1[1]*stepSize/tempnorm;
    
    tempPoint[0]=startPoint[0] - k1[0]*0.5;
    tempPoint[1]=startPoint[1] - k1[1]*0.5;
    
    if (!checkBounds2d(tempPoint, gradientArraySize)) return false;
    
    interpgrad2d(k2, gradientArray, gradientArraySize, tempPoint);
    tempnorm=norm2(k2);
    if(tempnorm == 0.0) return false;
    k2[0] = k2[0]*stepSize/tempnorm;
    k2[1] = k2[1]*stepSize/tempnorm;
    
    tempPoint[0]=startPoint[0] - k2[0]*0.5;
    tempPoint[1]=startPoint[1] - k2[1]*0.5;
    
    if (!checkBounds2d(tempPoint, gradientArraySize)) return false;
    
    interpgrad2d(k3, gradientArray, gradientArraySize, tempPoint);
    tempnorm=norm2(k3);
    if(tempnorm == 0.0) return false;
    k3[0] = k3[0]*stepSize/tempnorm;
    k3[1] = k3[1]*stepSize/tempnorm;
    
    tempPoint[0]=startPoint[0] - k3[0];
    tempPoint[1]=startPoint[1] - k3[1];
    
    if (!checkBounds2d(tempPoint, gradientArraySize)) return false;
    
    interpgrad2d(k4, gradientArray, gradientArraySize, tempPoint);
    tempnorm=norm2(k4);
    if(tempnorm == 0.0) return false;
    k4[0] = k4[0]*stepSize/tempnorm;
    k4[1] = k4[1]*stepSize/tempnorm;
    
    nextPoint[0] = startPoint[0] - (k1[0] + k2[0]*2.0 + k3[0]*2.0 + k4[0])/6.0;
    nextPoint[1] = startPoint[1] - (k1[1] + k2[1]*2.0 + k3[1]*2.0 + k4[1])/6.0;
    
    if (!checkBounds2d(nextPoint, gradientArraySize)) return false;
    
    return true;
}

int RK4STEP_3D(double *gradientArray, int *gradientArraySize, double *startPoint, double *nextPoint, double stepSize) {
    double k1[3], k2[3], k3[3], k4[3];
    double tempPoint[3];
    double tempnorm;
    
    interpgrad3d(k1, gradientArray, gradientArraySize, startPoint);
    tempnorm=norm3(k1);
    if(tempnorm == 0.0) return false;
    k1[0] = k1[0]*stepSize/tempnorm;
    k1[1] = k1[1]*stepSize/tempnorm;
    k1[2] = k1[2]*stepSize/tempnorm;
    
    tempPoint[0]=startPoint[0] - k1[0]*0.5;
    tempPoint[1]=startPoint[1] - k1[1]*0.5;
    tempPoint[2]=startPoint[2] - k1[2]*0.5;
            
    if (!checkBounds3d(tempPoint, gradientArraySize)) return false;
    
    interpgrad3d(k2, gradientArray, gradientArraySize, tempPoint);
    tempnorm=norm3(k2);
    if(tempnorm == 0.0) return false;
    k2[0] = k2[0]*stepSize/tempnorm;
    k2[1] = k2[1]*stepSize/tempnorm;
    k2[2] = k2[2]*stepSize/tempnorm;
    
    tempPoint[0]=startPoint[0] - k2[0]*0.5;
    tempPoint[1]=startPoint[1] - k2[1]*0.5;
    tempPoint[2]=startPoint[2] - k2[2]*0.5;
    
    if (!checkBounds3d(tempPoint, gradientArraySize)) return false;
    
    interpgrad3d(k3, gradientArray, gradientArraySize, tempPoint);
    tempnorm=norm3(k3);
    if(tempnorm == 0.0) return false;
    k3[0] = k3[0]*stepSize/tempnorm;
    k3[1] = k3[1]*stepSize/tempnorm;
    k3[2] = k3[2]*stepSize/tempnorm;
        
    tempPoint[0]=startPoint[0] - k3[0];
    tempPoint[1]=startPoint[1] - k3[1];
    tempPoint[2]=startPoint[2] - k3[2];
    
    if (!checkBounds3d(tempPoint, gradientArraySize)) return false;
    
    interpgrad3d(k4, gradientArray, gradientArraySize, tempPoint);
    tempnorm=norm3(k4);
    if(tempnorm == 0.0) return false;
    k4[0] = k4[0]*stepSize/tempnorm;
    k4[1] = k4[1]*stepSize/tempnorm;
    k4[2] = k4[2]*stepSize/tempnorm;
    
    nextPoint[0] = startPoint[0] - (k1[0] + k2[0]*2.0 + k3[0]*2.0 + k4[0])/6.0;
    nextPoint[1] = startPoint[1] - (k1[1] + k2[1]*2.0 + k3[1]*2.0 + k4[1])/6.0;
    nextPoint[2] = startPoint[2] - (k1[2] + k2[2]*2.0 + k3[2]*2.0 + k4[2])/6.0;
     
    if (!checkBounds3d(nextPoint, gradientArraySize)) return false;
    
    return true;
}
