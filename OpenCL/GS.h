#if CONFIG_USE_DOUBLE
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
typedef double real;
#else
typedef float real;
#endif

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <math.h>

#define MAX_SOURCE_SIZE (0x100000)

#define A(i,j) A[i+j*N]


int GS(void);
