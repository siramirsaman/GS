
#ifndef GS_DEF
#define GS_DEF

#include"precision.h"

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


#include "reset_arrays.h"

int GS(cl_context context, cl_device_id device_id, cl_command_queue command_queue,
	cl_mem x_mem_obj, cl_mem a_mem_obj, cl_mem b_mem_obj, cl_mem resid_mem_obj,
	cl_mem partial_sums_mem_obj, cl_mem partial_sums_mem_obj_num_blocks,
	real *x_out, real *A, real *x, real *b, real *resid, real *partial_sums,
	int N, real omega, real tol,
	size_t block_size, size_t num_blocks, size_t local_item_size, size_t global_item_size);

#endif

