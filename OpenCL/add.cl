#if CONFIG_USE_DOUBLE
	#if defined(cl_khr_fp64)  // Khronos extension
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	#elif defined(cl_amd_fp64)  // AMD extension
		#pragma OPENCL EXTENSION cl_amd_fp64 : enable
	#endif
	typedef double real;
#else
	typedef float real;
#endif


__kernel void vector_add(__global const real *A, __global const real *B,	__global real *C) 
{
	size_t i = get_global_id(0);
	C[i] = A[i] + B[i];
}