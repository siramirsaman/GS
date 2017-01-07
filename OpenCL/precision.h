#ifndef PRECISION
#define PRECISION


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


#endif