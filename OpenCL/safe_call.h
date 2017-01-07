#ifndef SAFE_CALL
#define SAFE_CALL


#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#define safe_call(a) safe_call_f(a, __LINE__)

/** @brief Error checking of OpenCL calls
*  @param code OpenCL routine error code
*  @param line code line number to be printed on error
*  @return void
*/
void safe_call_f(cl_int code, size_t line);


#endif