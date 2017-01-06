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

#define A(ix,j) A[ix+j*N]


/** @brief GPU Implementation of relaxed Gauss Seidel for solving a square
 *         system of linear equations A.x=b
 *  @param x array of unknowns [N]
 *  @param A array of coefficients [N*N]
 *  @param b array of constant terms [N]
 *  @param resid array of residuals
 *  @param N size of arrays
 *  @param omega Successive Over-Relaxation (SOR) relaxation factor
 *  @param x_shared Shared memory allocated dynamically
 *  @return void
 */
__kernel void kernel_GS(__global real* x, __global const real* A, __global const real* b,
	__global real* resid, const int N, const real omega, __local real *x_shared)
{
	size_t ix = get_global_id(0); /**< global work item ID */
	if (ix < N)
	{
		/* Copy \a x to shared memory \a x_shared */
		size_t ix_temp = get_local_id(0); /**< local work item ID */
		while (ix_temp < N)
		{
			x_shared[ix_temp] = x[ix_temp];
			ix_temp += get_local_size(0); /**< local work group size */
		}
		/* Synchronization */
		barrier(CLK_LOCAL_MEM_FENCE);

		real temp = 0; /**< dummy variable */
		for (size_t jy = 0; jy < N; jy++)
		{
			if (jy != ix)
			{
				temp += A(ix, jy) * x_shared[jy];
			}
		}
		temp = (1.0 - omega)*x_shared[ix] + omega / A(ix, ix)*(b[ix] - temp);

		/* Residual calculation */
		resid[ix] = fabs(temp - x_shared[ix]);
		/* Updating \x */
		x[ix] = temp;
	}
}


__kernel void block_sum(__global const real *in_array, 
	__global real *out_results, const int N, 
	__local real *in_array_shared)
{
	size_t ix = get_global_id(0); /**< global work item ID */
	/* Copy \a in_array to shared memory \a in_array_shared */
	real x = 0;
	if (ix < N)
	{
		x = in_array[ix];
	}
	in_array_shared[get_local_id(0)] = x;
	/* Synchronization */
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Reduction step */
	for (int shift = get_local_size(0) / 2; shift > 0; shift >>= 1)
	{
		if (get_local_id(0) < shift)
		{
			in_array_shared[get_local_id(0)] += in_array_shared[get_local_id(0) + shift];
		}
		/* Synchronization */
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	/* Worker 0 writes final result out */
	if (get_local_id(0) == 0)
	{
		out_results[get_group_id(0)] = in_array_shared[0];
	}
}