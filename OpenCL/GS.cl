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

#define A(i,j) A[i+j*N]


__kernel void kernel_GS(__global real* x, __global const real* A, __global const real* b, __global real* resid, const int N, const real omega, __local real *x_shared)
{
	size_t ix = get_global_id(0);
	if (ix < N)
	{
		size_t ix_temp = get_local_id(0);
		while (ix_temp < N)
		{
			x_shared[ix_temp] = x[ix_temp];
			ix_temp += get_local_size(0);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		real temp = 0;
		for (size_t jy = 0; jy < N; jy++)
		{
			if (jy != ix)
			{
				temp += A(ix, jy) * x_shared[jy];
			}
		}

		temp = (1.0 - omega)*x_shared[ix] + omega / A(ix, ix)*(b[ix] - temp);

		resid[ix] = fabs(temp - x_shared[ix]);
		x[ix] = temp;
	}
}


__kernel void block_sum(__global const real *input, __global real *per_block_results, const int n, __local real *sdata)
{
	size_t i = get_global_id(0);

	real x = 0;
	if (i < n)
	{
		x = input[i];
	}

	sdata[get_local_id(0)] = x;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for (int offset = get_local_size(0) / 2; offset > 0; offset >>= 1)
	{
		if (get_local_id(0) < offset)
		{
			sdata[get_local_id(0)] += sdata[get_local_id(0) + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (get_local_id(0) == 0)
	{
		per_block_results[get_group_id(0)] = sdata[0];
	}
}