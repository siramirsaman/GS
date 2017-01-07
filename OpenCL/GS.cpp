
#include "GS.h"

#include "safe_call.h"


/** @brief CPU Implementation of Gauss Seidel for solving a square 
 *         system of linear equations A.x=b 
 *  @param x array of unknowns [N]
 *  @param b array of constant terms [N]
 *  @param A array of coefficients [N*N]
 *  @param N size of arrays
 *  @param tol tolerance
 *  @return void
 */
void GS_CPU(real x[], const size_t N, const real A[], const real b[], const real tol)
{
	int counter = 1;
	real temp, resid = 10;

	while (resid > tol)
	{
		resid = 0;
		for (int i = 0; i < N; i++)
		{
			temp = 0;
			for (int j = 0; j < N; j++)
			{
				if (j != i){
					temp += A(i, j)*x[j];
				}
			}
			temp = 1 / A(i, i)*(b[i] - temp);
			resid += abs((real)(temp - x[i]));
			x[i] = temp;
		}
		counter++;
	}
}


/** @brief main wrapper
 */
int GS(cl_context context, cl_device_id device_id, cl_command_queue command_queue,
	cl_mem x_mem_obj, cl_mem a_mem_obj, cl_mem b_mem_obj, cl_mem resid_mem_obj,
	cl_mem partial_sums_mem_obj, cl_mem partial_sums_mem_obj_num_blocks,
	real *x_out, real *A, real *x, real *b, real *resid, real *partial_sums,
	int N, real omega, real tol,
	size_t block_size, size_t num_blocks, size_t local_item_size, size_t global_item_size)
{
	/* Loading kernel code */
	FILE *fp; /**< kernel file pointer */
	char *source_str; /**< kernel string to be loaded from file */
	size_t source_size; /**< source size */
	/* Reading kernel source file */
	fp = fopen("GS.cl", "r");
	if (!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		exit(EXIT_FAILURE);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	/* Creating program object for context 
	 * @param program program object (compilation target)
	 */
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, NULL);

	/* Building the target program */
	safe_call(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

	/* Creating OpenCL kernel
	* @param kernel_GS kernel object for \fn kernel_GS
	* @param kernel_block_sum_1 first kernel object for \fn block_sum
	* @param kernel_block_sum_2 second kernel object for \fn block_sum
	*/
	cl_kernel kernel_GS =		   clCreateKernel(program, "kernel_GS", NULL);
	cl_kernel kernel_block_sum_1 = clCreateKernel(program, "block_sum", NULL);
	cl_kernel kernel_block_sum_2 = clCreateKernel(program, "block_sum", NULL);

	/*
	 * kernel \a kernel_GS arguments
	 */
	safe_call(clSetKernelArg(kernel_GS, 0, sizeof(cl_mem),	(void *)&x_mem_obj));
	safe_call(clSetKernelArg(kernel_GS, 1, sizeof(cl_mem),	(void *)&a_mem_obj));
	safe_call(clSetKernelArg(kernel_GS, 2, sizeof(cl_mem),	(void *)&b_mem_obj));
	safe_call(clSetKernelArg(kernel_GS, 3, sizeof(cl_mem),	(void *)&resid_mem_obj));
	safe_call(clSetKernelArg(kernel_GS, 4, sizeof(int),		(void *)&N));
	safe_call(clSetKernelArg(kernel_GS, 5, sizeof(real), (void *)&omega));
	safe_call(clSetKernelArg(kernel_GS, 6, N * sizeof(real), NULL)); /**< dynamic shared memory */
	
	/*
	* kernel \a kernel_block_sum_1 arguments
	*/
	safe_call(clSetKernelArg(kernel_block_sum_1, 0, sizeof(cl_mem), (void *)&resid_mem_obj));
	safe_call(clSetKernelArg(kernel_block_sum_1, 1, sizeof(cl_mem), (void *)&partial_sums_mem_obj));
	safe_call(clSetKernelArg(kernel_block_sum_1, 2, sizeof(int), (void *)&N));
	safe_call(clSetKernelArg(kernel_block_sum_1, 3, block_size * sizeof(real), NULL)); /**< dynamic shared memory */

	/*
	* kernel \a kernel_block_sum_2 arguments
	*/
	safe_call(clSetKernelArg(kernel_block_sum_2, 0, sizeof(cl_mem), (void *)&resid_mem_obj));
	safe_call(clSetKernelArg(kernel_block_sum_2, 1, sizeof(cl_mem), (void *)&partial_sums_mem_obj_num_blocks));
	safe_call(clSetKernelArg(kernel_block_sum_2, 2, sizeof(int), (void *)&num_blocks));
	safe_call(clSetKernelArg(kernel_block_sum_2, 3, num_blocks * sizeof(real), NULL)); /**< dynamic shared memory */


	real resid_host = 10; /**< residual value on host */
	while (resid_host > tol)
	{
		/* Lunching kernel \a kernel_GS	*/
		safe_call(clEnqueueNDRangeKernel(command_queue, kernel_GS,			1, NULL, &global_item_size, &local_item_size,	0, NULL, NULL));
		/* Lunching kernel \a kernel_block_sum_1	*/
		safe_call(clEnqueueNDRangeKernel(command_queue, kernel_block_sum_1, 1, NULL, &global_item_size, &block_size,		0, NULL, NULL));
		/* Lunching kernel \a kernel_block_sum_2	*/
		safe_call(clEnqueueNDRangeKernel(command_queue, kernel_block_sum_2, 1, NULL, &num_blocks,		&local_item_size,	0, NULL, NULL));
		/* Read \a partial_sums_mem_obj buffer object to host \a resid_host */
		safe_call(clEnqueueReadBuffer(command_queue, partial_sums_mem_obj, CL_FALSE, 0, sizeof(real), &resid_host, 0, NULL, NULL));
	}

	
	/* Block till \a command_queue finishes */
	clFinish(command_queue);

	/* Read \a x from device to host variable \a x_out */
	safe_call(clEnqueueReadBuffer(command_queue, x_mem_obj, CL_TRUE, 0, N * sizeof(real), x_out, 0, NULL, NULL));
	
	/* Printing GPU results on screen */
	for (size_t i = 0; i < N; i++)
		printf("x_out[%d] = %f\n", i, x_out[i]);
	printf("###########################\n###########################\n###########################\n");
	/* Reset to the simple linear A.x=b */
	reset_arrays(A, x, b, N);
	/* Run CPU Implementation of Gauss Seidel */
	GS_CPU(x, N, A, b, tol);
	/* Printing CPU results on screen */
	for (size_t i = 0; i < N; i++)
		printf("CPU: x_out[%d] = %f\n", i, x[i]);

	/* OpenCL cleaning up */
	safe_call(clReleaseKernel(kernel_GS));
	safe_call(clReleaseKernel(kernel_block_sum_1));
	safe_call(clReleaseKernel(kernel_block_sum_2));
	safe_call(clReleaseProgram(program));

	return 0;
}