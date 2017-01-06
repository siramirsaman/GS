
#include "GS.h"


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


/** @brief Initialize or reset an example for a simple linear A.x=b
 *  @param x array of unknowns [N]
 *  @param b array of constant terms [N]
 *  @param A array of coefficients [N*N]
 *  @param N size of arrays
 *  @return void
 */
void reset_arrays(real* A, real* x, real* b, int N)
{
	b[0] = 1100;
	for (int i = 1; i < N; i++)
		b[i] = 100;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			A(i, j) = 0.0;

	for (int i = 0; i < N; i++)
		A(i, i) = 15.0;

	for (int i = 0; i < N; i++)
	{
		if (i + 1 < N)
			A(i, (i + 1)) = -5.0;
		if (i - 1 >= 0)
			A(i, (i - 1)) = -5.0;
	}

	A(0, 0) = 20;
	A((N - 1), (N - 1)) = 10;

	for (int i = 0; i < N; i++)
		x[i] = 0;
}


#define safe_call(a) safe_call_f(a, __LINE__)

/** @brief Error checking of OpenCL calls
 *  @param code OpenCL routine error code
 *  @param line code line number to be printed on error
 *  @return void
 */
void safe_call_f(cl_int code, size_t line)
{
	if (code)
	{
		printf("Call returned error code(%d): %d\n", line, code);
		exit(EXIT_FAILURE);
	}
}


/** @brief main wrapper
 */
int GS(void)
{
	int N = 2 * 64; /**< size of arrays */
	real tol = 1e-4; /**< tolerance */
	real omega = 1.0; /**< Successive Over-Relaxation (SOR) relaxation factor */

	size_t global_item_size = N; /**< global_work_size for kernel_GS */
	size_t local_item_size = 64; /**< local_work_size for kernel_GS */

	size_t block_size = 64; /**< global_work_size for block_sum */
	size_t num_blocks = N; /**< local_work_size for block_sum*/

	real * A = (real *)malloc(sizeof(real) *N*N); /**< array of coefficients [N*N] */
	real * x = (real *)malloc(sizeof(real) *N); /**< array of unknowns [N] */
	real * b = (real *)malloc(sizeof(real) *N); /**< array of constant terms [N] */
	real * resid = (real *)malloc(sizeof(real) *N); /**< array of residuals */
	real * partial_sums = (real *)malloc(sizeof(real) *num_blocks); /**< array of residual summations */
	
	/* Initialize a simple linear A.x=b */
	reset_arrays(A, x, b, N);

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

	/* Platform & device info */
	cl_platform_id platform_id = NULL; /**< list of OpenCL platforms */
	cl_device_id device_id = NULL; /**< list of OpenCL devices */
	cl_uint num_devices; /**< number of OpenCL devices */
	cl_uint num_platforms; /**< number of OpenCL platforms */
	cl_int errcode_ret; /**< error code, not checked here */
	safe_call(clGetPlatformIDs(1, &platform_id, &num_platforms));
	safe_call(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices));

	/* Creating OpenCL context 
	 * @param context OpenCL context
	 */
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &errcode_ret);

	/* Creating command queue
	* @param command_queue command queue
	*/
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &errcode_ret);

	/* Creating vector memory buffers
	* @param a_mem_obj memory buffer for \a A (read only)
	* @param x_mem_obj memory buffer for \a x (read and write)
	* @param b_mem_obj memory buffer for \a b (read only)
	* @param resid_mem_obj memory buffer for \a resid (read and write)
	* @param partial_sums_mem_obj memory buffer for \a partial_sums (read and write)
	*/
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,			 sizeof(real) *N*N,			NULL, &errcode_ret);
	cl_mem x_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,			 sizeof(real) *N,			NULL, &errcode_ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,			 sizeof(real) *N,			NULL, &errcode_ret);
	cl_mem resid_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,		 sizeof(real) *N,			NULL, &errcode_ret);
	cl_mem partial_sums_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real) *num_blocks,	NULL, &errcode_ret);
	
	/*
	 * @param region offset and size (bytes) in buffer
	 */
	cl_buffer_region region;
	region.origin = 0;
	region.size = num_blocks * sizeof(real);

	/* Creating a sub-buffer from existing buffers (offset in array)
	* @param partial_sums_mem_obj_num_blocks offset memory buffer for \a partial_sums_mem_obj by \a num_blocks bytes (read and write)
	*/
	cl_mem partial_sums_mem_obj_num_blocks = clCreateSubBuffer(partial_sums_mem_obj, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &errcode_ret);

	/* Copy arrays to device memory buffers */
	safe_call(clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, sizeof(real) *N*N, A, 0, NULL, NULL));
	safe_call(clEnqueueWriteBuffer(command_queue, x_mem_obj, CL_TRUE, 0, sizeof(real) *N,	x, 0, NULL, NULL));
	safe_call(clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, sizeof(real) *N,	b, 0, NULL, NULL));

	/* Creating program object for context 
	 * @param program program object (compilation target)
	 */
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode_ret);

	/* Building the target program */
	safe_call(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

	/* Creating OpenCL kernel
	* @param kernel_GS kernel object for \fn kernel_GS
	* @param kernel_block_sum_1 first kernel object for \fn block_sum
	* @param kernel_block_sum_2 second kernel object for \fn block_sum
	*/
	cl_kernel kernel_GS =		   clCreateKernel(program, "kernel_GS", &errcode_ret);
	cl_kernel kernel_block_sum_1 = clCreateKernel(program, "block_sum", &errcode_ret);
	cl_kernel kernel_block_sum_2 = clCreateKernel(program, "block_sum", &errcode_ret);

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

	real *x_out = (real*)malloc(sizeof(real) * N); /**< array of solved unknowns for \a x */
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
	safe_call(clFlush(command_queue));
	safe_call(clFinish(command_queue));
	safe_call(clReleaseKernel(kernel_GS));
	safe_call(clReleaseKernel(kernel_block_sum_1));
	safe_call(clReleaseKernel(kernel_block_sum_2));
	safe_call(clReleaseProgram(program));
	safe_call(clReleaseMemObject(a_mem_obj));
	safe_call(clReleaseMemObject(x_mem_obj));
	safe_call(clReleaseMemObject(b_mem_obj));
	safe_call(clReleaseMemObject(resid_mem_obj));
	safe_call(clReleaseMemObject(partial_sums_mem_obj));
	safe_call(clReleaseCommandQueue(command_queue));
	safe_call(clReleaseContext(context));

	/* Free memory */
	free(A);
	free(x);
	free(x_out);
	free(b);
	free(resid);
	free(partial_sums);

	return 0;
}