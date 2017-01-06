
#include "GS.h"


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

void safe_call_f(cl_int code, size_t line)
{
	if (code)
	{
		printf("Error(%d): %d\n", line, code);
		exit(1);
	}
}


int GS(void)
{
	int N = 5; //5*64
	real tol = 1e-4;
	real omega = 1.0;

	size_t global_item_size = N;
	size_t local_item_size = 5; //64

	size_t block_size = 5; //64
	size_t num_blocks = N;

	real * A = (real *)malloc(sizeof(real) *N*N);
	real * x = (real *)malloc(sizeof(real) *N);
	real * b = (real *)malloc(sizeof(real) *N);
	real * resid = (real *)malloc(sizeof(real) *N);
	real * partial_sums = (real *)malloc(sizeof(real) *num_blocks);
	
	reset_arrays(A, x, b, N);

	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("GS.cl", "r");
	if (!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;
	safe_call(clGetPlatformIDs(1, &platform_id, &ret_num_platforms));
	safe_call(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices));

	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create memory buffers on the device for each vector 
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,			 sizeof(real) *N*N,			NULL, &ret);
	cl_mem x_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,			 sizeof(real) *N,			NULL, &ret);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,			 sizeof(real) *N,			NULL, &ret);
	cl_mem resid_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,		 sizeof(real) *N,			NULL, &ret);
	cl_mem partial_sums_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real) *num_blocks,	NULL, &ret);
	

	cl_buffer_region region;
	region.origin = 0;
	region.size = num_blocks * sizeof(real);
	cl_mem partial_sums_mem_obj_num_blocks = clCreateSubBuffer(partial_sums_mem_obj, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret);

	// Copy the lists A and B to their respective memory buffers
	safe_call(clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, sizeof(real) *N*N, A, 0, NULL, NULL));
	safe_call(clEnqueueWriteBuffer(command_queue, x_mem_obj, CL_TRUE, 0, sizeof(real) *N,	x, 0, NULL, NULL));
	safe_call(clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, sizeof(real) *N,	b, 0, NULL, NULL));

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	safe_call(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

	// Create the OpenCL kernel
	cl_kernel kernel_GS =		   clCreateKernel(program, "kernel_GS", &ret);
	cl_kernel kernel_block_sum_1 = clCreateKernel(program, "block_sum", &ret);
	cl_kernel kernel_block_sum_2 = clCreateKernel(program, "block_sum", &ret);

	// Set the arguments of the kernel
	safe_call(clSetKernelArg(kernel_GS, 0, sizeof(cl_mem),	(void *)&x_mem_obj));
	safe_call(clSetKernelArg(kernel_GS, 1, sizeof(cl_mem),	(void *)&a_mem_obj));
	safe_call(clSetKernelArg(kernel_GS, 2, sizeof(cl_mem),	(void *)&b_mem_obj));
	safe_call(clSetKernelArg(kernel_GS, 3, sizeof(cl_mem),	(void *)&resid_mem_obj));
	safe_call(clSetKernelArg(kernel_GS, 4, sizeof(int),		(void *)&N));
	safe_call(clSetKernelArg(kernel_GS, 5, sizeof(real), (void *)&omega));
	safe_call(clSetKernelArg(kernel_GS, 6, N * sizeof(real), NULL));
	

	safe_call(clSetKernelArg(kernel_block_sum_1, 0, sizeof(cl_mem), (void *)&resid_mem_obj));
	safe_call(clSetKernelArg(kernel_block_sum_1, 1, sizeof(cl_mem), (void *)&partial_sums_mem_obj));
	safe_call(clSetKernelArg(kernel_block_sum_1, 2, sizeof(int), (void *)&N));
	safe_call(clSetKernelArg(kernel_block_sum_1, 3, block_size * sizeof(real), NULL));


	safe_call(clSetKernelArg(kernel_block_sum_2, 0, sizeof(cl_mem), (void *)&resid_mem_obj));
	safe_call(clSetKernelArg(kernel_block_sum_2, 1, sizeof(cl_mem), (void *)&partial_sums_mem_obj_num_blocks));
	safe_call(clSetKernelArg(kernel_block_sum_2, 2, sizeof(int), (void *)&num_blocks));
	safe_call(clSetKernelArg(kernel_block_sum_2, 3, num_blocks * sizeof(real), NULL));



	real resid_host = 10;
	while (resid_host > tol)
	{
		safe_call(clEnqueueNDRangeKernel(command_queue, kernel_GS,			1, NULL, &global_item_size, &local_item_size,	0, NULL, NULL));

		safe_call(clEnqueueNDRangeKernel(command_queue, kernel_block_sum_1, 1, NULL, &global_item_size, &block_size,		0, NULL, NULL));

		safe_call(clEnqueueNDRangeKernel(command_queue, kernel_block_sum_2, 1, NULL, &num_blocks,		&local_item_size,	0, NULL, NULL));

		safe_call(clEnqueueReadBuffer(command_queue, partial_sums_mem_obj, CL_TRUE, 0, sizeof(real), &resid_host, 0, NULL, NULL));

	}

	
	// Force the command queue to get processed, wait until all commands are complete
	clFinish(command_queue);

	// Read the memory buffer x on the device to the local variable x_out
	real *x_out = (real*)malloc(sizeof(real) * N);
	safe_call(clEnqueueReadBuffer(command_queue, x_mem_obj, CL_TRUE, 0, N * sizeof(real), x_out, 0, NULL, NULL));
	
	// Display the result to the screen
	for (size_t i = 0; i < N; i++)
		printf("x_out[%d] = %f\n", i, x_out[i]);

	printf("###########################\n###########################\n###########################\n");

	reset_arrays(A, x, b, N);
	GS_CPU(x, N, A, b, tol);
	for (size_t i = 0; i < N; i++)
		printf("CPU: x_out[%d] = %f\n", i, x[i]);


	// Clean up
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
	free(A);
	free(x);
	free(x_out);
	free(b);
	free(resid);
	free(partial_sums);

	return 0;
}