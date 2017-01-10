
#include "safe_call.h"

#include "GS.h"

#include "linear_system.h"

int main(void)
{	
	int N = 2 * 64; /**< size of arrays */
	real tol = (real)1e-4; /**< tolerance */
	real omega = 1.0; /**< Successive Over-Relaxation (SOR) relaxation factor */

	size_t global_item_size = N; /**< global_work_size for kernel_GS */
	size_t local_item_size = 64; /**< local_work_size for kernel_GS */

	size_t block_size = 64; /**< global_work_size for block_sum */
	size_t num_blocks = N; /**< local_work_size for block_sum*/

	Linear_System linear_system_1(N);

	real * resid = (real *)malloc(sizeof(real) *N); /**< array of residuals */
	real * partial_sums = (real *)malloc(sizeof(real) *num_blocks); /**< array of residual summations */

	/* Initialize a simple linear A.x=b */
	reset_arrays(linear_system_1.A, linear_system_1.x, linear_system_1.b, N);

	/* Platform & device info */
	cl_platform_id platform_id = NULL; /**< list of OpenCL platforms */
	cl_device_id device_id = NULL; /**< list of OpenCL devices */
	cl_uint num_devices; /**< number of OpenCL devices */
	cl_uint num_platforms; /**< number of OpenCL platforms */
	safe_call(clGetPlatformIDs(1, &platform_id, &num_platforms));
	safe_call(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices));

	/* Creating OpenCL context
	 * @param context OpenCL context
	 */
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);

	/* Creating command queue
	 * @param command_queue command queue
	 */
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);

	/* Creating vector memory buffers
	 * @param a_mem_obj memory buffer for \a A (read only)
	 * @param x_mem_obj memory buffer for \a x (read and write)
	 * @param b_mem_obj memory buffer for \a b (read only)
	 * @param resid_mem_obj memory buffer for \a resid (read and write)
	 * @param partial_sums_mem_obj memory buffer for \a partial_sums (read and write)
	 */
	cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(real) *N*N, NULL, NULL);
	cl_mem x_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real) *N, NULL, NULL);
	cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(real) *N, NULL, NULL);
	cl_mem resid_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real) *N, NULL, NULL);
	cl_mem partial_sums_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real) *num_blocks, NULL, NULL);

	/*
	 * @param region offset and size (bytes) in buffer
	 */
	cl_buffer_region region;
	region.origin = 0;
	region.size = num_blocks * sizeof(real);

	/* Creating a sub-buffer from existing buffers (offset in array)
	 * @param partial_sums_mem_obj_num_blocks offset memory buffer for \a partial_sums_mem_obj by \a num_blocks bytes (read and write)
	 */
	cl_mem partial_sums_mem_obj_num_blocks = clCreateSubBuffer(partial_sums_mem_obj, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, NULL);

	/* Copy arrays to device memory buffers */
	safe_call(clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, sizeof(real) *N*N, linear_system_1.A, 0, NULL, NULL));
	safe_call(clEnqueueWriteBuffer(command_queue, x_mem_obj, CL_TRUE, 0, sizeof(real) *N, linear_system_1.x, 0, NULL, NULL));
	safe_call(clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, sizeof(real) *N, linear_system_1.b, 0, NULL, NULL));

	real *x_out = (real*)malloc(sizeof(real) * N); /**< array of solved unknowns for \a x */

	GS(context, device_id, command_queue, 
		x_mem_obj, a_mem_obj, b_mem_obj, resid_mem_obj, 
		partial_sums_mem_obj, partial_sums_mem_obj_num_blocks, 
		x_out, linear_system_1.A, linear_system_1.x, linear_system_1.b, resid, partial_sums,
		N, omega, tol,
		block_size, num_blocks, local_item_size, global_item_size);

	/* OpenCL cleaning up */
	safe_call(clFlush(command_queue));
	safe_call(clFinish(command_queue));
	safe_call(clReleaseMemObject(a_mem_obj));
	safe_call(clReleaseMemObject(x_mem_obj));
	safe_call(clReleaseMemObject(b_mem_obj));
	safe_call(clReleaseMemObject(resid_mem_obj));
	safe_call(clReleaseMemObject(partial_sums_mem_obj));
	safe_call(clReleaseCommandQueue(command_queue));
	safe_call(clReleaseContext(context));

	/* Free memory */
	free(x_out);
	free(resid);
	free(partial_sums);

	return 0;
}