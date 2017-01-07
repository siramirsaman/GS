
#include "safe_call.h"

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
