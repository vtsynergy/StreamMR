#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/opencl.h>
int Scan(cl_context ctx, cl_mem *input_buffer, cl_mem *output_buffer, cl_uint count);
