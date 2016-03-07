
//#include <libc.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
//#include <mach/mach_time.h>
#include <math.h>
#include "scan.h"


    static char *
LoadProgramSourceFromFile(const char *filename)
{
    struct stat statbuf;
    FILE        *fh;
    char        *source;

    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;

    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

    return source;
}
cl_device_id            ComputeDeviceId;
cl_command_queue        ComputeCommands;
cl_context              ComputeContext;
cl_program              ComputeProgram;
cl_kernel*              ComputeKernels;
cl_mem*                 ScanPartialSums = 0;
unsigned int            ElementsAllocated = 0;
unsigned int            LevelsAllocated = 0;

int		GROUP_SIZE      = 256;
#define min(A,B) ((A) < (B) ? (A) : (B))

int Scan(cl_mem *input_buffer, cl_mem *output_buffer, cl_uint count)
{
    int i;
    int err = 0;
    const size_t local_wsize  = min(GROUP_SIZE, count);
    const size_t global_wsize = count; // i.e. 64 work groups
    const size_t num_work_groups = global_wsize / local_wsize;
    printf("scan: %d %d %d %d\n", count, local_wsize, global_wsize, num_work_groups);
    cl_platform_id platforms;
    err = clGetPlatformIDs(1, &platforms, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to locate a compute platform!\n");
        return EXIT_FAILURE;
    }

    // Connect to a GPU compute device
    //
    err = clGetDeviceIDs(platforms, CL_DEVICE_TYPE_GPU, 1, &ComputeDeviceId, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to locate a compute device!\n");
        return EXIT_FAILURE;
    }

    const char* filename = "scan.cl";

    char *source = LoadProgramSourceFromFile(filename);
    if(!source)
    {
        printf("Error: Failed to load compute program from file!\n");
        return EXIT_FAILURE;    
    }

    // Create a compute ComputeContext 
    //
    ComputeContext = clCreateContext(0, 1, &ComputeDeviceId, NULL, NULL, &err);
    if (!ComputeContext)
    {
        printf("Error: Failed to create a compute ComputeContext!\n");
        return EXIT_FAILURE;
    }

    // Create a command queue
    //
    ComputeCommands = clCreateCommandQueue(ComputeContext, ComputeDeviceId, 0, &err);
    if (!ComputeCommands)
    {
        printf("Error: Failed to create a command ComputeCommands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    //
    ComputeProgram = clCreateProgramWithSource(ComputeContext, 1, (const char **) & source, NULL, &err);
    if (!ComputeProgram || err != CL_SUCCESS)
    {
        printf("%s\n", source);
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    //
    err = clBuildProgram(ComputeProgram, 1, &ComputeDeviceId, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t length;
        char *build_log = NULL;
        size_t build_log_size = 0;
        printf("source: %s\n", source);
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(ComputeProgram, ComputeDeviceId, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, &build_log_size);
        build_log = (char *)malloc(build_log_size);
        clGetProgramBuildInfo(ComputeProgram, ComputeDeviceId, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
        printf("log: %s\n", build_log);
        fflush(stdout);
        free(build_log);
        return EXIT_FAILURE;
    }

    free(source);

    cl_kernel reduce = clCreateKernel(ComputeProgram, "reduce", &err);
    if (!reduce || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return EXIT_FAILURE;
    }

    cl_kernel top_scan = clCreateKernel(ComputeProgram, "top_scan", &err);
    if (!top_scan || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return EXIT_FAILURE;
    }

    cl_kernel bottom_scan = clCreateKernel(ComputeProgram, "bottom_scan", &err);
    if (!bottom_scan || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        return EXIT_FAILURE;
    }

    cl_mem d_isums = clCreateBuffer(ComputeContext, CL_MEM_READ_WRITE, 
            num_work_groups * sizeof(float), NULL, &err);
    // Set the kernel arguments for the reduction kernel
    err = clSetKernelArg(reduce, 0, sizeof(cl_mem), (void*)input_buffer);
    err |= clSetKernelArg(reduce, 1, sizeof(cl_mem), (void*)&d_isums);
    err |= clSetKernelArg(reduce, 2, sizeof(cl_uint), (void*)&count);
    err |= clSetKernelArg(reduce, 3, local_wsize * sizeof(float), NULL);
    if(err)
    {
        printf("Error: Failed clSetKernelArg for Reduce");
        return EXIT_FAILURE;
    }

    // Set the kernel arguments for the top-level scan
    err = clSetKernelArg(top_scan, 0, sizeof(cl_mem), (void*)&d_isums);
    err |= clSetKernelArg(top_scan, 1, sizeof(cl_int), (void*)&num_work_groups);
    err |= clSetKernelArg(top_scan, 2, local_wsize * 2 * sizeof(float), NULL);
    if(err)
    {
        printf("Error: Failed clSetKernelArg for top_scan");
        return EXIT_FAILURE;
    }
    // Set the kernel arguments for the bottom-level scan
    err = clSetKernelArg(bottom_scan, 0, sizeof(cl_mem), (void*)input_buffer);
    err |= clSetKernelArg(bottom_scan, 1, sizeof(cl_mem), (void*)&d_isums);
    err |= clSetKernelArg(bottom_scan, 2, sizeof(cl_mem), (void*)output_buffer);
    err |= clSetKernelArg(bottom_scan, 3, sizeof(cl_uint), (void*)&count);
    err |= clSetKernelArg(bottom_scan, 4, local_wsize * 2 * sizeof(float), NULL);
    if(err)
    {
        printf("Error: Failed clSetKernelArg for bottom_scan");
        return EXIT_FAILURE;
    }


    // Create the input buffer on the device
    //
    size_t buffer_size = sizeof(float) * count;
    err = clEnqueueNDRangeKernel(ComputeCommands, reduce, 1, NULL,
            &global_wsize, &local_wsize, 0, NULL, NULL);
    if(err)
    {
        printf("Error: Failed clEnqueueNDRangeKernel for reduce");
        return EXIT_FAILURE;
    }

    // Next, a top-level exclusive scan is performed on the array
    // of block sums
    err = clEnqueueNDRangeKernel(ComputeCommands, top_scan, 1, NULL,
            &local_wsize, &local_wsize, 0, NULL, NULL);
    if(err)
    {
        printf("Error: Failed clEnqueueNDRangeKernel for top_scan");
        return EXIT_FAILURE;
    }

    // Finally, a bottom-level scan is performed by each block
    // that is seeded with the scanned value in block sums
    err = clEnqueueNDRangeKernel(ComputeCommands, bottom_scan, 1, NULL,
            &global_wsize, &local_wsize, 0, NULL, NULL);
    if(err)
    {
        printf("Error: Failed clEnqueueNDRangeKernel for bottom_scan");
        return EXIT_FAILURE;
    }

    clReleaseKernel(reduce);
    clReleaseKernel(top_scan);
    clReleaseKernel(bottom_scan);
    clReleaseProgram(ComputeProgram);
    clReleaseCommandQueue(ComputeCommands);
    clReleaseContext(ComputeContext);

    free(ComputeKernels);
    //free(float_data);
    //free(reference);
    //free(result);


    return 0;
}

