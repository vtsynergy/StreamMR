/******************************************************************************************************
 * (c) Virginia Polytechnic Insitute and State University, 2011.
 * This is the source code for StreamMR, a MapReduce framework on graphics processing units.
 * Developer:  Marwa K. Elteir (City of Scientific Researches and Technology Applications, Egypt)
 *******************************************************************************************************/


/* ============================================================


   Copyright (c) 2009 Advanced Micro Devices, Inc.  All rights reserved.

   Redistribution and use of this material is permitted under the following
conditions:

Redistributions must retain the above copyright notice and all terms of this
license.

In no event shall anyone redistributing or accessing or using this material
commence or participate in any arbitration or legal action relating to this
material against Advanced Micro Devices, Inc. or any copyright holders or
contributors. The foregoing shall survive any expiration or termination of
this license or any agreement or access or use related to this material.

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERA TION, OR THAT IT IS FREE
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT.
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES,
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES,
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S.
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS,
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS,
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS.
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to
computer software and technical data, respectively. Use, duplication,
distribution or disclosure by the U.S. Government and/or DOD agencies is
subject to the full extent of restrictions in all applicable regulations,
including those found at FAR52.227 and DFARS252.227 et seq. and any successor
regulations thereof. Use of this material by the U.S. Government and/or DOD
agencies is acknowledgment of the proprietary rights of any copyright holders
and contributors, including those of Advanced Micro Devices, Inc., as well as
the provisions of FAR52.227-14 through 23 regarding privately developed and/or
commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and
supersedes all proposals and prior discussions and writings between the parties
with respect thereto. This license does not affect any ownership, rights, title,
     or interest in, or relating to, this material. No terms of this license can be
     modified or waived, and no breach of this license can be excused, unless done
     so in a writing signed by all affected parties. Each term of this license is
     separately enforceable. If any term of this license is determined to be or
     becomes unenforceable or illegal, such term shall be reformed to the minimum
     extent necessary in order for this license to remain in effect in accordance
     with its terms as modified by such reformation. This license shall be governed
     by and construed in accordance with the laws of the State of Texas without
     regard to rules on conflicts of law of any state or jurisdiction or the United
     Nations Convention on the International Sale of Goods. All disputes arising out
     of this license shall be subject to the jurisdiction of the federal and state
     courts in Austin, Tetas, and all defenses are hereby waived concerning personal
     jurisdiction and venue of these courts.
     ============================================================ */
#include "StreamMR.hpp"
#include <malloc.h>
#include <ctime>
#include <sys/time.h>
#include <errno.h>
#include "timeRec.h"
#include "rdtsc.h"
#include "scan.h"

     int quiet = 1;
#define CEIL(n,m) (n/m + (int)(n%m !=0))


int MapReduce::setupCL()
{
    cl_int status = CL_SUCCESS;
    cl_uint deviceListSize = 1;

    /* Now allocate memory for device list based on the size we got earlier */
    devices = (cl_device_id*)malloc(deviceListSize);
    if(devices == NULL)
    {
        error("Failed to allocate memory (devices).");
        return SDK_FAILURE;
    }

    devices[0] = GetDevice(0, 0);
    context = clCreateContext(0,
            1,
            devices,
            NULL,
            NULL,
            &status);

    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateContextFromType failed."))
    {
        return SDK_FAILURE;
    }
    /* Create command queue */

    commandQueue = clCreateCommandQueue(context,
            devices[0],
            CL_QUEUE_PROFILING_ENABLE,
            &status);

    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateCommandQueue failed."))
    {
        return SDK_FAILURE;
    }

    /* Get Device specific Information */
    status = clGetDeviceInfo(devices[0],
            CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(size_t),
            (void*)&maxWorkGroupSize,
            NULL);

    if(!checkVal(status,
                CL_SUCCESS,
                "clGetDeviceInfo"
                "CL_DEVICE_MAX_WORK_GROUP_SIZE failed."))
    {
        return SDK_FAILURE;
    }


    status = clGetDeviceInfo(devices[0],
            CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
            sizeof(cl_uint),
            (void*)&maxDimensions,
            NULL);

    if(!checkVal(status,
                CL_SUCCESS,
                "clGetDeviceInfo"
                "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS failed."))
    {
        return SDK_FAILURE;
    }


    maxWorkItemSizes = (size_t*)malloc(maxDimensions * sizeof(size_t));

    status = clGetDeviceInfo(devices[0],
            CL_DEVICE_MAX_WORK_ITEM_SIZES,
            sizeof(size_t) * maxDimensions,
            (void*)maxWorkItemSizes,
            NULL);

    if(!checkVal(status,
                CL_SUCCESS,
                "clGetDeviceInfo"
                "CL_DEVICE_MAX_WORK_ITEM_SIZES failed."))
    {
        return SDK_FAILURE;
    }


    status = clGetDeviceInfo(devices[0],
            CL_DEVICE_LOCAL_MEM_SIZE,
            sizeof(cl_ulong),
            (void*)&totalLocalMemory,
            NULL);

    if(!checkVal(status,
                CL_SUCCESS,
                "clGetDeviceInfo"
                "CL_DEVICE_LOCAL_MEM_SIZE failed."))
    {
        return SDK_FAILURE;
    }


    /* create a CL program using the kernel source */
    FILE *kernelFile;
    char *kernelSource;
    size_t kernelLength;


    std::string kernelPath = getPath();

    if (jobSpec->workflow == MAP_ONLY)
    {
        kernelPath.append(kernelfilename.c_str());	
    }
    else  //MAP_REDUCE
    {
        kernelPath.append(kernelfilename.c_str());	
    }

    kernelFile = fopen(kernelPath.c_str(), "r");
    fseek(kernelFile, 0, SEEK_END);
    if (!kernelFile)
    {
        std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << " Failed to open kernel file " << kernelPath << std::endl;
        std::cerr << "\tError reason: " << strerror(errno) << std::endl;
        exit(-1);
    }
    kernelLength = (size_t) ftell(kernelFile);
    kernelSource = (char *) malloc(sizeof(char)*kernelLength);
    rewind(kernelFile);
    if (!fread((void *) kernelSource, kernelLength, 1, kernelFile))
    {
        std::cerr << "ERROR: " << __FILE__ << ":" << __LINE__ << " Failed to read from kernel file: " << kernelPath << std::endl;
        std::cerr << "\tError reason: " << strerror(errno) << std::endl;
        exit(-1);
    }
    fclose(kernelFile);


    //     if(!kernelFile.open(kernelPath.c_str()))
    //     {
    //         std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
    //         return SDK_FAILURE;
    //     }
    //     printf("Successfully Load the Kernel File \n");
    //
    program = clCreateProgramWithSource(context,
            1,
            ((const char **)(&kernelSource)),
            (const size_t*) &kernelLength,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateProgramWithSource failed."))
    {
        return SDK_FAILURE;
    }

    /* create a cl program executable for all the devices specified */
    char *options = new char[100];
    strcpy( options, " ");
    if (jobSpec->perfectHashing)
    {
        strcat( options, "-D PERFECT ");
    }
    if (jobSpec->outputIntermediate)
    {
        strcat( options, "-D OUTPUTINTER ");
    }
    if (jobSpec->overflow == true)
    {
        strcat( options, "-D OVERFLOW ");
    }


    status = clBuildProgram(program,
            1,
            devices,
            options,
            NULL,
            NULL);
    if(status != CL_SUCCESS)
    {
        if(status == CL_BUILD_PROGRAM_FAILURE)
        {
            cl_int logStatus;
            char * buildLog = NULL;
            size_t buildLogSize = 0;
            logStatus = clGetProgramBuildInfo(program,
                    devices[0],
                    CL_PROGRAM_BUILD_LOG,
                    buildLogSize,
                    buildLog,
                    &buildLogSize);
            if(!checkVal(logStatus,
                        CL_SUCCESS,
                        "clGetProgramBuildInfo failed."))
            {
                return SDK_FAILURE;
            }

            buildLog = (char*)malloc(buildLogSize);
            if(buildLog == NULL)
            {
                error("Failed to allocate host memory.(buildLog)");
                return SDK_FAILURE;
            }
            memset(buildLog, 0, buildLogSize);

            logStatus = clGetProgramBuildInfo(program,
                    devices[0],
                    CL_PROGRAM_BUILD_LOG,
                    buildLogSize,
                    buildLog,
                    NULL);
            if(!checkVal(logStatus,
                        CL_SUCCESS,
                        "clGetProgramBuildInfo failed."))
            {
                free(buildLog);
                return SDK_FAILURE;
            }

            std::cout << " \n\t\t\tBUILD LOG\n";
            std::cout << " ************************************************\n";
            std::cout << buildLog << std::endl;
            std::cout << " ************************************************\n";
            free(buildLog);
        }

        if(!checkVal(status,
                    CL_SUCCESS,
                    "clBuildProgram failed."))
        {
            return SDK_FAILURE;
        }
    }

    printf("Successully build the kernel \n");


    /* get a kernel object handle for a kernel with the given name */

    mapperExtendedKernel = clCreateKernel(program,
            "mapperExtended",
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateKernel failed1."))
    {
        return SDK_FAILURE;
    }


    mapperKernel = clCreateKernel(program,
            "mapper",
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateKernel failed2."))
    {
        return SDK_FAILURE;
    }


    reducerKernel = clCreateKernel(program,
            "reducer",
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateKernel failed3."))
    {
        return SDK_FAILURE;
    }

    reducerInOverflowKernel = clCreateKernel(program,
            "reducerInOverflow",
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateKernel failed4."))
    {
        return SDK_FAILURE;
    }

    copyerKernel = clCreateKernel(program,
            "copyerHashToArray",
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateKernel failed7."))
    {
        return SDK_FAILURE;
    }

    copyerInOverflowKernel = clCreateKernel(program,
            "copyerHashToArrayInOverflow",
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateKernel failed8."))
    {
        return SDK_FAILURE;
    }

    /* Check whether specified groupSize is plausible on current kernel */

    status = clGetKernelWorkGroupInfo(mapperExtendedKernel,
            devices[0],
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &mapperExtendedWorkGroupSize,
            0);
    if(!checkVal(status,
                CL_SUCCESS,
                "clGetKernelWorkGroupInfo failed."))
    {
        return SDK_FAILURE;
    }

    status = clGetKernelWorkGroupInfo(mapperKernel,
            devices[0],
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &mapperWorkGroupSize,
            0);
    if(!checkVal(status,
                CL_SUCCESS,
                "clGetKernelWorkGroupInfo failed."))
    {
        return SDK_FAILURE;
    }


    status = clGetKernelWorkGroupInfo(reducerKernel,
            devices[0],
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &reducerWorkGroupSize,
            0);
    if(!checkVal(status,
                CL_SUCCESS,
                "clGetKernelWorkGroupInfo failed."))
    {
        return SDK_FAILURE;
    }


    status = clGetKernelWorkGroupInfo(reducerInOverflowKernel,
            devices[0],
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &reducerInOverflowWorkGroupSize,
            0);
    if(!checkVal(status,
                CL_SUCCESS,
                "clGetKernelWorkGroupInfo failed."))
    {
        return SDK_FAILURE;
    }
    status = clGetKernelWorkGroupInfo(copyerKernel,
            devices[0],
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &copyerWorkGroupSize,
            0);
    if(!checkVal(status,
                CL_SUCCESS,
                "clGetKernelWorkGroupInfo failed."))
    {
        return SDK_FAILURE;
    }

    status = clGetKernelWorkGroupInfo(copyerInOverflowKernel,
            devices[0],
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &copyerInOverflowWorkGroupSize,
            0);
    if(!checkVal(status,
                CL_SUCCESS,
                "clGetKernelWorkGroupInfo failed."))
    {
        return SDK_FAILURE;
    }

    /* If groupSize exceeds the maximum supported on kernel
     * fall back */
    printf("Successfully query the workgroup size of each kernel\n");
    size_t temp;
    if (kernelWorkGroupSize > mapperWorkGroupSize) 
        temp = mapperWorkGroupSize;
    else
        temp = kernelWorkGroupSize;

    if (temp > reducerWorkGroupSize) temp = reducerWorkGroupSize;

    if (temp >  mapperExtendedWorkGroupSize) temp = mapperExtendedWorkGroupSize;

    if (temp >  reducerInOverflowWorkGroupSize) temp = reducerInOverflowWorkGroupSize; 

    if (temp >  copyerWorkGroupSize) temp = copyerWorkGroupSize;

    if (temp >  copyerInOverflowWorkGroupSize) temp = copyerInOverflowWorkGroupSize;

    if(groupSize > temp)
    {
        if(!quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : " << groupSize << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                << temp << std::endl;
            std::cout << "Falling back to " << temp << std::endl;
        }
        groupSize = temp;
    }

    return SDK_SUCCESS;
}

int MapReduce::runCLKernels()
{
    return SDK_SUCCESS;
}


int MapReduce::initialize(JobSpecification* jobSpecification)
{

    //initialize the job Specification
    jobSpec = jobSpecification;

    // Call base class Initialize to get default configuration
    if(initialize())
        return SDK_FAILURE;

    return SDK_SUCCESS;
}

int MapReduce::setup()
{
    //int timer = createTimer();
    //resetTimer(timer);
    //startTimer(timer);

    if(setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    //stopTimer(timer);
    /* Compute setup time */
    //setupTime = (double)(readTimer(timer));

    return SDK_SUCCESS;
}


int MapReduce::cleanup()
{
    /* Releases OpenCL resources (Context, Memory etc.) */
    cl_int status;

    status = clReleaseKernel(mapperKernel);
    if(!checkVal(status,
                CL_SUCCESS,
                "clReleaseKernel failed."))
    {
        return SDK_FAILURE;
    }


    status = clReleaseKernel(mapperExtendedKernel);
    if(!checkVal(status,
                CL_SUCCESS,
                "clReleaseKernel failed."))
    {
        return SDK_FAILURE;
    }

    status = clReleaseKernel(reducerKernel);
    if(!checkVal(status,
                CL_SUCCESS,
                "clReleaseKernel failed."))
    {
        return SDK_FAILURE;
    }

    status = clReleaseKernel(copyerInOverflowKernel);
    if(!checkVal(status,
                CL_SUCCESS,
                "clReleaseKernel failed."))
    {
        return SDK_FAILURE;
    }

    status = clReleaseKernel(copyerKernel);
    if(!checkVal(status,
                CL_SUCCESS,
                "clReleaseKernel failed."))
    {
        return SDK_FAILURE;
    }

    status = clReleaseKernel(reducerInOverflowKernel);
    if(!checkVal(status,
                CL_SUCCESS,
                "clReleaseKernel failed."))
    {
        return SDK_FAILURE;
    }

    status = clReleaseProgram(program);
    if(!checkVal(status,
                CL_SUCCESS,
                "clReleaseProgram failed."))
    {
        return SDK_FAILURE;
    }

    status = clReleaseCommandQueue(commandQueue);
    if(!checkVal(status,
                CL_SUCCESS,
                "clReleaseCommandQueue failed."))
    {
        return SDK_FAILURE;
    }

    status = clReleaseContext(context);
    if(!checkVal(status,
                CL_SUCCESS,
                "clReleaseContext failed."))
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

MapReduce::~MapReduce()
{
    if (devices)
    {
        free(devices);
        devices = NULL;
    }

    if(maxWorkItemSizes)
    {
        free(maxWorkItemSizes);
        maxWorkItemSizes = NULL;
    }
}

//--------------------------------------------------------------------------------------
//Start Map phase for applications with Map and Reduce phase like KMeans and Wordcount 
//--------------------------------------------------------------------------------------
int MapReduce::startMapType2()
{

    timerStart();
    cl_int status;

    if (!jobSpec->Validate()) return -1;

    //1- Get map input data on host
    //------------------------------------------------------------------
    printf("\n(1)- Get map input data on host:\n");
    printf("----------------------------------\n");
    int	     h_inputRecordCount = jobSpec->inputRecordCount;
    int	     h_inputKeysBufSize = jobSpec->inputKeysBufSize;
    int	     h_inputValsBufSize = jobSpec->inputValsBufSize;
    cl_char*	 h_inputKeys = jobSpec->inputKeys;
    cl_char*	 h_inputVals = jobSpec->inputVals;
    cl_uint4* h_inputOffsetSizes = jobSpec->inputOffsetSizes;

    printf(" Map Input: keys size: %i bytes, values size: %i bytes, records: %i \n",h_inputKeysBufSize,h_inputValsBufSize,h_inputRecordCount);

    //2- Upload map input data to device memory
    //------------------------------------------------------------------
    printf("\n(2)- Upload map input data to device memory:\n");
    printf("----------------------------------------------\n");
    cl_mem d_inputRecordsMeta = NULL;
    cl_mem 	d_inputKeys = NULL;
    cl_mem  d_inputVals = NULL;
    cl_mem  d_inputOffsetSizes = NULL;

    d_inputKeys = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            h_inputKeysBufSize,
            h_inputKeys,
            &status);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_inputKeys)"))
        return SDK_FAILURE;

    d_inputVals = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            h_inputValsBufSize,
            h_inputVals,
            &status);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_inputVals)"))
        return SDK_FAILURE;

    d_inputOffsetSizes = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            sizeof(cl_int4)*h_inputRecordCount,
            h_inputOffsetSizes,
            &status);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_inputOffsetSizes)"))
        return SDK_FAILURE;

    //3- Determine the block size
    //------------------------------------------------------------------
    printf("\n(3)- Determine the block size:\n");
    printf("--------------------------------\n");
    int h_recordsPerTask = jobSpec->numRecTaskMap;
    int h_actualNumThreads=CEIL(h_inputRecordCount, h_recordsPerTask);
    size_t globalThreads[1]= {h_actualNumThreads};
    groupSize=jobSpec->userSize;
    size_t localThreads[1] = {groupSize};
    numGroups = h_actualNumThreads/groupSize;
    int wavefrontSize= isAMD? 64 : 32;
    printf("workgroup Size: %zu wavefrontSize: %d\n", groupSize, wavefrontSize);
    numWavefrontsPerGroup = groupSize/wavefrontSize;
    int localValuesSize = jobSpec->estimatedValSize *  groupSize;
    int localKeysSize =  jobSpec->estimatedKeySize * groupSize;

    printf("localKeysSize: %d  localValuesSize%d\n",localKeysSize,localValuesSize);
    numHashTables=numWavefrontsPerGroup*numGroups;


    // 6- Allocate intermediate memory on device memory
    //-----------------------------------------------
    printf("\n(6)- Allocate intermediate memory on device memory:\n");
    printf("---------------------------------------------------------------------\n");
    d_gOutputKeySize = NULL;
    d_gOutputValSize = NULL;
    d_gHashBucketSize = NULL;
    cl_int  *h_interAllKeys,*h_interAllVals, * h_outputAllVals, * h_outputAllKeys, 
            * h_hashTableAllVals ,  * h_hashBucketAllVals;

    h_estimatedInterValSize = (jobSpec->overflow)? jobSpec->estimatedInterRecords * jobSpec->estimatedInterValSize * 30: jobSpec->estimatedInterRecords * jobSpec->estimatedInterValSize ;
    h_estimatedInterKeySize = (jobSpec->overflow)? jobSpec->estimatedInterRecords * jobSpec->estimatedInterKeySize * 4: jobSpec->estimatedInterRecords * jobSpec->estimatedInterKeySize;
    printf("All Allocated Inter Keys Buffers: %i",h_estimatedInterKeySize);
    printf("All Allocated Inter vals Buffers: %i",h_estimatedInterValSize);

    d_interKeys = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            h_estimatedInterKeySize,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_interKeys1)"))
        return SDK_FAILURE;

    d_interVals = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            h_estimatedInterValSize,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_interVals)"))
        return SDK_FAILURE;

    d_interOffsets = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            sizeof(cl_int4)* jobSpec->estimatedInterRecords,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_interOffsets)"))
        return SDK_FAILURE;



    h_estimatedOutputValSize= (jobSpec->overflow)? jobSpec->estimatedRecords * jobSpec->estimatedValSize * numGroups* numWavefrontsPerGroup* 30 : jobSpec->estimatedRecords * jobSpec->estimatedValSize* numGroups;
    h_estimatedOutputKeySize= (jobSpec->overflow)? jobSpec->estimatedRecords * jobSpec->estimatedKeySize * numGroups* numWavefrontsPerGroup * 4 : jobSpec->estimatedRecords * jobSpec->estimatedKeySize * numGroups;
    printf("All Allocated Keys Buffers: %i\n", h_estimatedOutputKeySize);
    printf("All Allocated vals Buffers: %i\n",h_estimatedOutputValSize);
    printf("records: %i, valSize: %i, keySize: %i, numGroups: %i\n",jobSpec->estimatedRecords, jobSpec->estimatedValSize, jobSpec->estimatedKeySize, numGroups);

    d_outputKeys = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            h_estimatedOutputKeySize,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_outputKeys1)"))
        return SDK_FAILURE;

    d_outputVals = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            h_estimatedOutputValSize,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_outputVals1)"))
        return SDK_FAILURE;

    uint hashEntriesNum = jobSpec->estimatedRecords ;
    totalHashSize =  hashEntriesNum * numWavefrontsPerGroup * numGroups;
    cl_uint4* h_initial= (cl_uint4 *)malloc(sizeof(cl_uint4) * totalHashSize);
    printf("Hash Entries Num: %i numWavefrontsPerGroup: %d numGroups:%d Hash Table Size: %i\n",hashEntriesNum, numWavefrontsPerGroup, numGroups, totalHashSize);
    for (int i=0; i<totalHashSize ; i++)
    {
        h_initial[i]=(cl_uint4){0,0,0,0};
    }

    d_hashTable = clCreateBuffer(context,
            CL_MEM_READ_WRITE |CL_MEM_USE_HOST_PTR,
            sizeof(cl_uint4) * totalHashSize ,
            h_initial,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_hashTable)"))
        return SDK_FAILURE;

    //Create another data structure to hold the linked list of each hash bucket 
    totalHashBucketSize = (jobSpec->overflow)? hashEntriesNum* numWavefrontsPerGroup * numGroups  * 20 :hashEntriesNum* numGroups ; //assuming 20 collisions per hash entry

    d_hashBucket = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            sizeof(cl_int4) * totalHashBucketSize ,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_hashBucket)"))
        return SDK_FAILURE;

    //Create number of offsets equal to the number of workgroups
    //for each one to work independently from the other
    h_workgroupOutputValsizes = h_estimatedOutputValSize/numGroups;
    h_workgroupOutputKeysizes = h_estimatedOutputKeySize/numGroups;
    h_workgrouphashSizes =  totalHashSize / (numWavefrontsPerGroup * numGroups) ;
    h_workgrouphashBucketSizes =  totalHashBucketSize / ( numGroups) ;


    printf("Number of WorkGroups: %i\n",numGroups);
    printf("OutputVals Sizes per workgroup: %i\n",h_workgroupOutputValsizes);


    d_gOutputKeySize  = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            numGroups * sizeof(cl_uint) * 3,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_gOutputKeySize) "))
        return SDK_FAILURE;

    d_gOutputValSize  = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            numGroups * sizeof(cl_uint) * 3,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_gOutputValSize) "))
        return SDK_FAILURE;

    d_gHashBucketSize  = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            numGroups * sizeof(cl_uint) * 3,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_ghashBucketSize) "))
        return SDK_FAILURE;

    cl_uint * overflowWGId; // Ids of overflowWG Used to force only overflowed workgroups to continue their work
    overflowWGId = (cl_uint *) malloc(sizeof(cl_uint) * numGroups);

    cl_mem d_metaEmitted =  clCreateBuffer(context,
            CL_MEM_READ_WRITE, 
            h_inputRecordCount*sizeof(cl_int),
            NULL, 
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_metaEmitted)"))
        return SDK_FAILURE;

    cl_mem d_metaOverflow=  clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            h_inputRecordCount*sizeof(cl_int),
            NULL, 
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_metaOverflow)"))
        return SDK_FAILURE;

    cl_mem d_metaEmitted2=  clCreateBuffer(context,				//From second Map kernel
            CL_MEM_READ_WRITE,
            h_inputRecordCount*sizeof(cl_int),
            NULL, 
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_metaOverflow)"))
        return SDK_FAILURE;

    //Start Map
    //---------------------------------------------------------------------
    printf("\n(7)- Start Map:\n");
    printf("-----------------\n");
    cl_uint *h_gKeySizes,*h_gValSizes,*h_gCounts;
    cl_event events[3];

    //Set Kernel Arguments
    status = clSetKernelArg(
            mapperKernel,
            0,
            sizeof(cl_mem),
            (void*)&(d_inputDataSet));
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (inputDataSet)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            1,
            sizeof(cl_mem),
            (void*)&(d_constantData));
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (constantDataSet)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            2,
            sizeof(cl_mem),
            &d_inputKeys);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_inputKeys)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            3,
            sizeof(cl_mem),
            &d_inputVals);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_inputVals)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            4,
            sizeof(cl_mem),
            &d_inputOffsetSizes);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_inputOffsetSizes)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            5,
            sizeof(cl_mem),
            &d_interKeys);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_inputKeys)"))
        return SDK_FAILURE;


    status = clSetKernelArg(
            mapperKernel,
            6,
            sizeof(cl_mem),
            &d_interVals);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_interVals)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            7,
            sizeof(cl_mem),
            &d_interOffsets);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_interOffsets)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            8,
            sizeof(cl_mem),
            &d_outputKeys);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_outputKeys)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            9,
            sizeof(cl_mem),
            &d_outputVals);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_outputVals)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            10,
            sizeof(cl_mem),
            &d_hashTable);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_hashTable)"))
        return SDK_FAILURE;


    status = clSetKernelArg(
            mapperKernel,
            11,
            sizeof(cl_mem),
            &d_hashBucket);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_hashBucket)"))
        return SDK_FAILURE;


    status = clSetKernelArg(
            mapperKernel,
            12,
            sizeof(int),
            &h_inputRecordCount);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_inputRecordCount)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            13,
            sizeof(int),
            &h_recordsPerTask);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_recordsPerTask)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            14,
            sizeof(int),
            &h_actualNumThreads);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_actualNumThreads)"))
        return SDK_FAILURE;
    //Local memory
    status = clSetKernelArg(
            mapperKernel,
            15,
            sizeof(cl_uint)*3,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory16)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            16,
            sizeof(cl_uint)*3,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory17)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            17,
            sizeof(cl_uint)*3,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory18)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            18,
            sizeof(cl_uint) * groupSize,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory19)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            19,
            localKeysSize,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory20)"))
        return SDK_FAILURE;


    status = clSetKernelArg(
            mapperKernel,
            20,
            localValuesSize,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory21)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            21,
            localValuesSize,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory21)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            22,
            sizeof(cl_uint) * numWavefrontsPerGroup,   
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory22)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            23,
            sizeof(cl_uint)* numWavefrontsPerGroup, //Used scratch for each wavefront in a workgroup
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory22)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            24,
            sizeof(cl_uint) * groupSize,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory23)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            25,
            sizeof(cl_uint) * groupSize,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory24)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            26,
            sizeof(cl_uint) * groupSize,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory25)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            27,
            sizeof(cl_uint) * groupSize,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory26)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            28,
            sizeof(cl_mem),
            &d_gOutputKeySize);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_gOutputValSize)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            29,
            sizeof(cl_mem),
            &d_gOutputValSize);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_gOutputValSize)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            30,
            sizeof(cl_mem),
            &d_gHashBucketSize);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_gHashBucketSize)"))
        return SDK_FAILURE;

    uint localKeysBufferPerWavefront= localKeysSize/numWavefrontsPerGroup; 
    status = clSetKernelArg(
            mapperKernel,
            31,
            sizeof(int),
            &localKeysBufferPerWavefront);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (localKeysSize)"))
        return SDK_FAILURE;

    uint localScratchPerWavefront = localValuesSize/numWavefrontsPerGroup;
    status = clSetKernelArg(
            mapperKernel,
            32,
            sizeof(int),
            &localScratchPerWavefront);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (localScratchPerWavefront)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            33,
            sizeof(int),
            &h_workgroupOutputKeysizes);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_workgroupOutputKeysizes)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            34,
            sizeof(int),
            &h_workgroupOutputValsizes);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_workgroupOutputValsizes)"))
        return SDK_FAILURE;


    status = clSetKernelArg(
            mapperKernel,
            35,
            sizeof(int),
            &h_workgrouphashBucketSizes);  //Num of hash buckets
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (34: Num of hash entries)"))

        return SDK_FAILURE;

    printf("h_workgrouphashSizes : %i \n",h_workgrouphashSizes);

    status = clSetKernelArg(
            mapperKernel,
            36,
            sizeof(int),
            &h_workgrouphashSizes);  //Num of hash entries
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (35: Num of hashBuckets)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            37,
            sizeof(cl_uint) * numWavefrontsPerGroup ,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory36)"))
        return SDK_FAILURE;


    status = clSetKernelArg(
            mapperKernel,
            38,
            sizeof(cl_uint) * numWavefrontsPerGroup ,
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory37)"))
        return SDK_FAILURE;

    for (int i=0; i< numGroups; i++) 
        overflowWGId[i]=i;

    cl_mem  d_overflowWGId = clCreateBuffer(context,
            CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
            numGroups*sizeof(cl_uint),
            overflowWGId,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_overflowWGId) "))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            39,
            sizeof(cl_mem),
            &d_overflowWGId); //Just to pass compilation only used to handle overflow
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_overflowWGId)"))
        return SDK_FAILURE;

    cl_uint h_extended = 0;
    if (jobSpec->perfectHashing) h_extended = 2; 
    printf("perfect Hashing %i \n", h_extended);
    status = clSetKernelArg(
            mapperKernel,
            40,
            sizeof(cl_uint),
            &h_extended);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (43: extended)"))
        return SDK_FAILURE;
    cl_uint h_emitIntermediate = 0;
    if (jobSpec->outputIntermediate) h_emitIntermediate = 1;
    status = clSetKernelArg(
            mapperKernel,
            41,
            sizeof(cl_uint),
            &h_emitIntermediate);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (43: h_emitIntermediate)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            42,
            sizeof(cl_mem),
            &d_metaEmitted);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (45: metaemitted)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            43,
            sizeof(cl_mem),
            &d_metaOverflow);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (46: metaoverflow)"))
        return SDK_FAILURE;
    status = clSetKernelArg(
            mapperKernel,
            44,
            sizeof(cl_mem),
            &d_metaEmitted2);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (46: metaemitted2)"))
        return SDK_FAILURE;

    uint  h_interKeysPerWorkgroup = h_estimatedInterKeySize/numGroups;
    uint  h_interValsPerWorkgroup = h_estimatedInterValSize/numGroups;
    uint  h_interRecordsPerWorkgroup = jobSpec->estimatedInterRecords/numGroups;

    status = clSetKernelArg(
            mapperKernel,
            45,
            sizeof(int),
            &h_interKeysPerWorkgroup);  //
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (35: Num of hashBuckets)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            46,
            sizeof(int),
            &h_interValsPerWorkgroup);  //
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (35: Num of hashBuckets)"))
        return SDK_FAILURE;  

    status = clSetKernelArg(
            mapperKernel,
            47,
            sizeof(int),
            &h_interRecordsPerWorkgroup);  //
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (47: Num of hashBuckets)"))
        return SDK_FAILURE;
    //Enqueue a kernel run call
    printf("Before Run Kernel ...\n");
    status = clEnqueueNDRangeKernel(
            commandQueue,
            mapperKernel,
            1,
            NULL,
            globalThreads,
            localThreads,
            0,
            NULL,
            &events[0]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clEnqueueNDRangeKernel failed."))
        return SDK_FAILURE;

    printf("After Run Kernel ...\n");
    timerEnd();
    printf("Mapper Initialization: %.3f ms\n",elapsedTime());

    //wait for the kernel call to finish execution
    status = clFinish(commandQueue);
    if(!checkVal(status,
                CL_SUCCESS,
                "clFinish failed."))
    {
        return SDK_FAILURE;
    }
    cl_ulong startTime,endTime,diff;
    clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,NULL);
    clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,NULL);
    diff=endTime-startTime;
    printf("Mapper Kernel spent ** : %lu ns",diff);

    timerStart();

    /*         cl_int *  h_metae= (cl_int*)malloc( h_inputRecordCount*sizeof(cl_int));
               status = clEnqueueReadBuffer(commandQueue,
               d_metaEmitted,
               1,
               0,
               h_inputRecordCount*sizeof(cl_int),
               h_metae,
               0,
               0,
               &events[2]);
               if(!checkVal(
               status,
               CL_SUCCESS,
               "clReadBuffer failed. (h_meta)"))
               return SDK_FAILURE;
               int c=0, cg=0;
               for(int i=0; i <h_inputRecordCount; i++)
               {
               printf ("emitted[%i]= %i\n",i, h_metae[i]);
               cg+=h_metae[i];
               c+=  h_metae[i];	
               if ( (i+1)%64 == 0) 
               {
               printf("Group counts[%i]= %i\n",i, cg);
               cg = 0 ;
               }
               }
               printf("Total emitted: %i",c);

               status = clEnqueueReadBuffer(commandQueue,
               d_metaOverflow,
               1,
               0,
               h_inputRecordCount*sizeof(cl_int),
               h_metae,
               0,
               0,
               &events[2]);
               if(!checkVal(
               status,
               CL_SUCCESS,
               "clReadBuffer failed. (h_meta)"))
               return SDK_FAILURE;
               for(int i=0; i <h_inputRecordCount; i++)
               {
               printf ("overflow[%i]= %i\n",i, h_metae[i]);
               }

               free(h_metae);

*/
    timerStart();
    if (jobSpec->outputIntermediate == true)
    {
        jobSpec->interKeys= (cl_char*)malloc(h_estimatedInterKeySize);
        status = clEnqueueReadBuffer(commandQueue,
                d_interKeys,
                1,
                0,
                h_estimatedInterKeySize,
                jobSpec->interKeys,
                0,
                0,
                &events[2]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_interKeys)"))
            return SDK_FAILURE;

        jobSpec->interVals= (cl_char*)malloc(h_estimatedInterValSize);
        status = clEnqueueReadBuffer(commandQueue,
                d_interVals,
                1,
                0,
                h_estimatedInterValSize,
                jobSpec->interVals,
                0,
                0,
                &events[2]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_interVals)"))
            return SDK_FAILURE;

        jobSpec->interOffsetSizes= (cl_uint4*)malloc(jobSpec->estimatedInterRecords * sizeof(cl_uint4));
        status = clEnqueueReadBuffer(commandQueue,
                d_interOffsets,
                1,
                0,
                jobSpec->estimatedInterRecords * sizeof(cl_uint4),
                jobSpec->interOffsetSizes,
                0,
                0,
                &events[2]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_interOffsetSizes)"))
            return SDK_FAILURE;


    }
    timerEnd();
    printf("Reading intermediate output: %.3f ms\n",elapsedTime());

    cl_uint * h_gHashBucketSize =(cl_uint*)malloc(numGroups*sizeof(cl_uint)*3);
    status = clEnqueueReadBuffer(commandQueue,
            d_gHashBucketSize,
            1,
            0,
            numGroups*sizeof(cl_uint)* 3,
            h_gHashBucketSize,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (h_gHashBucketSize)"))
        return SDK_FAILURE;

    h_outputKeySizes=(cl_uint*)malloc(numGroups * sizeof(cl_uint)*3);
    status = clEnqueueReadBuffer(commandQueue,
            d_gOutputKeySize,
            1,
            0,
            numGroups*sizeof(cl_uint)*3,
            h_outputKeySizes,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (h_psCount)"))
        return SDK_FAILURE;

    h_outputValSizes=(cl_uint*)malloc(numGroups * sizeof(cl_uint)*3);
    status = clEnqueueReadBuffer(commandQueue,
            d_gOutputValSize,
            1,
            0,
            numGroups*sizeof(cl_uint)*3,
            h_outputValSizes,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (h_psCounts)"))
        return SDK_FAILURE;


    timerEnd();
    printf("Mapper Reading output: %.3f ms\n",elapsedTime());

    printf ("workgroup keys buffer %i\n", h_workgroupOutputKeysizes);
    printf ("workgroup values buffer %i\n", h_workgroupOutputValsizes);
    printf ("workgroup hash bucket buffer %i\n", h_workgrouphashBucketSizes);
    allOutputVals=0;

    //start overflow handling if it occurs
    timerStart();
    cl_int allKeys = 0 , allVals = 0 ,allCounts = 0;
    cl_int allKeysUsed = 0  , allValsUsed = 0 ,allCountsUsed = 0;
    overflowedWorkGroups = 0;
    printf("Num of groups : %i\n",numGroups);
    KeysDiff = 0, ValsDiff  = 0, CountsDiff = 0;
    h_interAllKeys= (cl_int*)malloc(sizeof(cl_int) * numGroups);
    h_interAllVals= (cl_int*)malloc(sizeof(cl_int) * numGroups);
    h_interAllOffsetSizes= (cl_int*)malloc(sizeof(cl_int) * numGroups);
    for (int i=0; i< numGroups; i++)
    {
        allKeys += h_outputKeySizes[i];
        allVals += h_outputValSizes[i];
        allCounts += h_gHashBucketSize[i];

        allKeysUsed += h_outputKeySizes[numGroups+i];
        allValsUsed += h_outputValSizes[numGroups+i];
        allCountsUsed += h_gHashBucketSize[numGroups+i];


        if ((h_gHashBucketSize[i] > h_workgrouphashBucketSizes ) || ( h_outputKeySizes[i] > h_workgroupOutputValsizes) ||( h_outputValSizes[i] >  h_workgroupOutputKeysizes) ){
            printf("Overflow!!!! WG ID: %i\n", i);
            printf("offsets: %i Keys: %i Vals: %i \n", h_gHashBucketSize[i], h_outputKeySizes[i], h_outputValSizes[i]);
            printf("offsets: %i Keys: %i Vals: %i \n", CountsDiff, KeysDiff, ValsDiff);
            overflowWGId[overflowedWorkGroups] = i;
            h_interAllKeys[overflowedWorkGroups]= ValsDiff;
            h_interAllVals[overflowedWorkGroups]= KeysDiff;
            h_interAllOffsetSizes[overflowedWorkGroups]= CountsDiff;
            overflowedWorkGroups++;
        }
        //Allocate all needed buffer.
        KeysDiff = allKeys - allKeysUsed;
        ValsDiff = allVals - allValsUsed;
        CountsDiff = allCounts - allCountsUsed;

        //printf("Output records of group%i: %i and maximum limit: %i\n",i,h_gHashBucketSize[i],h_workgrouphashBucketSizes);
    }

    printf("Overall offsets: %i Keys: %i Vals: %i \n", CountsDiff, KeysDiff, ValsDiff);
    printf("Output records: %i\n",allCountsUsed);
    jobSpec->interRecordCount = allCountsUsed;
    jobSpec->interDiffKeyCount = CountsDiff;
    jobSpec->interAllKeySize = allKeys;
    jobSpec->interAllValSize = allVals;

    extraHashBucketSize = CountsDiff; //used by the reducer kernel
    timerEnd();
    printf("Mapper pre overflow: %.3f ms\n",elapsedTime());


    for (int i=0; i<numGroups; i++)
    {
        /*
           if ( jobSpec->outputIntermediate)
           {
           allKeys += (h_gKeySizes[i] -  h_gKeySizes[numGroups+i]);
           allVals += (h_gValSizes[i] -  h_gValSizes[numGroups+i]);
           allCounts += (h_gCounts[i] -  h_gCounts[numGroups+i]);
           if (h_gCounts[i] >  h_gCounts[numGroups+i] )
           printf ("Counts Overflow workgroup %i %i %i!!!!\n",i, h_gCounts[i], h_gCounts[numGroups+i]);
           if (h_gKeySizes[i] > h_gKeySizes[numGroups+i] )
           printf ("Keys Overflow workgroup %i %i %i!!!!\n",i, h_gKeySizes[i], h_gKeySizes[numGroups+i]);
           if (h_gValSizes[i] >  h_gValSizes[numGroups+i] )
           printf ("Vals Overflow workgroup %i %i %i!!!!\n",i, h_gValSizes[i],h_gValSizes[numGroups+i]);

           }
           */
        allOutputVals += (h_outputValSizes[i] - h_outputValSizes[numGroups+i]);
        /*printf ("Output Keys Overflow workgroup %i %i %i!!!!\n",i, h_outputKeySizes[i],h_outputKeySizes[numGroups+i]);
        printf ("Output Vals Overflow workgroup %i %i %i!!!!\n",i, h_outputValSizes[i],h_outputValSizes[numGroups+i]);
        printf( "hash bucket sizes of workgroup %i: %i %i!!!!\n",i, h_gHashBucketSize[i], h_gHashBucketSize[numGroups+i]);*/
    }

    printFinalOutput(0,0,0,0,0,NULL);

    //8- Run a second map kernel if necessary
    //----------------------------------------

    printf("\n(8)- Start second map\n");
    printf("-----------------------------\n");
    if (overflowedWorkGroups > 0)
    {
        if ( (KeysDiff > 0) ||(ValsDiff > 0)||(CountsDiff > 0))
        {
            overflowOccurs = true;

            size_t globalThreads[1]= {overflowedWorkGroups * groupSize};

            printf("overflowed WorkGroups: %i",overflowedWorkGroups);
            timerStart();

            for (int i =0; i <overflowedWorkGroups ; i++)
                printf("overflowWGId[%i]= %i\n",i,overflowWGId[i]);

            cl_mem  d_overflowWGId = clCreateBuffer(context,
                    CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                    overflowedWorkGroups*sizeof(cl_uint),
                    overflowWGId,
                    &status);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clCreateBuffer failed. (d_gCounts) "))
                return SDK_FAILURE;

            d_outputKeysExtra = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    ValsDiff,
                    NULL,
                    &status);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clCreateBuffer failed. (d_outputKeysExtra)"))
                return SDK_FAILURE;

            d_outputValsExtra = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    KeysDiff,
                    NULL,
                    &status);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clCreateBuffer failed. (d_outputValsExtra)"))
                return SDK_FAILURE;


            totalHashSizeExtra = hashEntriesNum * numWavefrontsPerGroup * overflowedWorkGroups;
            extraHashTables =  numWavefrontsPerGroup * overflowedWorkGroups;
            cl_uint4* h_initial= (cl_uint4 *)malloc(sizeof(cl_uint4) * totalHashSizeExtra);
            printf("Extra Hash Table Size: %i\n",totalHashSizeExtra);
            for (int i=0; i<totalHashSizeExtra ; i++)
            {
                h_initial[i]=(cl_uint4){0,0,0,0};
            }

            d_hashTableExtra = clCreateBuffer(context,
                    CL_MEM_READ_WRITE |CL_MEM_USE_HOST_PTR,
                    sizeof(cl_uint4) * totalHashSizeExtra ,
                    h_initial,
                    &status);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clCreateBuffer failed. (d_hashTableExtra)"))
                return SDK_FAILURE;

            //Create another data structure to hold the linked list of each hash bucket
            d_hashBucketExtra = clCreateBuffer(context,
                    CL_MEM_READ_WRITE,
                    sizeof(cl_int4) * CountsDiff ,
                    NULL,
                    &status);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clCreateBuffer failed. (d_hashBucketExtra)"))
                return SDK_FAILURE;


            d_outputKeysOffsetsExtra = clCreateBuffer(context,
                    CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                    sizeof(cl_int)* overflowedWorkGroups,
                    h_interAllKeys,
                    &status);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clCreateBuffer failed. (d_interAllKeys)"))
                return SDK_FAILURE;

            d_outputValsOffsetsExtra = clCreateBuffer(context,
                    CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                    sizeof(cl_int)*overflowedWorkGroups,
                    h_interAllVals,
                    &status);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clCreateBuffer failed. (d_interAllVals)"))
                return SDK_FAILURE;

            d_hashBucketOffsetsExtra = clCreateBuffer(context,
                    CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                    sizeof(cl_int)* overflowedWorkGroups,
                    h_interAllOffsetSizes,
                    &status);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clCreateBuffer failed. (d_interAllKeys)"))
                return SDK_FAILURE;


            //set Kernel arguments and run the kernel
            status = clSetKernelArg(
                    mapperExtendedKernel,
                    0,
                    sizeof(cl_mem),
                    (void*)&(d_inputDataSet));
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (inputDataSet)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    1,
                    sizeof(cl_mem),
                    (void*)&(d_constantData));
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (constantDataSet)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    2,
                    sizeof(cl_mem),
                    &d_inputKeys);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (d_inputKeys)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    3,
                    sizeof(cl_mem),
                    &d_inputVals);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (d_inputVals)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    4,
                    sizeof(cl_mem),
                    &d_inputOffsetSizes);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (d_inputOffsetSizes)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    5,
                    sizeof(cl_mem),
                    &d_outputKeysExtra);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (d_interKeysExtra)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    6,
                    sizeof(cl_mem),
                    &d_outputValsExtra);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (d_outputValsExtra)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    7,
                    sizeof(cl_mem),
                    &d_hashTableExtra);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (d_hashTableExtra)"))
                return SDK_FAILURE;


            status = clSetKernelArg(
                    mapperExtendedKernel,
                    8,
                    sizeof(cl_mem),
                    &d_hashBucketExtra);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (d_hashBucketExtra)"))
                return SDK_FAILURE;
            status = clSetKernelArg(
                    mapperExtendedKernel,
                    9,
                    sizeof(cl_mem),
                    &d_outputKeysOffsetsExtra);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (d_interKeysOffsets)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    10,
                    sizeof(cl_mem),
                    &d_outputValsOffsetsExtra);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (d_interValsOffsets)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    11,
                    sizeof(cl_mem),
                    &d_hashBucketOffsetsExtra);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (d_interOffsetSizesOffsetsExtra)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    12,
                    sizeof(int),
                    &h_inputRecordCount);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (h_inputRecordCount)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    13,
                    sizeof(int),
                    &h_recordsPerTask);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (h_recordsPerTask)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    14,
                    sizeof(int),
                    &h_actualNumThreads);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (h_actualNumThreads)"))
                return SDK_FAILURE;
            //Local memory
            status = clSetKernelArg(
                    mapperExtendedKernel,
                    15,
                    sizeof(cl_uint)*2,
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory16)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    16,
                    sizeof(cl_uint)*2,
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory17)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    17,
                    sizeof(cl_uint)*2,
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory18)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    18,
                    sizeof(cl_uint) * groupSize,
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory19)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    19,
                    localKeysSize, //
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory20)"))
                return SDK_FAILURE;


            status = clSetKernelArg(
                    mapperExtendedKernel,
                    20,
                    localValuesSize, //
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory21)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    21,
                    localValuesSize, //
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory21)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    22,
                    sizeof(cl_uint) * numWavefrontsPerGroup ,
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory22)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    23,
                    sizeof(cl_uint) * numWavefrontsPerGroup ,
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory22)"))
                return SDK_FAILURE;
            status = clSetKernelArg(
                    mapperExtendedKernel,
                    24,
                    sizeof(cl_uint) * groupSize,
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory23)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    25,
                    sizeof(cl_uint) * groupSize,
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory24)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    26,
                    sizeof(cl_uint) * groupSize, //
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory25)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    27,
                    sizeof(cl_uint) * groupSize, //
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory26)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    28,
                    sizeof(cl_uint) * groupSize,
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory27 - localEmitted2)"))
                return SDK_FAILURE;

            uint localKeysBufferPerWavefront= localKeysSize/numWavefrontsPerGroup;
            status = clSetKernelArg(
                    mapperExtendedKernel,
                    29,
                    sizeof(int),
                    &localKeysBufferPerWavefront);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (localKeysSize)"))
                return SDK_FAILURE;

            uint localScratchPerWavefront = localValuesSize/numWavefrontsPerGroup;
            status = clSetKernelArg(
                    mapperExtendedKernel,
                    30,
                    sizeof(int),
                    &localScratchPerWavefront);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (localScratchPerWavefront)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    31,
                    sizeof(int),
                    &h_workgroupOutputKeysizes);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (h_workgroupOutputKeysizes)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    32,
                    sizeof(int),
                    &h_workgroupOutputValsizes);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (h_workgroupOutputValsizes)"))
                return SDK_FAILURE;


            status = clSetKernelArg(
                    mapperExtendedKernel,
                    33,
                    sizeof(int),
                    &h_workgrouphashBucketSizes);  //Num of hash buckets
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (34: Num of hash entries)"))

                return SDK_FAILURE;

            printf("h_workgrouphashSizes : %i \n",h_workgrouphashSizes);

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    34,
                    sizeof(int),
                    &h_workgrouphashSizes);  //Num of hash entries
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (35: Num of hashBuckets)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    35,
                    sizeof(cl_uint) * numWavefrontsPerGroup ,
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory36)"))
                return SDK_FAILURE;


            status = clSetKernelArg(
                    mapperExtendedKernel,
                    36,
                    sizeof(cl_uint) * numWavefrontsPerGroup ,
                    NULL);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (local memory37)"))
                return SDK_FAILURE;


            status = clSetKernelArg(
                    mapperExtendedKernel,
                    37,
                    sizeof(cl_mem),
                    &d_overflowWGId);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (d_overflowWGId)"))
                return SDK_FAILURE;


            cl_int h_extended = 1;
            status = clSetKernelArg(
                    mapperExtendedKernel,
                    38,
                    sizeof(int),
                    &h_extended);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (39: extended)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    39,
                    sizeof(cl_mem),
                    &d_metaEmitted);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (40: metaemitted)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    40,
                    sizeof(cl_mem),
                    &d_metaOverflow);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (41: metaoverflow)"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    mapperExtendedKernel,
                    41,
                    sizeof(cl_mem),
                    &d_metaEmitted2);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. (42: metaemitted2)"))
                return SDK_FAILURE;

            //Enqueue a kernel run call
            printf("Before Run Kernel ...\n");
            status = clEnqueueNDRangeKernel(
                    commandQueue,
                    mapperExtendedKernel,
                    1,
                    NULL,
                    globalThreads,
                    localThreads,
                    0,
                    NULL,
                    &events[0]);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clEnqueueNDRangeKernel failed."))
                return SDK_FAILURE;

            printf("After Run Kernel ...\n");
            timerEnd();
            printf("Mapper Initialization: %.3f ms\n",elapsedTime());

            //wait for the kernel call to finish execution
            status = clFinish(commandQueue);
            if(!checkVal(status,
                        CL_SUCCESS,
                        "clFinish failed."))
            {
                return SDK_FAILURE;
            }

            clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,NULL);
            clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,NULL);
            diff=endTime-startTime;
            printf("Second mapper kernel spent ** : %lu ns",diff);

            /*
               h_metae= (cl_int* ) malloc(sizeof(cl_int) * h_inputRecordCount);
               status = clEnqueueReadBuffer(commandQueue,
               d_metaOverflow,
               1,
               0,
               h_inputRecordCount*sizeof(cl_int),
               h_metae,
               0,
               0,
               &events[2]);
               if(!checkVal(
               status,
               CL_SUCCESS,
               "clReadBuffer failed. (h_meta)"))
               return SDK_FAILURE;
               int c=0, cg=0;
               for(int i=0; i <h_inputRecordCount; i++)
               {
               printf ("metaOverflow[%i]= %i\n",i, h_metae[i]);

               cg+=h_metae[i];
               c+=  h_metae[i];
               if ( (i+1)%64 == 0)
               {
               printf("Group counts[%i]= %i\n",i, cg);
               cg = 0 ;
               }

               }
               printf("Total emitted: %i",c);

               free(h_metae);

*/	 
            //printFinalOutput( 1, KeysDiff, ValsDiff, sizeof(cl_uint4) *totalHashSizeExtra, sizeof(cl_uint4) *CountsDiff, h_interAllOffsetSizes );

        }//End Extended Mapper
    }
    //9- Free allocated memory
    //----------------------------------------------
    printf("\n(9)- Free Allocated Memory:\n");
    printf("-----------------------------\n");

    clReleaseMemObject(d_inputKeys);
    clReleaseMemObject(d_inputVals);
    clReleaseMemObject(d_inputOffsetSizes);


    return 0;
}

//----------------------------------------------------------------------------------
//Start Reduce phase for applications with non-perfect hashing functions like wordcount
//----------------------------------------------------------------------------------
int MapReduce::startReduce()
{

    timerStart();

    cl_int status;
    cl_event events[3];
    cl_mem  d_keySizePerWG=NULL;
    cl_mem  d_valSizePerWG=NULL;
    cl_mem d_numPairsPerWG=NULL;

    printf("\n(10)- Start Reduce Phase:\n");
    printf("----------------------------------------\n");

    if (jobSpec->estimatedRecords == 0) { printf( "Error: invalid intermediate hash entries number");exit(0);}

    printf("Determine the block size:\n");
    printf("--------------------------------\n");
    int wavefrontSize= isAMD? 64 : 32;
    uint mapWavefrontsPerGroup = groupSize/wavefrontSize;
    int h_hashPartitionFactor = 16;//16384;
    int h_hashesPerThread = (numHashTables)/h_hashPartitionFactor;
    int h_recordsPerTask  = jobSpec->numRecTaskReduce;
    int h_actualNumThreads;
    size_t globalThreads[1] = {(extraHashTables + numHashTables) * jobSpec->estimatedRecords};
    size_t localThreads[1] = {64}; //Best 128 for Wordcoun 64 for KMeans
    int numWorkGroups;
    cl_ulong startTime,endTime,diff;

    printf("extraHashTables: %i numHashTables: %i\n",extraHashTables, numHashTables);
    printf("number of threads: %i\n", (extraHashTables + numHashTables) * jobSpec->estimatedRecords);

    cl_ulong totalTime=0;

    printf("\n(13)- Allocate output memory on device memory:\n");
    printf("---------------------------------------------------------------------\n");

    numWorkGroups = globalThreads[0]/localThreads[0];	 //Maximum number of work groups
    printf("Number of workgroups: %i\n",numWorkGroups);
    d_keySizePerWG = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            numWorkGroups * sizeof(cl_float),
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_psKeySizes)"))
        return SDK_FAILURE;

    d_valSizePerWG = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            numWorkGroups * sizeof(cl_float),
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_psValSizes)"))
        return SDK_FAILURE;

    d_numPairsPerWG = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            numWorkGroups * sizeof(cl_float),
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_psCounts) "))
        return SDK_FAILURE;


    //-------------------------------------------------------------------- Launch Reduce  --------------------------------------------------------------------//
    printf("\n(14)- Start Reduce:\n");
    printf("-----------------\n");

    uint stages;
    if (h_hashesPerThread > 1 ) stages=2; else stages=1;
    for (int i=0; i<stages; i++)
    {
        timerStart();

        if (i == 0)
        {
            //first stage
            if (h_hashesPerThread <= 1 )
            {
                //each thread handles all hash tables
                h_actualNumThreads = CEIL( jobSpec->estimatedRecords, h_recordsPerTask);
                h_hashesPerThread = 1; 
            }
            else
            {
                int threadsNum = (overflowOccurs == 1)?  jobSpec->estimatedRecords*h_hashPartitionFactor * 2:  jobSpec->estimatedRecords*h_hashPartitionFactor;
                h_actualNumThreads=CEIL( threadsNum, h_recordsPerTask); //When overflow occurs twice threads are executed to handle overflowed records
            }
        }
        else
        {
            //second stage
            h_actualNumThreads=CEIL( jobSpec->estimatedRecords, h_recordsPerTask);
        }


        numWorkGroups = h_actualNumThreads/localThreads[0];        //Maximum number of work groups
        printf("Number of workgroups: %i actual number of threads: %i hashes per thread : %i\n",numWorkGroups,h_actualNumThreads, h_hashesPerThread);
        if (numWorkGroups == 0) numWorkGroups =1;
        d_keySizePerWG = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                numWorkGroups * sizeof(cl_float),
                NULL,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_psKeySizes)"))
            return SDK_FAILURE;

        d_valSizePerWG = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                numWorkGroups * sizeof(cl_float),
                NULL,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_psValSizes)"))
            return SDK_FAILURE;

        d_numPairsPerWG = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                numWorkGroups * sizeof(cl_float),
                NULL,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_psCounts) "))
            return SDK_FAILURE;

        cl_kernel ker;
        if (overflowOccurs == 0)
            ker = reducerKernel;
        else
            ker = reducerInOverflowKernel;

        //Set Kernel Arguments
        status = clSetKernelArg(
                ker,
                0,
                sizeof(cl_mem),
                (void*)&(d_constantData));
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (constantDataSet)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                1,
                sizeof(cl_mem),
                &d_keySizePerWG);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_psKeySizes)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                2,
                sizeof(cl_mem),
                &d_valSizePerWG);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_psValSizes)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                3,
                sizeof(cl_mem),
                &d_numPairsPerWG);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_psCounts)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                4,
                sizeof(cl_mem),
                &d_outputKeys);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_outputKeys)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                5,
                sizeof(cl_mem),
                &d_outputVals);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_outputVals)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                6,
                sizeof(cl_mem),
                &d_hashTable);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_hashTable)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                7,
                sizeof(cl_mem),
                &d_hashBucket);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_hashBucket)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                8,
                sizeof(int),
                &jobSpec->estimatedRecords);  //number of hash entries
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (&jobSpec->estimatedRecords)"))
            return SDK_FAILURE;


        status = clSetKernelArg(
                ker,
                9,
                sizeof(int),
                &numHashTables);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (numHashTables)"))
            return SDK_FAILURE;

        printf("jobSpec->interDiffKeyCount: %i",jobSpec->interDiffKeyCount);
        printf("h_recordsPerTask: %i",h_recordsPerTask);

        status = clSetKernelArg(
                ker,
                10,
                sizeof(int),
                &jobSpec->interDiffKeyCount);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (jobSpec->interDiffKeyCount)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                11,
                sizeof(int),
                &h_recordsPerTask);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (h_recordsPerTask)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                12,
                sizeof(int),
                &h_actualNumThreads);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (h_actualNumThreads)"))
            return SDK_FAILURE;


        status = clSetKernelArg(
                ker,
                13,
                sizeof(int),
                &h_hashesPerThread);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (h_hashesPerThread)"))
            return SDK_FAILURE;

        uint mode = i+1;
        printf("Reduce Mode: %i HashesPerThread: %i\n",mode,h_hashesPerThread);
        status = clSetKernelArg(
                ker,
                14,
                sizeof(int),
                &mode); //mode
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (stageId)"))
            return SDK_FAILURE;
        status = clSetKernelArg(
                ker,
                15,
                sizeof(cl_uint),
                NULL);                                        
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (localKeySizesPerWorkgroup)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                16,
                sizeof(cl_uint),
                NULL);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (localKeySizesPerWorkgroup)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                17,
                sizeof(cl_uint),
                NULL);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (localKeySizesPerWorkgroup)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                18,
                sizeof(cl_uint),
                &mapWavefrontsPerGroup);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_hashBucketIsBaseForReduceExtra)"))
            return SDK_FAILURE;

        if (overflowOccurs == 1)
        {

            printf("Overflow Occur \n");
            status = clSetKernelArg(
                    ker,
                    19,
                    sizeof(cl_mem),
                    &d_outputKeysExtra);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. d_outputKeysExtra"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    ker,
                    20,
                    sizeof(cl_mem),
                    &d_outputValsExtra);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. outputValsExtra"))
                return SDK_FAILURE;

            status = clSetKernelArg(
                    ker,
                    21,
                    sizeof(cl_mem),
                    &d_hashTableExtra);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. hashTableExtra"))
                return SDK_FAILURE;
            status = clSetKernelArg(
                    ker,
                    22,
                    sizeof(cl_mem),
                    &d_hashBucketExtra);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. hashBucketExtra"))
                return SDK_FAILURE;
            status = clSetKernelArg(
                    ker,
                    23,
                    sizeof(cl_int),
                    &extraHashTables);
            if(!checkVal(
                        status,
                        CL_SUCCESS,
                        "clSetKernelArg failed. hashTableExtra"))
                return SDK_FAILURE;


        }


        //Enqueue a kernel run call
        printf("Before Run Kernel Reducer ...\n");
        globalThreads[0]= h_actualNumThreads;
        status = clEnqueueNDRangeKernel(
                commandQueue,
                ker,
                1,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clEnqueueNDRangeKernel failed."))
            return SDK_FAILURE;

        printf("After Run Kernel ...\n");
        timerEnd();
        printf("Reducer Initialization: %.3f ms\n",elapsedTime());

        //wait for the kernel call to finish execution
        status = clFinish(commandQueue);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clFinish failed."))
        {
            return SDK_FAILURE;
        }
        clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,NULL);
        clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,NULL);
        diff=endTime-startTime;
        totalTime+=diff;
        printf("Reducer Kernel spent **: %lu ns",diff);
        //if (overflowOccurs == 1) printFinalOutput( 4, KeysDiff, ValsDiff, sizeof(cl_uint4) *totalHashSizeExtra, sizeof(cl_uint4) *CountsDiff, h_interAllOffsetSizes );
        printFinalOutput(2,0,0,0,0,NULL);

    }//End for


    //--------------------------------------------- Reduce postprocessing: Copy Hash into Array ---------------------------------------------------------//
    printf("\n(15)- Copy Hash into Array for Efficient retrieval:\n");
    printf("-----------------------------------------------------\n");

    //Prefixsum for workgroups offsets
    timerStart();

    cl_mem d_keyOffsetPerWG, d_valOffsetPerWG, d_numPairsOffsetPerWG;

    if (numWorkGroups > 1)
    {
        //Create the buffers holding offsets per Bucket
        d_keyOffsetPerWG = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                sizeof(cl_float) * numWorkGroups,
                NULL,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_keyOffsetPerBucket) "))
            return SDK_FAILURE;

        d_valOffsetPerWG = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                sizeof(cl_float) * numWorkGroups,
                NULL,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_valOffsetPerBucket) "))
            return SDK_FAILURE;

        d_numPairsOffsetPerWG = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                sizeof(cl_float) * numWorkGroups,
                NULL,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_numPairsOffsetPerBucket) "))
            return SDK_FAILURE;


        timerEnd();
        printf("PrefixSum Initialization: %.3f ms\n",elapsedTime());

        //ScanLargeArrays clScanLarge(context,devices,commandQueue);
        //clScanLarge.initialize();
        //#ifdef PRINT_INFO

        cl_float * h_keySizePerWG= (cl_float*) malloc(sizeof(cl_float)* numWorkGroups);
        status = clEnqueueReadBuffer(commandQueue,
                d_keySizePerWG,
                1,
                0,
                sizeof(cl_float) * numWorkGroups,
                h_keySizePerWG,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_keySizePerWG)"))
            return SDK_FAILURE;

        //for (int i=0 ;i <numWorkGroups ; i++)
        //printf ("h_keySizePerWG[%i] = %f \n",i, h_keySizePerWG[i]);
        contiguousKeysSize =(int)h_keySizePerWG[numWorkGroups - 1];

        free(h_keySizePerWG);
        //printf("\n");
        cl_float * h_valSizePerWG= (cl_float*) malloc(sizeof(cl_float)* numWorkGroups);
        status = clEnqueueReadBuffer(commandQueue,
                d_valSizePerWG,
                1,
                0,
                sizeof(cl_float) * numWorkGroups,
                h_valSizePerWG,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_valSizePerWG)"))
            return SDK_FAILURE;

        //for (int i=0 ;i <numWorkGroups ; i++)
        //printf ("h_valSizePerWG[%i] = %f \n",i, h_valSizePerWG[i]);
        contiguousValsSize = (int)h_valSizePerWG[numWorkGroups - 1];

        free(h_valSizePerWG);
        //#endif
        cl_float * h_numPairsPerWG= (cl_float*) malloc(sizeof(cl_float)* numWorkGroups);
        status = clEnqueueReadBuffer(commandQueue,
                d_numPairsPerWG,
                1,
                0,
                sizeof(cl_float) * numWorkGroups,
                h_numPairsPerWG,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_valSizePerWG)"))
            return SDK_FAILURE;

        contiguousOffsets = (int) h_numPairsPerWG[numWorkGroups - 1];
        free(h_numPairsPerWG);


        Scan(&d_keySizePerWG, &d_keyOffsetPerWG, numWorkGroups);
        Scan(&d_valSizePerWG, &d_valOffsetPerWG, numWorkGroups);
        Scan(&d_numPairsPerWG, &d_numPairsOffsetPerWG, numWorkGroups);

        //contiguousKeysSize = (int)clScanLarge.MarsScan(&d_keySizePerWG, &d_keyOffsetPerWG, numWorkGroups, true);
        //d_keyOffsetPerWG = clScanLarge.outputBuffer[0];

        //contiguousValsSize = (int)clScanLarge.MarsScan(&d_valSizePerWG, &d_valOffsetPerWG, numWorkGroups, true);
        //d_valOffsetPerWG = clScanLarge.outputBuffer[0];

        //contiguousOffsets = clScanLarge.MarsScan(&d_numPairsPerWG, &d_numPairsOffsetPerWG, numWorkGroups, true);
        //d_numPairsOffsetPerWG = clScanLarge.outputBuffer[0];
        //#ifdef PRINT_INFO
        cl_float * h_keyOffsetPerWG= (cl_float*) malloc(sizeof(cl_float)* numWorkGroups);
        status = clEnqueueReadBuffer(commandQueue,
                d_keyOffsetPerWG,
                1,
                0,
                sizeof(cl_float) * numWorkGroups,
                h_keyOffsetPerWG,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_keySizePerWG)"))
            return SDK_FAILURE;

        //for (int i=0 ;i < numWorkGroups ; i++)
        //printf ("h_keyOffsetPerWG[%i] = %f \n",i, h_keyOffsetPerWG[i]);

        contiguousKeysSize +=(int)h_keyOffsetPerWG[numWorkGroups - 1];
        free(h_keyOffsetPerWG);

        cl_float * h_valOffsetPerWG= (cl_float*) malloc(sizeof(cl_float)* numWorkGroups);
        status = clEnqueueReadBuffer(commandQueue,
                d_valOffsetPerWG,
                1,
                0,
                sizeof(cl_float) * numWorkGroups,
                h_valOffsetPerWG,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_valSizePerWG)"))
            return SDK_FAILURE;


        contiguousValsSize += (int)h_valOffsetPerWG[numWorkGroups - 1];
        free(h_valOffsetPerWG);

        cl_float * h_countsOffsetPerWG= (cl_float*) malloc(sizeof(cl_float)* numWorkGroups);
        status = clEnqueueReadBuffer(commandQueue,
                d_numPairsOffsetPerWG,
                1,
                0,
                sizeof(cl_float) * numWorkGroups,
                h_countsOffsetPerWG,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_keySizePerWG)"))
            return SDK_FAILURE;

        //for (int i=0 ;i < numWorkGroups ; i++)
        //printf ("h_countsOffsetPerWG[%i] = %f \n",i, h_countsOffsetPerWG[i]);

        contiguousOffsets += (int) h_countsOffsetPerWG[numWorkGroups - 1];
        free(h_countsOffsetPerWG);

        //#endif

    }
    else
    {
        cl_float KeysSize, ValsSize,Offsets;
        status = clEnqueueReadBuffer(commandQueue,
                d_keySizePerWG,
                1,
                0,
                sizeof(cl_float),
                &KeysSize,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_keySizePerWG)"))
            return SDK_FAILURE;

        status = clEnqueueReadBuffer(commandQueue,
                d_valSizePerWG,
                1,
                0,
                sizeof(cl_float),
                &ValsSize,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_valSizePerWG)"))
            return SDK_FAILURE;

        status = clEnqueueReadBuffer(commandQueue,
                d_numPairsPerWG,
                1,
                0,
                sizeof(cl_float),
                &Offsets,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_valSizePerWG)"))
            return SDK_FAILURE;

        cl_int offset=0;
        d_keyOffsetPerWG = clCreateBuffer(context,
                CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                sizeof(cl_float) * numWorkGroups,
                &offset,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_keyOffsetPerBucket) "))
            return SDK_FAILURE;

        d_valOffsetPerWG = clCreateBuffer(context,
                CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                sizeof(cl_float) * numWorkGroups,
                &offset,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_valOffsetPerBucket) "))
            return SDK_FAILURE;

        d_numPairsOffsetPerWG = clCreateBuffer(context,
                CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                sizeof(cl_float) * numWorkGroups,
                &offset,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_numPairsOffsetPerBucket) "))
            return SDK_FAILURE;
        timerEnd();
        printf("PrefixSum Initialization: %.3f ms\n",elapsedTime());

        contiguousKeysSize =(int)KeysSize;
        contiguousValsSize = (int) ValsSize;
        contiguousOffsets= (int) Offsets;
    }
    printf("h_allKeySize %i , h_allKeySize  %i, h_allCounts %i\n ",contiguousKeysSize, contiguousValsSize, contiguousOffsets);

    timerStart();
    //Create buffers storing final outputs
    d_contiguousOutputKeys = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            contiguousKeysSize,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_contiguousOutputKeys)"))
        return SDK_FAILURE;

    d_contiguousOutputVals = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            contiguousValsSize,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_contiguousOutputVals)"))
        return SDK_FAILURE;

    d_outputKeyValOffsets = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            sizeof(cl_uint4) * contiguousOffsets,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_keyValOffsets)"))
        return SDK_FAILURE;

    //Launch corresponding copyer kernel
    cl_kernel ker;
    if (overflowOccurs == 0)
        ker = copyerKernel;
    else
        ker = copyerInOverflowKernel;

    status = clSetKernelArg(
            ker,
            0,
            sizeof(cl_mem),
            (void*)&(d_constantData));
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (constantDataSet)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            1,
            sizeof(cl_mem),
            &d_contiguousOutputKeys);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. d_contiguousOutputKeys)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            2,
            sizeof(cl_mem),
            &d_contiguousOutputVals);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_contiguousOutputVals)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            3,
            sizeof(cl_mem),
            &d_outputKeyValOffsets);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_outputKeyValOffsets)"))
        return SDK_FAILURE;


    status = clSetKernelArg(
            ker,
            4,
            sizeof(cl_mem),
            &d_outputKeys);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_outputKeys)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            5,
            sizeof(cl_mem),
            &d_outputVals);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_outputVals)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            6,
            sizeof(cl_mem),
            &d_hashTable);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_hashTable)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            7,
            sizeof(cl_mem),
            &d_hashBucket);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_hashBucket)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            8,
            sizeof(int),
            &h_workgrouphashBucketSizes);  //number of hash buckets per workgroup
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (&jobSpec->estimatedRecords)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            9,
            sizeof(int),
            &jobSpec->estimatedRecords);  //number of hash entries
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (&jobSpec->estimatedRecords)"))
        return SDK_FAILURE;



    status = clSetKernelArg(
            ker,
            10,
            sizeof(int),
            &numHashTables);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (numHashTables)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            11,
            sizeof(cl_uint),
            NULL);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (localKeySizesPerWorkgroup)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            12,
            sizeof(cl_uint),
            NULL);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (localKeySizesPerWorkgroup)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            13,
            sizeof(cl_uint),
            NULL);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (localKeySizesPerWorkgroup)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            14,
            sizeof(cl_mem),
            &d_keyOffsetPerWG);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_psKeySizes)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            15,
            sizeof(cl_mem),
            &d_valOffsetPerWG);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_psValSizes)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            16,
            sizeof(cl_mem),
            &d_numPairsOffsetPerWG);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_psCounts)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            ker,
            17,
            sizeof(cl_uint),
            &mapWavefrontsPerGroup);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_hashBucketIsBaseForReduceExtra)"))
        return SDK_FAILURE;


    if (overflowOccurs == 1)
    {

        status = clSetKernelArg(
                ker,
                18,
                sizeof(cl_mem),
                &d_outputKeysExtra);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. d_outputKeysExtra"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                19,
                sizeof(cl_mem),
                &d_outputValsExtra);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. outputValsExtra"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                20,
                sizeof(cl_mem),
                &d_hashTableExtra);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. hashTableExtra"))
            return SDK_FAILURE;
        status = clSetKernelArg(
                ker,
                21,
                sizeof(cl_mem),
                &d_hashBucketExtra);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. hashBucketExtra"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                ker,
                22,
                sizeof(cl_mem),
                &d_hashBucketOffsetsExtra);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. hashBucketOffsetsExtra"))
            return SDK_FAILURE;
    }

    status = clEnqueueNDRangeKernel(
            commandQueue,
            ker,
            1,
            NULL,
            globalThreads,
            localThreads,
            0,
            NULL,
            &events[0]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clEnqueueNDRangeKernel failed."))
        return SDK_FAILURE;


    timerEnd();
    printf("Copyer Initialization: %.3f ms\n",elapsedTime());

    //wait for the kernel call to finish execution
    status = clFinish(commandQueue);
    if(!checkVal(status,
                CL_SUCCESS,
                "clFinish failed."))
    {
        return SDK_FAILURE;
    }
    clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,NULL);
    clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,NULL);
    diff=endTime-startTime;
    printf("copyer Kernel spent **: %lu ns",diff);

    printFinalOutput(5,0,0,0,0,NULL);
    //if (overflowOccurs == 1) printFinalOutput( 4, KeysDiff, ValsDiff, sizeof(cl_uint4) *totalHashSizeExtra, sizeof(cl_uint4) *CountsDiff, h_interAllOffsetSizes );
    printFinalOutput(2,0,0,0,0,NULL);

    //----------------------------------------------
    //16, free allocated memory
    //----------------------------------------------
    printf("\n(16)- Free Allocated Memory:\n");
    printf("-----------------------------\n");


    clReleaseMemObject(d_keySizePerWG);
    clReleaseMemObject(d_valSizePerWG);
    clReleaseMemObject(d_numPairsPerWG);

    return 0;
}

//--------------------------------------------------------------------------
//Delegate the application to the appropriate Map and Reduce implementations
//--------------------------------------------------------------------------
int MapReduce::startMapReduce()
{
    int ret;

    //Create input and constant datasets Buffers
    cl_int status;
    d_inputDataSet = clCreateBuffer(context,
            CL_MEM_READ_ONLY| CL_MEM_USE_HOST_PTR,
            jobSpec->inputDataSetSize,
            jobSpec->inputDataSet,
            &status);
    if(status != CL_SUCCESS)
    {
        printf("clCreateBuffer failed. (d_filebuf)\n");
        exit(-1);
    }
    else
    {
        printf("clCreateBuffer succeed. File is on device now\n");
    }

    //Constants
    if (jobSpec->constantDataSize > 0)
    {
        d_constantData = clCreateBuffer(context,
                CL_MEM_READ_ONLY| CL_MEM_USE_HOST_PTR,
                jobSpec->constantDataSize,
                jobSpec->constantData,
                &status);
    }
    else
    {
        d_constantData = clCreateBuffer(context,
                CL_MEM_READ_ONLY,
                sizeof (cl_int),
                NULL,
                &status);
    }
    if(status != CL_SUCCESS)
    {
        printf("clCreateBuffer failed. (d_constantData)\n");
        exit(-1);
    }
    else
    {
        printf("clCreateBuffer succeed. File is on device now\n");
    }


    if (jobSpec->workflow == MAP_ONLY)
    {
        ret=startMapType1();

        if (ret != 0)
        {
            printf("Error in Map phase");
            return -1;
        }
    }
    else  //MAP_REDUCE
    {
        if (jobSpec->fixedSizeValue)
        {

            ret=startMapType2();

            if (ret != 0)
            {
                printf("Error in Map phase");
                return -1;	
            }
            ret=startReduce();

            if (ret != 0)
            {
                printf("Error in Reduce phase");
                return -1;
            }
        }
        else
        {
            //There should be another implementation for map and reduce phases to support these applications
            printf("Un-supported application!!");
            return -1;
        }
    }	
    return 0;
}

/*
   Move through the hash table and print all output key/value pairs
   extended=0 : Basic Map output
   extended=1 : Overflowed Map output
   extended=4: Overflowed Reduce output
   */
int MapReduce::printFinalOutput(uint stage, uint keysSizes, uint valsSizes, uint hashTablesSizes, uint hashBucketSizes, cl_int * h_hashBucketOffsets)
{
    int status;
    cl_event events[3];
    cl_int4 *htables;
    cl_int4* hbuckets= NULL;
    cl_int* hIsBaseForReduce= NULL;

    //Print final contiguous output 
    if (stage  == 5)
    {
        timerStart();
        jobSpec->outputKeys= (cl_char*)malloc(contiguousKeysSize);
        status = clEnqueueReadBuffer(commandQueue,
                d_contiguousOutputKeys,
                1,
                0,
                contiguousKeysSize,
                jobSpec->outputKeys,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_contiguousOutputKeys)"))
            return SDK_FAILURE;

        jobSpec->outputVals= (cl_char*)malloc(contiguousValsSize);
        status = clEnqueueReadBuffer(commandQueue,
                d_contiguousOutputVals,
                1,
                0,
                contiguousValsSize,
                jobSpec->outputVals,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_contiguousOutputVals)"))
            return SDK_FAILURE;

        jobSpec->outputOffsetSizes = (cl_uint4*)malloc(contiguousOffsets * sizeof (cl_uint4));
        status = clEnqueueReadBuffer(commandQueue,
                d_outputKeyValOffsets,
                1,
                0,
                contiguousOffsets * sizeof (cl_uint4),
                jobSpec->outputOffsetSizes,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_contiguousOutputVals)"))

            return SDK_FAILURE;

        timerEnd();
        printf("Contiguous output reading time: %.3f ms\n",elapsedTime());
        //int * keys=(int*)jobSpec->outputKeys;
        int * values=(int*)jobSpec->outputVals;
        //float * values=(float*)jobSpec->outputVals;
        char* s;

        for (int i=0 ; i < contiguousOffsets; i++)
        {
            //printf("%i %i %i %i\n",jobSpec->outputOffsetSizes[i].x, jobSpec->outputOffsetSizes[i].y,jobSpec->outputOffsetSizes[i].z,jobSpec->outputOffsetSizes[i].w);
            s=(char*)jobSpec->outputKeys+jobSpec->outputOffsetSizes[i].x;
            //printf("%i = ( %i,%i,%i) \n",keys[jobSpec->outputOffsetSizes[i].x/4],values[jobSpec->outputOffsetSizes[i].z/4],  values[jobSpec->outputOffsetSizes[i].z/4+1],values[jobSpec->outputOffsetSizes[i].z/4+2]);
            //printf("%i  %i \n",keys[i] , values[i]);
            printf("%s %i\n",s,values[jobSpec->outputOffsetSizes[i].z/4]);
            //printf("%f =( %i, %i ) \n", values[jobSpec->outputOffsetSizes[i].z/4],keys[jobSpec->outputOffsetSizes[i].x/4],keys[jobSpec->outputOffsetSizes[i].x/4+1]);
        }

        return 0;

    }

    printf ("numGroups: %i \n", numGroups);

    timerStart();

    cl_uint * h_gHashBucketSize =(cl_uint*)malloc(numGroups*sizeof(cl_uint)*3);
    status = clEnqueueReadBuffer(commandQueue,
            d_gHashBucketSize,
            1,
            0,
            numGroups*sizeof(cl_uint)* 3,
            h_gHashBucketSize,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (h_gHashBucketSize)"))
        return SDK_FAILURE;

    h_outputKeySizes=(cl_uint*)malloc(numGroups * sizeof(cl_uint)*3);
    status = clEnqueueReadBuffer(commandQueue,
            d_gOutputKeySize,
            1,
            0,
            numGroups*sizeof(cl_uint)*3,
            h_outputKeySizes,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (h_psCounts)"))
        return SDK_FAILURE;

    h_outputValSizes=(cl_uint*)malloc(numGroups * sizeof(cl_uint)*3);
    status = clEnqueueReadBuffer(commandQueue,
            d_gOutputValSize,
            1,
            0,
            numGroups*sizeof(cl_uint)*3,
            h_outputValSizes,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (h_psCounts)"))
        return SDK_FAILURE;

    if ((stage == 0)|| (stage == 2))  //No overflow
    {

        printf("0 and 2\n");
        jobSpec->outputKeys= (cl_char*)malloc(h_estimatedOutputKeySize);
        status = clEnqueueReadBuffer(commandQueue,
                d_outputKeys,
                1,
                0,
                h_estimatedOutputKeySize,
                jobSpec->outputKeys,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_outputKeys)"))
            return SDK_FAILURE;
        jobSpec->outputVals= (cl_char*)malloc(h_estimatedOutputValSize);
        status = clEnqueueReadBuffer(commandQueue,
                d_outputVals,
                1,
                0,
                h_estimatedOutputValSize,
                jobSpec->outputVals,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_interVals)"))
            return SDK_FAILURE;

        htables= (cl_int4*)malloc(sizeof(cl_int4) * totalHashSize);
        status = clEnqueueReadBuffer(commandQueue,
                d_hashTable,
                1,
                0,
                sizeof(cl_int4) * totalHashSize,
                htables,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. hashtables)"))
            return SDK_FAILURE;

        hbuckets= (cl_int4*)malloc( sizeof(cl_int4) * totalHashBucketSize);
        status = clEnqueueReadBuffer(commandQueue,
                d_hashBucket,
                1,
                0,
                sizeof(cl_int4) * totalHashBucketSize,
                hbuckets,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. hashbuckets)"))
            return SDK_FAILURE;
        /*
           if (stage == 2)
           {
           hIsBaseForReduce= (cl_int*)malloc(sizeof(cl_int) * totalHashBucketSize);
           status = clEnqueueReadBuffer(commandQueue,
           d_hashBucketIsBaseForReduce,
           1,
           0,
           sizeof(cl_int) * totalHashBucketSize,
           hIsBaseForReduce,
           0,
           0,
           &events[0]);
           if(!checkVal(
           status,
           CL_SUCCESS,
           "clReadBuffer failed. hIsBaseForReduce"))
           return SDK_FAILURE;

           }
           */
    }
    else     //Overflow
    {
        printf("1 and 4\n");
        jobSpec->outputVals= (cl_char*)malloc(keysSizes);
        status = clEnqueueReadBuffer(commandQueue,
                d_outputValsExtra,
                1,
                0,
                keysSizes,
                jobSpec->outputVals,
                0,
                0,
                &events[2]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_outputValsExtra)"))
            return SDK_FAILURE;

        jobSpec->outputKeys= (cl_char*)malloc(valsSizes);
        status = clEnqueueReadBuffer(commandQueue,
                d_outputKeysExtra,
                1,
                0,
                valsSizes,
                jobSpec->outputKeys,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_outputKeys)"))
            return SDK_FAILURE;

        htables= (cl_int4*)malloc(hashTablesSizes);
        status = clEnqueueReadBuffer(commandQueue,
                d_hashTableExtra,
                1,
                0,
                hashTablesSizes,
                htables,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. hashtables)"))
            return SDK_FAILURE;

        hbuckets= (cl_int4*)malloc(hashBucketSizes);
        status = clEnqueueReadBuffer(commandQueue,
                d_hashBucketExtra,
                1,
                0,
                hashBucketSizes,
                hbuckets,
                0,
                0,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. hashbuckets)"))
            return SDK_FAILURE;

        if (stage == 4)
        {

            printf("Number of hash buckets elements: %i\n",hashBucketSizes/4);
            /*	hIsBaseForReduce= (cl_int*)malloc(hashBucketSizes/4);
                status = clEnqueueReadBuffer(commandQueue,
                d_hashBucketIsBaseForReduceExtra,
                1,
                0,
                hashBucketSizes/4,
                hIsBaseForReduce,
                0,
                0,
                &events[0]);
                if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. hIsBaseForReduce"))
                return SDK_FAILURE;
                */
        }
    }

    timerEnd();
    printf("Read Reducer Output: %.3f ms\n",elapsedTime());

    uint extraHashedNeeded = 0 ;
    uint maxHashEntriesNeeded=0;
    int it;
    if ((stage == 1) || (stage == 4) )
        it = overflowedWorkGroups;
    else
        it = numWavefrontsPerGroup * numGroups;

    printf("Number of iterations : %i",it);
    timerStart();
    int count =0 ;
    cl_int * ok = (cl_int*)jobSpec->outputKeys;
    cl_int * o = (cl_int * )jobSpec->outputVals;
    for (int i= 0 ; i< it; i++)
    {
        //printf ("hashentries of wavefront %i  !!!!\n",i);

        for (int j=0; j < jobSpec->estimatedRecords; j++)
        {
            cl_int4 hashentry = htables[i*jobSpec->estimatedRecords + j];

            if ( hashentry.z <= 0) 
            {
                continue;
            }
            else
            {
                //printf("(%i- %i -%i- %i)   ",hashentry.x, hashentry.y, hashentry.z, hashentry.w);
            }

            if (hashentry.y < 0 )
            {
                //printf("y is negative\n");
                continue;
            }
            if (hashentry.x == -1) continue;

            cl_int4 hashbucket = hbuckets [ hashentry.x];
            //printf("(%i- %i -%i- %i)   ",hashbucket.x, hashbucket.y, hashbucket.z, hashbucket.w);

            if (hashbucket.y < 0 ) 
            {
                //printf("y is negative\n");
                continue;
            }
            //for (int k=0; k <  hashbucket.y; k++)
            //        printf("%c",jobSpec->outputKeys[hashbucket.x + k]);
            //printf("%i",ok[hashbucket.x/4]);

            //printf("  (%i,%i,%i ) ||",o[hashbucket.z/4],o[hashbucket.z/4+1],o[hashbucket.z/4+2]);//(extended == 2)||(extended == 4)? hbase: -1) ;
            //printf("  %i ||",o[hashbucket.z/4]);

            int next= hashbucket.w;
            int count= hashentry.z;

            //printf(" next: %i count: %i", next, count );

            if ( count > 1 )
            {
                for (int k=0; k < count-1; k++)
                {
                    hashbucket = hbuckets [  next];
                    //printf("(%i- %i -%i- %i)   ",hashbucket.x, hashbucket.y, hashbucket.z, hashbucket.w);
                    if (hashbucket.y < 0 ) 
                    {
                        //printf("y is negative\n");
                        break;
                    }
                    //for (int z=0; z <  hashbucket.y; z++)
                    //	printf("%c",jobSpec->outputKeys[hashbucket.x + z]);
                    //printf("%i",ok[hashbucket.x/4]);			

                    //printf("  (%i,%i,%i ) ||",o[hashbucket.z/4],o[hashbucket.z/4+1],o[hashbucket.z/4+2]);
                    //printf("  %i||",o[hashbucket.z/4]);

                    next= hashbucket.w;
                }
            }

            //if ( hashbucket.w != -1) 
            //printf("\n********************Error ( -1) ***********************\n");

            //printf("\n");

        }
        //printf("----------------------------------\n");
    }
    timerEnd();
    printf("Retrieval time: %.3f ms\n",elapsedTime());
    if (htables) free(htables);
    if (hbuckets) free(hbuckets);
    if (hIsBaseForReduce) free(hIsBaseForReduce);
    return 0;
}

/******************************************
 * Add record to the input key/value pairs.
 * Called by a job to prepare input dataset
 ******************************************/
void MapReduce::AddMapInputRecord(
        void*		key,
        void*		val,
        cl_int		keySize,
        cl_int		valSize)
{
    static cl_uint curOffset[2];  //key, value
    static cl_uint curChunkNum[3]; //key, value, offset

    int index = jobSpec->inputRecordCount;
    //printf("key:%s value:%s\n",key,val);
    //printf("inputRecordCount:%f\n",jobSpec->inputRecordCount);
    const int dataChunkSize = 1024*1024*256; 

    if (jobSpec->inputRecordCount > 0)
    {
        if (dataChunkSize*curChunkNum[0] < (curOffset[0] + keySize))
            jobSpec->inputKeys = (cl_char*)realloc(jobSpec->inputKeys, (++curChunkNum[0])*dataChunkSize);
        memcpy(jobSpec->inputKeys+curOffset[0], key, keySize);

        if (dataChunkSize*curChunkNum[1] < (curOffset[1] + valSize))
            jobSpec->inputVals = (cl_char*)realloc(jobSpec->inputVals, (++curChunkNum[1])*dataChunkSize);
        memcpy(jobSpec->inputVals+curOffset[1], val, valSize);

        if (dataChunkSize*curChunkNum[2] < (jobSpec->inputRecordCount+1)*sizeof(cl_int4))
            jobSpec->inputOffsetSizes = (cl_uint4*)realloc(jobSpec->inputOffsetSizes,
                    (++curChunkNum[2])*dataChunkSize);

        //printf("Done REAllocation \n");
        jobSpec->inputKeysBufSize+=keySize;
        jobSpec->inputValsBufSize+=valSize;

    }
    else
    {
        jobSpec->inputKeys = (cl_char*)malloc(dataChunkSize);
        if (NULL == jobSpec->inputKeys) exit(-1);
        memcpy(jobSpec->inputKeys, key, keySize);
        //printf("inputKeys buffer allocated\n");

        jobSpec->inputVals = (cl_char*)malloc(dataChunkSize);
        if (NULL == jobSpec->inputVals) exit(-1);
        memcpy(jobSpec->inputVals, val, valSize);
        //printf("inputVals buffer allocated\n");

        jobSpec->inputOffsetSizes = (cl_uint4*)memalign(16, dataChunkSize);
        //printf("inputOffsetSizes buffer allocated\n");

        //printf("Done Allocation \n");
        //printf("keySize:%i valueSize:%i\n",keySize,valSize);

        jobSpec->inputKeysBufSize=keySize;
        jobSpec->inputValsBufSize=valSize;

        curChunkNum[0]++;
        curChunkNum[1]++;
        curChunkNum[2]++;
    }

    jobSpec->inputOffsetSizes[index]= (cl_uint4){curOffset[0], keySize, curOffset[1], valSize};
    curOffset[0]+= keySize;
    curOffset[1]+= valSize;
    jobSpec->inputRecordCount++;
}

//--------------------------------------------------------------------------------------------
//Start Map phase for applications with MapOnly phase and with Overflow or without an overflow
//--------------------------------------------------------------------------------------------
int MapReduce::startMapType1()
{

    cl_int status;

    timerStart();	

    if (!jobSpec->Validate()) return -1;

    //1- Get map input data on host
    //------------------------------------------------------------------
    printf("\n(1)- Get map input data on host:\n");
    printf("----------------------------------\n");
    int	     h_inputRecordCount = jobSpec->inputRecordCount;
    int	     h_inputKeysBufSize = jobSpec->inputKeysBufSize;
    int	     h_inputValsBufSize = jobSpec->inputValsBufSize;
    cl_char*	 h_inputKeys = jobSpec->inputKeys;
    cl_char*	 h_inputVals = jobSpec->inputVals;
    cl_uint4* h_inputOffsetSizes = jobSpec->inputOffsetSizes;

    if ((h_inputRecordCount>=512)&&(h_inputRecordCount% 512 !=0))
    {
        printf("Error (Can not proceed): Input records number should be divisable by 512.");
        return -1;
    }
    printf(" Map Input: keys size: %i bytes, values size: %i bytes, records: %i \n",h_inputKeysBufSize,h_inputValsBufSize,h_inputRecordCount);

    //2- Upload map input data to device memory
    //------------------------------------------------------------------
    printf("\n(2)- Upload map input data to device memory:\n");
    printf("----------------------------------------------\n");
    cl_mem 	d_inputKeys = NULL;
    cl_mem  d_inputVals = NULL;
    cl_mem  d_inputOffsetSizes = NULL;

    d_inputKeys = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            h_inputKeysBufSize,
            h_inputKeys,
            &status);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_inputKeys)"))
        return SDK_FAILURE;

    d_inputVals = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            h_inputValsBufSize,
            h_inputVals,
            &status);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_inputVals)"))
        return SDK_FAILURE;

    d_inputOffsetSizes = clCreateBuffer(context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            sizeof(cl_int4)*h_inputRecordCount,
            h_inputOffsetSizes,
            &status);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_inputOffsetSizes)"))
        return SDK_FAILURE;

    //3- Determine the block size
    //------------------------------------------------------------------
    printf("\n(3)- Determine the block size:\n");
    printf("--------------------------------\n");
    int h_recordsPerTask = jobSpec->numRecTaskMap;
    int h_actualNumThreads=CEIL(h_inputRecordCount, h_recordsPerTask);
    size_t globalThreads[1]= {h_actualNumThreads};
    groupSize=jobSpec->userSize;
    printf("workgroup Size: %zu \n",groupSize);
    size_t localThreads[1] = {groupSize};
    int numGroups=h_actualNumThreads/groupSize;

    // 6- Allocate intermediate memory on device memory
    //-----------------------------------------------
    printf("\n(6)- Allocate intermediate memory on device memory:\n");
    printf("---------------------------------------------------------------------\n");
    cl_mem	d_interKeys = NULL;
    cl_mem	d_interVals = NULL;
    cl_mem	d_interOffsetSizes = NULL;
    cl_mem  d_gKeySizes=NULL;
    cl_mem  d_gValSizes=NULL;
    cl_mem  d_gCounts=NULL;
    cl_int  *h_interAllKeys,*h_interAllVals,*h_interAllOffsetSizes;

    uint h_estimatedKeySize,h_estimatedValSize,h_estimatCounts;
    h_estimatedKeySize = (jobSpec->overflow)? jobSpec->estimatedRecords*jobSpec->estimatedKeySize*5 : jobSpec->estimatedRecords*jobSpec->estimatedKeySize;
    h_estimatedValSize = (jobSpec->overflow)? jobSpec->estimatedRecords*jobSpec->estimatedValSize*5 : jobSpec->estimatedRecords*jobSpec->estimatedValSize;
    h_estimatCounts = (jobSpec->overflow)? jobSpec->estimatedRecords*5:jobSpec->estimatedRecords; //70

    d_interKeys = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            h_estimatedKeySize,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_interKeys)"))
        return SDK_FAILURE;
    d_interVals = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            h_estimatedValSize,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_interVals)"))
        return SDK_FAILURE;
    d_interOffsetSizes = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            sizeof(cl_uint4)*h_estimatCounts,
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_interOffsetSizes)"))
        return SDK_FAILURE;

    //Create number of offsets equal to the number of workgroups
    //for each one to work independently from the others
    cl_int h_workgroupKeySizes,h_workgroupValSizes,h_workgroupOffsetsizes;
    printf("h_estimatedKeySize %i h_estimatedValSize %i h_estimatCounts %i\n",h_estimatedKeySize,h_estimatedValSize, h_estimatCounts);
    h_workgroupKeySizes=h_estimatedKeySize/numGroups;
    h_workgroupValSizes=h_estimatedValSize/numGroups;
    h_workgroupOffsetsizes=h_estimatCounts/numGroups;

    h_interAllKeys=(cl_int*)malloc(sizeof(cl_int)*numGroups);
    h_interAllVals=(cl_int*)malloc(sizeof(cl_int)*numGroups);
    h_interAllOffsetSizes=(cl_int*)malloc(sizeof(cl_int)*numGroups);

    for (int i=0; i< numGroups; i++)
    {
        h_interAllKeys[i]=i*h_workgroupKeySizes;
        h_interAllVals[i]=i*h_workgroupValSizes;
        h_interAllOffsetSizes[i]=i*h_workgroupOffsetsizes;
        //printf("offsets group %i: %i - %i - %i\n",i,h_interAllKeys[i],h_interAllVals[i], h_interAllOffsetSizes[i]);
    } 	


    cl_mem  d_interKeysOffsets = clCreateBuffer(context,
            CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
            sizeof(cl_int)*numGroups,
            h_interAllKeys,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_interAllKeys)"))
        return SDK_FAILURE;
    cl_mem  d_interValsOffsets = clCreateBuffer(context,
            CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
            sizeof(cl_int)*numGroups,
            h_interAllVals,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_interAllVals)"))
        return SDK_FAILURE;
    cl_mem  d_interOffsetSizesOffsets = clCreateBuffer(context,
            CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
            sizeof(cl_int)*numGroups,
            h_interAllOffsetSizes,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_interAllKeys)"))
        return SDK_FAILURE;

    d_gKeySizes = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            2 * numGroups*sizeof(cl_uint),
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_gKeySizes)"))
        return SDK_FAILURE;

    d_gValSizes = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            2 * numGroups*sizeof(cl_uint),
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_gValSizes)"))
        return SDK_FAILURE;

    d_gCounts = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            2 * numGroups*sizeof(cl_uint),
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_gCounts) "))
        return SDK_FAILURE;

    cl_mem d_inputRecordsMeta = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            h_inputRecordCount * sizeof(cl_uint4),
            NULL,
            &status);
    if(!checkVal(status,
                CL_SUCCESS,
                "clCreateBuffer failed. (d_gCounts) "))
        return SDK_FAILURE;
    //Start Map
    //---------------------------------------------------------------------
    printf("\n(7)- Start Map:\n");
    printf("-----------------\n");
    cl_uint *h_gKeySizes,*h_gValSizes,*h_gCounts;
    cl_event events[3];

    //Set Kernel Arguments
    status = clSetKernelArg(
            mapperKernel,
            0,
            sizeof(cl_mem),
            (void*)&(d_inputDataSet));
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (inputDataSet)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            1,
            sizeof(cl_mem),
            (void*)&(d_constantData));
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (constantDataSet)"))
        return SDK_FAILURE;


    status = clSetKernelArg(
            mapperKernel,
            2,
            sizeof(cl_mem),
            &d_inputKeys);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_inputKeys)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            3,
            sizeof(cl_mem),
            &d_inputVals);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_inputVals)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            4,
            sizeof(cl_mem),
            &d_inputOffsetSizes);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_inputOffsetSizes)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            5,
            sizeof(cl_mem),
            &d_interKeys);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_interKeys)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            6,
            sizeof(cl_mem),
            &d_interVals);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_interVals)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            7,
            sizeof(cl_mem),
            &d_interOffsetSizes);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (7-d_interOffsetSizes)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            8,
            sizeof(cl_mem),
            &d_interKeysOffsets);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_interKeysOffsets)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            9,
            sizeof(cl_mem),
            &d_interValsOffsets);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_interValsOffsets)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            10,
            sizeof(cl_mem),
            &d_interOffsetSizesOffsets);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (10 - d_interOffsetSizesOffsets)"))
        return SDK_FAILURE;


    status = clSetKernelArg(
            mapperKernel,
            11,
            sizeof(int),
            &h_inputRecordCount);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_inputRecordCount)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            12,
            sizeof(int),
            &h_recordsPerTask);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_recordsPerTask)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            13,
            sizeof(int),
            &h_actualNumThreads);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_actualNumThreads)"))
        return SDK_FAILURE;

    //Local memory
    status = clSetKernelArg(
            mapperKernel,
            14,
            2*sizeof(cl_uint),
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            15,
            2*sizeof(cl_uint),
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            16,
            2*sizeof(cl_uint),
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            17,
            sizeof(cl_uint),
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            18,
            sizeof(cl_uint),
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            19,
            sizeof(cl_uint),
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            20,
            sizeof(cl_uint),
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            21,
            sizeof(cl_uint),
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            22,
            sizeof(cl_uint),
            NULL);
    if(!checkVal(status,
                CL_SUCCESS,
                "clSetKernelArg failed. (local memory)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            23,
            sizeof(cl_mem),
            &d_gKeySizes);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_gKeySizes)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            24,
            sizeof(cl_mem),
            &d_gValSizes);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_gValSizes)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            25,
            sizeof(cl_mem),
            &d_gCounts);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_gCounts)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            26,
            sizeof(int),
            &h_workgroupKeySizes);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_actualNumThreads)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            27,
            sizeof(int),
            &h_workgroupValSizes);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_actualNumThreads)"))
        return SDK_FAILURE;

    status = clSetKernelArg(
            mapperKernel,
            28,
            sizeof(int),
            &h_workgroupOffsetsizes);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_actualNumThreads)"))
        return SDK_FAILURE;



    status = clSetKernelArg(
            mapperKernel,
            29,
            sizeof(cl_mem),
            &d_inputRecordsMeta);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (d_gCounts)"))
        return SDK_FAILURE;

    cl_uint extendedKernel = 0;
    status = clSetKernelArg(
            mapperKernel,
            30,
            sizeof(uint),
            &extendedKernel);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clSetKernelArg failed. (h_actualNumThreads)"))
        return SDK_FAILURE;

    //Enqueue a kernel run call
    printf("Before Run Kernel ...\n");
    status = clEnqueueNDRangeKernel(
            commandQueue,
            mapperKernel,
            1,
            NULL,
            globalThreads,
            localThreads,
            0,
            NULL,
            &events[0]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clEnqueueNDRangeKernel failed."))
        return SDK_FAILURE;

    printf("After Run Kernel ...\n");
    timerEnd();
    printf("Mapper Initialization: %.3f ms\n",elapsedTime());

    //wait for the kernel call to finish execution
    status = clFinish(commandQueue);
    if(!checkVal(status,
                CL_SUCCESS,
                "clFinish failed."))
    {
        return SDK_FAILURE;
    }
    cl_ulong startTime,endTime,diff;
    clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,NULL);
    clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,NULL);
    diff=endTime-startTime;
    printf("Mapper Kernel spent: %lu ns",diff);


    //Read Output
    timerStart();
    h_gKeySizes= (cl_uint*)malloc(2 * numGroups * sizeof(cl_uint));
    status = clEnqueueReadBuffer(commandQueue,
            d_gKeySizes,
            1,
            0,
            2 * numGroups *sizeof(cl_uint),
            h_gKeySizes,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (h_psKeySizes)"))
        return SDK_FAILURE;

    h_gValSizes= (cl_uint*)malloc(2 * numGroups * sizeof(cl_uint));
    status = clEnqueueReadBuffer(commandQueue,
            d_gValSizes,
            1,
            0,
            2 * numGroups*sizeof(cl_uint),
            h_gValSizes,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (h_psValSizes)"))
        return SDK_FAILURE;

    h_gCounts=(cl_uint*)malloc(2 * numGroups * sizeof(cl_uint));
    status = clEnqueueReadBuffer(commandQueue,
            d_gCounts,
            1,
            0,
            2 * numGroups*sizeof(cl_uint),
            h_gCounts,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (h_psCounts)"))
        return SDK_FAILURE;


    jobSpec->interKeys= (cl_char*)malloc(h_estimatedKeySize);
    status = clEnqueueReadBuffer(commandQueue,
            d_interKeys,
            1,
            0,
            h_estimatedKeySize,
            jobSpec->interKeys,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (d_interKeys)"))
        return SDK_FAILURE;

    jobSpec->interVals= (cl_char*)malloc(h_estimatedValSize);
    status = clEnqueueReadBuffer(commandQueue,
            d_interVals,
            1,
            0,
            h_estimatedValSize,
            jobSpec->interVals,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (d_interVals)"))
        return SDK_FAILURE;

    jobSpec->interOffsetSizes= (cl_uint4*)malloc(h_estimatCounts*sizeof(cl_uint4));
    status = clEnqueueReadBuffer(commandQueue,
            d_interOffsetSizes,
            1,
            0,
            h_estimatCounts*sizeof(cl_uint4),
            jobSpec->interOffsetSizes,
            0,
            0,
            &events[2]);
    if(!checkVal(
                status,
                CL_SUCCESS,
                "clReadBuffer failed. (d_interOffsets)"))
        return SDK_FAILURE;


    timerEnd();
    printf("Mapper Read Output: %.3f ms\n",elapsedTime());

    timerStart();
    cl_int allKeys = 0 , allVals = 0 ,allCounts = 0, KeysDiff = 0, ValsDiff  = 0, CountsDiff = 0;
    cl_int allKeysUsed = 0  , allValsUsed = 0 ,allCountsUsed = 0;
    cl_int overflowedWorkGroups = 0;
    cl_int * overflowWGId; // Ids of overflowWG
    // Used to force only overflowed workgroups to continue their work
    overflowWGId = (cl_int *) malloc(sizeof(cl_int) * numGroups);
    printf("Num of groups : %i\n",numGroups);
    for (int i=0; i< numGroups; i++)
    {
        allKeys += h_gKeySizes[i];
        allVals += h_gValSizes[i];
        allCounts += h_gCounts[i];

        allKeysUsed += h_gKeySizes[numGroups+i];
        allValsUsed += h_gValSizes[numGroups+i];
        allCountsUsed += h_gCounts[numGroups+i];


        if ((h_gCounts[i] > h_workgroupOffsetsizes ) || ( h_gKeySizes[i] > h_workgroupKeySizes) ||( h_gValSizes[i] > h_workgroupValSizes) ){
            printf("Overflow!!!! WG ID: %i\n", i);
            printf("offsets: %i Keys: %i Vals: %i \n", CountsDiff, KeysDiff, ValsDiff);

            printf("Actual offsets: %i Keys: %i Vals: %i \n", h_gCounts[i], h_gKeySizes[i], h_gValSizes[i]);
            printf("Allocated offsets: %i Keys: %i Vals: %i \n",h_workgroupOffsetsizes, h_workgroupKeySizes, h_workgroupValSizes);
            overflowWGId[overflowedWorkGroups] = i;
            h_interAllKeys[overflowedWorkGroups]= KeysDiff;
            h_interAllVals[overflowedWorkGroups]= ValsDiff;
            h_interAllOffsetSizes[overflowedWorkGroups]= CountsDiff;
            overflowedWorkGroups++;
        }
        KeysDiff = allKeys - allKeysUsed;
        ValsDiff = allVals - allValsUsed;
        CountsDiff = allCounts - allCountsUsed;

        //printf("Output records of group%i: %i and maximum limit: %i\n",i,h_gCounts[i],h_workgroupOffsetsizes);
    }

    //printf("offsets: %i Keys: %i Vals: %i \n", CountsDiff, KeysDiff, ValsDiff);
    printf("Output records: %i\n",allCountsUsed);
    jobSpec->interRecordCount = allCountsUsed; 
    jobSpec->interDiffKeyCount = CountsDiff; 
    jobSpec->interAllKeySize = allKeys;
    jobSpec->interAllValSize = allVals;

    timerEnd();
    printf("Mapper pre overflow: %.3f ms\n",elapsedTime());

    printf( "MapOutput Records: %u \n",static_cast<unsigned int>(jobSpec->interRecordCount));
    cl_int * keys=(cl_int *)jobSpec->interKeys;
    cl_int * values=(cl_int *)jobSpec->interVals;
    cl_uint4 * offsets=(cl_uint4 *)jobSpec->interOffsetSizes;
    /*
       for (int i=0;i < h_estimatCounts; i++)
       {
       printf("offset: %i size: %i \n",keys[i],values[i]);
       }
       */
    printf("End \n");



    if ( (KeysDiff > 0) ||(ValsDiff > 0)||(CountsDiff > 0))
    {
        printf("overflowed WorkGroups: %i",overflowedWorkGroups);
        timerStart();
        //Overflow Handling
        cl_mem  d_overflowWGId = clCreateBuffer(context,
                CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                numGroups*sizeof(cl_uint),
                overflowWGId,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_gCounts) "))
            return SDK_FAILURE;

        d_interKeys = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                KeysDiff,
                NULL,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_interKeys)"))
            return SDK_FAILURE;

        d_interVals = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                ValsDiff,
                NULL,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_interVals)"))
            return SDK_FAILURE;

        d_interOffsetSizes = clCreateBuffer(context,
                CL_MEM_READ_WRITE,
                sizeof(cl_uint4)* CountsDiff,
                NULL,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_interoffsetSizes)"))
            return SDK_FAILURE;

        cl_mem  d_interKeysOffsets = clCreateBuffer(context,
                CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                sizeof(cl_int)*numGroups,
                h_interAllKeys,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_interAllKeys)"))
            return SDK_FAILURE;

        cl_mem  d_interValsOffsets = clCreateBuffer(context,
                CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                sizeof(cl_int)*numGroups,
                h_interAllVals,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_interAllVals)"))
            return SDK_FAILURE;

        cl_mem  d_interOffsetSizesOffsets = clCreateBuffer(context,
                CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                sizeof(cl_int)*numGroups,
                h_interAllOffsetSizes,
                &status);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clCreateBuffer failed. (d_interAllKeys)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                mapperKernel,
                5,
                sizeof(cl_mem),
                &d_interKeys);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_interKeys)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                mapperKernel,
                6,
                sizeof(cl_mem),
                &d_interVals);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_interVals)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                mapperKernel,
                7,
                sizeof(cl_mem),
                &d_interOffsetSizes);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_interOffsetSizes)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                mapperKernel,
                8,
                sizeof(cl_mem),
                &d_interKeysOffsets);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_interKeysOffsets)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                mapperKernel,
                9,
                sizeof(cl_mem),
                &d_interValsOffsets);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_interValsOffsets)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                mapperKernel,
                10,
                sizeof(cl_mem),
                &d_interOffsetSizesOffsets);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_interOffsetSizesOffsets)"))
            return SDK_FAILURE;

        status = clSetKernelArg(
                mapperKernel,
                25,
                sizeof(cl_mem),
                &d_overflowWGId);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (d_overflowWGId)"))
            return SDK_FAILURE;


        cl_uint extendedKernel = 1;
        status = clSetKernelArg(
                mapperKernel,
                30,
                sizeof(uint),
                &extendedKernel);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clSetKernelArg failed. (h_actualNumThreads)"))
            return SDK_FAILURE;

        size_t globalThreads[1]= {overflowedWorkGroups * groupSize};

        //Enqueue a kernel run call
        printf("Before Run Kernel ...\n");
        status = clEnqueueNDRangeKernel(
                commandQueue,
                mapperKernel,
                1,
                NULL,
                globalThreads,
                localThreads,
                0,
                NULL,
                &events[0]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clEnqueueNDRangeKernel failed."))
            return SDK_FAILURE;

        printf("After Run Kernel ...\n");
        timerEnd();
        printf("Second Mapper Initialization: %.3f ms\n",elapsedTime());

        //wait for the kernel call to finish execution
        status = clFinish(commandQueue);
        if(!checkVal(status,
                    CL_SUCCESS,
                    "clFinish failed."))
        {
            return SDK_FAILURE;
        }
        cl_ulong startTime,endTime,diff;
        clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&startTime,NULL);
        clGetEventProfilingInfo (events[0],CL_PROFILING_COMMAND_END,sizeof(cl_ulong),&endTime,NULL);
        diff=endTime-startTime;
        printf("Second Mapper Kernel spent: %lu ns",diff);


        timerStart();
        jobSpec->interKeysExtra= (cl_char*)malloc(KeysDiff);
        status = clEnqueueReadBuffer(commandQueue,
                d_interKeys,
                1,
                0,
                KeysDiff,
                jobSpec->interKeysExtra,
                0,
                0,
                &events[2]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_interKeys)"))
            return SDK_FAILURE;

        jobSpec->interValsExtra= (cl_char*)malloc(ValsDiff);
        status = clEnqueueReadBuffer(commandQueue,
                d_interVals,
                1,
                0,
                ValsDiff,
                jobSpec->interValsExtra,
                0,
                0,
                &events[2]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_interVals)"))
            return SDK_FAILURE;
        jobSpec->interOffsetSizesExtra= (cl_uint4*)malloc(CountsDiff*sizeof(cl_uint4));
        status = clEnqueueReadBuffer(commandQueue,
                d_interOffsetSizes,
                1,
                0,
                CountsDiff*sizeof(cl_uint4),
                jobSpec->interOffsetSizesExtra,
                0,
                0,
                &events[2]);
        if(!checkVal(
                    status,
                    CL_SUCCESS,
                    "clReadBuffer failed. (d_interOffsets)"))
            return SDK_FAILURE;

        timerEnd();
        printf("Second Mapper Copy output: %.3f ms\n",elapsedTime());

        /*
           keys=(cl_int *)jobSpec->interKeysExtra;
           values=(cl_int *)jobSpec->interValsExtra;
           cl_uint4 * offsets=(cl_uint4 *)jobSpec->interOffsetSizesExtra;

           printf(" Extra records:%i \n",CountsDiff);
           for (int i=0;i < CountsDiff; i++)
           {
           printf(" offset: %i size:%i \n",keys[i],values[i]);
           printf("%i- %i- %i- %i\n\n",offsets[i].x,offsets[i].y,offsets[i].z,offsets[i].w);
           }
           */
    }

    //9- Free allocated memory
    //----------------------------------------------
    printf("\n(8)- Free Allocated Memory:\n");
    printf("-----------------------------\n");

    clReleaseMemObject(d_inputKeys);
    clReleaseMemObject(d_inputVals);
    clReleaseMemObject(d_inputOffsetSizes);

    clReleaseMemObject(d_gKeySizes);
    clReleaseMemObject(d_gValSizes);
    clReleaseMemObject(d_gCounts);

    return 0;
}




int MapReduce::checkVal(cl_int& states, int check, const std::string *info) 
{
    if (states==check)
        return 1;
    else
    {
        std::cout << info << " Error: " << get_error_string(states) <<  std::endl;
        return 0;
    }

}
int MapReduce::checkVal(cl_int& states, int check, const char *info)
{
    if (states==check)
        return 1;
    else
    {
        std::cout << info << " Error: " << get_error_string(states) << std::endl;
        return 0;
    }
}
void MapReduce::error(const char* info)
{
    std::cout << info << std::endl;
}

int MapReduce::initialize()
{
    return 0;
}
    std::string
MapReduce::getPath()
{
#ifdef _WIN32
    char buffer[MAX_PATH];
#ifdef UNICODE
    if(!GetModuleFileName(NULL, (LPWCH)buffer, sizeof(buffer)))
        throw std::string("GetModuleFileName() failed!");
#else
    if(!GetModuleFileName(NULL, buffer, sizeof(buffer)))
        throw std::string("GetModuleFileName() failed!");
#endif
    std::string str(buffer);
    /* '\' == 92 */
    int last = (int)str.find_last_of((char)92);
#else
    char buffer[PATH_MAX + 1];
    ssize_t len;
    if((len = readlink("/proc/self/exe",buffer, sizeof(buffer) - 1)) == -1)
        throw std::string("readlink() failed!");
    buffer[len] = '\0';
    std::string str(buffer);
    /* '/' == 47 */
    int last = (int)str.find_last_of((char)47);
#endif
    return str.substr(0, last + 1);
}
