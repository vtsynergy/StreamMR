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
     courts in Austin, Texas, and all defenses are hereby waived concerning personal
     jurisdiction and venue of these courts.

     ============================================================ */


#ifndef MAPREDUCE_H_
#define MAPREDUCE_H_

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <unistd.h>
#define SDK_SUCCESS 0
#define SDK_FAILURE 1
#define CL_SUCCESS 0
#define CL_FAILURE 1
#define MAP_ONLY		0x01
#define MAP_GROUP		0x02
#define MAP_REDUCE		0x03
#define PATH_MAX 128
#define GROUP_SIZE 64


//------------------------------- JobSpecification Class ---------------------//
class JobSpecification
{


    public:
        //for input data on host
        cl_char*		inputKeys;
        cl_char*		inputVals;
        cl_uint4*               inputOffsetSizes;
        cl_float		inputRecordCount;
        cl_float 		inputKeysBufSize;
        cl_float 		inputValsBufSize;
        void* 		inputDataSet;
        void*           constantData;
        cl_int          inputDataSetSize;
        cl_int		constantDataSize;


        //for output data on host
        cl_char*		outputKeys;
        cl_char*		outputVals;
        cl_char* 		outputValsExtra;
        cl_uint4*               outputOffsetSizes;
        cl_int2*		outputKeyListRange;
        cl_int		        outputRecordCount;
        cl_float		outputAllKeySize;
        cl_float		outputAllValSize;
        cl_float		outputDiffKeyCount;



        //for intermediate data on host
        cl_char*		interKeys;
        cl_char*		interVals;
        cl_uint4*		interOffsetSizes;
        cl_int2*		interKeyListRange;
        cl_float		interRecordCount;
        cl_int  		interDiffKeyCount;
        cl_float		interAllKeySize;
        cl_float		interAllValSize;
        cl_char*		interKeysExtra;
        cl_char* 		interValsExtra;
        cl_uint4*		interOffsetSizesExtra;

        //user specification
        cl_char		workflow;
        cl_char		outputToHost;
        cl_int		numRecTaskMap;
        cl_int		numRecTaskReduce;
        cl_int 		userSize; //Workgroup size assigned by the user
        cl_uint estimatedRecords,estimatedInterRecords;
        cl_uint estimatedValSize, estimatedInterValSize;
        cl_uint estimatedKeySize, estimatedInterKeySize;

        bool LoadKeyVal;
        bool perfectHashing;
        bool outputIntermediate;
        bool overflow;
        bool fixedSizeValue;

        JobSpecification()
        {
            userSize = 64;
            numRecTaskReduce = 1;
            numRecTaskMap = 1;
            inputRecordCount = 0;
            workflow = MAP_REDUCE;
            perfectHashing = false;
            outputIntermediate = false;
            fixedSizeValue = true;
            estimatedInterKeySize = 1;
            estimatedInterValSize = 1;
            estimatedInterRecords = 1;
        }

        bool Validate()
        {
            if (inputKeys == NULL)
            {
                std::cout<<"Error: no any input keys";
                return false;
            }
            if (inputVals == NULL)
            {
                std::cout<<"Error: no any input values";
                return false;
            }
            if (inputOffsetSizes == NULL)
            {
                std::cout<<"Error: no any input pointer info";
                return false;
            }
            if (inputRecordCount == 0)
            {
                std::cout<<"Error: invalid input record count";
                return false;
            }
            return true;
        }
};

/**
 * MapReduce
 * Class implements OpenCL MapReduce
 * Derived from SDKSample base class
 */

class MapReduce
{
    cl_double setupTime;            /**< time taken to setup OpenCL resources and building kernel */
    cl_double kernelTime;           /**< time taken to run kernel and read result back */

    size_t maxWorkGroupSize;        /**< Max allowed work-items in a group */
    cl_uint maxDimensions;          /**< Max group dimensions allowed */
    size_t *maxWorkItemSizes;       /**< Max work-items sizes in each dimensions */
    cl_ulong totalLocalMemory;      /**< Max local memory allowed */
    cl_ulong usedLocalMemory;       /**< Used local memory */

    cl_device_id *devices;          /**< CL device list */
    cl_program program;             /**< CL program  */
    cl_kernel mapperExtendedKernel;
    cl_kernel mapperKernel;
    cl_kernel reducerKernel;
    cl_kernel reducerInOverflowKernel;
    cl_kernel copyerKernel;
    cl_kernel copyerInOverflowKernel;
    size_t kernelWorkGroupSize;     /**< Group size returned by kernel */
    size_t mapperWorkGroupSize;     /**< Group size returned by kernel */
    size_t mapperExtendedWorkGroupSize;
    size_t reducerWorkGroupSize;     /**< Group size returned by kernel */
    size_t reducerInOverflowWorkGroupSize;     /**< Group size returned by kernel */
    size_t copyerWorkGroupSize;
    size_t copyerInOverflowWorkGroupSize;
    size_t groupSize;               /**< Work-group size */
    bool verbose;
    cl_int numHashTables;
    cl_int extraHashTables; 

    cl_uint h_estimatedOutputValSize, h_estimatedOutputKeySize,  h_estimatedInterValSize, h_estimatedInterKeySize;
    int numGroups;
    cl_int h_workgroupOutputValsizes, h_workgroupOutputKeysizes;
    cl_int h_workgrouphashSizes, h_workgrouphashBucketSizes;
    cl_uint * h_outputKeySizes, *h_outputValSizes; 
    cl_uint* h_hashEntriies;
    cl_uint totalHashSize;
    cl_uint totalHashSizeExtra;
    cl_uint totalHashBucketSize;
    cl_uint extraHashBucketSize;
    cl_int * h_interAllOffsetSizes;
    cl_int KeysDiff, ValsDiff, CountsDiff;
    int numWavefrontsPerGroup;
    uint     allOutputVals;
    cl_mem   d_inputDataSet;
    cl_mem   d_constantData;
    cl_mem   d_interKeys;
    cl_mem   d_interVals;
    cl_mem   d_interOffsets;
    cl_mem   d_hashTable;
    cl_mem   d_hashBucket;
    cl_mem   d_hashTableOffsets;
    cl_mem   d_keysPointers;	
    cl_mem   d_keysOffsets;
    cl_mem   d_valStart;
    cl_mem   d_valEnd;
    cl_mem   d_valsPointers;
    cl_mem   d_valsOffsets;

    cl_mem   d_hashBucketOffsets;
    cl_mem   d_hashBucketPointers;  
    cl_mem   d_outputVals;
    cl_mem   d_outputValsOffsets;	
    cl_mem   d_outputKeys;
    cl_mem   d_outputKeysOffsets;

    bool overflowOccurs;
    cl_int overflowedWorkGroups;
    cl_mem  d_hashTableExtra;
    cl_mem  d_hashTableOffsetsExtra;
    cl_mem  d_hashBucketExtra;
    cl_mem  d_hashBucketOffsetsExtra;
    cl_mem  d_outputValsExtra;
    cl_mem  d_outputValsOffsetsExtra;
    cl_mem  d_interValsExtra;
    cl_mem  d_interValsOffsetsExtra;
    cl_mem  d_outputKeysExtra;
    cl_mem  d_outputKeysOffsetsExtra;

    cl_mem  d_gOutputKeySize;
    cl_mem  d_gOutputValSize;
    cl_mem  d_gNumHashentries;
    cl_mem   d_gHashBucketSize;

    cl_mem d_hashBucketIsBaseForReduce;
    cl_mem d_hashBucketIsBaseForReduceExtra;

    cl_mem d_contiguousOutputKeys, d_contiguousOutputVals, d_outputKeyValOffsets;
    cl_uint contiguousKeysSize, contiguousValsSize, contiguousOffsets;
    JobSpecification* jobSpec;

    public:
    std::string kernelfilename;
    bool isAMD;
    /**
     * Constructor
     * Initialize member variables
     * @param name name of sample (string)
     */
    cl_context context;             /**< CL context */
    cl_command_queue commandQueue;  /**< CL command queue */


    explicit MapReduce(std::string name)
        : maxWorkItemSizes(NULL),
        devices(NULL)
    {
        groupSize = GROUP_SIZE;
        overflowOccurs = false;
        d_hashTableExtra=NULL;
        d_hashTableOffsetsExtra=NULL;
        d_outputValsExtra=NULL;
        d_outputValsOffsetsExtra=NULL;
        overflowOccurs = false;
        extraHashTables = 0;
    }

    /**
     * Constructor
     * Initialize member variables
     * @param name name of sample (const char*)
     */
    explicit MapReduce(const char* name)
        : maxWorkItemSizes(NULL),
        devices(NULL)
    {
        groupSize = GROUP_SIZE;
        overflowOccurs = false;
        d_hashTableExtra=NULL;
        d_hashTableOffsetsExtra=NULL;
        d_outputValsExtra=NULL;
        d_outputValsOffsetsExtra=NULL;
        overflowOccurs = false;
        extraHashTables = 0;
    }

    ~MapReduce();

    /**
     * Allocate and initialize host memory array with random values
     * @return 1 on success and 0 on failure
     */
    int setupMapReduce(){return 0;};

    /**
     * OpenCL related initialisations.
     * Set up Context, Device list, Command Queue, Memory buffers
     * Build CL kernel program executable
     * @return 1 on success and 0 on failure
     */
    int setupCL();

    /**
     * Set values for kernels' arguments, enqueue calls to the kernels
     * on to the command queue, wait till end of kernel execution.
     * Get kernel start and end time if timing is enabled
     * @return 1 on success and 0 on failure
     */
    int runCLKernels();

    /**
     * Override from SDKSample. Initialize
     * command line parser, add custom options
     */
    int initialize(JobSpecification* jobSpec);

    /**
     * Override from SDKSample, adjust width and height
     * of execution domain, perform all sample setup
     */
    int setup();

    /**
     * Override from SDKSample
     * Run OpenCL Reduction
     */
    int run(){return 0;};

    /**
     * Override from SDKSample
     * Cleanup memory allocations
     */
    int cleanup();

    int genBinaryImage(){return 0; };

    int verifyResults(){return 0; };

    int printFinalOutput(uint extended, uint keysSizes, uint valsSizes, uint hashTableSizes, uint hashBucketSizes, cl_int * h_hashBucketOffsets);

    unsigned int upper_power_of_two(unsigned int x)
    {
        int v= x;
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v++;
        return v;
    }

    void AddMapInputRecord(
            void*		key,
            void*		val,
            cl_int		keySize,
            cl_int		valSize);

    int startMapReduce();

    int startMapType1();

    int startMapType2();

    int startReduce();
    int checkVal(cl_int&, int, const std::string *); 
    int checkVal(cl_int&, int, const char *); 
    void error(const char*);
    int initialize();
    std::string getPath();

};


#endif // MAPREDUCE_H_
