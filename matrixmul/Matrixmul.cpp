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

#include "Matrixmul.hpp"
#include "../StreamMR.hpp"
#include <malloc.h>
#include <ctime>
#include <sys/time.h>
#include "../timeRec.h"
#include <getopt.h>
#include <math.h>
#define CEIL(n,m) (n/m + (int)(n%m !=0))

     static struct option long_options[] = {
         /* name, has_arg, flag, val */
         {"rowsNumber", 1, NULL, 'r'},
         {"colsNumber", 1, NULL, 'c'},
         {"userSize", 1, NULL, 'g'},
         {0,0,0,0}
     };

int Matrixmul::setupMatrixmul()
{
    printf("Generating two %dx%d matrices ...\n", rowsNumber, colsNumber);

    h_matrix = (float*)malloc(sizeof(float)*rowsNumber*2*colsNumber);

    srand(time(0));
    for (int i = 0; i < rowsNumber*2; i++)
        for (int j = 0; j < colsNumber; j++)
            h_matrix[i*colsNumber+j] = (float)(rand() % 100);
            //h_matrix[i*colsNumber+j] = (float)rand()/(float)RAND_MAX;


    return SDK_SUCCESS;

}

int Matrixmul::setupCL()
{

    return SDK_SUCCESS;
}


int Matrixmul::runCLKernels()
{
    printf("(1)- Create the specification of Matrix Muktiplication:\n");

    JobSpecification* jobSpec;
    jobSpec=new JobSpecification();
    jobSpec->workflow = MAP_ONLY;
    jobSpec->overflow = false;
    jobSpec->outputToHost = 1;
    jobSpec->userSize=userSize;

    jobSpec->inputDataSet =  h_matrix;
    jobSpec->inputDataSetSize = sizeof(float)*rowsNumber*2*colsNumber;

    //Constants
    cl_uint *h_constantData = (cl_uint*)malloc(sizeof(cl_uint)*3);
    h_constantData[0] = rowsNumber*colsNumber*sizeof(float);
    h_constantData[1] = rowsNumber;
    h_constantData[2] = colsNumber;

    jobSpec->constantData = h_constantData;
    jobSpec->constantDataSize =  3* sizeof(cl_uint);

    jobSpec->estimatedRecords=(rowsNumber) * (colsNumber);
    jobSpec->estimatedKeySize=sizeof(cl_float); //intermediate key
    jobSpec->estimatedValSize=sizeof(cl_int2); //intermediate val


    printf("(2)- Initialize MapReduce Engine ...:\n");
    printf("----------------------------\n");
    MapReduce clMapReduce("StreamMR");
    clMapReduce.initialize(jobSpec);
    clMapReduce.kernelfilename = "MapReduce_MM.cl";
    if(clMapReduce.setup() != SDK_SUCCESS)  return SDK_FAILURE;

    KEY_T key;
    VAL_T val;

    for (int i = 0; i < rowsNumber; i++)
    {
        key.row = i;
        for (int j = 0; j < colsNumber; j++)
        {
            val.col = j;
            clMapReduce.AddMapInputRecord(&key, &val, sizeof(KEY_T), sizeof(VAL_T));
        }
    }

    clMapReduce.startMapReduce();

    /*
       printf("\nMatrix 1:\n");
       printMatrix(h_matrix1,clReduction.rowsNumber,clReduction.colsNumber);
       printf("\nMatrix 2:\n");
       printMatrix(h_matrix2,clReduction.rowsNumber,clReduction.colsNumber);
       */
    //printf("\nCPU Resulting Multiplication:\n");

    printf("\nResulting Multiplication:\n");
    float * keys=(float*)jobSpec->interKeys;
    //cl_int2 * values=(cl_int2*)jobSpec->interVals;
    //cl_int4 *kvo4=(cl_int4*)jobSpec->interOffsetSizes;
    verifyResults(keys);
    //for (int i=0;i <colsNumber * rowsNumber; i++)
    //{
        //printf("c[%i,%i]=%f \n ",values[i].x, values[i].y, keys[i]);
    //}

    //cl_int *kvo=(cl_int*)jobSpec->interOffsetSizes;
    //for (int i=0;i <1024; i+=4)
    //{
        //printf("gKeys %i, lKeys %i, gVals %i, lVals %i \n",kvo[i],kvo[i+1],kvo[i+2],kvo[i+3]);
    //}

    if(clMapReduce.cleanup() != SDK_SUCCESS)
        return SDK_FAILURE;


    return SDK_SUCCESS;
}

/*
 * Reduces the input array (in place)
 * length specifies the length of the array
 */
void Matrixmul::matrixmulCPUReference( int num, float *keys)
{
    printf("Verifying results....\n");
    float* result = (float*)malloc(colsNumber*rowsNumber*sizeof(float));
    cl_int2* pos = (cl_int2*)malloc(colsNumber*rowsNumber*sizeof(cl_int2));
    int count = 0;
    for (int i = 0; i < rowsNumber; i++)
    {
        for (int k = 0; k < rowsNumber; k++)
        {
            result[count] = 0.0f;
            for (int j = 0; j < colsNumber; j++)
                result[count] += (h_matrix[i*colsNumber+j] * h_matrix[(k+rowsNumber)*colsNumber+j]);
            pos[count].x = i;
            pos[count].y = k;
            ++count;
        }
    }

    char err = 0;
    for (int i = 0; i < rowsNumber * rowsNumber; i++)
        if(fabs(result[i] - keys[i]) > 0.0001*result[i])
        {
            printf("( %d, %d ) : CPU:%f - GPU:%f Dif:%f\n", pos[i].x, pos[i].y, result[i], keys[i], result[i] - keys[i]);
            err = 1;
        }
    if(err == 1)
        printf("The result is incorrect!!!");
    else
        printf("Pass!");
    printf("-----------------\n");
    free(result);
    free(pos);
}


int Matrixmul::initialize(int argc, char * argv[])
{
    // Call base class Initialize to get default configuration
    if(argc < 6){
        fprintf(stderr, "Usage %s <-c colsNumber> <-r rowsNumber> <-g userSize>\n",argv[0]);
        exit(EXIT_FAILURE);
    }
    int opt, option_index=0;
    while ((opt = getopt_long(argc, argv, "::c:r:g:",
                    long_options, &option_index)) != -1 ) {
        switch(opt){
            case 'c': 
                colsNumber = atoi(optarg);
                break;
            case 'r':
                rowsNumber = atoi(optarg);
                break;
            case 'g':
                userSize = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage %s <-c colsNumber> <-r rowsNumber> <-g userSize>\n",argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    printf("rowsNumber: %d colsNumber: %d\n", rowsNumber, colsNumber);
}

int Matrixmul::setup()
{
    if(setupMatrixmul() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    printf("Inside setup\n");

    if(setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    /* Compute setup time */

    return SDK_SUCCESS;
}


int Matrixmul::run()
{
    /* Arguments are set and execution call is enqueued on command buffer */
    if(runCLKernels() != SDK_SUCCESS)
        return SDK_FAILURE;

    return SDK_SUCCESS;
}

int Matrixmul::verifyResults(float *keys)
{
    matrixmulCPUReference(colsNumber, keys);

    return SDK_SUCCESS;
}

int Matrixmul::cleanup()
{
    /* Releases OpenCL resources (Context, Memory etc.) */
    if (h_matrix) 
        free(h_matrix);
    return SDK_SUCCESS;
}

Matrixmul::~Matrixmul()
{
    /* release program resources (input memory etc.) */
    if (h_matrix)
    {
#ifdef _WIN32
        _aligned_free(h_matrix);
#else
        free(h_matrix);
#endif
        h_matrix = NULL;
    }

}
int printMatrix(float * matrix,int M_ROW_COUNT, int M_COL_COUNT)
{
    for (int i = 0; i < M_ROW_COUNT; i++)
        for (int j = 0; j < M_COL_COUNT; j++)
            printf ("m[%i,%i]=%f  ",i,j,matrix[i * M_COL_COUNT + j]);

    return 0;
}

int main(int argc, char * argv[])
{
    Matrixmul mm("Matrix Multiplication");
    mm.initialize(argc, argv);
    if(mm.setup() != SDK_SUCCESS)  return SDK_FAILURE;
    mm.run();
    return SDK_SUCCESS;
}
