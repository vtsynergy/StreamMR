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

#include "Wordcount.hpp"
#include "../StreamMR.hpp"
#include <malloc.h>
#include <ctime>
#include <sys/time.h>
#include "../timeRec.h"
#include <getopt.h>
     static struct option long_options[] = {
         /* name, has_arg, flag, val */
         {"inputFile", 1, NULL, 'f'},
         {"numOfChuncks", 1, NULL, 'c'},
         {"workgroupSize", 1, NULL, 'g'},
         {0,0,0,0}
     };


int verify = 0;

#define CEIL(n,m) (n/m + (int)(n%m !=0))

    int
Wordcount::setupWordcount()
{

    FILE* fp = fopen(filename.c_str(), "r");
    if(fp == NULL)
        printf("cannot open file %s\n", filename.c_str());
    fseek(fp, 0, SEEK_END);
    fileSize = ftell(fp) + 1;

    rewind(fp);
    h_filebuf = (char*)malloc(fileSize);
    fread(h_filebuf, fileSize, 1, fp);
    for (int i = 0; i < fileSize; i++)
        h_filebuf[i] = toupper(h_filebuf[i]);

    fclose(fp);

    if(h_filebuf == NULL)
    {
        printf("Failed to allocate host memory. (h_filebuf)");
        return SDK_FAILURE;
    }


    /* random initialisation of input */
    //sampleCommon->fillRandom<cl_uint>(input, length * VECTOR_SIZE, 1, 0, 5);

    /*
     * Unless quiet mode has been enabled, print the INPUT array
     */
    /*
       if(!quiet)
       {
       sampleCommon->printArray<cl_uint>("Input",
       input,
       length * VECTOR_SIZE,
       1);
       }
       */
    return SDK_SUCCESS;
}

    int
Wordcount::setupCL()
{
    return SDK_SUCCESS;
}


int Wordcount::runCLKernels()
{
    JobSpecification* jobSpec;

    // Start WordCount Application
    //----------------------------------------------
    printf("(0)- Prepare Wordcount data:\n");
    printf("----------------------------\n");
    jobSpec=new JobSpecification();
    jobSpec->workflow = MAP_REDUCE;
    jobSpec->overflow = true;
    jobSpec->outputToHost = 1;
    jobSpec->userSize = wgSize;

    printf("Input file size: %i\n",fileSize);
    jobSpec->inputDataSet = h_filebuf;
    jobSpec->inputDataSetSize = fileSize;

    cl_uint *h_constantData = (cl_uint*)malloc(sizeof(cl_uint));
    h_constantData[0] = 1;
    jobSpec->constantData = h_constantData;
    jobSpec->constantDataSize =  1 * sizeof(cl_uint);	

    jobSpec->estimatedRecords = 2048;
    jobSpec->estimatedKeySize = 50;
    jobSpec->estimatedValSize = sizeof(cl_int);


    //Start StreamMR framework
    //-----------------------------------------
    MapReduce clMapReduce("StreamMR");
    clMapReduce.initialize(jobSpec);
    clMapReduce.kernelfilename = "MapReduce_WC.cl";
    if(clMapReduce.setup() != SDK_SUCCESS)
        return SDK_FAILURE;

    KEY_T key;
    key.line_num = 0;
    VAL_T val;
    int offset = 0;
    char* p = h_filebuf;
    char* start = h_filebuf;
    //choose blocksize so that the total number of chunks is multiple of 2
    int blockSize = fileSize/numChunks;
    int lineSize;

    while (1)
    {
        if (offset + blockSize > fileSize) blockSize = fileSize - offset;
        if (blockSize < 0) break;
        p += blockSize;
        for (; *p >= 'A' && *p <= 'Z'; p++);

        if (*p != '\0')
        {
            *p = '\0';
            ++p;
            lineSize = (int)(p - start);
            val.line_offset = offset;
            val.line_size = lineSize;
            key.line_num++;
            /*      	printf("offset: %i size:%i\n",offset,lineSize);

                        for (int i=offset; i< offset+lineSize; i++)
                        {
                        printf("%c",h_filebuf[i]);

                        }
                        printf("\n");
                        */
            clMapReduce.AddMapInputRecord(&key, &val, sizeof(KEY_T), sizeof(VAL_T));
            offset += lineSize;
            start = p;
        }
        else
        {
            *p = '\0';
            lineSize = (int)(fileSize - offset);
            val.line_offset = offset;
            val.line_size = lineSize;
            key.line_num++;
            /*              printf("offset: %i size:%i\n",offset,lineSize);
                            for (int i=offset; i< offset+lineSize; i++)
                            {
                            printf("%c",h_filebuf[i]);

                            }
                            printf("\n");
                            */
            clMapReduce.AddMapInputRecord(&key, &val, sizeof(KEY_T), sizeof(VAL_T));
            break;
        }
    }
    /* pad dummy records that doesn't affect the final results
     * for the size of input to be power of 2
     */
    int padRecords= clMapReduce.upper_power_of_two(jobSpec->inputRecordCount)-jobSpec->inputRecordCount;
    printf("Input records: %f, Padding records: %i",jobSpec->inputRecordCount,padRecords);
    val.line_offset = 0;
    val.line_size = 0;
    key.line_num++;
    for (int i=0; i<padRecords; i++)
        clMapReduce.AddMapInputRecord(&key, &val, sizeof(KEY_T), sizeof(VAL_T));


    clMapReduce.startMapReduce();

    if(clMapReduce.cleanup() != SDK_SUCCESS)
        return SDK_FAILURE;


    return SDK_SUCCESS;
}

void Wordcount::wordCountCPUReference(cl_uint * input,
        const cl_uint length,
        cl_uint& output)
{
}

int Wordcount::initialize(int argc, char * argv[])
{
    if(argc < 7){
        fprintf(stderr, "Usage %s <-f input_file> <-c chunckSize > <-g workGroupSize>\n",argv[0]);
        exit(EXIT_FAILURE);
    }
    int opt, option_index=0;
    while ((opt = getopt_long(argc, argv, "::f:c:g:",
                    long_options, &option_index)) != -1 ) {
        switch(opt){
            case 'f':
                filename = optarg;
                break;
            case 'c':
                numChunks = atoi(optarg);
                break;
            case 'g':
                wgSize = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage %s <-d dimensions> <-p pointsNumbers> <-c numClusters> <-g workGroupSize>\n",argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    return SDK_SUCCESS;
}

int Wordcount::setup()
{

    if(setupWordcount() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }
    //int timer = sampleCommon->createTimer();
    //sampleCommon->resetTimer(timer);
    //sampleCommon->startTimer(timer);

    if(setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    //sampleCommon->stopTimer(timer);
    /* Compute setup time */
    //setupTime = (double)(sampleCommon->readTimer(timer));

    return SDK_SUCCESS;
}


int Wordcount::run()
{
    /* Arguments are set and execution call is enqueued on command buffer */
    if(runCLKernels() != SDK_SUCCESS)
        return SDK_FAILURE;

    //    if(!quiet)
    //        sampleCommon->printArray<cl_uint>("Output", &output, 1, 1);

    return SDK_SUCCESS;
}

int Wordcount::verifyResults()
{
    if(verify)
    {
        /* reference implementation
         * it overwrites the input array with the output
         */
        //      wordcountCPUReference(input, length * VECTOR_SIZE, refOutput);
    }


    return SDK_SUCCESS;
}

void Wordcount::printStats()
{
    /*
       std::string strArray[3] = {"Elements", "Time(sec)", "kernelTime(sec)"};
       std::string stats[3];

       totalTime = setupTime + kernelTime;
       stats[0]  = sampleCommon->toString(length * VECTOR_SIZE, std::dec);
       stats[1]  = sampleCommon->toString(totalTime, std::dec);
       stats[2]  = sampleCommon->toString(kernelTime, std::dec);

       this->SDKSample::printStats(strArray, stats, 3);
       */
}

int Wordcount::cleanup()
{
    return SDK_SUCCESS;
}

Wordcount::~Wordcount()
{
    /* release program resources (input memory etc.) */
    if(h_filebuf)
    {
#ifdef _WIN32
        _aligned_free(h_filebuf);
#else
        free(h_filebuf);
#endif
        h_filebuf = NULL;
    }

}

int main(int argc, char * argv[])
{

    Wordcount wc("Wordcount");
    wc.initialize(argc, argv);
    if(wc.setup() != SDK_SUCCESS)  return SDK_FAILURE;
    wc.run();
    return SDK_SUCCESS;
    return 0;
}
