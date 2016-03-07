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
#include "StringMatch.hpp"
#include "../StreamMR.hpp"
#include <malloc.h>
#include <ctime>
#include <sys/time.h>
#include "../timeRec.h"
#include <iostream>
#include <getopt.h>
     using namespace std;

     static struct option long_options[] = {
         /* name, has_arg, flag, val */
         {"filename", 1, NULL, 'f'},
         {"keyWord", 1, NULL, 'w'},
         {"sizePerMap", 1, NULL, 's'},
         {"userSize", 1, NULL, 'g'},
         {0,0,0,0}
     };



int StringMatch::setupStringMatch()
{
    cout <<"Inside setupStringMatch ..."<<endl;
    cout<<"Input file: "<<fileName<<" keyWord: "<<keyWord<<endl;

    //Read the input file, and the keyword and load it to the device
    FILE *fp = fopen(fileName.c_str(), "r");
    fseek(fp, 0, SEEK_END);
    fileSize = ftell(fp);
    kwordSize=strlen(keyWord.c_str())+1;

    h_filebuf = (char*)malloc(fileSize+kwordSize+1);
    rewind(fp);
    fread(h_filebuf, fileSize, 1, fp);
    for (int i=0; i<kwordSize; i++)	h_filebuf[fileSize+i]=keyWord[i];
    h_filebuf[fileSize+kwordSize]='\0';
    printf("Input file size: %i keywordsize: %i\n",fileSize,kwordSize);
    fclose(fp);

    return SDK_SUCCESS;
}

int StringMatch::setupCL()
{

    return SDK_SUCCESS;
}


int StringMatch::runCLKernels()
{

    printf("(1)- Create StringMatch Job Specification...:\n");
    printf("----------------------------\n");
    JobSpecification* jobSpec;
    jobSpec=new JobSpecification();
    jobSpec->workflow = MAP_ONLY;
    jobSpec->outputToHost = 1;
    jobSpec->overflow = true;
    jobSpec->userSize=userSize;

    jobSpec->inputDataSet = h_filebuf;
    jobSpec->inputDataSetSize =  fileSize+kwordSize+1;

    jobSpec->constantData = NULL;
    jobSpec->constantDataSize =  0;

    //-----------------------------------------------------------------
    jobSpec->estimatedRecords=102400;   //Should be multiple of workgroups number to produce correct results
    //-----------------------------------------------------------------

    jobSpec->estimatedKeySize=sizeof(cl_int); //intermediate key
    jobSpec->estimatedValSize=sizeof(cl_int); //intermediate val

    printf("(2)- Initialize MapReduce Engine ...:\n");
    printf("----------------------------\n");
    MapReduce clMapReduce("StreamMR");
    clMapReduce.initialize(jobSpec);
    clMapReduce.kernelfilename = "MapReduce_SM.cl";
    if(clMapReduce.setup() != SDK_SUCCESS)  return SDK_FAILURE;

    KEY_T key;
    key.keywordOffset = fileSize;
    key.keywordSize = kwordSize;
    VAL_T val;

    int offset = 0;
    char* p = h_filebuf;
    char* start = h_filebuf;

    while (1)
    {
        int blockSize=sizePerMap;
        if (offset + blockSize > fileSize) blockSize = fileSize - offset;
        p += blockSize;
        for (; *p != '\n' && *p != '\0'; p++);
        if (*p != '\0')
        {
            ++p;
            blockSize = (int)(p - start);
            val.lineOffset = offset;
            val.lineSize =blockSize;
            //                      printf("lineOffset: %i lineSize: %i\n",offset,blockSize);
            clMapReduce.AddMapInputRecord( &key, &val, sizeof(KEY_T), sizeof(VAL_T));
            offset += blockSize;
            start = p;
        }
        else
        {
            *p = '\n';
            blockSize = (int)(fileSize - offset);
            val.lineOffset = offset;
            val.lineSize = blockSize;
            //                       printf("lineOffset: %i lineSize: %i\n",offset,blockSize);
            clMapReduce.AddMapInputRecord(&key, &val, sizeof(KEY_T), sizeof(VAL_T));
            break;
        }
    }


    /* pad dummy records that doesn't affect the final results
     * for the size of input to be power of 2
     */
    int padRecords=clMapReduce.upper_power_of_two(jobSpec->inputRecordCount)-jobSpec->inputRecordCount;
    printf("Input records: %f, Padding records: %i",jobSpec->inputRecordCount,padRecords);
    val.lineOffset = 0;
    val.lineSize = 0;
    for (int i=0; i<padRecords; i++)
        clMapReduce.AddMapInputRecord( &key, &val, sizeof(KEY_T), sizeof(VAL_T));

    clMapReduce.startMapReduce();

    printf( "Finished MapReduce .....\n");
    //printf( "MapOutput Records: %i \n",jobSpec->interRecordCount);

    cl_int * keys=(cl_int *)jobSpec->interKeys;
    cl_int * values=(cl_int *)jobSpec->interVals;
    //for (int i=0;i < jobSpec->interRecordCount; i++)
    //{
    //printf("offset %i size:%i\n",keys[i], values[i]);
    //}

    keys=(cl_int *)jobSpec->interKeysExtra;
    values=(cl_int *)jobSpec->interValsExtra;
    //for (int i=0;i < jobSpec->interDiffKeyCount; i++)
    //{
    //printf("offset %i size:%i\n",keys[i],values[i]);
    //}



    if(clMapReduce.cleanup() != SDK_SUCCESS)
        return SDK_FAILURE;

    return SDK_SUCCESS;
}

/*
 * Reduces the input array (in place)
 * length specifies the length of the array
 */
void StringMatch::stringMatchCPUReference()
{
}

int StringMatch::initialize(int argc, char * argv[])
{
    // Call base class Initialize to get default configuration
    if(argc < 8){
        fprintf(stderr, "Usage %s <-f filename> <-s sizePerMap> <-w keyWord> <-g userSize>\n",argv[0]);
        exit(EXIT_FAILURE);
    }
    int opt, option_index=0;
    while ((opt = getopt_long(argc, argv, "::s:f:g:w:",
                    long_options, &option_index)) != -1 ) {
        switch(opt){
            case 'f': 
                fileName = optarg;
                break;
            case 's': 
                sizePerMap = atoi(optarg);
                break;
            case 'w':
                keyWord = optarg;
                break;
            case 'g':
                userSize = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage %s <-f filename> <-s sizePerMap> <-w keyWord> <-g userSize>\n",argv[0]);
                exit(EXIT_FAILURE);

        }
    }

    //if(userSize < 32)
    {
        printf("userSize should larger than 32\n");
        //exit(EXIT_FAILURE);
    }

    printf("filename: %s keyWord: %s sizePerMap: %d userSize: %d\n", fileName.c_str(), keyWord.c_str(), sizePerMap, userSize);
}


int StringMatch::setup()
{
    if(setupStringMatch() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }


    if(setupCL() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}


int StringMatch::run()
{
    /* Arguments are set and execution call is enqueued on command buffer */
    if(runCLKernels() != SDK_SUCCESS)
        return SDK_FAILURE;

    return SDK_SUCCESS;
}

int StringMatch::cleanup()
{
    return SDK_SUCCESS;
}

StringMatch::~StringMatch()
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


    int
main(int argc, char * argv[])
{
    StringMatch sm("String Match");
    sm.initialize(argc, argv);
    if(sm.setup() != SDK_SUCCESS)  return SDK_FAILURE;
    sm.run();
    return SDK_SUCCESS;
}

