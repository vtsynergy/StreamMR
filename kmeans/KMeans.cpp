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

#include "KMeans.hpp"
#include "../StreamMR.hpp"
#include <malloc.h>
#include <ctime>
#include <sys/time.h>
#include "../timeRec.h"
#include <getopt.h>
#define verify 1
     static struct option long_options[] = {
         /* name, has_arg, flag, val */
         {"Dimensions", 1, NULL, 'd'},
         {"numOfPoints", 1, NULL, 'p'},
         {"numOfClusters", 1, NULL, 'c'},
         {"workgroupSize", 1, NULL, 'g'},
         {0,0,0,0}
     };



int KMeans::setupKMeans()
{

#if defined (_WIN32)
    h_data = (cl_int*)_aligned_malloc(sizeof(cl_int)*((numPoints+ numClusters)*dim+1,16));
#else
    h_data = (cl_int*)memalign(16,sizeof(cl_int)*((numPoints+ numClusters)*dim+1));
#endif


    srand(0);//time(0));
    //Generate points
    for (int i = 0; i < numPoints; i++)
    {
        //printf("point%i: ",i);
        for (int j = 0; j < dim; j++)
        {
            h_data[i*dim+j] = (cl_int)(rand() % 100);
            //printf("%i - ",h_data[i*dim+j]);
        }
        //printf("\n");
    }

    //Generate clusters
    for (int i = numPoints; i < numClusters + numPoints; i++)
    {
        //      printf("cluster%i: ",i);
        for (int j = 0; j < dim; j++)
        {
            h_data[i*dim+j] = h_data[(i-numPoints)*dim + j];
            //      printf("%i - ",data[i*dim+j]);
        }
        //      printf("\n");
    }
    //change=0
    h_data[(numPoints + numClusters)*dim]=0;


    return SDK_SUCCESS;
}

int KMeans::setupCL()
{
    return SDK_SUCCESS;
}


int KMeans::runCLKernels()
{

    JobSpecification * jobSpec;

    // Start Kmeans Application
    //-------------------------
    printf("(1)- Prepare Kmeans Specification:\n");
    printf("----------------------------\n");
    jobSpec=new JobSpecification();
    jobSpec->workflow = MAP_REDUCE;
    jobSpec->overflow = false;
    jobSpec->outputToHost = 1;
    jobSpec->perfectHashing = true;
    jobSpec->outputIntermediate = true;

    jobSpec->estimatedInterRecords = numPoints;
    jobSpec->estimatedInterValSize =  sizeof(cl_int);
    jobSpec->estimatedInterKeySize = sizeof(cl_int);

    jobSpec->LoadKeyVal=true;
    jobSpec->userSize = wgSize;

    jobSpec->inputDataSet = h_data;
    jobSpec->inputDataSetSize = ((numPoints+ numClusters)*dim + 1)*sizeof(cl_int);

    //Constants
    cl_uint *h_constantData = (cl_uint*)malloc(sizeof(cl_uint)*4);
    h_constantData[0] = dim;
    h_constantData[1] = numClusters;
    h_constantData[2] = numPoints * dim * sizeof(cl_int); //Clusters Offsets
    h_constantData[3] = (numPoints + numClusters)* dim*sizeof(cl_int); //Change Offset

    jobSpec->constantData = h_constantData;
    jobSpec->constantDataSize =  4 * sizeof(cl_uint);

    jobSpec->estimatedRecords = numClusters; //control hashtable size per each wavefront
    jobSpec->estimatedValSize =  (dim + 1 ) * sizeof(cl_int);
    jobSpec->estimatedKeySize = sizeof(KEY_T);


    printf("(2)- Create MapReduce instance .\n");
    printf("----------------------------\n");
    MapReduce clMapReduce("OpenCL MapReduce");
    clMapReduce.initialize(jobSpec);
    clMapReduce.kernelfilename = "MapReduce_KM.cl";
    if(clMapReduce.setup() != SDK_SUCCESS)
        return SDK_FAILURE;

    //Input Key/Value pairs
    VAL_T val;
    KEY_T key;

    key.clusterId = 0;
    for (int i = 0; i < numPoints; i++)
    {
        val.pointId = i;
        clMapReduce.AddMapInputRecord(&key, &val, sizeof(KEY_T), sizeof(VAL_T));
    }

    printf ("Initial Input .............\n");
    KEY_T* keys=(KEY_T*)jobSpec->inputKeys;
    VAL_T* vals=(VAL_T*)jobSpec->inputVals;
    for (int i = 0; i <numPoints; i++)
    {
        //           printf("point %i - cluster %i \n",vals[i].pointId,keys[i].clusterId);
    }
    /*
       VAL_T* vals=(VAL_T*)jobSpec->inputVals;
       for (int i = 0; i < clReduction.numPoints; i++)
       {
       printf("offset %i \n",vals[i].clustersOffset);
       }
       */
    cl_int status;
    cl_event events[1];
    //      while (1)
    //      {
    //cl_int h_change=0;
    /*
       status = clEnqueueWriteBuffer(clMapReduce.commandQueue,
       d_data,
       1,
       ((numPoints + numClusters)* dim)*sizeof(cl_int),
       sizeof(cl_int),
       &h_change,
       0,
       0,
       &events[1]);
       if(status!= NULL)
       {
       printf("clwriteBuffer failed. (d_data)\n");
       exit(-1);
       }
       else
       {
       printf("clwriteBuffer succeed. Data is on host\n");
       }
       */

    clMapReduce.startMapReduce();
    jobSpec->LoadKeyVal=false; //to keep the changes from previous iterations

    printf ("\nMap Output ................. \n");
    keys=(KEY_T*)jobSpec->interKeys;
    vals=(VAL_T*)jobSpec->interVals;
    //for (int i = 0; i <numPoints; i++)
    //{
    //printf("point %i - cluster %i \n", vals[i].pointId, keys[i].clusterId);
        //}	
    
    //status = clEnqueueReadBuffer(clMapReduce.commandQueue,
            //d_data,
            //1,
            //((numPoints + numClusters)* dim)*sizeof(cl_int),
            //sizeof(cl_int),
            //&h_change,
            //0,
            //0,
            //&events[1]);
    //if(status!= NULL)
    //{
        //printf("clreadBuffer failed. (d_data)\n");
        //exit(-1);
    //}
    //else
    //{
        //printf("clreadBuffer succeed. Data is on host\n");
    //}
    //printf("Map output change: %i\n",h_change);
    //              if (h_change == 0 ) break;
    //      } //End While

    verifyResults(keys);
    /*
       status = clEnqueueReadBuffer(clMapReduce.commandQueue,
       d_data,
       1,
       0,
       ((numPoints + numClusters)* dim)*sizeof(cl_int),
       h_data,
       0,
       0,
       &events[1]);
       if(status!= NULL)
       {
       printf("clreadBuffer failed. (d_data)\n");
       exit(-1);
       }
       else
       {
       printf("clreadBuffer succeed. Data is on host\n");
       }

       for (int i = numPoints; i < numPoints + numClusters; i++)
       {
       printf("%i: ",i);
       for (int j = 0; j < dim; j++)
       {
       printf("%i - ",h_data[i * dim + j]);
       }
       printf("\n");
       }


       cl_int* c=(cl_int*) jobSpec->outputVals;
       for (int i=0 ; i< numClusters* (dim+1); i+=3)
       {
       printf("cluster%i: (%i,%i) ",i/3,c[i+1],c[i+2]);
       }

       printf("\n");

       c=(cl_int*) jobSpec->outputValsExtra;
       for (int i=0 ; i< numClusters* (dim+1); i+=3)
       {
       printf("cluster%i: (%i,%i) ",i/3,c[i+1],c[i+2]);
       }
       */

    if(clMapReduce.cleanup() != SDK_SUCCESS)     return SDK_FAILURE;


    return SDK_SUCCESS;
}

/*
 * Reduces the input array (in place)
 * length specifies the length of the array
 */
void KMeans::kmeansCPUReference(KEY_T *keys)
{

    printf("verifying result....\n");
    int* clusterId = (int*)malloc(sizeof(int)*numPoints);

    memset(clusterId, 0, sizeof(int)*numPoints);

    int* clusters=(int*)h_data + numPoints * dim;

    //printf("Validate Initial Clusters\n");
    //        for (int i = 0; i < numClusters; i++)
    //                {
    //                        printf("cluster%i: ",i);
    //                        for (int j = 0; j < dim; j++)
    //                        {
    //                                printf("%i - ",clusters[i*dim+j]);
    //                        }
    //                        //printf("\n");
    //                }
    int iter = 0;
    while (iter<1) //(1)
    {
        int change = 0;
        //printf("========iteration:%d===========\n", iter);
        for (int i = 0; i < numPoints; i++)
        {
            int minMean = 0;
            int* curPoint = h_data + dim * i;
            int* originCluster = clusters + clusterId[i] * dim;
            for (int j = 0; j < dim; ++j)
                minMean += (originCluster[j] - curPoint[j])* (originCluster[j] - curPoint[j]);

            int curClusterId = clusterId[i];
            for (int k = 0; k < numClusters; ++k)
            {
                int* curCluster = clusters + k*dim;
                int curMean = 0;
                for (int x = 0; x < dim; ++x)
                    curMean += (curCluster[x] - curPoint[x]) * (curCluster[x] - curPoint[x]);

                //printf("pt:%d, cl:%d, minDist:%d, curDist:%d\n", curPoint[0], curCluster[0], minMean, curMean);
                if (minMean > curMean)
                {
                    curClusterId = k;
                    minMean = curMean;
                }
            }

            //printf("point:%d, curClusterId:%d, clusterId[i]:%d\n", i, curClusterId, clusterId[i]);

            if (curClusterId != clusterId[i])
            {
                change = 1;
                clusterId[i] = curClusterId;
            }

        }
        if (change == 0) break;

        int* tmpClusters = (int*)malloc(sizeof(int)* numClusters * dim);
        memset(tmpClusters, 0, sizeof(int) * numClusters * dim);

        int* counter = (int*)malloc(sizeof(int)* numClusters);
        memset(counter, 0, numClusters * sizeof(int));
        for (int i = 0; i < numPoints ; i++)
        {
            for (int j = 0; j < dim; j++)
                tmpClusters[clusterId[i] * dim + j] += h_data[i*dim + j];
            counter[clusterId[i]]++;
        }
        for (int i = 0; i < numClusters; i++)
        {
            //                        printf("cluster %d: ", i);
            if (counter[i] !=0)
            {
                for (int j =0; j < dim; j++)
                {
                    tmpClusters[i*dim + j] /= counter[i];
                    clusters[i*dim + j] = tmpClusters[i*dim + j];
                }
            }
            //                        for (int j = 0; j < dim; j++)
            //                                printf("%d ", clusters[i*dim + j]);
            //                        printf("\n");
        }

        free(tmpClusters);
        free(counter);
        iter++;
    }//while
    int err = 0;
    for(int i = 0; i < numPoints; i++)
        if(clusterId[i] != keys[i].clusterId)
            err = 1;
    if(err == 1) printf("result is invalid!!!\n"); else printf("pass!!!\n");
    free(clusterId);

}

int KMeans::initialize(int argc, char * argv[])
{
    if(argc < 8){
        fprintf(stderr, "Usage %s <-d dimensions> <-p pointsNumbers> <-c numClusters> <-g workGroupSize>\n",argv[0]);
        exit(EXIT_FAILURE);
    }
    int opt, option_index=0;
    while ((opt = getopt_long(argc, argv, "::d:p:c:g:",
                    long_options, &option_index)) != -1 ) {
        switch(opt){
            case 'd': 
                dim = atoi(optarg);
                break;
            case 'p': 
                numPoints = atoi(optarg);
                break;
            case 'c':
                numClusters = atoi(optarg);
                break;
            case 'g':
                wgSize = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage %s <-d dimensions> <-p pointsNumbers> <-c numClusters> <-g workGroupSize>\n",argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    if(numClusters <= wgSize)
    {
        printf("numClusters shoud be larger than workGroupSize\n");
        exit(EXIT_FAILURE);
    }

    if(wgSize < 64)
    {
        printf("workgroupSize should larger than 64\n");
        exit(EXIT_FAILURE);
    }

    return SDK_SUCCESS;
}

int KMeans::setup()
{
    if(setupKMeans() != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}


int KMeans::run()
{
    /* Arguments are set and execution call is enqueued on command buffer */
    if(runCLKernels() != SDK_SUCCESS)
        return SDK_FAILURE;


    return SDK_SUCCESS;
}

int KMeans::verifyResults(KEY_T * keys)
{
    if(verify)
    {
        /* reference implementation
         * it overwrites the input array with the output
         */
        kmeansCPUReference(keys);

    }

    return SDK_SUCCESS;
}

void KMeans::printStats()
{
}

int KMeans::cleanup()
{

    return SDK_SUCCESS;
}

KMeans::~KMeans()
{
    /* release program resources (input memory etc.) */
    if(h_data)
    {
#ifdef _WIN32
        _aligned_free(h_data);
#else
        free(h_data);
#endif
        h_data = NULL;
    }
}



int main(int argc, char * argv[])
{
    KMeans km("KMeans");
    km.initialize(argc, argv);
    if(km.setup() != SDK_SUCCESS)  return SDK_FAILURE;
    km.run();
    return SDK_SUCCESS;
}
