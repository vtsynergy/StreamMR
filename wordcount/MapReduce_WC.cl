/*****************************************************************************************************?
 * (c) Virginia Polytechnic Insitute and State University, 2011.
 * This is the source code for StreamMR, a MapReduce framework on graphics processing units.
 * Developer:  Marwa K. Elteir (City of Scientific Researches and Technology Applications, Egypt)
 *******************************************************************************************************/

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

typedef struct
{
    int line_num;
} KEY_T;

typedef struct
{
    int line_offset;
    int line_size;
} VAL_T;

__constant int  warpSize = 64; //32 for NVIDIA should 64 for AMD

/****************************************************************************/
//Map phase kernels
/****************************************************************************/

uint hash(__constant uint* constantDataset,void* key, uint keysize)
{
    unsigned long hash = 0; //5381;

    char *str = (char *)key;

    for (int i = 0; i < keysize; i++)
    {
        hash += (int)str[i]; //((hash << 5) + hash) + ((int)str[i]); //  hash * 33 + c
    }

    return hash;

}


uint combineSize(__constant uint* constantDataset)
{
    return  sizeof(int); 
}

//------------------------------------------------------------------------------------------------------------------------
// Getting two values combineLL, combineGL, and combineGG combine both values and insert the result into the first value
// stage = 1 when v1 is null, and v2 is used to initialize v1
// stage = 0 when v1, and v2 to be combined into v1
// stage = 2 when v2 is null, and there is special processing to be done to v1
//------------------------------------------------------------------------------------------------------------------------
void combineLL (__global void* inputDataset,__constant uint* constantDataset,uint key, __local void * v1, __local void* v2, int stage)
{
    __local int * iv1= (__local int*) v1;
    __local int * iv2= (__local int*) v2;

    if (stage == 1 )
    {
        //just initialize v1
        *iv1 = *iv2;
    }
    else
    {
        if (stage == 0 )     *iv1 += *iv2;
    }

}

void combineGL (__global void* inputDataset,__constant uint* constantDataset,__global void * v1, __local void* v2, int stage)
{
    __global int * iv1= (__global int*) v1;     //cluster dimensions
    __local int * iv2= (__local int*) v2;

    if (stage == 1 )
    {
        //just initialize v1
        *iv1 = *iv2;
    }
    else
    {
        if (stage == 0 )  *iv1 += *iv2;
    }

}

void  combineGG(__constant uint* constantDataset, __global void * v1, __global void * v2, uint stage)
{
    __global int * iv1 =  (__global void *)v1;
    __global int * iv2 =  (__global void *)v2;

    if (stage == 1 )
        *iv1 = *iv2;
    else
    {
        if (stage == 0 )   *iv1 += *iv2;
    }

}

// 2 k1 > k2 - 3 k2 > k1  -  1 equal
uint keyEqualGG(__global void * k1, uint k1Size, __global void * k2, uint k2Size)
{
    __global char* word1 = (__global char*)k1;
    __global char* word2 = (__global char*)k2;

    for (int i=0; i < k1Size; i++)
    {
        int iword1, iword2;
        iword1=(int)*word1;
        iword2=(int)*word2;
        if (iword1 == iword2)
        {
            word1++;
            word2++;
        }
        else
        {
            if (iword1 > iword2)
                return 2;  //1 > 2
            else
                return 3; //1 < 2
        }
    }

    return 1;
}

// 0 Not equal -  1 equal
uint keyEqualLL(__local void * k1, uint k1Size, __local void * k2, uint k2Size)
{
    __local char* word1 = (__local char*)k1;
    __local char* word2 = (__local char*)k2;

    if (k1Size != k2Size) return 0;

    for (int i=0; i < k1Size; i++)
    {
        int iword1, iword2;
        iword1=(int)*word1;
        iword2=(int)*word2;
        if (iword1 == iword2)
        {
            word1++;
            word2++;
        }
        else
            return 0;
    }

    return 1;

}

// 2 k1 > k2 - 3 k2 > k1 -  1 equal
uint keyEqualLG(__local void * k1, uint k1Size, __global void * k2, uint k2Size)
{

    __local char* word1 = (__local char*)k1;
    __global char* word2 = (__global char*)k2;

    //   if (k1Size != k2Size) return 0;

    for (int i=0; i < k1Size; i++)
    {
        int iword1, iword2;
        iword1=(int)*word1;
        iword2=(int)*word2;
        if (iword1 == iword2)
        {
            word1++;
            word2++;
        }
        else
        {
            if (iword1 > iword2)
                return 2;  //1 > 2
            else
                return 3; //1 < 2			
        }
    }
    return 1;

}



//-------------------------------------------------------------------------------
//    Handle local memory overflow if it occurs by working on stages i.e., half
//    threads in a wavefront work first then the other half
//-------------------------------------------------------------------------------
int handleLocalOverflow(__local uint * lUsedLocalBuffer,
        __local uint * activeThreadsPerWavefront,
        __local uint * activePortion,
        uint localBufferSize)
{
    uint tidInGroup = get_local_id(0);
    uint widInGroup = tidInGroup >> 6; 

    mem_fence(CLK_LOCAL_MEM_FENCE);
    if ( lUsedLocalBuffer[widInGroup] > localBufferSize)
    {

        //Reset the wavefrontBuffer
        lUsedLocalBuffer[widInGroup] = 0;

        //Adjust the number of active threads in the wavefront 
        // activeThreads all threads: 0 - half threads: 1 - quarter threads: 2 
        // active portion first half: 0 - second half: 1
        // first quarter: 0 - second quarter: 1 - third quarter: 2 - fourth quarter: 3
        activeThreadsPerWavefront[widInGroup]++;
        activePortion[widInGroup] = -1;
        mem_fence(CLK_LOCAL_MEM_FENCE);
        return 1;

    }
    return 0;
}

void emitIntermediateExtended(__global void* inputDataset,
        __constant uint* constantDataset,
        void*  key,
        void*  val,
        uint           keySize,
        uint           valSize,

        __global char *  outputKeys,
        __global char *  outputVals,
        __global int4 *  hashTable,
        __global int4 *  hashBucket, //x: keyOffset y:keySize z:valOffset w: next pointer

        __constant uint * outputKeysOffsets,
        __constant uint * outputValsOffsets,  
        __constant uint * hashBucketOffsets,

        __local uint *  lsOutKeySizes,
        __local uint *  lsOutValSizes,
        __local uint *  lsHashBucketSizes,

        __local uint *  lHashedInterKeys,
        __local char *  lLocalInterKeys,
        __local char *  lLocalInterVals,				
        __local char *  lLocalScratch,
        __local uint *  lUsedLocalBuffer,
        __local uint *  lUsedScratch,
        __local uint *  lLocalEmitted,
        __local uint *  lLocalInterSizes,
        __local uint *  lLocalInterOffsets,
        __local uint *  lslaveThreads,
        __local uint *  lLocalEmitted2,

        uint localBufferSizePerWavefront,
        uint localScratchPerWavefront,
        uint outputKeysBufferSize,
        uint outputValsBufferSize,
        uint hashBucketBufferSize,
        uint hashEntriesNum,
        __local uint *  activeThreadsPerWavefront,
        __local uint *  activePortion,

        uint inputRecordId,
        __global int * metaEmitted,
        __global int * metaOverflow,
        __global int* metaEmitted2)
{
    uint keySizes, valSizes, counts, combValSizes, combSize, localKeyStart, localValStart;
    int4  hashEntry;
    uint tid=get_global_id(0);
    uint groupId=get_group_id(0);
    uint groupsNum=get_num_groups(0);
    uint tidInGroup = get_local_id(0);
    uint widInGroup = tidInGroup >> 6; 	//Wavefront Id in a workgroup
    uint tidInWavefront = tidInGroup % 64; 
    uint wavefrontStart =  widInGroup << 6;
    uint wid = tid >> 6;
    uint wavefrontSize = 64;

    activeThreadsPerWavefront[widInGroup] = 0;
    activePortion[widInGroup] = -1 ;
    uint startThread, endThread;
    lUsedLocalBuffer[widInGroup] = 0;
    lUsedScratch[widInGroup] = 0;

    while(1)
    {
        //------------------------------------------------------------------------------------
        // 0- Handle local overflow by reducing the number of active threads per wavefront
        //------------------------------------------------------------------------------------
        lUsedLocalBuffer[widInGroup] = 0;
        lUsedScratch[widInGroup] = 0;
        activePortion[widInGroup]++;

        if  (activeThreadsPerWavefront[widInGroup] == 0) // No overflow
        {
            startThread = 0; 	
            endThread = wavefrontSize - 1;
        }
        else
        {
            if  (activeThreadsPerWavefront[widInGroup] == 1) //Overflow half wavefront
            {
                if ( activePortion[widInGroup] == 0) // first half 0-31
                {
                    startThread = 0;
                    endThread = (wavefrontSize >> 1) - 1;
                }
                else  //second half 32 -63
                {
                    startThread = wavefrontSize >> 1;
                    endThread = wavefrontSize - 1;
                }
            }
            else // Overflow quarter wavefront
            {
                if (activePortion[widInGroup] == 0) //first quarter 0-15
                {
                    startThread = 0;
                    endThread = (wavefrontSize >> 2) - 1;
                }
                else
                {
                    if (activePortion[widInGroup] == 1) //second quarter 16-31
                    {
                        startThread = wavefrontSize >> 2;
                        endThread = (wavefrontSize >> 1) - 1;

                    }
                    else
                    {
                        if (activePortion[widInGroup] == 2) // third quarter 32- 47
                        {
                            startThread = wavefrontSize >> 1;
                            endThread = ((wavefrontSize >> 2) * 3) - 1;

                        }
                        else //fourth quarter 48 -64
                        {
                            startThread = (wavefrontSize >> 2) * 3;
                            endThread = wavefrontSize - 1 ;
                        }
                    }						
                }
            }

        }

        startThread += wavefrontStart;
        endThread += wavefrontStart;
        if ( ( tidInGroup >= startThread) && ( tidInGroup <= endThread))
        {		
            //----------------------------------------------------
            // 1- Each thread write hash of its key to local memory
            //--------------------------------------------------
            //Insert hashed key	
            int hashedk;
            hashedk = fmod ((float)hash(constantDataset, key, keySize) , hashEntriesNum) ;
            lLocalEmitted2[tidInGroup]++;
            lHashedInterKeys[tidInGroup]  = lLocalEmitted2[tidInGroup] <=  lLocalEmitted[tidInGroup] ? -1 : hashedk;//successfully emitted in the first map phase
            lslaveThreads[tidInGroup]= 0;
            mem_fence(CLK_LOCAL_MEM_FENCE);

            //--------------------------------------------------------------------------------
            // 2- Each thread pass through the previous hashed keyes and the first thread with
            //    any key is marked as master and other threads with the same key as slaves
            //--------------------------------------------------------------------------------
            for (int i = startThread; i <= endThread; i++)
            {
                if ( lHashedInterKeys[i] == lHashedInterKeys[tidInGroup] )
                    if ( tidInGroup < i )  lslaveThreads[i] = 1;

            }
            mem_fence(CLK_LOCAL_MEM_FENCE);

            //--------------------------------------------------------------------------------
            // 3- All threads write their keys and values to the local memory 
            //    Here Key is variable and Values is constant size
            //--------------------------------------------------------------------------------
            uint lkeySizes = atom_add(&lUsedLocalBuffer[widInGroup],keySize);
            uint ret = handleLocalOverflow(lUsedLocalBuffer, activeThreadsPerWavefront, activePortion, localBufferSizePerWavefront);
            if (ret == 1)
            {
                lLocalEmitted2[tidInGroup]--;
                continue;
            }	
            lLocalInterOffsets[tidInGroup]= lkeySizes;
            lLocalInterSizes[tidInGroup] = keySize;

            //Begining From interValsOffsets[groupsNum] the local offsets of each wavefront are stored
            __local char *lVals = (__local char*)(lLocalInterVals +  localScratchPerWavefront * widInGroup + tidInWavefront * valSize);
            char* sVal = (char*)val;
            for (int i = 0; i <valSize; ++i)    lVals[i] = sVal[i];

            //Begining From interKeysOffsets[groupsNum] the local offsets of each wavefront are stored
            __local char *lKeys = (__local char*)(lLocalInterKeys + localBufferSizePerWavefront * widInGroup + lkeySizes);
            char* sKey = (char*)key;
            for (int i = 0; i <keySize; ++i)    lKeys[i] = sKey[i];

            mem_fence(CLK_LOCAL_MEM_FENCE);

            //--------------------------------------------------------------------------------
            // 4- Master thread begins combining the values of slave threads 
            //-------------------------------------------------------------------------------

            if ( lslaveThreads[tidInGroup] == 0)
            {	
                combSize  = combineSize(constantDataset);
                combValSizes = atom_add(&lUsedScratch[widInGroup] , combSize);
                combValSizes+= localScratchPerWavefront*widInGroup;
                uint ret = handleLocalOverflow(lUsedScratch, activeThreadsPerWavefront, activePortion, localScratchPerWavefront);
                if (ret == 1)  lLocalEmitted2[tidInGroup]--; 

                if (ret == 0) //No local Overflow
                {

                    uint revertToCounting = 0;
                    localKeyStart = lLocalInterKeys + localBufferSizePerWavefront * widInGroup;
                    localValStart = lLocalInterVals + localScratchPerWavefront * widInGroup;	

                    int remaining = 1;					
                    while ( remaining == 1)
                    {
                        remaining = 0;
                        int masterKeyId = -1;
                        uint entryExist = 0;
                        int before = -1,after = -1; //indicate where exactly to insert a new element in the linked lisi
                        int4 masterHashBucket;
                        uint outValSizes, outKeySizes, hashBucketSizes;
                        //-----------------------------------------------------------------------------------
                        // 4.1 - pass through all slave threads and generates possibly several combine values
                        //-----------------------------------------------------------------------------------

                        for (int i = tidInGroup; i <= endThread; i++)
                        {
                            if ( ( lHashedInterKeys[i] == hashedk ) && (lHashedInterKeys[i] != -1))
                            {
                                if (masterKeyId == -1 ) 
                                {
                                    masterKeyId = i;
                                    combineLL( inputDataset, constantDataset,  lHashedInterKeys[i],lLocalScratch + combValSizes,
                                            localValStart + (i % 64 )*  valSize, 1); 
                                    lHashedInterKeys[i]= -1;

                                    //Check whether the hash entry exist or not
                                    //-----------------------------------------
                                    mem_fence(CLK_GLOBAL_MEM_FENCE);
                                    __global int4 *hTable= (__global int4*) hashTable + hashEntriesNum * wid;
                                    hashEntry = hTable[hashedk]; //x: first pointer y:last pointer z: num of elements

                                    if ( hashEntry.z == 0 )
                                    {
                                        entryExist = 0;
                                    }
                                    else
                                    {
                                        //pass through all elements in linked list and see whether exist or not
                                        //---------------------------------------------------------------------
                                        masterHashBucket =  hashBucket[hashEntry.x];
                                        after = hashEntry.x;								
                                        for (int j=0; j < hashEntry.z ; j++)
                                        {
                                            uint keyOffset =  outputKeys + masterHashBucket.x;
                                            uint comp = keyEqualLG(localKeyStart+lLocalInterOffsets[masterKeyId],
                                                    lLocalInterSizes[masterKeyId], keyOffset, masterHashBucket.y);

                                            if ( comp == 1)
                                            {
                                                entryExist = 1;
                                                break;
                                            }

                                            else
                                            { 
                                                if (comp == 3) //G > L
                                                {
                                                    entryExist = 0	;
                                                    break;
                                                }

                                            }	

                                            before = after;
                                            after = masterHashBucket.w;
                                            masterHashBucket = hashBucket[masterHashBucket.w];
                                        }  
                                    }
                                    if (entryExist == 0)
                                    {
                                        outValSizes = atom_add(&lsOutValSizes[1],combSize); //used size
                                        outKeySizes = atom_add(&lsOutKeySizes[1],lLocalInterSizes[masterKeyId]); //used size
                                        hashBucketSizes = atom_inc(&lsHashBucketSizes[1]); //used size
                                    }
                                }
                                else
                                {
                                    //Compare the master key with the current key
                                    //-------------------------------------------
                                    if (keyEqualLL(localKeyStart + lLocalInterOffsets[masterKeyId],lLocalInterSizes[masterKeyId],
                                                localKeyStart + lLocalInterOffsets[i],lLocalInterSizes[i]))
                                    {
                                        //Calculate the combined value
                                        combineLL( inputDataset, constantDataset, lHashedInterKeys[i],  lLocalScratch + combValSizes ,
                                                localValStart + (i % 64 )*  valSize,0 );
                                        lHashedInterKeys[i]= -1;
                                    }
                                    else
                                        remaining = 1;
                                }
                            }//End if
                        }//End for


                        //if no overflow write the key/value pair to the global buffer
                        //------------------------------------------------------------
                        if (masterKeyId != -1)
                        {
                            if (entryExist == 0)
                            {
                                __global char *pOutKeySet = (__global char*)( outputKeys + outputKeysOffsets[groupId] + outKeySizes);
                                __local char *lKeys = (__local char*)(localKeyStart + lLocalInterOffsets[masterKeyId]);
                                for (int i = 0; i < lLocalInterSizes[masterKeyId] ; ++i)   pOutKeySet[i] = lKeys[i];


                                __global char *pOutValSet = (__global char*)( outputVals + outputValsOffsets[groupId] + outValSizes);    
                                __local char *lVals = (__local char*)(lLocalScratch + combValSizes);
                                for (int i = 0; i <combSize; ++i)   pOutValSet[i] = lVals[i];

                                int4 bucketElement;
                                bucketElement.x = outputKeysOffsets[groupId] + outKeySizes;
                                bucketElement.y = lLocalInterSizes[masterKeyId];
                                bucketElement.z = outputValsOffsets[groupId] + outValSizes;
                                bucketElement.w = after;

                                __global int4* hTable= (__global int4*) hashTable +  hashEntriesNum * wid;
                                hashEntry=  hTable[hashedk];

                                if (before == -1)	
                                    hashEntry.x =  hashBucketOffsets[groupId]+ hashBucketSizes;
                                else
                                    hashBucket[before].w =  hashBucketOffsets[groupId] + hashBucketSizes;

                                hashEntry.z++;
                                hashEntry.y = 1;
                                hashEntry.w = hashedk;

                                hTable[hashedk] = hashEntry;
                                hashBucket[ hashBucketOffsets[groupId] + hashBucketSizes] = bucketElement;	
                                mem_fence(CLK_GLOBAL_MEM_FENCE);
                            }
                            else
                            {
                                //Update the already existed entry
                                //---------------------------------
                                combineGL(inputDataset, constantDataset, outputVals + masterHashBucket.z,
                                        lLocalScratch + combValSizes, 0);
                                mem_fence(CLK_GLOBAL_MEM_FENCE);
                            }
                        }


                        if (remaining == 0 ) break;
                    }//End while


                } //End if local overflow


            } // End if Master thread	

        }//End if active threads



        // Exiting condition 	

        if  (activeThreadsPerWavefront[widInGroup] == 0)
            break;
        else
            if  (activeThreadsPerWavefront[widInGroup] == 1)
            {	
                if ( activePortion[widInGroup] == 1) break;
            }
            else
                if ( activePortion[widInGroup] == 3) break;		  

    }// End While
}
//---------------------------------------------------------------------------
// Called by user-defined map function
// Threads in a wavefront collaborate to reduce the values of each unique key
// and write the results to a hash table per wavefront
//---------------------------------------------------------------------------
void emitIntermediatePerfect(__global void* inputDataset,
        __constant uint* constantDataset,
        void*  key,
        void*  val,
        uint           keySize,
        uint           valSize,

        __global char * interKeys,
        __global char *  interVals,
        __global int4 * interOffsets,

        __global char *  outputKeys,
        __global char *  outputVals,
        __global int4 *  hashTable,
        __global int4 *  hashBucket, //x: keyOffset y:keySize z:valOffset w: next pointer

        __constant uint * outputKeysOffsets,
        __constant uint * outputValsOffsets,
        __constant uint * hashBucketOffsets,

        __local uint *  lsOutKeySizes,
        __local uint *  lsOutValSizes,
        __local uint *  lsHashBucketSizes,

        __local uint *  lHashedInterKeys,
        __local char *  lLocalInterKeys,
        __local char *  lLocalInterVals,				
        __local char *  lLocalScratch,
        __local uint *  lUsedLocalBuffer,
        __local uint *  lUsedScratch,
        __local uint *  lLocalInputRecordId,
        __local uint *  lLocalInterSizes,
        __local uint *  lLocalInterOffsets,
        __local uint *  lslaveThreads,
        __local uint *  lLocalEmitted2,

        uint localBufferSizePerWavefront,
        uint localScratchPerWavefront,
        uint outputKeysBufferSizePerWG,
        uint outputValsBufferSizePerWG,
        uint hashBucketBufferSizePerWG,
        uint hashEntriesNum,
        __local uint *  activeThreadsPerWavefront,
        __local uint *  activePortion,

        uint inputRecordId,
        int extended,
        int outputIntermediate,
        __global int * metaEmitted,
        __global int * metaOverflow,
        __global int* metaEmitted2,
        uint interKeysPerWorkgroup,
        uint interValsPerWorkgroup,
        uint interCountsPerWorkgroup)
{


    uint keySizes, valSizes, counts, combValSizes, combSize, localKeyStart, localValStart;
    int4  hashEntry;
    uint tid=get_global_id(0);
    uint groupId=get_group_id(0);
    uint groupsNum=get_num_groups(0);
    uint tidInGroup = get_local_id(0);
    uint widInGroup = tidInGroup >> 6; 	//Wavefront Id in a workgroup
    uint tidInWavefront = tidInGroup % 64; 
    uint wavefrontStart =  widInGroup << 6;
    uint wid = tid >> 6;
    uint wavefrontSize = 64;


    if ( metaOverflow[inputRecordId] == 1) //Stop emitting when overflow occurs
    {
        atom_add(&lsOutValSizes[0],valSize); //need size
        atom_add(&lsOutKeySizes[0],keySize); //need size
        atom_inc(&lsHashBucketSizes[0]); //need size
        mem_fence(CLK_LOCAL_MEM_FENCE);
        return;
    }


    activeThreadsPerWavefront[widInGroup] = 0;
    activePortion[widInGroup] = -1 ;
    int it = 0;
    uint startThread, endThread;

    while( 1)
    {
        //------------------------------------------------------------------------------------
        // 0- Handle local overflow by reducing the number of active threads per wavefront
        //------------------------------------------------------------------------------------
        it++;
        lUsedLocalBuffer[widInGroup] = 0;
        lUsedScratch[widInGroup]= 0;
        activePortion[widInGroup]++;
        mem_fence(CLK_LOCAL_MEM_FENCE);

        if  (activeThreadsPerWavefront[widInGroup] == 0) // No overflow
        {
            startThread = 0; 	
            endThread = wavefrontSize - 1;
        }
        else
        {
            if  (activeThreadsPerWavefront[widInGroup] == 1) //Overflow half wavefront
            {
                if ( activePortion[widInGroup] == 0) // first half 0-31
                {
                    startThread = 0;
                    endThread = (wavefrontSize >> 1) - 1;
                }
                else  //second half 32 -63
                {
                    startThread = wavefrontSize >> 1;
                    endThread = wavefrontSize - 1;
                }
            }
            else // Overflow quarter wavefront
            {
                if (activePortion[widInGroup] == 0) //first quarter 0-15
                {
                    startThread = 0;
                    endThread = (wavefrontSize >> 2) - 1;
                }
                else
                {
                    if (activePortion[widInGroup] == 1) //second quarter 16-31
                    {
                        startThread = wavefrontSize >> 2;
                        endThread = (wavefrontSize >> 1) - 1;

                    }
                    else
                    {
                        if (activePortion[widInGroup] == 2) // third quarter 32- 47
                        {
                            startThread = wavefrontSize >> 1;
                            endThread = ((wavefrontSize >> 2) * 3) - 1;

                        }
                        else //fourth quarter 48 -64
                        {
                            startThread = (wavefrontSize >> 2) * 3;
                            endThread = wavefrontSize - 1 ;
                        }
                    }						
                }
            }

        }
        startThread += wavefrontStart;
        endThread += wavefrontStart;

        if ( ( tidInGroup >= startThread) && ( tidInGroup <= endThread))
        {		

            //----------------------------------------------------
            // 1- Each thread write hash of its key to local memory
            //--------------------------------------------------
            //Insert hashed key	
            int hashedk;
            hashedk = fmod ((float)hash(constantDataset, key, keySize) , hashEntriesNum) ;
            lHashedInterKeys[tidInGroup]  = hashedk;
            lslaveThreads[tidInGroup]= 0;
            mem_fence(CLK_LOCAL_MEM_FENCE);

            //--------------------------------------------------------------------------------
            // 2- Each thread pass through the previous hashed keyes and the first thread with
            //    any key is marked as master and other threads with the same key as slaves
            //--------------------------------------------------------------------------------
            for (int i = startThread; i <= endThread; i++)
            {
                if ( lHashedInterKeys[i] == lHashedInterKeys[tidInGroup] )
                    if ( tidInGroup < i )  lslaveThreads[i]=1;

            }
            mem_fence(CLK_LOCAL_MEM_FENCE);

            //--------------------------------------------------------------------------------
            // 3- All threads write their keys and values to the local memory 
            //    Here Key is variable and Values is constant size
            //--------------------------------------------------------------------------------
            lLocalInputRecordId[tidInGroup]= inputRecordId;

            //use localScratchPerWavefront and widInGroup to zoomin to my portion of the local values
            __local char *lVals = (__local char*)(lLocalInterVals + localScratchPerWavefront*widInGroup + tidInWavefront * valSize);
            char* sVal = (char*)val;
            for (int i = 0; i <valSize; ++i)    lVals[i] = sVal[i];

            mem_fence(CLK_LOCAL_MEM_FENCE);

            //--------------------------------------------------------------------------------
            // 4- Master thread begins combining the values of slave threads 
            //-------------------------------------------------------------------------------

            if ( lslaveThreads[tidInGroup] == 0)
            {	
                combSize  = combineSize(constantDataset);
                combValSizes = atom_add(&lUsedScratch[widInGroup] , combSize);
                combValSizes+= localScratchPerWavefront*widInGroup;
                uint ret = handleLocalOverflow(lUsedScratch, activeThreadsPerWavefront, activePortion, localScratchPerWavefront);

                uint outValSizes, outKeySizes, hashBucketSizes;

                if (ret == 0) //No local Overflow
                {
                    uint revertToCounting = 0;
                    localKeyStart = lLocalInterKeys +  localBufferSizePerWavefront* widInGroup ;
                    localValStart = lLocalInterVals +  localScratchPerWavefront* widInGroup;	

                    //----------------------------------------
                    //4.1 Check whether the hash entry exist or not
                    //-----------------------------------------

                    mem_fence(CLK_GLOBAL_MEM_FENCE);
                    __global int4 * hTable= (__global int4*) hashTable +  hashEntriesNum * wid;
                    hashEntry = hTable[hashedk]; //x: first pointer y:last pointer z: num of elements
                    uint entryExist = 1;

                    if ( hashEntry.z == 0 )  entryExist = 0;

                    // If hashbucket does not exist check whether global overflow will occur or not
                    //-----------------------------------------------------------------------------
                    if (entryExist == 0)
                    {
                        outValSizes = atom_add(&lsOutValSizes[0],combSize); //need size
                        outKeySizes = atom_add(&lsOutKeySizes[0],keySize); //need size
                        hashBucketSizes = atom_inc(&lsHashBucketSizes[0]); //need size
                        mem_fence(CLK_LOCAL_MEM_FENCE);

                        if (( outValSizes >= outputValsBufferSizePerWG ) || (outKeySizes >= outputKeysBufferSizePerWG)||(hashBucketSizes >= hashBucketBufferSizePerWG))
                        {
                            revertToCounting = 1;
                        }
                        else
                        {
                            revertToCounting = 0;
                            outValSizes = atom_add(&lsOutValSizes[1],combSize); //used size
                            outKeySizes = atom_add(&lsOutKeySizes[1],keySize); //used size
                            hashBucketSizes = atom_inc(&lsHashBucketSizes[1]); //used size
                        }
                    }
                    else
                    {
                        revertToCounting = 0;
                    }

                    //----------------------------------------------------------------------
                    // 4.1 - pass through all slave threads and generates the combined value
                    //----------------------------------------------------------------------
                    for (int i = tidInGroup; i <= endThread; i++)
                    {
                        if ( ( lHashedInterKeys[i] == hashedk ) && (lHashedInterKeys[i] != -1))
                        {
                            combineLL( inputDataset, constantDataset, lHashedInterKeys[i], lLocalScratch + combValSizes,
                                    localValStart + (i % 64 )*  valSize, (i==tidInGroup)?1:0); //localValStart + 63*  valSize,

                            lHashedInterKeys[i]= -1;
                            if (revertToCounting == 0)
                                metaEmitted[lLocalInputRecordId[i]] += 1;
                            else
                                metaOverflow[lLocalInputRecordId[i]] = 1;
                        }
                    }//End for


                    //if no overflow write the key/value pair to the global buffer
                    //------------------------------------------------------------
                    hTable= (__global int4*) hashTable +  hashEntriesNum * wid;
                    hashEntry=  hTable[hashedk];
                    if (entryExist == 0)
                    {
                        if ( revertToCounting == 0)
                        {

                            __global char *pOutKeySet = (__global char*)( outputKeys +  groupId * outputKeysBufferSizePerWG  + outKeySizes);
                            char* sKey = (char*)key;
                            for (int i = 0; i <keySize; ++i)  pOutKeySet[i] = sKey[i];

                            __global char *pOutValSet = (__global char*)( outputVals + groupId * outputValsBufferSizePerWG + outValSizes);   
                            __local char *lVals = (__local char*)(lLocalScratch + combValSizes);
                            for (int i = 0; i < combSize; ++i)   pOutValSet[i] = lVals[i]; 

                            int4 bucketElement;
                            bucketElement.x =  groupId * outputKeysBufferSizePerWG + outKeySizes; 
                            bucketElement.y =  keySize;
                            bucketElement.z =  groupId * outputValsBufferSizePerWG + outValSizes; 
                            bucketElement.w =  -1;

                            hashEntry.x = hashBucketBufferSizePerWG * groupId + hashBucketSizes;
                            hashEntry.z++;
                            hashEntry.y = 1;    
                            hashEntry.w = hashedk; 
                            hTable[hashedk] = hashEntry;
                            hashBucket[hashBucketBufferSizePerWG * groupId + hashBucketSizes] = bucketElement;						    

                            mem_fence(CLK_GLOBAL_MEM_FENCE);

                        } 
                    }

                    else
                    {


                        //Update the already existed entry
                        //---------------------------------
                        combineGL(inputDataset, constantDataset, outputVals + hashBucket[hTable[hashedk].x].z,
                                lLocalScratch + combValSizes, 0);
                        mem_fence(CLK_GLOBAL_MEM_FENCE);


                    }

                } //End if local overflow
            } // End if Master thread	
        }//End if active threads

        // Exiting condition 	
        if  (activeThreadsPerWavefront[widInGroup] == 0)
            break;
        else
            if  (activeThreadsPerWavefront[widInGroup] == 1)
            {	
                if ( activePortion[widInGroup] == 1) break;
            }
            else
                if ( activePortion[widInGroup] == 3) break;		  
    }// End While

}

//---------------------------------------------------------------------------
// Called by user-defined map function
// Threads in a wavefront collaborate to reduce the values of each unique key
// and write the results to a hash table per wavefront
//---------------------------------------------------------------------------
void emitIntermediate(__global void* inputDataset,
        __constant uint* constantDataset,
        void*  key,
        void*  val,
        uint           keySize,
        uint           valSize,

        __global char * interKeys,
        __global char *  interVals,
        __global int4 * interOffsets,

        __global char *  outputKeys,
        __global char *  outputVals,
        __global int4 *  hashTable,
        __global int4 *  hashBucket, //x: keyOffset y:keySize z:valOffset w: next pointer

        __constant uint * outputKeysOffsets,
        __constant uint * outputValsOffsets,
        __constant uint * hashBucketOffsets,

        __local uint *  lsOutKeySizes,
        __local uint *  lsOutValSizes,
        __local uint *  lsHashBucketSizes,

        __local uint *  lHashedInterKeys,
        __local char *  lLocalInterKeys,
        __local char *  lLocalInterVals,				
        __local char *  lLocalScratch,
        __local uint *  lUsedLocalBuffer,
        __local uint *  lUsedScratch,
        __local uint *  lLocalInputRecordId,
        __local uint *  lLocalInterSizes,
        __local uint *  lLocalInterOffsets,
        __local uint *  lslaveThreads,
        __local uint *  lLocalEmitted2,

        uint localBufferSizePerWavefront,
        uint localScratchPerWavefront,
        uint outputKeysBufferSizePerWG,
        uint outputValsBufferSizePerWG,
        uint hashBucketBufferSizePerWG,
        uint hashEntriesNum,
        __local uint *  activeThreadsPerWavefront,
        __local uint *  activePortion,

        uint inputRecordId,
        int extended,
        int outputIntermediate,
        __global int * metaEmitted,
        __global int * metaOverflow,
        __global int* metaEmitted2,
        uint interKeysPerWorkgroup,
        uint interValsPerWorkgroup,
        uint interCountsPerWorkgroup)
{
    uint keySizes, valSizes, counts, combValSizes, combSize, localKeyStart, localValStart;
    int4 hashEntry;
    uint tid=get_global_id(0);
    uint groupId=get_group_id(0);
    uint groupsNum=get_num_groups(0);
    uint tidInGroup = get_local_id(0);
    uint widInGroup = tidInGroup >> 6; 	//Wavefront Id in a workgroup
    uint tidInWavefront = tidInGroup % 64; 
    uint wavefrontStart =  widInGroup << 6;
    uint wid = tid >> 6;
    uint wavefrontSize = 64;

#ifdef OUTPUTINTER
    //if (outputIntermediate == 1)
    //{
    keySizes = atom_add(&lsOutKeySizes[2],keySize); //used size
    valSizes = atom_add(&lsOutValSizes[2],valSize);
    counts = atom_inc(&lsHashBucketSizes[2]);

    __global char *pKeySet = (__global char*)(interKeys + interKeysPerWorkgroup * groupId + keySizes);
    __global char *pValSet = (__global char*)(interVals + interValsPerWorkgroup * groupId + valSizes);

    char* sKey = (char*)key;
    for (int i = 0; i <keySize; ++i)   pKeySet[i] = sKey[i];

    char* sVal = (char*)val;
    for (int i = 0; i <valSize; ++i)   pValSet[i] = sVal[i];

    __global int4 * offset= (__global int4*)(interOffsets + interCountsPerWorkgroup * groupId);
    int4 l_interOffsetSizes;
    l_interOffsetSizes.x = interKeysPerWorkgroup * groupId  + keySizes;  //keySizes;
    l_interOffsetSizes.y = keySize;
    l_interOffsetSizes.z = interValsPerWorkgroup * groupId + valSizes;
    l_interOffsetSizes.w = valSize;
    offset[counts] = l_interOffsetSizes;

    //}
#endif

#ifdef PERFECT
    // if (extended == 2)
    // {
    //efficient handle for applications with perfect hashing functions
    emitIntermediatePerfect(inputDataset,constantDataset,  key, val, keySize, valSize, interKeys,interVals, interOffsets, outputKeys, outputVals
            ,hashTable,
            hashBucket, outputKeysOffsets,outputValsOffsets, hashBucketOffsets,
            lsOutKeySizes, lsOutValSizes,
            lsHashBucketSizes,lHashedInterKeys, lLocalInterKeys, lLocalInterVals,lLocalScratch,lUsedLocalBuffer, lUsedScratch,
            lLocalInputRecordId, lLocalInterSizes, lLocalInterOffsets, lslaveThreads, lLocalEmitted2, localBufferSizePerWavefront,
            localScratchPerWavefront,
            outputKeysBufferSizePerWG,
            outputValsBufferSizePerWG,hashBucketBufferSizePerWG, hashEntriesNum, activeThreadsPerWavefront,activePortion,
            inputRecordId, extended,outputIntermediate,metaEmitted,metaOverflow,metaEmitted2,interKeysPerWorkgroup, interValsPerWorkgroup,
            interCountsPerWorkgroup);
    return;
    // }
#else

    if ( metaOverflow[inputRecordId] == 1) //Stop emitting when overflow occurs
    {
        atom_add(&lsOutValSizes[0],valSize); //need size
        atom_add(&lsOutKeySizes[0],keySize); //need size
        atom_inc(&lsHashBucketSizes[0]); //need size
        mem_fence(CLK_LOCAL_MEM_FENCE);
        return;
    }


    activeThreadsPerWavefront[widInGroup] = 0;
    activePortion[widInGroup] = -1 ;
    int it = 0;
    uint startThread, endThread;

    while( 1)
    {
        //------------------------------------------------------------------------------------
        // 0- Handle local overflow by reducing the number of active threads per wavefront
        //------------------------------------------------------------------------------------
        it++;
        lUsedLocalBuffer[widInGroup] = 0;
        lUsedScratch[widInGroup]= 0;
        activePortion[widInGroup]++;
        mem_fence(CLK_LOCAL_MEM_FENCE);

        if  (activeThreadsPerWavefront[widInGroup] == 0) // No overflow
        {
            startThread = 0; 	
            endThread = wavefrontSize - 1;
        }
        else
        {
            if  (activeThreadsPerWavefront[widInGroup] == 1) //Overflow half wavefront
            {
                if ( activePortion[widInGroup] == 0) // first half 0-31
                {
                    startThread = 0;
                    endThread = (wavefrontSize >> 1) - 1;
                }
                else  //second half 32 -63
                {
                    startThread = wavefrontSize >> 1;
                    endThread = wavefrontSize - 1;
                }
            }
            else // Overflow quarter wavefront
            {
                if (activePortion[widInGroup] == 0) //first quarter 0-15
                {
                    startThread = 0;
                    endThread = (wavefrontSize >> 2) - 1;
                }
                else
                {
                    if (activePortion[widInGroup] == 1) //second quarter 16-31
                    {
                        startThread = wavefrontSize >> 2;
                        endThread = (wavefrontSize >> 1) - 1;

                    }
                    else
                    {
                        if (activePortion[widInGroup] == 2) // third quarter 32- 47
                        {
                            startThread = wavefrontSize >> 1;
                            endThread = ((wavefrontSize >> 2) * 3) - 1;

                        }
                        else //fourth quarter 48 -64
                        {
                            startThread = (wavefrontSize >> 2) * 3;
                            endThread = wavefrontSize - 1 ;
                        }
                    }						
                }
            }

        }
        startThread += wavefrontStart;
        endThread += wavefrontStart;

        if ( ( tidInGroup >= startThread) && ( tidInGroup <= endThread))
        {		

            //----------------------------------------------------
            // 1- Each thread write hash of its key to local memory
            //--------------------------------------------------
            //Insert hashed key	
            int hashedk;
            hashedk = fmod ((float)hash(constantDataset, key, keySize) , hashEntriesNum) ;
            lHashedInterKeys[tidInGroup]  = hashedk;
            lslaveThreads[tidInGroup]= 0;
            mem_fence(CLK_LOCAL_MEM_FENCE);

            //--------------------------------------------------------------------------------
            // 2- Each thread pass through the previous hashed keyes and the first thread with
            //    any key is marked as master and other threads with the same key as slaves
            //--------------------------------------------------------------------------------
            for (int i = startThread; i <= endThread; i++)
            {
                if ( lHashedInterKeys[i] == lHashedInterKeys[tidInGroup] )
                    if ( tidInGroup < i )  lslaveThreads[i]=1;

            }
            mem_fence(CLK_LOCAL_MEM_FENCE);

            //--------------------------------------------------------------------------------
            // 3- All threads write their keys and values to the local memory 
            //    Here Key is variable and Values is constant size
            //--------------------------------------------------------------------------------
            uint lkeySizes = atom_add(&lUsedLocalBuffer[widInGroup],keySize);
            uint ret = handleLocalOverflow(lUsedLocalBuffer, activeThreadsPerWavefront, activePortion, localBufferSizePerWavefront);
            if (ret == 1) continue;

            lLocalInterOffsets[tidInGroup]= lkeySizes;
            lLocalInterSizes[tidInGroup] = keySize;
            lLocalInputRecordId[tidInGroup]= inputRecordId;

            //use localScratchPerWavefront and widInGroup to zoomin to my portion of the local values
            __local char *lVals = (__local char*)(lLocalInterVals + localScratchPerWavefront*widInGroup + tidInWavefront * valSize);
            char* sVal = (char*)val;
            for (int i = 0; i <valSize; ++i)    lVals[i] = sVal[i];

            //use localBufferSizePerWavefront and widInGroup to zoomin to my portion of the local keys
            __local char *lKeys = (__local char*)(lLocalInterKeys + localBufferSizePerWavefront* widInGroup + lkeySizes);
            char* sKey = (char*)key;
            for (int i = 0; i <keySize; ++i)    lKeys[i] = sKey[i];

            mem_fence(CLK_LOCAL_MEM_FENCE);

            //--------------------------------------------------------------------------------
            // 4- Master thread begins combining the values of slave threads 
            //-------------------------------------------------------------------------------

            if ( lslaveThreads[tidInGroup] == 0)
            {	
                combSize  = combineSize(constantDataset);
                combValSizes = atom_add(&lUsedScratch[widInGroup] , combSize);
                combValSizes+= localScratchPerWavefront*widInGroup;
                uint ret = handleLocalOverflow(lUsedScratch, activeThreadsPerWavefront, activePortion, localScratchPerWavefront);

                uint outValSizes, outKeySizes, hashBucketSizes;

                if (ret == 0) //No local Overflow
                {
                    uint revertToCounting = 0;
                    localKeyStart = lLocalInterKeys +  localBufferSizePerWavefront* widInGroup ;
                    localValStart = lLocalInterVals +  localScratchPerWavefront* widInGroup;	

                    int remaining = 1;					
                    int it2 = 0;
                    while ( remaining == 1)
                    {
                        it2++;
                        remaining = 0;
                        int masterKeyId = -1;
                        uint entryExist = 0;
                        int before = -1,after = -1; //indicate where exactly to insert a new element in the linked lisi
                        int4 masterHashBucket;
                        //-----------------------------------------------------------------------------------
                        // 4.1 - pass through all slave threads and generates possibly several combine values
                        //-----------------------------------------------------------------------------------

                        for (int i = tidInGroup; i <= endThread; i++)
                        {
                            if ( ( lHashedInterKeys[i] == hashedk ) && (lHashedInterKeys[i] != -1))
                            {
                                if (masterKeyId == -1 ) 
                                {
                                    combineLL( inputDataset, constantDataset, lHashedInterKeys[i], lLocalScratch + combValSizes,
                                            localValStart + (i % 64 )*  valSize, 1); //localValStart + 63*  valSize,

                                    lHashedInterKeys[i]= -1;
                                    masterKeyId = i;


                                    //Check whether the hash entry exist or not
                                    //-----------------------------------------
                                    mem_fence(CLK_GLOBAL_MEM_FENCE);
                                    __global int4 * hTable= (__global int4*) hashTable +  hashEntriesNum * wid;
                                    hashEntry = hTable[hashedk]; //x: first pointer y:last pointer z: num of elements
                                    entryExist=0;

                                    if ( hashEntry.z == 0 )
                                    {
                                        entryExist = 0;
                                    }
                                    else
                                    {
                                        //pass through all elements in linked list and see whether exist or not
                                        //---------------------------------------------------------------------
                                        masterHashBucket =  hashBucket[hashEntry.x];
                                        after = hashEntry.x;
                                        for (int j=0; j < hashEntry.z ; j++)
                                        {
                                            uint keyOffset =  outputKeys + masterHashBucket.x;
                                            uint comp = keyEqualLG(localKeyStart+lLocalInterOffsets[masterKeyId],
                                                    lLocalInterSizes[masterKeyId], keyOffset, masterHashBucket.y);

                                            if ( comp == 1)
                                            {
                                                entryExist = 1;
                                                break;
                                            }
                                            else
                                            { 
                                                if (comp == 3) //G > L
                                                {
                                                    entryExist = 0	;
                                                    break;
                                                }
                                            }	
                                            before = after;
                                            after = masterHashBucket.w;
                                            masterHashBucket = hashBucket[masterHashBucket.w];
                                        }  

                                    }


                                    // If hashbucket does not exist check whether global overflow will occur or not
                                    //-----------------------------------------------------------------------------
                                    if (entryExist == 0)
                                    {
                                        outValSizes = atom_add(&lsOutValSizes[0],combSize); //need size
                                        outKeySizes = atom_add(&lsOutKeySizes[0],lLocalInterSizes[masterKeyId]); //need size
                                        hashBucketSizes = atom_inc(&lsHashBucketSizes[0]); //need size
                                        mem_fence(CLK_LOCAL_MEM_FENCE);

                                        if (( outValSizes >= outputValsBufferSizePerWG ) || (outKeySizes >= outputKeysBufferSizePerWG)||(hashBucketSizes >= hashBucketBufferSizePerWG))
                                        {
                                            revertToCounting = 1;
                                            metaOverflow[lLocalInputRecordId[i]] = 1;
                                        }
                                        else
                                        {
                                            revertToCounting = 0;
                                            outValSizes = atom_add(&lsOutValSizes[1],combSize); //used size
                                            outKeySizes = atom_add(&lsOutKeySizes[1],lLocalInterSizes[masterKeyId]); //used size
                                            hashBucketSizes = atom_inc(&lsHashBucketSizes[1]); //used size
                                            metaEmitted[lLocalInputRecordId[i]] +=  1;
                                        }
                                    }
                                    else
                                    {
                                        revertToCounting = 0;
                                        metaEmitted[lLocalInputRecordId[i]] += 1;
                                    }
                                }
                                else
                                {
                                    //Compare the master key with the current key
                                    //-------------------------------------------
                                    if (keyEqualLL(localKeyStart + lLocalInterOffsets[masterKeyId],lLocalInterSizes[masterKeyId],
                                                localKeyStart + lLocalInterOffsets[i],lLocalInterSizes[i]))
                                    {
                                        if ( revertToCounting == 0)
                                        {
                                            //Calculate the combined value
                                            combineLL( inputDataset, constantDataset, lHashedInterKeys[i], lLocalScratch + combValSizes ,
                                                    localValStart + (i % 64 )*  valSize,0 );
                                            metaEmitted[lLocalInputRecordId[i]] += 1;
                                            lHashedInterKeys[i]= -1;
                                        }
                                        else
                                            metaOverflow[lLocalInputRecordId[i]] = 1;
                                    }
                                    else
                                        remaining = 1;
                                }
                            }//End if
                        }//End for


                        //if no overflow write the key/value pair to the global buffer
                        //------------------------------------------------------------
                        if (masterKeyId != -1)
                        {
                            if (entryExist == 0)
                            {
                                if ( revertToCounting == 0)
                                {

                                    __global char *pOutKeySet = (__global char*)( outputKeys + groupId * outputKeysBufferSizePerWG  + outKeySizes);
                                    __local char *lKeys = (__local char*)(localKeyStart + lLocalInterOffsets[masterKeyId]);
                                    for (int i = 0; i < lLocalInterSizes[masterKeyId] ; ++i)   pOutKeySet[i] = lKeys[i];

                                    __global char *pOutValSet = (__global char*)( outputVals + groupId * outputValsBufferSizePerWG + outValSizes);   
                                    __local char *lVals = (__local char*)(lLocalScratch + combValSizes);
                                    for (int i = 0; i < combSize; ++i)   pOutValSet[i] = lVals[i]; 

                                    __global int4* hTable= (__global int4*) hashTable +  hashEntriesNum * wid;
                                    hashEntry=  hTable[hashedk];	

                                    int4 bucketElement;
                                    bucketElement.x =  groupId * outputKeysBufferSizePerWG + outKeySizes; 
                                    bucketElement.y = lLocalInterSizes[masterKeyId];
                                    bucketElement.z = groupId * outputValsBufferSizePerWG + outValSizes; 
                                    bucketElement.w = after;

                                    if (before == -1)	
                                        hashEntry.x = hashBucketBufferSizePerWG * groupId + hashBucketSizes;
                                    else
                                        hashBucket[before].w = hashBucketBufferSizePerWG * groupId + hashBucketSizes;

                                    hashEntry.z++;
                                    hashEntry.y =  1;    
                                    hashEntry.w = hashedk; 
                                    hTable[hashedk] = hashEntry;
                                    hashBucket[ hashBucketBufferSizePerWG * groupId + hashBucketSizes] = bucketElement;						    
                                    mem_fence(CLK_GLOBAL_MEM_FENCE);

                                }
                            }

                            else
                            {


                                //Update the already existed entry
                                //---------------------------------
                                combineGL(inputDataset, constantDataset, outputVals + masterHashBucket.z,
                                        lLocalScratch + combValSizes, 0);
                                mem_fence(CLK_GLOBAL_MEM_FENCE);


                            }
                        }
                        if (remaining == 0 ) break;
                    }//End while
                } //End if local overflow
            } // End if Master thread	
        }//End if active threads
        // Exiting condition 	
        if  (activeThreadsPerWavefront[widInGroup] == 0)
            break;
        else
            if  (activeThreadsPerWavefront[widInGroup] == 1)
            {	
                if ( activePortion[widInGroup] == 1) break;
            }
            else
                if ( activePortion[widInGroup] == 3) break;		  
    }// End While
#endif
}

void mapExtended(__global void* inputDataset,
        __constant uint* constantDataset,
        __global void*  key,
        __global void*  val,


        __global char * interKeys,
        __global char *  interVals,
        __global int4 * interOffsets,

        __global char *  outputKeys,
        __global char *  outputVals,
        __global int4 *  hashTable,
        __global int4 *  hashBucket, //x: keyOffset y:keySize z:valOffset w: next pointer

        __constant uint * outputKeysOffsets,
        __constant uint * outputValsOffsets,
        __constant uint * hashBucketOffsets,

        __local uint *  lsOutKeySizes,
        __local uint *  lsOutValSizes,
        __local uint *  lsHashBucketSizes,

        __local uint *  lHashedInterKeys,
        __local char *  lLocalInterKeys,                
        __local char *  lLocalInterVals,
        __local char *  lLocalScratch,
        __local uint *  lUsedLocalBuffer,
        __local uint *  lUsedScratch,
        __local uint *  lLocalInputRecordId,
        __local uint *  lLocalInterSizes,
        __local uint *  lLocalInterOffsets,
        __local uint *  lslaveThreads,
        __local uint *  lLocalEmitted2,

        uint localBufferSizePerWavefront,
        uint localScratchPerWavefront,
        uint outputKeysBufferSizePerWG,
        uint outputValsBufferSizePerWG,
        uint hashBucketBufferSizePerWG,
        uint hashEntriesNum,
        __local uint *  activeThreadsPerWavefront,
        __local uint *  activePortion,

        uint inputRecordId,
        int extended,
        int outputIntermediate,
        __global int* metaEmitted,
        __global int* metaOverflow,
        __global int* metaEmitted2,
        uint interKeysPerWorkgroup,
        uint interValsPerWorkgroup,
        uint interCountsPerWorkgroup)
{
    int tid=get_global_id(0);
    __global VAL_T* pVal = (__global VAL_T*)val;

    __global char* ptrBuf = (__global char*)inputDataset + pVal->line_offset;
    int line_size = pVal->line_size;


    __global char* p = ptrBuf;
    __global char* start =  ptrBuf;
    int indexOf4  = 4;
    char4 c4;
    char word[100],c[4],cc;
    int wordindex = 0;
    int it = line_size >> 2;
    int rem = line_size & 0x00000003;
    int oval = 1;
    int lsize = 0;

    while (lsize <= line_size)
    {

        cc = *p; //c[indexOf4];
        p++;

        if ( (cc == ' ') || (cc == '\0') || (cc == '\n') || (cc == '/') || (cc == '_'))
        {
            word[wordindex] = '\0';
            wordindex++;
            lsize++;

            if ( (wordindex > 6))
            {

                emitIntermediateExtended(inputDataset,constantDataset, word, &oval, wordindex, sizeof(int),  outputKeys, outputVals, hashTable,
                        hashBucket, outputKeysOffsets, outputValsOffsets, hashBucketOffsets,
                        lsOutKeySizes, lsOutValSizes,
                        lsHashBucketSizes,
                        lHashedInterKeys, lLocalInterKeys, lLocalInterVals,lLocalScratch,lUsedLocalBuffer, lUsedScratch,
                        lLocalInputRecordId, lLocalInterSizes, lLocalInterOffsets, lslaveThreads, lLocalEmitted2, localBufferSizePerWavefront,
                        localScratchPerWavefront,
                        outputKeysBufferSizePerWG,
                        outputValsBufferSizePerWG,hashBucketBufferSizePerWG, hashEntriesNum, activeThreadsPerWavefront,activePortion,
                        inputRecordId,metaEmitted,metaOverflow,metaEmitted2);

            }
            wordindex = 0;
        }
        else
        {
            word[wordindex]= cc;
            wordindex++;
            lsize++;
        }
    }
}


void map(__global void* inputDataset,
        __constant uint* constantDataset,
        __global void*  key,
        __global void*  val,


        __global char * interKeys,
        __global char *  interVals,
        __global int4 * interOffsets,

        __global char *  outputKeys,
        __global char *  outputVals,
        __global int4 *  hashTable,
        __global int4 *  hashBucket, //x: keyOffset y:keySize z:valOffset w: next pointer

        __constant uint * outputKeysOffsets,
        __constant uint * outputValsOffsets,
        __constant uint * hashBucketOffsets,

        __local uint *  lsOutKeySizes,
        __local uint *  lsOutValSizes,
        __local uint *  lsHashBucketSizes,

        __local uint *  lHashedInterKeys,
        __local char *  lLocalInterKeys,                
        __local char *  lLocalInterVals,
        __local char *  lLocalScratch,
        __local uint *  lUsedLocalBuffer,
        __local uint *  lUsedScratch,
        __local uint *  lLocalInputRecordId,
        __local uint *  lLocalInterSizes,
        __local uint *  lLocalInterOffsets,
        __local uint *  lslaveThreads,
        __local uint *  lLocalEmitted2,

        uint localBufferSize,
        uint localScratchPerWavefront,
        uint outputKeysBufferSize,
        uint outputValsBufferSize,
        uint hashBucketBufferSize,
        uint hashEntriesNum,
        __local uint *  activeThreadsPerWavefront,
        __local uint *  activePortion,

        uint inputRecordId,
        int extended,
        int outputIntermediate,
        __global int* metaEmitted,
        __global int* metaOverflow,
        __global int* metaEmitted2,
        uint interKeysPerWorkgroup,
        uint interValsPerWorkgroup,
        uint interCountsPerWorkgroup)
{
    int tid=get_global_id(0);
    __global VAL_T* pVal = (__global VAL_T*)val;

    __global char* ptrBuf = (__global char*)inputDataset + pVal->line_offset;
    int line_size = pVal->line_size;


    __global char* p = ptrBuf;
    __global char* start =  ptrBuf;
    int indexOf4  = 4;
    char4 c4;
    char word[100],c[4],cc;
    int wordindex = 0;
    int it = line_size >> 2;
    int rem = line_size & 0x00000003;
    int oval = 1;
    int lsize = 0;

    while (lsize <= line_size)
    {

        cc = *p; //c[indexOf4];
        p++;

        if ( (cc == ' ') || (cc == '\0') || (cc == '\n') || (cc == '/') || (cc == '_'))
        {
            word[wordindex] = '\0';
            wordindex++;
            lsize++;

            if ( (wordindex > 6))
            {

                emitIntermediate(inputDataset,constantDataset,  word, &oval, wordindex, sizeof(int), interKeys,interVals, interOffsets, outputKeys, outputVals
                        ,hashTable,
                        hashBucket, outputKeysOffsets,outputValsOffsets, hashBucketOffsets,
                        lsOutKeySizes, lsOutValSizes,
                        lsHashBucketSizes,lHashedInterKeys, lLocalInterKeys, lLocalInterVals,lLocalScratch,lUsedLocalBuffer, lUsedScratch,
                        lLocalInputRecordId, lLocalInterSizes, lLocalInterOffsets, lslaveThreads, lLocalEmitted2, localBufferSize, localScratchPerWavefront,
                        outputKeysBufferSize,
                        outputValsBufferSize,hashBucketBufferSize, hashEntriesNum, activeThreadsPerWavefront,activePortion,
                        inputRecordId, extended,outputIntermediate, metaEmitted,metaOverflow,metaEmitted2,interKeysPerWorkgroup, interValsPerWorkgroup,
                        interCountsPerWorkgroup);

            }
            wordindex = 0;
        }
        else
        {
            word[wordindex]= cc;
            wordindex++;
            lsize++;
        }
    }
}

__kernel void mapperExtended(__global void* inputDataset,
        __constant uint* constantDataset,

        __global char*       inputKeys,
        __global char*       inputVals,
        __global int4*       inputOffsetSizes,


        __global char *  outputKeys,
        __global char *  outputVals,
        __global int4 *  hashTable,
        __global int4 *  hashBucket,

        __constant uint * outputKeysOffsets,
        __constant uint * outputValsOffsets,
        __constant uint * hashBucketOffsets,

        int  recordNum,
        int  recordsPerTask,
        int  taskNum,
        __local uint * lsOutKeySizes,
        __local uint *  lsOutValSizes,
        __local uint *  lsHashBucketSizes,

        __local uint * lHashedInterKeys,
        __local char *  lLocalInterKeys,
        __local char *  lLocalInterVals,
        __local char *  lLocalScratch,
        __local uint *  lUsedLocalBuffer,
        __local uint *  lUsedScratch,
        __local uint *  lLocalInputRecordId,
        __local uint *  lLocalInterSizes,
        __local uint *  lLocalInterOffsets,
        __local uint *  lslaveThreads,
        __local uint *  lLocalEmitted2,

        uint localBufferSize,
        uint localScratchPerWavefront,
        uint outputKeysBufferSize,
        uint outputValsBufferSize,
        uint hashBucketBufferSize,
        uint hashEntriesNum,
        __local uint *  activeThreadsPerWavefront,
        __local uint * activePortion,

        __global int * overflowedGroupIds,
        int extended,
        __global int * metaEmitted,
        __global int * metaOverflow,
        __global int * metaEmitted2)    //extended = 0 for basic mapper and 1 for overflow handling mapper
{

    int tid = get_global_id(0);
    int lid = get_local_id(0);
    int groupSize = get_local_size(0);
    uint widInGroup= lid >> 6;   //Wavefront Id in a workgroup
    uint numGroups=get_num_groups(0);

    //To zoomin to overflowed workgroup
    int bid = overflowedGroupIds[get_group_id(0)];

    //if (tid*recordsPerTask >= recordNum) return;
    int recordBase = bid * recordsPerTask * groupSize;
    int terminate = (bid + 1) * (recordsPerTask * groupSize);
    if (terminate > recordNum) terminate = recordNum;

    // Iniitialize global variables carrying keysize, valsizes, keyvaloffests to zero
    if (lid == 0)
    {
        lsOutKeySizes[0] = 0;
        lsOutKeySizes[1] = 0;
        lsOutValSizes[0] = 0;
        lsOutValSizes[1] = 0;
        lsHashBucketSizes[0] = 0;
        lsHashBucketSizes[1] = 0 ;
    }
    lLocalInterVals[widInGroup] = 0 ;
    lHashedInterKeys[lid] = -1;
    lLocalEmitted2[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = recordBase + lid; i < terminate; i+=groupSize)
    {

        metaEmitted2[i] = 0 ;
        if ( metaOverflow[i] == 0 )  continue; //no overflow

        lLocalInputRecordId[lid] = metaEmitted[i]; //lid;
        mem_fence(CLK_LOCAL_MEM_FENCE);

        int4 offsetSize = inputOffsetSizes[i];
        __global char *key = inputKeys + offsetSize.x;
        __global char *val = inputVals + offsetSize.z;

        mapExtended ( inputDataset,
                constantDataset,
                key,
                val,

                0,
                0,
                0,

                outputKeys,
                outputVals,
                hashTable,
                hashBucket, //x: keyOffset y:keySize z:valOffset w: next pointer

                outputKeysOffsets,
                outputValsOffsets,
                hashBucketOffsets,

                lsOutKeySizes,
                lsOutValSizes,
                lsHashBucketSizes,

                lHashedInterKeys,
                lLocalInterKeys,
                lLocalInterVals,
                lLocalScratch,
                lUsedLocalBuffer,
                lUsedScratch,
                lLocalInputRecordId,
                lLocalInterSizes,
                lLocalInterOffsets,
                lslaveThreads,
                lLocalEmitted2,

                localBufferSize,
                localScratchPerWavefront,
                outputKeysBufferSize,
                outputValsBufferSize,
                hashBucketBufferSize,
                hashEntriesNum,

                activeThreadsPerWavefront,
                activePortion,
                i,

                1,
                0,
                metaEmitted,
                metaOverflow,
                metaEmitted2,

                0,
                0,
                0);

    }

}

__kernel void mapper(__global void* inputDataset,
        __constant uint* constantDataset,

        __global char*       inputKeys,
        __global char*       inputVals,
        __global int4*       inputOffsetSizes,

        __global char * interKeys,
        __global char * interVals,
        __global int4 * interOffsets,

        __global char *  outputKeys,
        __global char *  outputVals,
        __global int4 *  hashTable,
        __global int4 *  hashBucket,

        int  recordNum,
        int  recordsPerTask,
        int  taskNum,

        __local uint * lsOutKeySizes,
        __local uint *  lsOutValSizes,
        __local uint *  lsHashBucketSizes,

        __local uint * lHashedInterKeys,
        __local char *  lLocalInterKeys,
        __local char *  lLocalInterVals,
        __local char *  lLocalScratch,
        __local uint *  lUsedLocalBuffer,
        __local uint *  lUsedScratch,
        __local uint *  lLocalInputRecordId,
        __local uint *  lLocalInterSizes,
        __local uint *  lLocalInterOffsets,
        __local uint *  lslaveThreads,

        __global uint * gOutValSizes,
        __global uint * gOutKeySizes,
        __global uint * gHashBucketNum,

        uint localBufferSize,
        uint localScratchPerWavefront,
        uint outputKeysBufferSize,
        uint outputValsBufferSize,
        uint hashBucketBufferSize,
        uint hashEntriesNum,
        __local uint *  activeThreadsPerWavefront,
        __local uint * activePortion,

        __global int * overflowedGroupIds,
        int extended,
        int emitIntermediate,
        __global int* metaEmitted,
        __global int* metaOverflow,
        __global int* metaEmitted2,
        uint interKeysPerWorkgroup,
        uint interValsPerWorkgroup,
        uint interCountsPerWorkgroup)    //extended = 0 for basic mapper and 1 for overflow handling mapper
{
    int tid = get_global_id(0);
    int lid = get_local_id(0);
    int groupSize = get_local_size(0);
    uint widInGroup= lid >> 6;   //Wavefront Id in a workgroup
    uint wid = tid >> 6;
    uint numGroups=get_num_groups(0);

    int bid = get_group_id(0);

    if (tid*recordsPerTask >= recordNum) return;
    int recordBase = bid * recordsPerTask * groupSize;
    int terminate = (bid + 1) * (recordsPerTask * groupSize);
    if (terminate > recordNum) terminate = recordNum;

    // Iniitialize global variables carrying keysize, valsizes, keyvaloffests to zero
    if (lid == 0)
    {
        lsOutKeySizes[0] = 0;
        lsOutKeySizes[1] = 0;
        lsOutKeySizes[2] = 0;

        lsOutValSizes[0] = 0;
        lsOutValSizes[1] = 0;
        lsOutValSizes[2] = 0;

        lsHashBucketSizes[0] = 0;
        lsHashBucketSizes[1] = 0 ;
        lsHashBucketSizes[2] = 0 ;
    }
    lLocalInterVals[widInGroup] = 0 ;
    lHashedInterKeys[lid] = -1;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = recordBase + lid; i < terminate; i+= groupSize)
    {
        int cindex =  i;

        metaEmitted[i] = 0;
        metaOverflow[i]= 0;

        int4 offsetSize = inputOffsetSizes[cindex];
        __global char *key = inputKeys + offsetSize.x;
        __global char *val = inputVals + offsetSize.z;

        map ( inputDataset,
                constantDataset,
                key,
                val,

                interKeys,	
                interVals,
                interOffsets,

                outputKeys,
                outputVals,
                hashTable,
                hashBucket, //x: keyOffset y:keySize z:valOffset w: next pointer

                0,
                0,
                0,

                lsOutKeySizes,
                lsOutValSizes,
                lsHashBucketSizes,

                lHashedInterKeys,
                lLocalInterKeys,
                lLocalInterVals,
                lLocalScratch,
                lUsedLocalBuffer,
                lUsedScratch,
                lLocalInputRecordId,
                lLocalInterSizes,
                lLocalInterOffsets,
                lslaveThreads,
                0,

                localBufferSize,
                localScratchPerWavefront,
                outputKeysBufferSize,
                outputValsBufferSize,
                hashBucketBufferSize,
                hashEntriesNum,

                activeThreadsPerWavefront,
                activePortion,
                cindex,

                extended,
                emitIntermediate,
                metaEmitted,
                metaOverflow,
                metaEmitted2,

                interKeysPerWorkgroup,
                interValsPerWorkgroup,
                interCountsPerWorkgroup);



    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    mem_fence(CLK_LOCAL_MEM_FENCE);

    if ( lid == 0 )
    {
        //needed sizes
        gOutKeySizes[bid] = lsOutKeySizes[0];
        gOutValSizes[bid] = lsOutValSizes[0];
        gHashBucketNum[bid] = lsHashBucketSizes[0];

        //used sizes
        gOutKeySizes[numGroups + bid] = lsOutKeySizes[1];
        gOutValSizes[numGroups + bid] = lsOutValSizes[1];
        gHashBucketNum[numGroups + bid] = lsHashBucketSizes[1];
    }

}



/****************************************************************************/
//Reduce phase kernels
/****************************************************************************/
/*-----------------------------------------------------------------------------------------------
 * Execute reduce phase using one of three modes:
 * 0: Normal mode where each thread is responsible for reducing agiven key among all hash tables
 * 1: First stage of the hierarchical mode
 * 2: Second stage of the hierarchical mode
 * hierarchical mode is applicable and efficient for application with both perfect and non-perfect 
 * hashing functions 
 --------------------------------------------------------------------------------------------------*/
__kernel void  reducer( __constant uint* constantDataset,
        __global float* gKeySizes,
        __global float* gValSizes,
        __global float* gOffsetCounts,

        __global char *  outputKeys,
        __global char *  outputVals,
        __global int4 *  hashTable,
        __global int4 *  hashBucket, //x: keyOffset y:keySize z:valOffset w: next pointer

        uint hashEntriesNum,

        uint	        hashTablesNum,
        uint		recordNum,
        uint		recordsPerTask,
        uint		taskNum,
        uint hashTablesPerThread,
        int mode,
        __local uint* localKeySizesPerWorkgroup,
        __local uint* localValSizesPerWorkgroup,
        __local uint* localOffsetsPerWorkgroup,
        uint mapWavefrontsPerGroup)
{

    uint tid = get_global_id(0);
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);

    if (lid == 0 ) 
    {
        *localKeySizesPerWorkgroup = 0;
        *localValSizesPerWorkgroup = 0;
        *localOffsetsPerWorkgroup = 0;
    }	
    barrier(CLK_LOCAL_MEM_FENCE);

    int hashEntryId = tid%hashEntriesNum;
    int startHashTableId =  (mode == 1 )? tid/hashEntriesNum*hashTablesPerThread:0; 

    //read entry tid from every hashtable i and reduce the retrieved values
    //---------------------------------------------------------------------
    uint EndHashTableId = (mode == 1)? (hashTablesPerThread <= 1)? hashTablesNum  : startHashTableId + hashTablesPerThread  : hashTablesNum;
    uint step = (mode == 1)? 1: hashTablesPerThread;
    for (int i=startHashTableId+step; i< EndHashTableId; i+=step)
    {
        __global int4* hashTablei = hashTable + hashEntriesNum * i ;
        int4 hashEntry2 = hashTablei[hashEntryId];
        int  associatedHashBuckets2= hashEntry2.z;

        int slaveHashBucketIndex = -1;
        int4 hashBucket2;
        int4 hashBucket1;
        int foundMatch = 1; 
        int beforeMasterIndex;
        int masterHashBucketIndex = -1;

        hashTablei = hashTable + hashEntriesNum * startHashTableId;
        int4 hashEntry = hashTablei[hashEntryId];
        int  associatedHashBuckets = hashEntry.z;

        beforeMasterIndex = -1 ;
        if (associatedHashBuckets ==0 ) 
            masterHashBucketIndex =-1;
        else
        {
            masterHashBucketIndex =  hashEntry.x;
            hashBucket1 = hashBucket[masterHashBucketIndex];
        }
        for (int k=0; k < associatedHashBuckets2; k++)
        {

            if (foundMatch == 1)
            {
                slaveHashBucketIndex= (k == 0)?  hashEntry2.x :  hashBucket2.w;
                hashBucket2 = hashBucket[slaveHashBucketIndex];
            }
            __global char* otherKey = outputKeys + hashBucket2.x;
            int otherKeySize = hashBucket2.y;


            foundMatch = 0;  //hashBucket1 < hashBucket2
            while ( masterHashBucketIndex != -1)
            {
                __global char* currentKey = outputKeys + hashBucket1.x;
                int currentKeySize = hashBucket1.y;

                //If similar key exist update the value of the current key and remove this hashBucket from the hashtable
                int comp = keyEqualGG(currentKey,  currentKeySize , otherKey, otherKeySize);
                if (comp == 1)
                {
                    foundMatch=1;
                    combineGG(  constantDataset, outputVals + hashBucket1.z, outputVals + hashBucket2.z, 0 );				     

                }
                else
                {
                    if (comp == 2) //hashBucket1 > hashBucket2
                    {
                        foundMatch = -1;
                    }

                }
                if (foundMatch == -1) break;

                beforeMasterIndex =  masterHashBucketIndex;
                masterHashBucketIndex =  hashBucket1.w;
                hashBucket1 = hashBucket[masterHashBucketIndex];

                if ( foundMatch ==1 )  break;

            }
            if (foundMatch != 1)
            {
                int toBeMovedHashedBucket = slaveHashBucketIndex;
                slaveHashBucketIndex= hashBucket[slaveHashBucketIndex].w;
                hashBucket2=  hashBucket[slaveHashBucketIndex];

                //Adjust master hash buckets
                //--------------------------	
                if ((masterHashBucketIndex == -1) && (beforeMasterIndex == -1)) //hashBucket empty
                {
                    __global int4* hashTablei = hashTable + hashEntriesNum * startHashTableId;
                    hashTablei[hashEntryId].x = toBeMovedHashedBucket;
                    hashBucket[toBeMovedHashedBucket].w = -1 ;
                    hashTablei[hashEntryId].z++;
                    beforeMasterIndex = toBeMovedHashedBucket;
                    masterHashBucketIndex = -1;

                }
                else
                {
                    if (foundMatch == 0 )
                    {
                        //insert to the last
                        hashBucket[beforeMasterIndex].w = toBeMovedHashedBucket;
                        hashBucket[toBeMovedHashedBucket].w = -1;
                        __global int4* hashTablei = hashTable + hashEntriesNum * startHashTableId;
                        hashTablei[hashEntryId].z++;				
                        masterHashBucketIndex = -1;
                        beforeMasterIndex = toBeMovedHashedBucket;
                    }
                    else
                    {
                        //insert inbetween
                        __global int4* hashTablei = hashTable + hashEntriesNum * startHashTableId;
                        if ( beforeMasterIndex == -1)
                            hashTablei[hashEntryId].x = toBeMovedHashedBucket;
                        else
                            hashBucket[beforeMasterIndex].w = toBeMovedHashedBucket;
                        hashBucket[toBeMovedHashedBucket].w = masterHashBucketIndex;
                        hashTablei[hashEntryId].z++;
                        beforeMasterIndex =toBeMovedHashedBucket;
                    }

                }
            } 
        }
        mem_fence(CLK_GLOBAL_MEM_FENCE);
    }

    if ( (( mode == 1 ) &&( hashTablesPerThread <= 1 )) || ( mode == 2))
    {

        __global int4* hashTablei = hashTable + hashEntriesNum * startHashTableId;
        int4 hashEntry = hashTablei[hashEntryId];
        int  associatedHashBuckets = hashEntry.z;
        for ( int j = 0; j < associatedHashBuckets; j++)
        {
            int4 hashBucket1 = (j == 0)?  hashBucket[hashEntry.x] :  hashBucket[hashBucket1.w];
            atom_add(localKeySizesPerWorkgroup, hashBucket1.y);
            atom_add(localValSizesPerWorkgroup, combineSize(constantDataset));
            atom_inc(localOffsetsPerWorkgroup);
            combineGG(  constantDataset, outputVals + hashBucket1.z, 0, 2 ); //Post processing
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0 )
        {
            gKeySizes[gid]  = (float)*localKeySizesPerWorkgroup;
            gValSizes[gid] = (float)*localValSizesPerWorkgroup;
            gOffsetCounts[gid] = (float)*localOffsetsPerWorkgroup;
        }
    }

}

/*--------------------------------------------------------------------------------------------------------------- 
 *  Note: to differentiate between base entries and extra entries (allocated to handle overflow), we make use of 
 *  the length field in each int4 entry E i.e., E.y. If E.y >= 0, the entry is considered as base entry. 
 *  If E.y < 0, the entry is condidered as extra entry. This is an efficient way to differentiate both entries 
 *  without maintaining another data structure.
 * --------------------------------------------------------------------------------------------------------------*/
__kernel void  reducerInOverflow( __constant uint* constantDataset,
        __global float* gKeySizes,
        __global float* gValSizes,
        __global float* gOffsetCounts,

        __global char *  outputKeys,
        __global char *  outputVals,
        __global int4 *  hashTable,
        __global int4 *  hashBucket, //x: keyOffset y:keySize z:valOffset w: next pointer

        uint hashEntriesNum,

        int                     hashTablesNum,
        int             recordNum,
        int             recordsPerTask,
        int             taskNum,
        int hashTablesPerThread,
        int mode,
        __local uint* localKeySizesPerWorkgroup,
        __local uint* localValSizesPerWorkgroup,
        __local uint* localOffsetsPerWorkgroup,
        uint mapWavefrontsPerGroup,
        __global char *  outputKeysExtra,
        __global char *  outputValsExtra,
        __global int4 *  hashTableExtra,
        __global int4 *  hashBucketExtra, //x: keyOffset y:keySize z:valOffset w: next pointer
        int extraHashTablesNum)
{

    uint tid = get_global_id(0);
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);

    if (lid == 0 )
    {
        *localKeySizesPerWorkgroup = 0;
        *localValSizesPerWorkgroup = 0;
        *localOffsetsPerWorkgroup = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int hashEntryId = tid%hashEntriesNum;
    int startHashTableId =  (mode == 1 )? tid/hashEntriesNum*hashTablesPerThread:0;
    if (startHashTableId >= hashTablesNum + extraHashTablesNum)  return;

    int extraMasterHashTable = (startHashTableId > hashTablesNum - 1)? 1: 0;
    __global int4* hashTable1 = (extraMasterHashTable == 0)? hashTable + hashEntriesNum * startHashTableId : hashTableExtra + hashEntriesNum * (startHashTableId -hashTablesNum);

    //read entry tid from every hashtable i and reduce the retrieved values
    //---------------------------------------------------------------------
    uint EndHashTableId = (mode == 1)? (hashTablesPerThread <= 1)? hashTablesNum + extraHashTablesNum : startHashTableId + hashTablesPerThread  : hashTablesNum + extraHashTablesNum;
    uint step = (mode == 1)? 1: hashTablesPerThread;
    int masterHashBucketIndex;

    for (int i = startHashTableId + step; i < EndHashTableId; i+=step)
    {

        if ( i >= (hashTablesNum + extraHashTablesNum)) return;

        int extraSlaveHashTable = (i >( hashTablesNum - 1))? 1: 0;
        __global int4* hashTablei = (extraSlaveHashTable == 0)? hashTable+ hashEntriesNum * i : hashTableExtra + hashEntriesNum * (i - hashTablesNum);
        __global int4* hashBucketi =  (extraSlaveHashTable == 0)? hashBucket : hashBucketExtra ;
        int4 hashEntry2 = hashTablei[hashEntryId];
        int  associatedHashBuckets2= hashEntry2.z;

        int slaveHashBucketIndex = -1;
        int4 hashBucket2;
        int4 hashBucket1;
        int foundMatch = 1;
        int beforeMasterIndex = -1;
        masterHashBucketIndex = -1;

        int4 hashEntry = hashTable1[hashEntryId];
        int  associatedHashBuckets = hashEntry.z;

        int isMasterExtra =  (extraMasterHashTable == 1)? 1: 0;
        int isBeforeMasterExtra =  (extraMasterHashTable == 1)? 1: 0;

        if (associatedHashBuckets == 0 )
            masterHashBucketIndex = -1;
        else
        {
            masterHashBucketIndex =  hashEntry.x;
            hashBucket1 =   (extraMasterHashTable == 1)? hashBucketExtra[masterHashBucketIndex] : (hashEntry.y < 0)?  hashBucketExtra[masterHashBucketIndex] : hashBucket[masterHashBucketIndex];
            isMasterExtra=  (extraMasterHashTable == 1)? 1 : (hashEntry.y < 0)? 1:0;
        }
        for (int k = 0; k < associatedHashBuckets2; k++)
        {
            if (foundMatch == 1)
            {
                slaveHashBucketIndex= (k == 0)?  hashEntry2.x :  hashBucket2.w;
                hashBucket2 = hashBucketi[slaveHashBucketIndex];
            }
            __global char* otherKey = (extraSlaveHashTable == 0)? outputKeys + hashBucket2.x:  outputKeysExtra+ hashBucket2.x;
            int otherKeySize = abs(hashBucket2.y);

            //Compare the current slave key to master keys
            foundMatch = 0;  //hashBucket1 < hashBucket2

            while ( masterHashBucketIndex != -1)
            {
                __global char* currentKey = (extraMasterHashTable == 1)? outputKeysExtra+ hashBucket1.x : (isMasterExtra == 0)? outputKeys + hashBucket1.x:  outputKeysExtra+ hashBucket1.x;
                int currentKeySize = abs(hashBucket1.y );

                //If similar key exist update the value of the current key and remove this hashBucket from the hashtable
                int comp = keyEqualGG(currentKey,  currentKeySize , otherKey, otherKeySize);
                if (comp == 1)
                {
                    foundMatch=1;
                    combineGG(  constantDataset, (extraMasterHashTable == 1)? outputValsExtra + hashBucket1.z :  (isMasterExtra == 0)? outputVals + hashBucket1.z:  outputValsExtra+ hashBucket1.z, (extraSlaveHashTable == 0)? outputVals + hashBucket2.z:  outputValsExtra + hashBucket2.z, 0 );                             
                    beforeMasterIndex =  masterHashBucketIndex;
                    masterHashBucketIndex =  hashBucket1.w;
                    isBeforeMasterExtra= isMasterExtra;
                    isMasterExtra = (extraMasterHashTable == 1)? 1 : (hashBucket1.y < 0)? 1 : 0;
                    hashBucket1 =   (extraMasterHashTable == 1)?  hashBucketExtra[masterHashBucketIndex] : (hashBucket1.y < 0)? hashBucketExtra[masterHashBucketIndex] : hashBucket[masterHashBucketIndex] ;
                    break;
                }
                else
                {
                    if (comp == 2) //hashBucket1 > hashBucket2
                    {
                        foundMatch = -1;
                        break;
                    }

                }

                beforeMasterIndex =  masterHashBucketIndex;
                masterHashBucketIndex =  hashBucket1.w;
                isBeforeMasterExtra= isMasterExtra;
                isMasterExtra = (extraMasterHashTable == 1)? 1 : (hashBucket1.y < 0)? 1 : 0;
                hashBucket1 =   (extraMasterHashTable == 1)?  hashBucketExtra[masterHashBucketIndex] : (hashBucket1.y < 0)? hashBucketExtra[masterHashBucketIndex] : hashBucket[masterHashBucketIndex] ;
            }//End While

            if (foundMatch != 1)
            {
                int toBeMovedHashedBucket = slaveHashBucketIndex;
                slaveHashBucketIndex = hashBucketi[slaveHashBucketIndex].w;
                hashBucket2=  hashBucketi[slaveHashBucketIndex];

                __global int4 * hashBucketBefore = (isBeforeMasterExtra == 0)? hashBucket : hashBucketExtra;

                //Adjust master hash buckets
                //--------------------------
                if ((masterHashBucketIndex == -1) && (beforeMasterIndex == -1)) //hashBucket empty
                {
                    hashTable1[hashEntryId].x = toBeMovedHashedBucket;
                    hashTable1[hashEntryId].y = (extraSlaveHashTable == 0)? 1 : -1;
                    hashTable1[hashEntryId].z++;
                    hashTable1[hashEntryId].w = hashEntryId;
                    hashBucketi[toBeMovedHashedBucket].w = -1 ;
                    beforeMasterIndex = toBeMovedHashedBucket;
                    masterHashBucketIndex = -1;
                    isBeforeMasterExtra = (extraSlaveHashTable == 0)? 0 : 1;

                }
                else
                {
                    if (foundMatch == 0 )
                    {
                        //insert to the end
                        hashBucketBefore[beforeMasterIndex].w =  toBeMovedHashedBucket;
                        hashBucketBefore[beforeMasterIndex].y = (extraSlaveHashTable == 0)? abs(hashBucketBefore[beforeMasterIndex].y) : -1 *  abs(hashBucketBefore[beforeMasterIndex].y);
                        hashBucketi[toBeMovedHashedBucket].w = -1;
                        hashTable1[hashEntryId].z++;
                        masterHashBucketIndex = -1;
                        beforeMasterIndex = toBeMovedHashedBucket;
                        isBeforeMasterExtra = (extraSlaveHashTable == 0)? 0 : 1;
                    }

                    else
                    {
                        //insert inbetween
                        if ( beforeMasterIndex == -1)
                        {
                            //Adjust extra indicator before overriding
                            hashBucketi[toBeMovedHashedBucket].y = ( hashTable1[hashEntryId].y < 0)? -1 * abs(hashBucketi[toBeMovedHashedBucket].y):  abs(hashBucketi[toBeMovedHashedBucket].y);
                            hashTable1[hashEntryId].x = toBeMovedHashedBucket;
                            hashTable1[hashEntryId].y = (extraSlaveHashTable == 0)? 1 : -1;
                        }
                        else
                        {
                            //Adjust extra indicator before overriding
                            hashBucketi[toBeMovedHashedBucket].y = (hashBucketBefore[beforeMasterIndex].y < 0)? -1 *  abs (hashBucketi[toBeMovedHashedBucket].y) :  abs(hashBucketi[toBeMovedHashedBucket].y);
                            hashBucketBefore[beforeMasterIndex].w = toBeMovedHashedBucket;
                            hashBucketBefore[beforeMasterIndex].y = (extraSlaveHashTable == 0)? abs(hashBucketBefore[beforeMasterIndex].y) :-1 *abs ( hashBucketBefore[beforeMasterIndex].y);
                        }
                        hashBucketi[toBeMovedHashedBucket].w = masterHashBucketIndex;
                        hashTable1[hashEntryId].z++;
                        beforeMasterIndex =toBeMovedHashedBucket;
                        isBeforeMasterExtra = (extraSlaveHashTable  == 0)? 0 : 1;
                    }

                }

            }

            mem_fence(CLK_GLOBAL_MEM_FENCE);
        }//End for k
    } //End for i

    if ( (( mode == 1 ) &&( hashTablesPerThread <= 1 )) || ( mode == 2))
    {
        int4 hashEntry = hashTable1[hashEntryId];
        int  associatedHashBuckets = hashEntry.z;
        int4 hashBucket1;
        for ( int j = 0; j < associatedHashBuckets; j++)
        {

            int Extra = (j == 0)?  ((hashEntry.y < 0)? 1 : 0) : ( (hashBucket1.y < 0)? 1: 0 );
            hashBucket1 = (j == 0)?  (hashEntry.y < 0)? hashBucketExtra[ hashEntry.x] : hashBucket[hashEntry.x] :  (hashBucket1.y < 0)?  hashBucketExtra[ hashBucket1.w]: hashBucket[hashBucket1.w];
            atom_add(localKeySizesPerWorkgroup, abs(hashBucket1.y));
            atom_add(localValSizesPerWorkgroup, combineSize(constantDataset));
            atom_inc(localOffsetsPerWorkgroup);

            combineGG(  constantDataset, (Extra == 1)?  outputValsExtra + hashBucket1.z : outputVals + hashBucket1.z, 0, 2 ); //Post processing
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0 )
        {
            gKeySizes[gid]  = (float)*localKeySizesPerWorkgroup;
            gValSizes[gid] = (float)*localValSizesPerWorkgroup;
            gOffsetCounts[gid] = (float)*localOffsetsPerWorkgroup;
        }
    }

}

//This kernel aims at copying the output into a buffer contiguously
__kernel void copyerHashToArray(  __constant uint* constantDataset,
        __global char *  outputKeys,
        __global char *  outputVals,	
        __global uint4*  keyValOffsets,
        __global char *  oldOutputKeys,
        __global char *  oldOutputVals,
        __global int4 *  hashTable,
        __global int4 *  hashBucket, //x: keyOffset y:keySize z:valOffset w: next pointer

        uint hashBucketBufferSizePerWG,
        uint hashEntriesNum,

        int			hashTablesNum,
        __local uint* localKeySizesPerWorkgroup,
        __local uint* localValSizesPerWorkgroup,
        __local uint* localOffsetsPerWorkgroup,
        __global float* gKeySizesOffset,
        __global float* gValSizesOffset,
        __global float* gOffsetCountsOffset,
        uint mapWavefrontsPerGroup )
{

    uint tid = get_global_id(0);
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);

    if (lid == 0 ) 
    {
        *localKeySizesPerWorkgroup = (uint)gKeySizesOffset[gid];
        *localValSizesPerWorkgroup = (uint)gValSizesOffset[gid];
        *localOffsetsPerWorkgroup = (uint)gOffsetCountsOffset[gid];
    }	
    barrier(CLK_LOCAL_MEM_FENCE);

    int hashEntryId = tid%hashEntriesNum;
    int startHashTableId = tid/hashEntriesNum;

    //read entry tid from every hashtable i and reduce the retrieved values
    //---------------------------------------------------------------------
    // This code handles only the case where there is one wavefront per workgroup, need to be modified to
    // handle any number of wavefronts per workgroup

    __global int4* hashTablei = hashTable +  hashEntriesNum * startHashTableId;
    int4 hashEntry = hashTablei[hashEntryId];
    int  associatedHashBuckets = hashEntry.z;

    uint valSize = combineSize(constantDataset) ;
    for (int j=0; j < associatedHashBuckets; j++)
    {
        int4 hashBucket1;
        if (j == 0)
            hashBucket1 = hashBucket[hashEntry.x];
        else
            hashBucket1 = hashBucket[ hashBucket1.w ];

        //Update the workgroup counts
        uint keyOffset = atom_add(localKeySizesPerWorkgroup,hashBucket1.y);
        uint valOffset= atom_add(localValSizesPerWorkgroup, valSize); //hashBucket1.w);
        uint counts= atom_inc(localOffsetsPerWorkgroup);

        //Copy Key
        __global char * skey = oldOutputKeys + hashBucket1.x;
        __global char * dkey = outputKeys + keyOffset;
        for (int i=0 ; i< hashBucket1.y; i++)
            dkey[i] = skey[i];

        //Copy Value
        __global char * sval = oldOutputVals + hashBucket1.z;
        __global char * dval = outputVals + valOffset;
        for (int i=0 ; i< valSize; i++)
            dval[i] = sval[i];

        uint4 keyValOffset;
        keyValOffset.x =  keyOffset;
        keyValOffset.y =  hashBucket1.y;  
        keyValOffset.z =  valOffset;
        keyValOffset.w =  valSize;

        keyValOffsets[counts] = keyValOffset;
    }

}

__kernel void copyerHashToArrayInOverflow( __constant uint* constantDataset,
        __global char *  outputKeys,
        __global char *  outputVals,	
        __global uint4*  keyValOffsets,
        __global char *  oldOutputKeys,
        __global char *  oldOutputVals,
        __global int4 *  hashTable,
        __global int4 *  hashBucket, //x: keyOffset y:keySize z:valOffset w: next pointer

        uint hashBucketBufferSizePerWG,
        uint hashEntriesNum,


        int hashTablesNum,
        __local uint* localKeySizesPerWorkgroup,
        __local uint* localValSizesPerWorkgroup,
        __local uint* localOffsetsPerWorkgroup,
        __global float* gKeySizesOffset,
        __global float* gValSizesOffset,
        __global float* gOffsetCountsOffset,
        uint mapWavefrontsPerGroup,
        __global char *  outputKeysExtra,
        __global char *  outputValsExtra,
        __global int4 *  hashTableExtra,
        __global int4 *  hashBucketExtra, //x: keyOffset y:keySize z:valOffset w: next pointer
        __constant uint * hashBucketOffsetsExtra)
{


    uint tid = get_global_id(0);
    uint lid = get_local_id(0);
    uint gid = get_group_id(0);

    if (lid == 0 ) 
    {
        *localKeySizesPerWorkgroup = (uint)gKeySizesOffset[gid];
        *localValSizesPerWorkgroup = (uint)gValSizesOffset[gid];
        *localOffsetsPerWorkgroup = (uint)gOffsetCountsOffset[gid];
    }	
    barrier(CLK_LOCAL_MEM_FENCE);

    int hashEntryId = tid%hashEntriesNum;
    int startHashTableId = tid/hashEntriesNum;

    int extraHashTables = (startHashTableId > hashTablesNum - 1)? 1: 0;
    startHashTableId = (startHashTableId > hashTablesNum - 1)? startHashTableId - hashTablesNum : startHashTableId;

    //read entry tid from every hashtable i and reduce the retrieved values
    //---------------------------------------------------------------------
    // This code handles only the case where there is one wavefront per workgroup, need to be modified to
    // handle any number of wavefronts per workgroup
    __global int4* hashTablei = (extraHashTables == 0)? hashTable + hashEntriesNum * startHashTableId :  hashTableExtra + hashEntriesNum * startHashTableId;
    int4 hashEntry = hashTablei[hashEntryId];
    int  associatedHashBuckets = hashEntry.z;


    uint valSize = combineSize(constantDataset) ;
    for (int j=0; j < associatedHashBuckets; j++)
    {
        int4 hashBucket1;
        int Extra = (j == 0)?  ((hashEntry.y < 0)? 1 : 0) : ( (hashBucket1.y < 0)? 1: 0 );
        hashBucket1 = (j == 0)?  (hashEntry.y < 0)? hashBucketExtra[ hashEntry.x] : hashBucket[hashEntry.x] :  (hashBucket1.y < 0)?  hashBucketExtra[ hashBucket1.w]: hashBucket[hashBucket1.w];

        __global char* sourceKeys =  (Extra == 0)? oldOutputKeys : outputKeysExtra;
        __global char* sourceVals =  (Extra == 0)? oldOutputVals : outputValsExtra;

        //Update the workgroup counts
        uint keyOffset = atom_add(localKeySizesPerWorkgroup,abs(hashBucket1.y));
        uint valOffset= atom_add(localValSizesPerWorkgroup,valSize);
        uint counts= atom_inc(localOffsetsPerWorkgroup);

        //Copy Key
        __global char * skey = sourceKeys + hashBucket1.x;
        __global char * dkey = outputKeys + keyOffset;
        for (int i=0 ; i< abs(hashBucket1.y); i++)
            dkey[i] = skey[i];

        //Copy Value
        __global char * sval = sourceVals + hashBucket1.z;
        __global char * dval = outputVals + valOffset;
        for (int i=0 ; i< valSize; i++)
            dval[i] = sval[i];

        uint4 keyValOffset;
        keyValOffset.x =  keyOffset;
        keyValOffset.y =  abs(hashBucket1.y);  
        keyValOffset.z =  valOffset;
        keyValOffset.w =  valSize;

        keyValOffsets[counts] = keyValOffset;
    }

}



