/******************************************************************************************************
* (c) Virginia Polytechnic Insitute and State University, 2011.
* This is the source code for StreamMR, a MapReduce framework on graphics processing units.
* Developer:  Marwa K. Elteir (City of Scientific Researches and Technology Applications, Egypt)
*******************************************************************************************************/

#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

typedef struct
{
        int keywordOffset;
        int keywordSize;
} KEY_T;

typedef struct
{
        int lineOffset;
        int lineSize;
} VAL_T;

__constant uint warpSize=64; //32 for NVIDIA should 64 for AMD

/****************************************************************************/
//Map phase kernels
/****************************************************************************/

// Called by user-defined map function
void emitIntermediate(void*  key,
                                 void*          val,
                                 uint           keySize,
                                 uint           valSize,
				 __global char*  interKeys,
		                 __global char*  interVals,
               			 __global int4*  interOffsetSizes,
                                 __constant uint *     interKeysOffsets,  //include the start write location of each workgroup
                                 __constant uint *     interValsOffsets,  //include the start write location of each workgroup
                                 __constant uint *     interOffsetSizesOffsets, //include the start write location of each workgroup
                                 __local uint *  lsKeySizes,
                                 __local uint *  lsValSizes,
                                 __local uint *  lsCounts,
                                 __local uint *  lsPrefixKeySizes,
                                 __local uint *  lsPrefixValSizes,
                                 __local uint *  lsPrefixCounts,                                 
                             __local uint *  lCurKeySizes,
                             __local uint *  lCurValSizes,
                             __local uint *  lCurCounts,
                             __global uint *  gKeySizes,
                             __global uint *  gValSizes,
                             __global uint * gCounts,
			     uint keysBufferSize,
                             uint valsBufferSize,
                             uint offsetsBufferSize,
			      __global uint4 * inputRecordsMeta,
			     uint inputRecordId)
{
    uint keySizes, valSizes, counts;

    uint tid=get_global_id(0);
    uint groupId=get_group_id(0);
    uint groupsNum=get_num_groups(0);
    uint tidInGroup = get_local_id(0);
    uint groupSize=get_local_size(0);	

    //prefixSum to sizes of a workgroup
    atom_add(&lsKeySizes[0],keySize);
    atom_add(&lsValSizes[0],valSize);
    atom_inc(&lsCounts[0]);

    if ((lsKeySizes[0] > keysBufferSize) || (lsValSizes[0]> valsBufferSize) ||(lsCounts[0] > offsetsBufferSize))
    {
        //update meta data indicating the status of each input record
        //1- Overflow
        //2- emitted records before overflow of the first mapper kernel
        //3- emitted records of the second mapper kernel
        //This data is used to indicate when to write the emitted record to the global buffer
        inputRecordsMeta[inputRecordId].x = 1; // Overflow in this record
    }
    else
    {
	//update meta data of this input record
         inputRecordsMeta[inputRecordId].x = 0; // No Overflow in this record
         inputRecordsMeta[inputRecordId].y++; //successfully emitted records

	keySizes = atom_add(&lsKeySizes[1],keySize);
	valSizes = atom_add(&lsValSizes[1],valSize);
	counts = atom_inc(&lsCounts[1]);
    
    	__global char *pKeySet = (__global char*)(interKeys + interKeysOffsets[groupId] + keySizes);
	__global char *pValSet = (__global char*)(interVals + interValsOffsets[groupId] + valSizes);

    	char* sKey = (char*)key;
	char* sVal = (char*)val;

    	for (int i = 0; i <keySize; ++i)
            pKeySet[i] = sKey[i];

    	for (int i = 0; i <valSize; ++i)
            pValSet[i] = sVal[i];


   	 __global int4 * offset= (__global int4*)(interOffsetSizes +  interOffsetSizesOffsets[groupId]);
    	int4 l_interOffsetSizes;
    	l_interOffsetSizes.x = interKeysOffsets[groupId] + keySizes;  //keySizes;
    	l_interOffsetSizes.z =  interValsOffsets[groupId] + valSizes; //valSizes;
    	l_interOffsetSizes.y = *((int*)key);//keySize;
    	l_interOffsetSizes.w = *((int*)val); //valSize;
    	offset[counts] = l_interOffsetSizes; 

    }

}

void emitIntermediateExtended(void*  key,
                                 void*          val,
                                 uint           keySize,
                                 uint           valSize,
                                 __global char*  interKeys,
                                 __global char*  interVals,
                                 __global int4*  interOffsetSizes,
                                 __constant uint *     interKeysOffsets,  //include the start write location of each workgroup
                                 __constant uint *     interValsOffsets,  //include the start write location of each workgroup
                                 __constant uint *     interOffsetSizesOffsets, //include the start write location of each workgroup
                                 __local uint *  lsKeySizes,
                                 __local uint *  lsValSizes,
                                 __local uint *  lsCounts,
                                 __local uint *  lsPrefixKeySizes,
                                 __local uint *  lsPrefixValSizes,
                                 __local uint *  lsPrefixCounts,
                             __local uint *  lCurKeySizes,
                             __local uint *  lCurValSizes,
                             __local uint *  lCurCounts,
                             __global uint *  gKeySizes,
                             __global uint *  gValSizes,
                             __global uint * gCounts,
                             uint keysBufferSize,
                             uint valsBufferSize,
                             uint offsetsBufferSize,
                              __global uint4 * inputRecordsMeta,
			      uint inputRecordId)
{
    uint keySizes, valSizes, counts;

    uint tid=get_global_id(0);
    uint groupId=get_group_id(0);
    uint groupsNum=get_num_groups(0);
    uint tidInGroup = get_local_id(0);
    uint groupSize=get_local_size(0);

   //To handle applications with multiple emits per one record
   inputRecordsMeta[inputRecordId].z++;
   if (inputRecordsMeta[inputRecordId].z > inputRecordsMeta[inputRecordId].y)
   {

        //Insert, we are sure there will be no overflow 
	//prefixSum to sizes of a workgroup
        keySizes = atom_add(&lsKeySizes[1],keySize);
        valSizes = atom_add(&lsValSizes[1],valSize);
        counts = atom_inc(&lsCounts[1]);

        __global char *pKeySet = (__global char*)(interKeys + interKeysOffsets[groupId] + keySizes);
        __global char *pValSet = (__global char*)(interVals + interValsOffsets[groupId] + valSizes);

        char* sKey = (char*)key;
        char* sVal = (char*)val;

        for (int i = 0; i <keySize; ++i)
            pKeySet[i] = sKey[i];

        for (int i = 0; i <valSize; ++i)
            pValSet[i] = sVal[i];


         __global int4 * offset= (__global int4*)(interOffsetSizes +  interOffsetSizesOffsets[groupId]);
        int4 l_interOffsetSizes;
        l_interOffsetSizes.x = interKeysOffsets[groupId] + keySizes;  //keySizes;
        l_interOffsetSizes.z = interValsOffsets[groupId] + valSizes; //valSizes;
        l_interOffsetSizes.y = keySize;
        l_interOffsetSizes.w = valSize;
        offset[counts] = l_interOffsetSizes;
   }

}


void map(__global void* inputDataset, __global void *key, __global void *val, int keySize, int valSize, 
		__global char*	interKeys,
		__global char*	interVals,
		__global int4*	interOffsetSizes,
		__constant uint *     interKeysOffsets,  //include the start write location of each workgroup
                __constant uint *     interValsOffsets,  //include the start write location of each workgroup
                __constant uint *     interOffsetSizesOffsets, //include the start write location of each workgroup
		__local uint *  lsKeySizes,
	    __local uint *  lsValSizes,
	    __local uint *  lsCounts,
	    __local uint *  lsPrefixKeySizes,
	    __local uint *  lsPrefixValSizes,
	    __local uint *  lsPrefixCounts,			     
	    __local uint *  lCurKeySizes,
	    __local uint *  lCurValSizes,
	    __local uint *  lCurCounts,
	    __global uint *  gKeySizes,
	    __global uint *  gValSizes,
	    __global uint * gCounts,
	    uint keysBufferSize,
            uint valsBufferSize,
            uint offsetsBufferSize,
            __global uint4 * inputRecordsMeta,
	    uint inputRecordId)
{

        __global KEY_T* pKey = (__global KEY_T*)key;
        __global VAL_T* pVal = (__global VAL_T*)val;

        int bufOffset = pVal->lineOffset;
        int bufSize = pVal->lineSize;
        __global char* buf = (__global char*) (inputDataset+ bufOffset);

        __global char* word =  (__global char*) (inputDataset+ pKey->keywordOffset);
        int keywordSize = pKey->keywordSize;

       int cur = 0;
        __global char* p = buf;
        __global char* start = buf;

        while(1)
        {
                while (1)
                {

                        if ( (*p ==' ') || (*p =='\0') || (*p =='\n') || (*p =='_')) break;
                        p++;
                        cur++;
                }

                p++;
                cur++;
                int wordSize = (int)(p - start);

                if (cur >= bufSize) break;

                __global char* k = word;
                __global char* s = start;

                if (wordSize == keywordSize)
                {
                        int i;
                        for (i=0 ; (i < wordSize); s++, k++,i++)
                        {
                                if (*s == *k)
                                        {  }
                                else
                                        break;
                        }

		        if ( i >=  wordSize -1)

				 emitIntermediate(&bufOffset, &wordSize, sizeof(int), sizeof(int),interKeys,interVals, interOffsetSizes,interKeysOffsets,
		                     interValsOffsets,interOffsetSizesOffsets, lsKeySizes, lsValSizes,
                		     lsCounts, lsPrefixKeySizes, lsPrefixValSizes, lsPrefixCounts, lCurKeySizes, lCurValSizes, lCurCounts,
		                     gKeySizes, gValSizes, gCounts, keysBufferSize, valsBufferSize, offsetsBufferSize, inputRecordsMeta,inputRecordId);



                }
                start = p;
                bufOffset += wordSize;
        }

}

void mapExtended(__global void* inputDataset, __global void *key, __global void *val, int keySize, int valSize,
                __global char*  interKeys,
                __global char*  interVals,
                __global int4*  interOffsetSizes,
                __constant uint *     interKeysOffsets,  //include the start write location of each workgroup
                __constant uint *     interValsOffsets,  //include the start write location of each workgroup
                __constant uint *     interOffsetSizesOffsets, //include the start write location of each workgroup
                __local uint *  lsKeySizes,
            __local uint *  lsValSizes,
            __local uint *  lsCounts,
            __local uint *  lsPrefixKeySizes,
            __local uint *  lsPrefixValSizes,
            __local uint *  lsPrefixCounts,
            __local uint *  lCurKeySizes,
            __local uint *  lCurValSizes,
            __local uint *  lCurCounts,
            __global uint *  gKeySizes,
            __global uint *  gValSizes,
            __global uint * gCounts,
            uint keysBufferSize,
            uint valsBufferSize,
            uint offsetsBufferSize,
            __global uint4 * inputRecordsMeta,
	    uint inputRecordId)
{

        __global KEY_T* pKey = (__global KEY_T*)key;
        __global VAL_T* pVal = (__global VAL_T*)val;

        int bufOffset = pVal->lineOffset;
        int bufSize = pVal->lineSize;
        __global char* buf = (__global char*) (inputDataset+ bufOffset);

        __global char* word =  (__global char*) (inputDataset+ pKey->keywordOffset);
        int keywordSize = pKey->keywordSize;

       if (inputRecordsMeta[inputRecordId].x == 0) return; // No overflow

       int cur = 0;
        __global char* p = buf;
        __global char* start = buf;

        while(1)
        {
                while (1)
                {

                        if ( (*p ==' ') || (*p =='\0') || (*p =='\n') || (*p =='_')) break;
                    	p++;
                        cur++;
                }

                p++;
                cur++;
                int wordSize = (int)(p - start);

                if (cur >= bufSize) break;

                __global char* k = word;
                __global char* s = start;

                if (wordSize == keywordSize)
                {
                        int i;
                        for (i=0 ; (i < wordSize); s++, k++,i++)
                        {
                                if (*s == *k)
                                        {  }
                                else
                                        break;
                        }

                        if ( i >=  wordSize -1)

                                 emitIntermediateExtended(&bufOffset, &wordSize, sizeof(int), sizeof(int),interKeys,interVals, interOffsetSizes,interKeysOffsets,
                                     interValsOffsets,interOffsetSizesOffsets, lsKeySizes, lsValSizes,
                                     lsCounts, lsPrefixKeySizes, lsPrefixValSizes, lsPrefixCounts, lCurKeySizes, lCurValSizes, lCurCounts,
                                     gKeySizes, gValSizes, gCounts, keysBufferSize, valsBufferSize, offsetsBufferSize, inputRecordsMeta,inputRecordId);



                }
                start = p;
                bufOffset += wordSize;
        }
}

__kernel void mapper(__global uint* inputDataset,
		  __constant uint * constantDatset,
		   __global char*	inputKeys,
		   __global char*	inputVals,
		   __global int4*	inputOffsetSizes,
		   __global char*	interKeys,
		   __global char*	interVals,
		   __global int4*	interOffsetSizes,
		   __constant uint *     interKeysOffsets,  //include the start write location of each workgroup
	           __constant uint *     interValsOffsets,  //include the start write location of each workgroup
                   __constant uint *     interOffsetSizesOffsets, //include the start write location of each workgroup
		   int	recordNum,
		   int	recordsPerTask,
		   int	taskNum,
		   __local uint * lsKeySizes,
	           __local uint *  lsValSizes,
	           __local uint *  lsCounts,
		   __local uint *  lsPrefixKeySizes,
		   __local uint *  lsPrefixValSizes,
		   __local uint *  lsPrefixCounts,			     
	           __local uint *  lCurKeySizes,
	           __local uint *  lCurValSizes,
	           __local uint *  lCurCounts,
	           __global uint *  gKeySizes,
	           __global uint *  gValSizes,
	           __global uint * gCounts,
		   uint keysBufferSize,
                   uint valsBufferSize,
                   uint offsetsBufferSize,
                   __global uint4 * inputRecordsMeta,
		   int extendedKernel)
{
	int index = get_global_id(0);
	int bid = get_group_id(0);
	int tid = get_local_id(0);
	int blockDimx=get_local_size(0);
	int numGroups=get_num_groups(0);
	
        if (extendedKernel == 1 )
        {
		bid = gCounts[bid];
		
	}
	// Initialize global variables carrying keysize, valsizes, keyvaloffests to zero
        if (tid == 0)
        	{
                	lsKeySizes[0] = 0;
	                lsValSizes[0] = 0;
        	        lsCounts[0] = 0;

                	lsKeySizes[1] = 0;
	                lsValSizes[1] = 0;
        	        lsCounts[1] = 0;

	        }
        	//barrier
	        mem_fence(CLK_GLOBAL_MEM_FENCE);
	

	if (index*recordsPerTask >= recordNum) return;
	int recordBase = bid * recordsPerTask * blockDimx;
	int terminate = (bid + 1) * (recordsPerTask * blockDimx);
	if (terminate > recordNum) terminate = recordNum;

	 
	for (int i = recordBase + tid; i < terminate; i+=blockDimx)
	{
		int cindex =  i;

		int4 offsetSize = inputOffsetSizes[cindex];
		__global char *key = inputKeys + offsetSize.x;
		__global char *val = inputVals + offsetSize.z;



	if (extendedKernel == 1 )
        {
		inputRecordsMeta[cindex].z = 0;
		
		mapExtended(inputDataset,
                key,
                val,
                offsetSize.y,
                offsetSize.w,
                interKeys,
                interVals,
                interOffsetSizes,
                interKeysOffsets,
                interValsOffsets,
                interOffsetSizesOffsets,
                lsKeySizes,
                lsValSizes,
                lsCounts,
                lsPrefixKeySizes,
                lsPrefixValSizes,
                lsPrefixCounts,
            	lCurKeySizes,
            	lCurValSizes,
		lCurCounts,
            	gKeySizes,
            	gValSizes,
            	gCounts,
            	keysBufferSize,
            	valsBufferSize,
            	offsetsBufferSize,
            	inputRecordsMeta,	
		cindex);
        }
	else
	{

              inputRecordsMeta[cindex].y = 0;
	
		 map(inputDataset,
                key,
                val,
                offsetSize.y,
                offsetSize.w,
                interKeys,
                interVals,
                interOffsetSizes,
		interKeysOffsets,
		interValsOffsets,
		interOffsetSizesOffsets,
                lsKeySizes,
                lsValSizes,
                lsCounts,
                lsPrefixKeySizes,
                lsPrefixValSizes,
                lsPrefixCounts,
            lCurKeySizes, 
            lCurValSizes, 
            lCurCounts,
            gKeySizes,
            gValSizes, 
            gCounts, 
            keysBufferSize, 
            valsBufferSize, 
            offsetsBufferSize, 
	    inputRecordsMeta,
	    cindex);
	
	}
       }
	barrier(CLK_GLOBAL_MEM_FENCE);

        if ( tid == 0 )
        {
            gKeySizes[bid]= lsKeySizes[0];
            gValSizes[bid]= lsValSizes[0];
            gCounts[bid]= lsCounts[0];

	    //used sizes
            gKeySizes[numGroups + bid] = lsKeySizes[1];
            gValSizes[numGroups + bid] = lsValSizes[1];
            gCounts[numGroups + bid] = lsCounts[1];

        }
}

	
/****************************************************************************/
//Reduce phase kernels
/****************************************************************************/

void reduce(__global void *key, 
            __global void *val, 
            int keySize, 
            int valCount, 
		    __global float*	psKeySizes,
		    __global float*	psValSizes,
		    __global float*	psCounts,
		    __global int4*  interOffsetSizes,
		    __global char*  interVals,
		    __global char*	outputKeys,
		    __global char*	outputVals,
		    __global int4*	outputOffsetSizes,
		    int valStartIndex)
{
}

__kernel void  reducer( __global char*		interKeys,
		        __global char*		interVals,
			__global int4*		interOffsetSizes,
			__global int2*		interKeyListRange,
		        __global uint*		psKeySizes,
		        __global uint*		psValSizes,
	  	        __global uint*		psCounts,
		        __global char*		outputKeys,
			__global char*		outputVals,
			__global int4*		outputOffsetSizes,
			int		recordNum,
			int		recordsPerTask,
			int		taskNum)
{
	int index = get_global_id(0);
	int bid = get_group_id(0);
	int tid = get_local_id(0);
	int blockDimx=get_local_size(0);
	
	if (index*recordsPerTask >= recordNum) return;
	int recordBase = bid * recordsPerTask * blockDimx;
	int terminate = (bid + 1) * (recordsPerTask * blockDimx);
	if (terminate > recordNum) terminate = recordNum;

        // Initialize global variables carrying keysize, valsizes, keyvaloffests to zero
        if (index == 0)
        {
                *psKeySizes = 0;
                *psValSizes=0;
                *psCounts=0;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = recordBase + tid; i < terminate; i+=blockDimx)
	{
		int cindex = i;

		int valStartIndex = interKeyListRange[cindex].x;
		int valCount = interKeyListRange[cindex].y - interKeyListRange[cindex].x;
		int keySize = interOffsetSizes[interKeyListRange[cindex].x].y;

		__global char *key = interKeys + interOffsetSizes[valStartIndex].x;
		__global char *vals = interVals + interOffsetSizes[valStartIndex].z;

		reduce(key,
			   vals,
			   keySize,
			   valCount,
			   psKeySizes,
			   psValSizes,
			   psCounts,
			   interOffsetSizes,
			   interVals,
			   outputKeys,
			   outputVals,
			   outputOffsetSizes,
			   valStartIndex);
	}
}


__kernel void mapperExtended()
{
}


__kernel void hashTableCleanup()
{
}

__kernel void hashTableCleanupInOverflow()
{
}

__kernel void reducerInOverflow()
{
}
__kernel void copyerHashToArray()
{
}

__kernel void copyerHashToArrayInOverflow()
{
}
	
