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
    uint row;
} KEY_T;

typedef struct
{
    uint col;
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
    /*
#ifdef OVERFLOW
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


    __global char *pKeySet = (__global char*)(interKeys + interKeysOffsets[groupId] +  keySizes);
    __global char *pValSet = (__global char*)(interVals + interValsOffsets[groupId] +  valSizes);

    char* sKey = (char*)key;
    char* sVal = (char*)val;

    for (int i = 0; i < keySize; ++i)
    pKeySet[i] = sKey[i];
    for (int i = 0; i < valSize; ++i)
    pValSet[i] = sVal[i];

    __global int4 * offset= (__global int4*)(interOffsetSizes);
    offset+= interOffsetSizesOffsets[groupId] >> 2 +  counts;
    int4 l_interOffsetSizes;
    l_interOffsetSizes.x = interKeysOffsets[groupId]  + keySizes;  
    l_interOffsetSizes.z =  interValsOffsets[groupId] + valSizes; 
    l_interOffsetSizes.y = keySize;
    l_interOffsetSizes.w = valSize;
    offset[0] = l_interOffsetSizes;
    } 
#else
     */
    keySizes = atom_add(&lsKeySizes[1],keySize);
    valSizes = atom_add(&lsValSizes[1],valSize);
    counts = atom_inc(&lsCounts[1]);

    __global char *pKeySet = (__global char*)(interKeys + interKeysOffsets[groupId] +  keySizes);
    __global char *pValSet = (__global char*)(interVals + interValsOffsets[groupId] +  valSizes);

    char* sKey = (char*)key;
    char* sVal = (char*)val;

    for (int i = 0; i < keySize; ++i)
        pKeySet[i] = sKey[i];
    for (int i = 0; i < valSize; ++i)
        pValSet[i] = sVal[i];

    //   __global int4 * offset= (__global int4*)(interOffsetSizes);
    //   offset+= interOffsetSizesOffsets[groupId] >> 2 +  counts;

    __global int4 * offset= (__global int4*)(interOffsetSizes + interOffsetSizesOffsets[groupId]);
    int4 l_interOffsetSizes;
    l_interOffsetSizes.x = interKeysOffsets[groupId]  + keySizes;  
    l_interOffsetSizes.z =  interValsOffsets[groupId] + valSizes; 
    l_interOffsetSizes.y = keySize;
    l_interOffsetSizes.w = valSize;
    //offset[0] = l_interOffsetSizes;

    offset[counts] = l_interOffsetSizes;

    //#endif
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


void map(__global void* inputDataset,
        __constant uint* constantDataset,
        __global void *key, __global void *val, int keySize, int valSize,
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

    uint matrix2Offset = constantDataset[0];
    int M_ROW_COUNT = constantDataset[1];
    int M_COL_COUNT = constantDataset[2];

    __global VAL_T *pVal = (__global VAL_T*)val;
    __global KEY_T *pKey = (__global KEY_T*)key;
    int rowId = pKey->row;
    int colId = pVal->col;

    __global float4 *matrix1 = (__global float*)inputDataset+rowId*M_COL_COUNT;
    __global float4 *matrix2 = (__global float*)inputDataset+(colId+M_ROW_COUNT)*M_COL_COUNT;

    float newVal = 0.0f;

    int col4 = M_COL_COUNT>> 2;
    int remainder = M_COL_COUNT& 0x00000003;
    for (int i = 0; i < col4; i++)
    {

        float4 v1 = matrix1[i];
        float4 v2 = matrix2[i];

        newVal += v1.x * v2.x;
        newVal += v1.y * v2.y;
        newVal += v1.z * v2.z;
        newVal += v1.w * v2.w;
    }

    __global float *rMatrix1 = (__global float*)matrix1+col4;
    __global float *rMatrix2 = (__global float*)matrix2+col4;

    for (int i = 0; i < remainder; i++)
    {
        float f1 = rMatrix1[i];
        float f2 = rMatrix2[i];
        newVal += (f1 * f2);
    }

    float o_result= newVal;
    int2 o_pos;
    o_pos.x = rowId;
    o_pos.y = colId;

    emitIntermediate(&o_result, &o_pos, sizeof(float), sizeof(int2),interKeys,interVals, interOffsetSizes,interKeysOffsets,
            interValsOffsets,interOffsetSizesOffsets, lsKeySizes, lsValSizes,
            lsCounts, lsPrefixKeySizes, lsPrefixValSizes, lsPrefixCounts, lCurKeySizes, lCurValSizes, lCurCounts,
            gKeySizes, gValSizes, gCounts,  keysBufferSize, valsBufferSize, offsetsBufferSize, inputRecordsMeta,inputRecordId);                    
}



void mapExtended(__global void* inputDataset, __global void *key, __global void *val, int keySize, int valSize, 
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

}

__kernel void mapper(__global uint* inputDataset,
        __constant uint* constantDataset,
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
    int numGroups= get_num_groups(0);

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
        /*
#ifdef OVERFLOW

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
constantDataset,
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
#else
*/
inputRecordsMeta[cindex].y = 0;

map(inputDataset,
        constantDataset,
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

//#endif

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

//called by user defined reduce function
//---------------------------------------------------------
void emit  (__global void*		key,
        void*		val,
        int		keySize,
        int		valSize,
        __global uint*		psKeySizes,
        __global uint*		psValSizes,
        __global uint*		psCounts,
        __global char*		outputKeys,
        __global char*		outputVals,
        __global int4*		outputOffsetSizes)
{
    uint index = get_global_id(0);
    uint keySizes, valSizes, counts;

    //Atomic add to sizes
    keySizes=atom_add(psKeySizes,keySize);
    valSizes=atom_add(psValSizes,valSize);
    counts=atom_inc(psCounts);

    __global char *pKeySet = (__global char*)(outputKeys + keySizes);
    __global char *pValSet = (__global char*)(outputVals + valSizes);

    __global char* sKey= (__global char*)key;
    char* sVal = (char*)val;

    for (int i = 0; i < keySize; i++)
        pKeySet[i] = sKey[i];
    for (int i = 0; i < valSize; i++)
        pValSet[i] = sVal[i];

    int4 l_outputOffsetSizes;
    l_outputOffsetSizes.x = keySizes;
    l_outputOffsetSizes.z = valSizes;
    l_outputOffsetSizes.y = keySize;
    l_outputOffsetSizes.w = valSize;
    outputOffsetSizes[counts] = l_outputOffsetSizes;
}


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
