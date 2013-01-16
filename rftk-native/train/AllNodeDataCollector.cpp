#include "assert_util.h"
#include "AllNodeDataCollector.h"


AllNodeDataCollector::AllNodeDataCollector()
: mNumberOfCollectedSamples(0)
{ 
}

void AllNodeDataCollector::Collect( BufferCollection& data,
                                    const MatrixBufferInt& sampleIndices,
                                    const MatrixBufferFloat& featureValues ) 
{
    printf("AllNodeDataCollector::Collect\n");

    ASSERT_ARG_DIM_1D(sampleIndices.GetN(), 1)
    ASSERT_ARG_DIM_1D(sampleIndices.GetM(), featureValues.GetM())

    mNumberOfCollectedSamples += sampleIndices.GetM();

    if( data.HasMatrixBufferFloat(FEATURE_VALUES) )
    {
        printf("AllNodeDataCollector::Collect FEATURE_VALUES\n");
        MatrixBufferFloat featureValues = data.GetMatrixBufferFloat(FEATURE_VALUES).Slice(sampleIndices);
        printf("AllNodeDataCollector::Collect FEATURE_VALUES append\n");
        mData.AppendVerticalMatrixBufferFloat(FEATURE_VALUES, featureValues);
    }

    if( data.HasMatrixBufferFloat(SAMPLE_WEIGHTS) )
    {
        printf("AllNodeDataCollector::Collect SAMPLE_WEIGHTS\n");
        MatrixBufferFloat sampleWeights = data.GetMatrixBufferFloat(SAMPLE_WEIGHTS).Slice(sampleIndices);
        mData.AppendVerticalMatrixBufferFloat(SAMPLE_WEIGHTS, sampleWeights);
    }

    if( data.HasMatrixBufferInt(CLASS_LABELS) )
    {
        MatrixBufferInt classLabels = data.GetMatrixBufferInt(CLASS_LABELS).Slice(sampleIndices);        
        mData.AppendVerticalMatrixBufferInt(CLASS_LABELS, classLabels);
    }    
}

BufferCollection AllNodeDataCollector::GetCollectedData() 
{
    return mData; 
}

int AllNodeDataCollector::GetNumberOfCollectedSamples()
{
    return mData.GetMatrixBufferFloat(FEATURE_VALUES).GetM(); 
}


NodeDataCollectorI* AllNodeDataCollectorFactory::Create() const 
{
    printf("AllNodeDataCollectorFactory::Create\n");
    return new AllNodeDataCollector(); 
}


