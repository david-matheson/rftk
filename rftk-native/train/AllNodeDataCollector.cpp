#include "assert_util.h"
#include "AllNodeDataCollector.h"


AllNodeDataCollector::AllNodeDataCollector()
{
}

AllNodeDataCollector::~AllNodeDataCollector()
{
}

void AllNodeDataCollector::Collect( const BufferCollection& data,
                                    const MatrixBufferInt& sampleIndices,
                                    const MatrixBufferFloat& featureValues,
                                    boost::mt19937& gen )
{
    ASSERT_ARG_DIM_1D(sampleIndices.GetN(), 1)
    ASSERT_ARG_DIM_1D(sampleIndices.GetM(), featureValues.GetM())

    mData.AppendVerticalMatrixBufferFloat(FEATURE_VALUES, featureValues);

    if( data.HasMatrixBufferFloat(SAMPLE_WEIGHTS) )
    {
        const MatrixBufferFloat& sampleWeights = data.GetMatrixBufferFloat(SAMPLE_WEIGHTS).Slice(sampleIndices);
        mData.AppendVerticalMatrixBufferFloat(SAMPLE_WEIGHTS, sampleWeights);
    }

    if( data.HasMatrixBufferInt(CLASS_LABELS) )
    {
        const MatrixBufferInt& classLabels = data.GetMatrixBufferInt(CLASS_LABELS).Slice(sampleIndices);
        mData.AppendVerticalMatrixBufferInt(CLASS_LABELS, classLabels);
    }
}

const BufferCollection& AllNodeDataCollector::GetCollectedData()
{
    return mData;
}

int AllNodeDataCollector::GetNumberOfCollectedSamples()
{
    return mData.GetMatrixBufferFloat(FEATURE_VALUES).GetM();
}


NodeDataCollectorFactoryI* AllNodeDataCollectorFactory::Clone() const
{
    return new AllNodeDataCollectorFactory(*this);
}

NodeDataCollectorI* AllNodeDataCollectorFactory::Create() const
{
    // printf("AllNodeDataCollectorFactory::Create\n");
    return new AllNodeDataCollector();
}


