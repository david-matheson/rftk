#include "assert_util.h"
#include "AllNodeDataCollector.h"


AllNodeDataCollector::AllNodeDataCollector()
{
}

AllNodeDataCollector::~AllNodeDataCollector()
{
}

void AllNodeDataCollector::Collect( const BufferCollection& data,
                                    const Int32MatrixBuffer& sampleIndices,
                                    const Float32MatrixBuffer& featureValues,
                                    boost::mt19937& gen )
{
    ASSERT_ARG_DIM_1D(sampleIndices.GetN(), 1)
    ASSERT_ARG_DIM_1D(sampleIndices.GetM(), featureValues.GetM())

    mData.AppendVerticalFloat32MatrixBuffer(FEATURE_VALUES, featureValues);

    if( data.HasFloat32MatrixBuffer(SAMPLE_WEIGHTS) )
    {
        const Float32MatrixBuffer& sampleWeights = data.GetFloat32MatrixBuffer(SAMPLE_WEIGHTS).Slice(sampleIndices);
        mData.AppendVerticalFloat32MatrixBuffer(SAMPLE_WEIGHTS, sampleWeights);
    }

    if( data.HasInt32MatrixBuffer(CLASS_LABELS) )
    {
        const Int32MatrixBuffer& classLabels = data.GetInt32MatrixBuffer(CLASS_LABELS).Slice(sampleIndices);
        mData.AppendVerticalInt32MatrixBuffer(CLASS_LABELS, classLabels);
    }
}

const BufferCollection& AllNodeDataCollector::GetCollectedData()
{
    return mData;
}

int AllNodeDataCollector::GetNumberOfCollectedSamples()
{
    return mData.GetFloat32MatrixBuffer(FEATURE_VALUES).GetM();
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


