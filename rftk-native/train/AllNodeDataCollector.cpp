#include "assert_util.h"
#include "AllNodeDataCollector.h"


AllNodeDataCollector::AllNodeDataCollector()
{
}

AllNodeDataCollector::~AllNodeDataCollector()
{
}

void AllNodeDataCollector::Collect( const BufferCollection& data,
                                    const Int32VectorBuffer& sampleIndices,
                                    const Float32MatrixBuffer& featureValues,
                                    boost::mt19937& gen )
{
    ASSERT_ARG_DIM_1D(sampleIndices.GetN(), featureValues.GetM())

    mData.AppendFloat32MatrixBuffer(FEATURE_VALUES, featureValues);

    if( data.HasFloat32VectorBuffer(SAMPLE_WEIGHTS) )
    {
        const Float32VectorBuffer& sampleWeights = data.GetFloat32VectorBuffer(SAMPLE_WEIGHTS).Slice(sampleIndices);
        mData.AppendFloat32VectorBuffer(SAMPLE_WEIGHTS, sampleWeights);
    }

    if( data.HasInt32VectorBuffer(CLASS_LABELS) )
    {
        const Int32VectorBuffer& classLabels = data.GetInt32VectorBuffer(CLASS_LABELS).Slice(sampleIndices);
        mData.AppendInt32VectorBuffer(CLASS_LABELS, classLabels);
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


