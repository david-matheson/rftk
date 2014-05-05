#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "SplitBuffersI.h"

template <class BufferTypes>
class SplitBuffersIndices : public SplitBuffersI
{
public:
    SplitBuffersIndices(const BufferId& indicesBufferId);
    virtual ~SplitBuffersIndices();

    virtual void SplitBuffers(const SplitSelectorBuffers& splitSelectorBuffers,
                              int bestFeature,
                              int bestSplitpoint,
                              const BufferCollectionStack& readBuffers,
                              BufferCollection& leftBuffers, 
                              BufferCollection& rightBuffers) const;

    virtual SplitBuffersI* Clone() const;

private:
    BufferId mIndicesBufferId;
};

template <class BufferTypes>
SplitBuffersIndices<BufferTypes>::SplitBuffersIndices(const BufferId& indicesBufferId)
: mIndicesBufferId(indicesBufferId)
{}

template <class BufferTypes>
SplitBuffersIndices<BufferTypes>::~SplitBuffersIndices()
{}

template <class BufferTypes>
void SplitBuffersIndices<BufferTypes>::SplitBuffers(const SplitSelectorBuffers& splitSelectorBuffers,
                                                        int bestFeature,
                                                        int bestSplitpoint,
                                                        const BufferCollectionStack& readBuffers,
                                                        BufferCollection& leftBuffers, 
                                                        BufferCollection& rightBuffers) const
{
    const VectorBufferTemplate<typename BufferTypes::Index>& indices
          = readBuffers.GetBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(mIndicesBufferId);

    const MatrixBufferTemplate<typename BufferTypes::FeatureValue>& featureValuesMatrix
          = readBuffers.GetBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(splitSelectorBuffers.mFeatureValuesBufferId);

    const MatrixBufferTemplate<typename BufferTypes::FeatureValue>& splitpoints
          = readBuffers.GetBuffer< MatrixBufferTemplate<typename BufferTypes::FeatureValue> >(splitSelectorBuffers.mSplitpointsBufferId);

    const typename BufferTypes::FeatureValue bestSplitpointValue = splitpoints.Get(bestFeature, bestSplitpoint);
    VectorBufferTemplate<typename BufferTypes::FeatureValue> featureValues;
    if( splitSelectorBuffers.mOrdering == FEATURES_BY_DATAPOINTS )
    {
        featureValues = featureValuesMatrix.SliceRowAsVector(bestFeature);
    }
    else if ( splitSelectorBuffers.mOrdering == DATAPOINTS_BY_FEATURES )
    {
        featureValues = featureValuesMatrix.SliceColumnAsVector(bestFeature);
    }
    ASSERT_ARG_DIM_1D(featureValues.GetN(), indices.GetN())

    std::vector<typename BufferTypes::Index> leftIndices;
    std::vector<typename BufferTypes::Index> rightIndices;
    for(int i=0; i<indices.GetN(); i++)
    {
        const typename BufferTypes::FeatureValue featureValue = featureValues.Get(i);
        const typename BufferTypes::Index index = indices.Get(i);
        if( featureValue > bestSplitpointValue )
        {
            leftIndices.push_back(index);
        }
        else
        {
            rightIndices.push_back(index);
        }
    }
    VectorBufferTemplate<typename BufferTypes::Index> leftIndicesBuf(&leftIndices[0], leftIndices.size());
    leftBuffers.AddBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(mIndicesBufferId, leftIndicesBuf );

    VectorBufferTemplate<typename BufferTypes::Index> rightIndicesBuf(&rightIndices[0], rightIndices.size());
    rightBuffers.AddBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(mIndicesBufferId, rightIndicesBuf );
}

template <class BufferTypes>
SplitBuffersI* SplitBuffersIndices<BufferTypes>::Clone() const
{
    SplitBuffersIndices<BufferTypes>* clone = new SplitBuffersIndices<BufferTypes>(*this);
    return clone;
}
