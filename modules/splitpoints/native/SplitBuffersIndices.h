#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "SplitBuffersI.h"

template <class FloatType, class IntType>
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

template <class FloatType, class IntType>
SplitBuffersIndices<FloatType, IntType>::SplitBuffersIndices(const BufferId& indicesBufferId)
: mIndicesBufferId(indicesBufferId)
{}

template <class FloatType, class IntType>
SplitBuffersIndices<FloatType, IntType>::~SplitBuffersIndices()
{}

template <class FloatType, class IntType>
void SplitBuffersIndices<FloatType, IntType>::SplitBuffers(const SplitSelectorBuffers& splitSelectorBuffers,
                                                        int bestFeature,
                                                        int bestSplitpoint,
                                                        const BufferCollectionStack& readBuffers,
                                                        BufferCollection& leftBuffers, 
                                                        BufferCollection& rightBuffers) const
{
    const VectorBufferTemplate<IntType>& indices
          = readBuffers.GetBuffer< VectorBufferTemplate<IntType> >(mIndicesBufferId);

    const MatrixBufferTemplate<FloatType>& featureValuesMatrix
          = readBuffers.GetBuffer< MatrixBufferTemplate<FloatType> >(splitSelectorBuffers.mFeatureValuesBufferId);

    const MatrixBufferTemplate<FloatType>& splitpoints
          = readBuffers.GetBuffer< MatrixBufferTemplate<FloatType> >(splitSelectorBuffers.mSplitpointsBufferId);

    const FloatType bestSplitpointValue = splitpoints.Get(bestFeature, bestSplitpoint);
    VectorBufferTemplate<FloatType> featureValues;
    if( splitSelectorBuffers.mOrdering == FEATURES_BY_DATAPOINTS )
    {
        featureValues = featureValuesMatrix.SliceRowAsVector(bestFeature);
    }
    else if ( splitSelectorBuffers.mOrdering == DATAPOINTS_BY_FEATURES )
    {
        featureValues = featureValuesMatrix.SliceColumnAsVector(bestFeature);
    }
    ASSERT_ARG_DIM_1D(featureValues.GetN(), indices.GetN())

    std::vector<IntType> leftIndices;
    std::vector<IntType> rightIndices;
    for(int i=0; i<indices.GetN(); i++)
    {
        const FloatType featureValue = featureValues.Get(i);
        const IntType index = indices.Get(i);
        if( featureValue > bestSplitpointValue )
        {
            leftIndices.push_back(index);
        }
        else
        {
            rightIndices.push_back(index);
        }
    }
    VectorBufferTemplate<IntType> leftIndicesBuf(&leftIndices[0], leftIndices.size());
    leftBuffers.AddBuffer< VectorBufferTemplate<IntType> >(mIndicesBufferId, leftIndicesBuf );

    VectorBufferTemplate<IntType> rightIndicesBuf(&rightIndices[0], rightIndices.size());
    rightBuffers.AddBuffer< VectorBufferTemplate<IntType> >(mIndicesBufferId, rightIndicesBuf );
}

template <class FloatType, class IntType>
SplitBuffersI* SplitBuffersIndices<FloatType, IntType>::Clone() const
{
    SplitBuffersIndices<FloatType, IntType>* clone = new SplitBuffersIndices<FloatType, IntType>(*this);
    return clone;
}