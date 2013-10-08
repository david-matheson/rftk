#pragma once

#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "LinearMatrixFeature.h"

// ----------------------------------------------------------------------------
//
// Slice a buffer by a collection of feature indices
//
// ----------------------------------------------------------------------------
template <class BufferTypes, class DataBufferType>
class SliceAxisAlignedFeaturesStep: public PipelineStepI
{
public:
    SliceAxisAlignedFeaturesStep(  const BufferId& buffer,
                                    const BufferId& axisAlignedIntParamsBufferId );
    virtual ~SliceAxisAlignedFeaturesStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    // Read only output buffer
    const BufferId SlicedBufferId;
private:
    const BufferId mBufferBufferId;
    const BufferId mAxisAlignedIntParamsBufferId;
};


template <class BufferTypes, class DataBufferType>
SliceAxisAlignedFeaturesStep<BufferTypes, DataBufferType>::SliceAxisAlignedFeaturesStep(const BufferId& buffer,
                                                                const BufferId& axisAlignedIntParamsBufferId)
: PipelineStepI("SliceAxisAlignedFeaturesStep")
, SlicedBufferId(buffer+GetBufferId("-Sliced"))
, mBufferBufferId(buffer)
, mAxisAlignedIntParamsBufferId(axisAlignedIntParamsBufferId)
{}

template <class BufferTypes, class DataBufferType>
SliceAxisAlignedFeaturesStep<BufferTypes, DataBufferType>::~SliceAxisAlignedFeaturesStep()
{}

template <class BufferTypes, class DataBufferType>
PipelineStepI* SliceAxisAlignedFeaturesStep<BufferTypes, DataBufferType>::Clone() const
{
    SliceAxisAlignedFeaturesStep<BufferTypes, DataBufferType>* clone = new SliceAxisAlignedFeaturesStep<BufferTypes, DataBufferType>(*this);
    return clone;
}

template <class BufferTypes, class DataBufferType>
void SliceAxisAlignedFeaturesStep<BufferTypes, DataBufferType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                                BufferCollection& writeCollection,
                                                                boost::mt19937& gen,
                                                                BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(gen);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);
    const DataBufferType& buffer = readCollection.GetBuffer<DataBufferType>(mBufferBufferId);
    const MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& axisAlignedIntParams =
          readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mAxisAlignedIntParamsBufferId);

    const int numberOfIndices = axisAlignedIntParams.GetM();
    VectorBufferTemplate<int> indices(numberOfIndices);
    for(int i=0; i<numberOfIndices; i++)
    {
        ASSERT(axisAlignedIntParams.Get(i,0) == MATRIX_FEATURES);
        ASSERT(axisAlignedIntParams.Get(i,1) == 1);
        indices.Set(i, axisAlignedIntParams.Get(i,2));
    }

    const DataBufferType& slicedBuffer = buffer.Slice(indices);
    writeCollection.AddBuffer<DataBufferType>(SlicedBufferId, slicedBuffer);
}