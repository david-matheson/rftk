#pragma once

#include <asserts.h>

#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
//
// Produces index and weight buffers for all datapoints (all weights are one)
//
// ----------------------------------------------------------------------------
template <class BufferTypes, class DataMatrixType>
class AllSamplesStep: public PipelineStepI
{
public:
    AllSamplesStep( const BufferId& dataBuffer );
    virtual ~AllSamplesStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    // Read only output buffer
    const BufferId IndicesBufferId;
    const BufferId WeightsBufferId;
private:
    const BufferId mDataBufferId;

};


template <class BufferTypes, class DataMatrixType>
AllSamplesStep<BufferTypes, DataMatrixType>::AllSamplesStep(const BufferId& dataBuffer)
: PipelineStepI("AllSamplesStep")
, IndicesBufferId(GetBufferId("IndicesBuffer"))
, WeightsBufferId(GetBufferId("WeightsBuffer"))
, mDataBufferId(dataBuffer)
{}

template <class BufferTypes, class DataMatrixType>
AllSamplesStep<BufferTypes, DataMatrixType>::~AllSamplesStep()
{}

template <class BufferTypes, class DataMatrixType>
PipelineStepI* AllSamplesStep<BufferTypes, DataMatrixType>::Clone() const
{
    AllSamplesStep<BufferTypes, DataMatrixType>* clone = new AllSamplesStep<BufferTypes, DataMatrixType>(*this);
    return clone;
}

template <class BufferTypes, class DataMatrixType>
void AllSamplesStep<BufferTypes, DataMatrixType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                                BufferCollection& writeCollection,
                                                                boost::mt19937& gen,
                                                                BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(gen);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);

    const DataMatrixType & buffer
          = readCollection.GetBuffer <DataMatrixType >(mDataBufferId);
    VectorBufferTemplate<typename BufferTypes::Index>& indices
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(IndicesBufferId);
    VectorBufferTemplate<typename BufferTypes::ParamsContinuous>& weights
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename BufferTypes::ParamsContinuous> >(WeightsBufferId);

    const typename BufferTypes::Index numberOfSamples = buffer.GetM();
    weights.Resize(numberOfSamples);
    weights.SetAll(typename BufferTypes::ParamsContinuous(1));
    indices.Resize(numberOfSamples);
    for(int i=0; i<numberOfSamples; i++)
    {
        indices.Set(i, i);
    }
}