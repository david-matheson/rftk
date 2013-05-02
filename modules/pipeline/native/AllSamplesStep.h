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
template <class MatrixType, class FloatType, class IntType>
class AllSamplesStep: public PipelineStepI
{
public:
    AllSamplesStep( const BufferId& dataBuffer );
    virtual ~AllSamplesStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection) const;

    // Read only output buffer
    const BufferId IndicesBufferId;
    const BufferId WeightsBufferId;
private:
    const BufferId mDataBufferId;

};


template <class MatrixType, class FloatType, class IntType>
AllSamplesStep<MatrixType, FloatType, IntType>::AllSamplesStep(const BufferId& dataBuffer)
: IndicesBufferId(GetBufferId("IndicesBuffer"))
, WeightsBufferId(GetBufferId("WeightsBuffer"))
, mDataBufferId(dataBuffer)
{}

template <class MatrixType, class FloatType, class IntType>
AllSamplesStep<MatrixType, FloatType, IntType>::~AllSamplesStep()
{}

template <class MatrixType, class FloatType, class IntType>
PipelineStepI* AllSamplesStep<MatrixType, FloatType, IntType>::Clone() const
{
    AllSamplesStep<MatrixType, FloatType, IntType>* clone = new AllSamplesStep<MatrixType, FloatType, IntType>(*this);
    return clone;
}

template <class MatrixType, class FloatType, class IntType>
void AllSamplesStep<MatrixType, FloatType, IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                                BufferCollection& writeCollection) const
{
    ASSERT(readCollection.HasBuffer< MatrixType >(mDataBufferId));

    const MatrixType & buffer
          = readCollection.GetBuffer <MatrixType >(mDataBufferId);
    VectorBufferTemplate<IntType>& indices
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<IntType> >(IndicesBufferId);
    VectorBufferTemplate<FloatType>& weights
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<FloatType> >(WeightsBufferId);

    const IntType numberOfSamples = buffer.GetM();
    weights.Resize(numberOfSamples);
    weights.SetAll(FloatType(1));
    indices.Resize(numberOfSamples);
    for(int i=0; i<numberOfSamples; i++)
    {
        indices.Set(i, i);
    }
}