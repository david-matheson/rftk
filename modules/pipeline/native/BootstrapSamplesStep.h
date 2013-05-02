#pragma once

#include <asserts.h>
#include <bootstrap.h>

#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
//
// Bootstrap data (sample datapoints without replacement) to determine weights
// and indices of datapoints to include
//
// ----------------------------------------------------------------------------
template <class MatrixType, class FloatType, class IntType>
class BootstrapSamplesStep: public PipelineStepI
{
public:
    BootstrapSamplesStep( const BufferId& dataBuffer );
    virtual ~BootstrapSamplesStep();

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
BootstrapSamplesStep<MatrixType, FloatType, IntType>::BootstrapSamplesStep(const BufferId& dataBuffer)
: IndicesBufferId(GetBufferId("IndicesBuffer"))
, WeightsBufferId(GetBufferId("WeightsBuffer"))
, mDataBufferId(dataBuffer)
{}

template <class MatrixType, class FloatType, class IntType>
BootstrapSamplesStep<MatrixType, FloatType, IntType>::~BootstrapSamplesStep()
{}

template <class MatrixType, class FloatType, class IntType>
PipelineStepI* BootstrapSamplesStep<MatrixType, FloatType, IntType>::Clone() const
{
    BootstrapSamplesStep<MatrixType, FloatType, IntType>* clone = new BootstrapSamplesStep<MatrixType, FloatType, IntType>(*this);
    return clone;
}

template <class MatrixType, class FloatType, class IntType>
void BootstrapSamplesStep<MatrixType, FloatType, IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                                BufferCollection& writeCollection) const
{
    ASSERT(readCollection.HasBuffer< MatrixType >(mDataBufferId));

    const MatrixType & buffer
          = readCollection.GetBuffer< MatrixType >(mDataBufferId);
    VectorBufferTemplate<IntType>& indices
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<IntType> >(IndicesBufferId);
    VectorBufferTemplate<FloatType>& weights
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<FloatType> >(WeightsBufferId);

    const IntType numberOfSamples = buffer.GetM();
    weights.Resize(numberOfSamples);
    weights.Zero();

    // This implementation could be much simpler if bootstrap.h was cleaned up
    std::vector<int> counts(numberOfSamples);
    sampleWithReplacement( &counts[0],
                          numberOfSamples,
                          numberOfSamples);

    std::vector<IntType> sampledIndices;
    for(unsigned int i=0; i<counts.size(); i++)
    {
        if( counts[i] > 0 )
        {
            weights.Set(i,static_cast<float>(counts[i]));
            sampledIndices.push_back(i);
        }
    }
    indices = VectorBufferTemplate<IntType>(&sampledIndices[0], sampledIndices.size());

}