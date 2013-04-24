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
template <class FloatType, class IntType>
class BootstrapSamplesStep: public PipelineStepI
{
public:
    BootstrapSamplesStep( const UniqueBufferId::BufferId& dataBuffer );
    virtual ~BootstrapSamplesStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection) const;

    // Read only output buffer
    const UniqueBufferId::BufferId IndicesBufferId;
    const UniqueBufferId::BufferId WeightsBufferId;
private:
    const UniqueBufferId::BufferId mDataBufferId;

};


template <class FloatType, class IntType>
BootstrapSamplesStep<FloatType, IntType>::BootstrapSamplesStep(const UniqueBufferId::BufferId& dataBuffer)
: IndicesBufferId(UniqueBufferId::GetBufferId("IndicesBuffer"))
, WeightsBufferId(UniqueBufferId::GetBufferId("WeightsBuffer"))
, mDataBufferId(dataBuffer)
{}

template <class FloatType, class IntType>
BootstrapSamplesStep<FloatType, IntType>::~BootstrapSamplesStep()
{}

template <class FloatType, class IntType>
PipelineStepI* BootstrapSamplesStep<FloatType, IntType>::Clone() const
{
    BootstrapSamplesStep<FloatType, IntType>* clone = new BootstrapSamplesStep<FloatType, IntType>(*this);
    return clone;
}

template <class FloatType, class IntType>
void BootstrapSamplesStep<FloatType, IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                                BufferCollection& writeCollection) const
{
    ASSERT(readCollection.HasBuffer< MatrixBufferTemplate<FloatType> >(mDataBufferId));

    const MatrixBufferTemplate<FloatType> & buffer 
          = readCollection.GetBuffer <MatrixBufferTemplate<FloatType> >(mDataBufferId);
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