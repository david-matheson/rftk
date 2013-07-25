#pragma once

#include <asserts.h>

#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
//
// Sample the weight of each index from a poisson distribution this is used for
// online bagging
//
// ----------------------------------------------------------------------------
template <class BufferTypes, class DataMatrixType>
class PoissonSamplesStep: public PipelineStepI
{
public:
    PoissonSamplesStep( const BufferId& dataBuffer, const typename BufferTypes::SourceContinuous mean );
    virtual ~PoissonSamplesStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    // Read only output buffer
    const BufferId IndicesBufferId;
    const BufferId WeightsBufferId;
private:
    const BufferId mDataBufferId;
    const typename BufferTypes::SourceContinuous mMean;

};


template <class BufferTypes, class DataMatrixType>
PoissonSamplesStep<BufferTypes, DataMatrixType>::PoissonSamplesStep(const BufferId& dataBuffer, const typename BufferTypes::SourceContinuous mean)
: IndicesBufferId(GetBufferId("IndicesBuffer"))
, WeightsBufferId(GetBufferId("WeightsBuffer"))
, mDataBufferId(dataBuffer)
, mMean(mean)
{}

template <class BufferTypes, class DataMatrixType>
PoissonSamplesStep<BufferTypes, DataMatrixType>::~PoissonSamplesStep()
{}

template <class BufferTypes, class DataMatrixType>
PipelineStepI* PoissonSamplesStep<BufferTypes, DataMatrixType>::Clone() const
{
    PoissonSamplesStep<BufferTypes, DataMatrixType>* clone = new PoissonSamplesStep<BufferTypes, DataMatrixType>(*this);
    return clone;
}

template <class BufferTypes, class DataMatrixType>
void PoissonSamplesStep<BufferTypes, DataMatrixType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                                BufferCollection& writeCollection,
                                                                boost::mt19937& gen) const
{
    const DataMatrixType & buffer
          = readCollection.GetBuffer< DataMatrixType >(mDataBufferId);
    VectorBufferTemplate<typename BufferTypes::Index>& indices
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(IndicesBufferId);
    VectorBufferTemplate<typename BufferTypes::ParamsContinuous>& weights
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename BufferTypes::ParamsContinuous> >(WeightsBufferId);

    boost::poisson_distribution<> poisson(mMean);
    boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(gen, poisson);

    const typename BufferTypes::Index numberOfSamples = buffer.GetM();
    weights.Resize(numberOfSamples);
    weights.Zero();
    std::vector<typename BufferTypes::Index> sampledIndices;
    for(typename BufferTypes::Index s=0; s<numberOfSamples; s++)
    {
        const typename BufferTypes::Index weight = var_poisson();
        if(weight > 0)
        {
            weights.Set(s, static_cast<typename BufferTypes::ParamsContinuous>(weight));
            sampledIndices.push_back(s);
        }
    }

    indices = VectorBufferTemplate<typename BufferTypes::Index>(&sampledIndices[0], sampledIndices.size());
}