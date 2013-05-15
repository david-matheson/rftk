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
template <class MatrixType, class FloatType, class IntType>
class PoissonSamplesStep: public PipelineStepI
{
public:
    PoissonSamplesStep( const BufferId& dataBuffer, const FloatType mean );
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
    const FloatType mMean;

};


template <class MatrixType, class FloatType, class IntType>
PoissonSamplesStep<MatrixType, FloatType, IntType>::PoissonSamplesStep(const BufferId& dataBuffer,const FloatType mean)
: IndicesBufferId(GetBufferId("IndicesBuffer"))
, WeightsBufferId(GetBufferId("WeightsBuffer"))
, mDataBufferId(dataBuffer)
, mMean(mean)
{}

template <class MatrixType, class FloatType, class IntType>
PoissonSamplesStep<MatrixType, FloatType, IntType>::~PoissonSamplesStep()
{}

template <class MatrixType, class FloatType, class IntType>
PipelineStepI* PoissonSamplesStep<MatrixType, FloatType, IntType>::Clone() const
{
    PoissonSamplesStep<MatrixType, FloatType, IntType>* clone = new PoissonSamplesStep<MatrixType, FloatType, IntType>(*this);
    return clone;
}

template <class MatrixType, class FloatType, class IntType>
void PoissonSamplesStep<MatrixType, FloatType, IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                                BufferCollection& writeCollection,
                                                                boost::mt19937& gen) const
{
    const MatrixType & buffer
          = readCollection.GetBuffer< MatrixType >(mDataBufferId);
    VectorBufferTemplate<IntType>& indices
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<IntType> >(IndicesBufferId);
    VectorBufferTemplate<FloatType>& weights
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<FloatType> >(WeightsBufferId);

    boost::poisson_distribution<> poisson(mMean);
    boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(gen, poisson);

    const IntType numberOfSamples = buffer.GetM();
    weights.Resize(numberOfSamples);
    weights.Zero();
    std::vector<IntType> sampledIndices;
    for(IntType s=0; s<numberOfSamples; s++)
    {
        const IntType weight = var_poisson();
        if(weight > 0)
        {
            weights.Set(s, static_cast<FloatType>(weight));
            sampledIndices.push_back(s);
        }
    }

    indices = VectorBufferTemplate<IntType>(&sampledIndices[0], sampledIndices.size());
}