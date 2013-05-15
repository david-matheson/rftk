#pragma once

#include <asserts.h>

#include "BufferCollection.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
//
// Sample a poisson
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class PoissonStep: public PipelineStepI
{
public:
    PoissonStep( const FloatType mean, const IntType dimension );
    virtual ~PoissonStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    // Read only output buffer
    const BufferId OutputBufferId;
private:
    const FloatType mMean;
    const IntType mDimension;

};


template <class FloatType, class IntType>
PoissonStep<FloatType, IntType>::PoissonStep(const FloatType mean, const IntType dimension)
: OutputBufferId(GetBufferId("PoissonStep"))
, mMean(mean)
, mDimension(dimension)
{}

template <class FloatType, class IntType>
PoissonStep<FloatType, IntType>::~PoissonStep()
{}

template <class FloatType, class IntType>
PipelineStepI* PoissonStep<FloatType, IntType>::Clone() const
{
    PoissonStep<FloatType, IntType>* clone = new PoissonStep<FloatType, IntType>(*this);
    return clone;
}

template <class FloatType, class IntType>
void PoissonStep<FloatType, IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                                BufferCollection& writeCollection,
                                                                boost::mt19937& gen) const
{
    UNUSED_PARAM(readCollection)

    VectorBufferTemplate<IntType>& output
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<IntType> >(OutputBufferId);
    output.Resize(mDimension);

    boost::poisson_distribution<> poisson(mMean);
    boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(gen, poisson);

    for(IntType d=0; d<mDimension; d++)
    {
        const IntType weight = var_poisson();
        output.Set(d, weight);
    }
}