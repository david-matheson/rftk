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
template <class BufferTypes>
class PoissonStep: public PipelineStepI
{
public:
    PoissonStep( const double mean, const int dimension );
    virtual ~PoissonStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    // Read only output buffer
    const BufferId OutputBufferId;
private:
    const typename BufferTypes::SourceContinuous mMean;
    const typename BufferTypes::Index mDimension;

};


template <class BufferTypes>
PoissonStep<BufferTypes>::PoissonStep(const double mean, const int dimension)
: OutputBufferId(GetBufferId("PoissonStep"))
, mMean(mean)
, mDimension(dimension)
{}

template <class BufferTypes>
PoissonStep<BufferTypes>::~PoissonStep()
{}

template <class BufferTypes>
PipelineStepI* PoissonStep<BufferTypes>::Clone() const
{
    PoissonStep<BufferTypes>* clone = new PoissonStep<BufferTypes>(*this);
    return clone;
}

template <class BufferTypes>
void PoissonStep<BufferTypes>::ProcessStep(const BufferCollectionStack& readCollection,
                                                                BufferCollection& writeCollection,
                                                                boost::mt19937& gen) const
{
    UNUSED_PARAM(readCollection)

    VectorBufferTemplate<typename BufferTypes::SourceInteger>& output
          = writeCollection.GetOrAddBuffer< VectorBufferTemplate<typename BufferTypes::SourceInteger> >(OutputBufferId);
    output.Resize(mDimension);

    boost::poisson_distribution<> poisson(mMean);
    boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(gen, poisson);

    for(typename BufferTypes::Index d=0; d<mDimension; d++)
    {
        const typename BufferTypes::SourceInteger weight = var_poisson();
        output.Set(d, weight);
    }
}