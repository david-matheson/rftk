#pragma once

#include "asserts.h"
#include "VectorBuffer.h"
#include "FinalizerI.h"

// ----------------------------------------------------------------------------
//
// Compute the mean and variance from the sufficient statistics
//
// ----------------------------------------------------------------------------
template <class FloatType>
class MeanVarianceEstimatorFinalizer: public FinalizerI<FloatType>
{
public:
    virtual void Finalize(FloatType count, VectorBufferTemplate<FloatType>& estimator) const;
    virtual FinalizerI<FloatType>* Clone() const;
    virtual ~MeanVarianceEstimatorFinalizer() {}
};

template <class FloatType>
void MeanVarianceEstimatorFinalizer<FloatType>::Finalize(FloatType count, VectorBufferTemplate<FloatType>& estimator) const
{
    UNUSED_PARAM(count);
    UNUSED_PARAM(estimator);
    // Switched to online method that maintains the mean
    // for(int d=0; d<estimator.GetN()/2; d++)
    // {
    //     FloatType value = estimator.Get(d);
    //     estimator.Set(d, value/count);
    // }
}

template <class FloatType>
FinalizerI<FloatType>* MeanVarianceEstimatorFinalizer<FloatType>::Clone() const
{
    MeanVarianceEstimatorFinalizer<FloatType>* clone = new MeanVarianceEstimatorFinalizer<FloatType>(*this);
    return clone;
}