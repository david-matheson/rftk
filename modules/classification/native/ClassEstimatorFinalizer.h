#pragma once

#include "VectorBuffer.h"
#include "FinalizerI.h"

// ----------------------------------------------------------------------------
//
// Normalize the class histograms to probabilities
//
// ----------------------------------------------------------------------------
template <class FloatType>
class ClassEstimatorFinalizer: public FinalizerI<FloatType>
{
public:
    virtual void Finalize(FloatType count, VectorBufferTemplate<FloatType>& estimator) const;
    virtual FinalizerI<FloatType>* Clone() const;
};

template <class FloatType>
void ClassEstimatorFinalizer<FloatType>::Finalize(FloatType count, VectorBufferTemplate<FloatType>& estimator) const
{
    estimator.Normalize();
}

template <class FloatType>
FinalizerI<FloatType>* ClassEstimatorFinalizer<FloatType>::Clone() const
{
    ClassEstimatorFinalizer<FloatType>* clone = new ClassEstimatorFinalizer<FloatType>(*this);
    return clone; 
}