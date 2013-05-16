#pragma once

#include "asserts.h"
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
    virtual ~ClassEstimatorFinalizer() {}
};

template <class FloatType>
void ClassEstimatorFinalizer<FloatType>::Finalize(FloatType count, VectorBufferTemplate<FloatType>& estimator) const
{
    UNUSED_PARAM(count);
    estimator.Normalize();
}

template <class FloatType>
FinalizerI<FloatType>* ClassEstimatorFinalizer<FloatType>::Clone() const
{
    ClassEstimatorFinalizer<FloatType>* clone = new ClassEstimatorFinalizer<FloatType>(*this);
    return clone; 
}