#pragma once

#include "asserts.h"
#include "VectorBuffer.h"
#include "FinalizerI.h"

// ----------------------------------------------------------------------------
//
// Normalize the class histograms to probabilities
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class ClassEstimatorFinalizer: public FinalizerI<BufferTypes>
{
public:
    virtual VectorBufferTemplate<typename BufferTypes::TreeEstimator> Finalize(
    						const typename BufferTypes::DatapointCounts count, 
    						const VectorBufferTemplate<typename BufferTypes::SufficientStatsContinuous>& estimator) const;
    virtual FinalizerI<BufferTypes>* Clone() const;
    virtual ~ClassEstimatorFinalizer() {}
};

template <class BufferTypes>
VectorBufferTemplate<typename BufferTypes::TreeEstimator> ClassEstimatorFinalizer<BufferTypes>::Finalize(
    						const typename BufferTypes::DatapointCounts count, 
    						const VectorBufferTemplate<typename BufferTypes::SufficientStatsContinuous>& estimator) const
{
    UNUSED_PARAM(count);
    VectorBufferTemplate<typename BufferTypes::TreeEstimator> result = 
    		ConvertVectorBufferTemplate<typename BufferTypes::SufficientStatsContinuous, typename BufferTypes::TreeEstimator>(estimator);
    result.Normalize();
    return result;

}

template <class BufferTypes>
FinalizerI<BufferTypes>* ClassEstimatorFinalizer<BufferTypes>::Clone() const
{
    ClassEstimatorFinalizer<BufferTypes>* clone = new ClassEstimatorFinalizer<BufferTypes>(*this);
    return clone; 
}