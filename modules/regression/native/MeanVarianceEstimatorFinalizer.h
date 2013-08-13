#pragma once

#include "asserts.h"
#include "VectorBuffer.h"
#include "FinalizerI.h"

// ----------------------------------------------------------------------------
//
// Compute the mean and variance from the sufficient statistics
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class MeanVarianceEstimatorFinalizer: public FinalizerI<BufferTypes>
{
public:
    virtual VectorBufferTemplate<typename BufferTypes::TreeEstimator> Finalize(
                            const typename BufferTypes::DatapointCounts count, 
                            const VectorBufferTemplate<typename BufferTypes::SufficientStatsContinuous>& estimator) const;
    virtual FinalizerI<BufferTypes>* Clone() const;
    virtual ~MeanVarianceEstimatorFinalizer() {}
};

template <class BufferTypes>
VectorBufferTemplate<typename BufferTypes::TreeEstimator> MeanVarianceEstimatorFinalizer<BufferTypes>::Finalize(
                            const typename BufferTypes::DatapointCounts count, 
                            const VectorBufferTemplate<typename BufferTypes::SufficientStatsContinuous>& estimator) const
{
    UNUSED_PARAM(count)
    // Switched to online method that maintains the mean
    // for(int d=0; d<estimator.GetN()/2; d++)
    // {
    //     FloatType value = estimator.Get(d);
    //     estimator.Set(d, value/count);
    // }
    VectorBufferTemplate<typename BufferTypes::TreeEstimator> result = 
            ConvertVectorBufferTemplate<typename BufferTypes::SufficientStatsContinuous, typename BufferTypes::TreeEstimator>(estimator);
    return result;
}

template <class BufferTypes>
FinalizerI<BufferTypes>* MeanVarianceEstimatorFinalizer<BufferTypes>::Clone() const
{
    MeanVarianceEstimatorFinalizer<BufferTypes>* clone = new MeanVarianceEstimatorFinalizer<BufferTypes>(*this);
    return clone;
}