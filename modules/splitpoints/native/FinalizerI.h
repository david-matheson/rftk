#pragma once

#include "VectorBuffer.h"
#include "BufferTypes.h"

// ----------------------------------------------------------------------------
//
// Finalize estimator parameters (ie sufficient statistics) before writing them 
// to the tree
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class FinalizerI
{
public:
    virtual VectorBufferTemplate<typename BufferTypes::TreeEstimator> Finalize(
    						const typename BufferTypes::DatapointCounts count, 
    						const VectorBufferTemplate<typename BufferTypes::SufficientStatsContinuous>& estimator) const=0;
    virtual FinalizerI<BufferTypes>* Clone() const=0;
    virtual ~FinalizerI() {}
};