#pragma once

#include "VectorBuffer.h"

// ----------------------------------------------------------------------------
//
// Finalize estimator parameters (ie sufficient statistics) before writing them 
// to the tree
//
// ----------------------------------------------------------------------------
template <class FloatType>
class FinalizerI
{
public:
    virtual void Finalize(FloatType count, VectorBufferTemplate<FloatType>& estimator) const=0;
    virtual FinalizerI<FloatType>* Clone() const=0;
};