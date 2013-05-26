#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"

template <class FloatType, class IntType>
class FeatureEqualI
{
public:
    virtual bool IsEqual(const MatrixBufferTemplate<FloatType>& floatParams,
                        const MatrixBufferTemplate<IntType>& intParams,
                        const int featureIndex,
                        const MatrixBufferTemplate<FloatType>& otherFloatParams,
                        const MatrixBufferTemplate<IntType>& otherIntParams,
                        const int otherFeatureIndex) const = 0;

    virtual FeatureEqualI<FloatType, IntType>* Clone() const = 0;
    virtual ~FeatureEqualI() {}
};