#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"

template <class BufferTypes>
class FeatureEqualI
{
public:
    virtual bool IsEqual(const MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                        const MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams,
                        const int featureIndex,
                        const MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& otherFloatParams,
                        const MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& otherIntParams,
                        const int otherFeatureIndex) const = 0;

    virtual FeatureEqualI<BufferTypes>* Clone() const = 0;
    virtual ~FeatureEqualI() {}
};