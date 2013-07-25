#pragma once

#include "BufferTypes.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "FeatureEqualI.h"

// Determines if two features are equal if their int params are equal and if
// their float params x floatPrecision are equal

template <class BufferTypes>
class FeatureEqualQuantized: public FeatureEqualI<BufferTypes>
{
public:
    FeatureEqualQuantized(typename BufferTypes::ParamsContinuous floatPrecision); 
    virtual ~FeatureEqualQuantized();
    virtual FeatureEqualI<BufferTypes>* Clone() const;

    virtual bool IsEqual(const MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                        const MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams,
                        const int featureIndex,
                        const MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& otherFloatParams,
                        const MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& otherIntParams,
                        const int otherFeatureIndex) const;
private:
    typename BufferTypes::ParamsContinuous mFloatPrecision;
};

template <class BufferTypes>
FeatureEqualQuantized<BufferTypes>::FeatureEqualQuantized(typename BufferTypes::ParamsContinuous floatPrecision)
: mFloatPrecision(floatPrecision)
{}

template <class BufferTypes>
FeatureEqualQuantized<BufferTypes>::~FeatureEqualQuantized()
{}

template <class BufferTypes>
FeatureEqualI<BufferTypes>* FeatureEqualQuantized<BufferTypes>::Clone() const
{
    FeatureEqualI<BufferTypes>* clone = new FeatureEqualQuantized<BufferTypes>(*this);
    return clone;
}

template <class BufferTypes>
bool FeatureEqualQuantized<BufferTypes>::IsEqual(const MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                        const MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams,
                        const int featureIndex,
                        const MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& otherFloatParams,
                        const MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& otherIntParams,
                        const int otherFeatureIndex) const
{
    const bool sameDimension = ((floatParams.GetN() == otherFloatParams.GetN())
                                 && (intParams.GetN() == intParams.GetN()));
    if(!sameDimension)
    {
        return false;
    }

    for(int i=0; i<floatParams.GetN(); i++)
    {
        const typename BufferTypes::ParamsInteger intFeature = intParams.Get(featureIndex, i);
        const typename BufferTypes::ParamsInteger otherIntFeature = otherIntParams.Get(otherFeatureIndex, i);
        const bool intEqual = (intFeature == otherIntFeature);

        const typename BufferTypes::ParamsInteger floatFeature = typename BufferTypes::ParamsInteger(mFloatPrecision*floatParams.Get(featureIndex, i));
        const typename BufferTypes::ParamsInteger otherFloatFeature = typename BufferTypes::ParamsInteger(mFloatPrecision*otherFloatParams.Get(otherFeatureIndex, i));
        const bool floatEqual = (floatFeature == otherFloatFeature);

        if(!intEqual || !floatEqual)
        {
            return false;
        }
    }

    return true;
}