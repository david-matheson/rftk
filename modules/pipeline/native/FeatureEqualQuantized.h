#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "FeatureEqualI.h"

// Determines if two features are equal if their int params are equal and if
// their float params x floatPrecision are equal

template <class FloatType, class IntType>
class FeatureEqualQuantized: public FeatureEqualI<FloatType, IntType>
{
public:
    FeatureEqualQuantized(FloatType floatPrecision); 
    virtual ~FeatureEqualQuantized();
    virtual FeatureEqualI<FloatType, IntType>* Clone() const;

    virtual bool IsEqual(const MatrixBufferTemplate<FloatType>& floatParams,
                        const MatrixBufferTemplate<IntType>& intParams,
                        const int featureIndex,
                        const MatrixBufferTemplate<FloatType>& otherFloatParams,
                        const MatrixBufferTemplate<IntType>& otherIntParams,
                        const int otherFeatureIndex) const;
private:
    FloatType mFloatPrecision;
};

template <class FloatType, class IntType>
FeatureEqualQuantized<FloatType, IntType>::FeatureEqualQuantized(FloatType floatPrecision)
: mFloatPrecision(floatPrecision)
{}

template <class FloatType, class IntType>
FeatureEqualQuantized<FloatType, IntType>::~FeatureEqualQuantized()
{}

template <class FloatType, class IntType>
FeatureEqualI<FloatType, IntType>* FeatureEqualQuantized<FloatType, IntType>::Clone() const
{
    FeatureEqualI<FloatType, IntType>* clone = new FeatureEqualQuantized<FloatType, IntType>(*this);
    return clone;
}

#include <stdio.h>

template <class FloatType, class IntType>
bool FeatureEqualQuantized<FloatType, IntType>::IsEqual(const MatrixBufferTemplate<FloatType>& floatParams,
                        const MatrixBufferTemplate<IntType>& intParams,
                        const int featureIndex,
                        const MatrixBufferTemplate<FloatType>& otherFloatParams,
                        const MatrixBufferTemplate<IntType>& otherIntParams,
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
        const IntType intFeature = intParams.Get(featureIndex, i);
        const IntType otherIntFeature = otherIntParams.Get(otherFeatureIndex, i);
        const bool intEqual = intFeature == otherIntFeature;

        const IntType floatFeature = IntType(mFloatPrecision*floatParams.Get(featureIndex, i));
        const IntType otherFloatFeature = IntType(mFloatPrecision*otherFloatParams.Get(otherFeatureIndex, i));
        const bool floatEqual = floatFeature == otherFloatFeature;

        if(!intEqual || !floatEqual)
        {
            return false;
        }
    }

    return true;
}