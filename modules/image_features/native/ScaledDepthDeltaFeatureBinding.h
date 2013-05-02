#pragma once

#include "asserts.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "Constants.h"
#include "ImageUtils.h"

// ----------------------------------------------------------------------------
//
// ScaledDepthDeltaFeature is the depth delta of a pair of pixels specified by
// offsets
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class ScaledDepthDeltaFeatureBinding
{
public:
    ScaledDepthDeltaFeatureBinding( MatrixBufferTemplate<FloatType> const* floatParams,
                                 MatrixBufferTemplate<IntType> const* intParams,
                                 VectorBufferTemplate<IntType> const* indices,
                                 MatrixBufferTemplate<IntType> const* pixelIndices,
                                 Tensor3BufferTemplate<FloatType> const* depthImgs,
                                 MatrixBufferTemplate<FloatType> const* scales);
    ScaledDepthDeltaFeatureBinding();
    ~ScaledDepthDeltaFeatureBinding();

    ScaledDepthDeltaFeatureBinding(const ScaledDepthDeltaFeatureBinding& other);
    ScaledDepthDeltaFeatureBinding & operator=(const ScaledDepthDeltaFeatureBinding & other);

    FloatType FeatureValue( const int featureIndex, const int relativeSampleIndex) const;

    IntType GetNumberOfFeatures() const;
    IntType GetNumberOfDatapoints() const;

private:
    MatrixBufferTemplate<FloatType> const* mFloatParams;
    MatrixBufferTemplate<IntType> const* mIntParams;
    VectorBufferTemplate<IntType> const* mIndices;
    MatrixBufferTemplate<IntType> const* mPixelIndices;
    Tensor3BufferTemplate<FloatType> const* mDepthImgs;
    MatrixBufferTemplate<FloatType> const* mScales;
};

template <class FloatType, class IntType>
ScaledDepthDeltaFeatureBinding<FloatType, IntType>::ScaledDepthDeltaFeatureBinding( MatrixBufferTemplate<FloatType> const* floatParams,
                                                                                   MatrixBufferTemplate<IntType> const* intParams,
                                                                                   VectorBufferTemplate<IntType> const* indices,
                                                                                   MatrixBufferTemplate<IntType> const* pixelIndices,
                                                                                   Tensor3BufferTemplate<FloatType> const* depthImgs,
                                                                                   MatrixBufferTemplate<FloatType> const* scales )
: mFloatParams(floatParams)
, mIntParams(intParams)
, mIndices(indices)
, mPixelIndices(pixelIndices)
, mDepthImgs(depthImgs)
, mScales(scales)
{}

template <class FloatType, class IntType>
ScaledDepthDeltaFeatureBinding<FloatType, IntType>::ScaledDepthDeltaFeatureBinding()
: mFloatParams(NULL)
, mIntParams(NULL)
, mIndices(NULL)
, mPixelIndices(NULL)
, mDepthImgs(NULL)
, mScales(NULL)
{}

template <class FloatType, class IntType>
ScaledDepthDeltaFeatureBinding<FloatType, IntType>::ScaledDepthDeltaFeatureBinding( const ScaledDepthDeltaFeatureBinding& other )
: mFloatParams(other.mFloatParams)
, mIntParams(other.mIntParams)
, mIndices(other.mIndices)
, mPixelIndices(other.mPixelIndices)
, mDepthImgs(other.mDepthImgs)
, mScales(other.mScales)
{}

template <class FloatType, class IntType>
ScaledDepthDeltaFeatureBinding<FloatType, IntType>& ScaledDepthDeltaFeatureBinding<FloatType, IntType>::operator=(const ScaledDepthDeltaFeatureBinding & other)
{
    mFloatParams = other.mFloatParams;
    mIntParams = other.mIntParams;
    mIndices = other.mIndices;
    mPixelIndices = other.mPxelIndices;
    mDepthImgs = other.mDepthImgs;
    mScales = other.mScales;
    return *this;
}

template <class FloatType, class IntType>
ScaledDepthDeltaFeatureBinding<FloatType, IntType>::~ScaledDepthDeltaFeatureBinding()
{}


template <class FloatType, class IntType>
FloatType ScaledDepthDeltaFeatureBinding<FloatType, IntType>::FeatureValue( const int featureIndex, const int relativeSampleIndex) const
{
    const IntType index = mIndices->Get(relativeSampleIndex);

    const IntType imgIndex = mPixelIndices->Get(index, 0);
    const IntType pixelM = mPixelIndices->Get(index, 1);
    const IntType pixelN = mPixelIndices->Get(index, 2);

    const FloatType scaleM = (mScales != NULL) ? mScales->Get(index, 0) : FloatType(1.0);
    const FloatType scaleN = (mScales != NULL) ? mScales->Get(index, 1) : FloatType(1.0);

    const FloatType um = mFloatParams->Get(featureIndex,FEATURE_SPECIFIC_PARAMS_START);
    const FloatType un = mFloatParams->Get(featureIndex,FEATURE_SPECIFIC_PARAMS_START+1);
    const FloatType vm = mFloatParams->Get(featureIndex,FEATURE_SPECIFIC_PARAMS_START+2);
    const FloatType vn = mFloatParams->Get(featureIndex,FEATURE_SPECIFIC_PARAMS_START+3);

    const FloatType featureValue = PixelDepthDelta<FloatType, IntType>(*mDepthImgs, imgIndex, pixelM, pixelN, um*scaleM, un*scaleN, vm*scaleM, vn*scaleN);
    return featureValue;
}

template <class FloatType, class IntType>
IntType ScaledDepthDeltaFeatureBinding<FloatType, IntType>::GetNumberOfFeatures() const
{
    return mIntParams->GetM();
}

template <class FloatType, class IntType>
IntType ScaledDepthDeltaFeatureBinding<FloatType, IntType>::GetNumberOfDatapoints() const
{
    return mIndices->GetN();
}
