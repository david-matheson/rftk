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
template <class BufferTypes>
class ScaledDepthDeltaFeatureBinding
{
public:
    ScaledDepthDeltaFeatureBinding( MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> const* floatParams,
                                 MatrixBufferTemplate<typename BufferTypes::ParamsInteger> const* intParams,
                                 VectorBufferTemplate<typename BufferTypes::Index> const* indices,
                                 MatrixBufferTemplate<typename BufferTypes::Index> const* pixelIndices,
                                 Tensor3BufferTemplate<typename BufferTypes::SourceContinuous> const* depthImgs,
                                 MatrixBufferTemplate<typename BufferTypes::SourceContinuous> const* scales);
    ScaledDepthDeltaFeatureBinding();
    ~ScaledDepthDeltaFeatureBinding();

    ScaledDepthDeltaFeatureBinding(const ScaledDepthDeltaFeatureBinding& other);
    ScaledDepthDeltaFeatureBinding & operator=(const ScaledDepthDeltaFeatureBinding & other);

    typename BufferTypes::FeatureValue FeatureValue( const int featureIndex, const int relativeSampleIndex) const;

    typename BufferTypes::Index GetNumberOfFeatures() const;
    typename BufferTypes::Index GetNumberOfDatapoints() const;

private:
    MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> const* mFloatParams;
    MatrixBufferTemplate<typename BufferTypes::ParamsInteger> const* mIntParams;
    VectorBufferTemplate<typename BufferTypes::Index> const* mIndices;
    MatrixBufferTemplate<typename BufferTypes::Index> const* mPixelIndices;
    Tensor3BufferTemplate<typename BufferTypes::SourceContinuous> const* mDepthImgs;
    MatrixBufferTemplate<typename BufferTypes::SourceContinuous> const* mScales;
};

template <class BufferTypes>
ScaledDepthDeltaFeatureBinding<BufferTypes>::ScaledDepthDeltaFeatureBinding( MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> const* floatParams,
                                                                                   MatrixBufferTemplate<typename BufferTypes::ParamsInteger> const* intParams,
                                                                                   VectorBufferTemplate<typename BufferTypes::Index> const* indices,
                                                                                   MatrixBufferTemplate<typename BufferTypes::Index> const* pixelIndices,
                                                                                   Tensor3BufferTemplate<typename BufferTypes::SourceContinuous> const* depthImgs,
                                                                                   MatrixBufferTemplate<typename BufferTypes::SourceContinuous> const* scales )
: mFloatParams(floatParams)
, mIntParams(intParams)
, mIndices(indices)
, mPixelIndices(pixelIndices)
, mDepthImgs(depthImgs)
, mScales(scales)
{}

template <class BufferTypes>
ScaledDepthDeltaFeatureBinding<BufferTypes>::ScaledDepthDeltaFeatureBinding()
: mFloatParams(NULL)
, mIntParams(NULL)
, mIndices(NULL)
, mPixelIndices(NULL)
, mDepthImgs(NULL)
, mScales(NULL)
{}

template <class BufferTypes>
ScaledDepthDeltaFeatureBinding<BufferTypes>::ScaledDepthDeltaFeatureBinding( const ScaledDepthDeltaFeatureBinding& other )
: mFloatParams(other.mFloatParams)
, mIntParams(other.mIntParams)
, mIndices(other.mIndices)
, mPixelIndices(other.mPixelIndices)
, mDepthImgs(other.mDepthImgs)
, mScales(other.mScales)
{}

template <class BufferTypes>
ScaledDepthDeltaFeatureBinding<BufferTypes>& ScaledDepthDeltaFeatureBinding<BufferTypes>::operator=(const ScaledDepthDeltaFeatureBinding & other)
{
    mFloatParams = other.mFloatParams;
    mIntParams = other.mIntParams;
    mIndices = other.mIndices;
    mPixelIndices = other.mPixelIndices;
    mDepthImgs = other.mDepthImgs;
    mScales = other.mScales;
    return *this;
}

template <class BufferTypes>
ScaledDepthDeltaFeatureBinding<BufferTypes>::~ScaledDepthDeltaFeatureBinding()
{}


template <class BufferTypes>
typename BufferTypes::FeatureValue ScaledDepthDeltaFeatureBinding<BufferTypes>::FeatureValue( const int featureIndex, const int relativeSampleIndex) const
{
    const typename BufferTypes::Index index = mIndices->Get(relativeSampleIndex);

    const typename BufferTypes::Index imgIndex = mPixelIndices->Get(index, 0);
    const typename BufferTypes::Index pixelM = mPixelIndices->Get(index, 1);
    const typename BufferTypes::Index pixelN = mPixelIndices->Get(index, 2);

    const typename BufferTypes::SourceContinuous scaleM = 
        (mScales != NULL) ? mScales->Get(index, 0) : typename BufferTypes::SourceContinuous(1.0);
    const typename BufferTypes::SourceContinuous scaleN = 
        (mScales != NULL) ? mScales->Get(index, 1) : typename BufferTypes::SourceContinuous(1.0);

    const typename BufferTypes::ParamsContinuous um = mFloatParams->Get(featureIndex,FEATURE_SPECIFIC_PARAMS_START);
    const typename BufferTypes::ParamsContinuous un = mFloatParams->Get(featureIndex,FEATURE_SPECIFIC_PARAMS_START+1);
    const typename BufferTypes::ParamsContinuous vm = mFloatParams->Get(featureIndex,FEATURE_SPECIFIC_PARAMS_START+2);
    const typename BufferTypes::ParamsContinuous vn = mFloatParams->Get(featureIndex,FEATURE_SPECIFIC_PARAMS_START+3);

    const typename BufferTypes::FeatureValue featureValue = PixelDepthDelta<BufferTypes>(*mDepthImgs, imgIndex, pixelM, pixelN, um*scaleM, un*scaleN, vm*scaleM, vn*scaleN);
    return featureValue;
}

template <class BufferTypes>
typename BufferTypes::Index ScaledDepthDeltaFeatureBinding<BufferTypes>::GetNumberOfFeatures() const
{
    return mIntParams->GetM();
}

template <class BufferTypes>
typename BufferTypes::Index ScaledDepthDeltaFeatureBinding<BufferTypes>::GetNumberOfDatapoints() const
{
    return mIndices->GetN();
}
