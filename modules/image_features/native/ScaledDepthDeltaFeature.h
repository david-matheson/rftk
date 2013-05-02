#pragma once

#include "asserts.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Tensor3Buffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "UniqueBufferId.h"
#include "ImageUtils.h"
#include "ScaledDepthDeltaFeatureBinding.h"

// ----------------------------------------------------------------------------
//
// ScaledDepthDeltaFeature is the depth delta of a pair of pixels
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class ScaledDepthDeltaFeature
{
public:
    ScaledDepthDeltaFeature( const BufferId& floatParamsBufferId,
                            const BufferId& intParamsBufferId,
                            const BufferId& indicesBufferId,
                            const BufferId& pixelIndicesBufferId,
                            const BufferId& scalesBufferId,
                            const BufferId& depthsDataBufferId );

    ScaledDepthDeltaFeature( const BufferId& floatParamsBufferId,
                            const BufferId& intParamsBufferId,
                            const BufferId& indicesBufferId,
                            const BufferId& pixelIndicesBufferId,
                            const BufferId& depthsDataBufferId );

    ScaledDepthDeltaFeature( const BufferId& indicesBufferId,
                            const BufferId& pixelIndicesBufferId,
                            const BufferId& depthsDataBufferId  );

    ~ScaledDepthDeltaFeature();

    ScaledDepthDeltaFeatureBinding<FloatType, IntType> Bind(const BufferCollectionStack& readCollection) const;


    typedef FloatType Float;
    typedef IntType Int;
    typedef ScaledDepthDeltaFeatureBinding<FloatType, IntType> FeatureBinding;

    const BufferId mFloatParamsBufferId;
    const BufferId mIntParamsBufferId;
    const BufferId mIndicesBufferId;
    const BufferId mPixelIndicesBufferId;
    const BufferId mScalesBufferId;
    const BufferId mDepthsImgsBufferId;
};

template <class FloatType, class IntType>
ScaledDepthDeltaFeature<FloatType, IntType>::ScaledDepthDeltaFeature( const BufferId& floatParamsBufferId,
                                                                      const BufferId& intParamsBufferId,
                                                                      const BufferId& indicesBufferId,
                                                                      const BufferId& pixelIndicesBufferId,
                                                                      const BufferId& depthsDataBufferId,
                                                                      const BufferId& scalesBufferId )
: mFloatParamsBufferId(floatParamsBufferId)
, mIntParamsBufferId(intParamsBufferId)
, mIndicesBufferId(indicesBufferId)
, mPixelIndicesBufferId(pixelIndicesBufferId)
, mScalesBufferId(scalesBufferId)
, mDepthsImgsBufferId(depthsDataBufferId)
{}

template <class FloatType, class IntType>
ScaledDepthDeltaFeature<FloatType, IntType>::ScaledDepthDeltaFeature( const BufferId& floatParamsBufferId,
                                                                      const BufferId& intParamsBufferId,
                                                                      const BufferId& indicesBufferId,
                                                                      const BufferId& pixelIndicesBufferId,
                                                                      const BufferId& depthsDataBufferId )
: mFloatParamsBufferId(floatParamsBufferId)
, mIntParamsBufferId(intParamsBufferId)
, mIndicesBufferId(indicesBufferId)
, mPixelIndicesBufferId(pixelIndicesBufferId)
, mScalesBufferId(NullKey)
, mDepthsImgsBufferId(depthsDataBufferId)
{}


template <class FloatType, class IntType>
ScaledDepthDeltaFeature<FloatType, IntType>::ScaledDepthDeltaFeature( const BufferId& indicesBufferId,
                                                                      const BufferId& pixelIndicesBufferId,
                                                                      const BufferId& depthsDataBufferId )
: mFloatParamsBufferId(GetBufferId("floatParams"))
, mIntParamsBufferId(GetBufferId("intParams"))
, mIndicesBufferId(indicesBufferId)
, mPixelIndicesBufferId(pixelIndicesBufferId)
, mScalesBufferId(NullKey)
, mDepthsImgsBufferId(depthsDataBufferId)
{}

template <class FloatType, class IntType>
ScaledDepthDeltaFeature<FloatType, IntType>::~ScaledDepthDeltaFeature()
{}

template <class FloatType, class IntType>
ScaledDepthDeltaFeatureBinding<FloatType, IntType> ScaledDepthDeltaFeature<FloatType, IntType>::Bind(const BufferCollectionStack& readCollection) const
{
    MatrixBufferTemplate<FloatType> const* floatParams = readCollection.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mFloatParamsBufferId);
    MatrixBufferTemplate<IntType> const* intParams = readCollection.GetBufferPtr< MatrixBufferTemplate<IntType> >(mIntParamsBufferId);
    VectorBufferTemplate<IntType> const* indices = readCollection.GetBufferPtr< VectorBufferTemplate<IntType> >(mIndicesBufferId);
    MatrixBufferTemplate<IntType> const* pixelIndices = readCollection.GetBufferPtr< MatrixBufferTemplate<IntType> >(mPixelIndicesBufferId);
    Tensor3BufferTemplate<FloatType> const* depthImgs = readCollection.GetBufferPtr< Tensor3BufferTemplate<FloatType> >(mDepthsImgsBufferId);
    
    MatrixBufferTemplate<FloatType> const* scales = NULL;
    if( mScalesBufferId != NullKey )
    {
        scales = readCollection.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mScalesBufferId);
    }

    ASSERT_ARG_DIM_1D(floatParams->GetN(), intParams->GetN());

    return ScaledDepthDeltaFeatureBinding<FloatType, IntType>(floatParams, intParams, indices, pixelIndices, depthImgs, scales);
}

