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
template <class BufferTypes>
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

    ScaledDepthDeltaFeatureBinding<BufferTypes> Bind(const BufferCollectionStack& readCollection) const;


    typedef typename BufferTypes::FeatureValue Float;
    typedef typename BufferTypes::Index Int;
    typedef ScaledDepthDeltaFeatureBinding<BufferTypes> FeatureBinding;

    const BufferId mFloatParamsBufferId;
    const BufferId mIntParamsBufferId;
    const BufferId mIndicesBufferId;
    const BufferId mPixelIndicesBufferId;
    const BufferId mScalesBufferId;
    const BufferId mDepthsImgsBufferId;
};

template <class BufferTypes>
ScaledDepthDeltaFeature<BufferTypes>::ScaledDepthDeltaFeature( const BufferId& floatParamsBufferId,
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

template <class BufferTypes>
ScaledDepthDeltaFeature<BufferTypes>::ScaledDepthDeltaFeature( const BufferId& floatParamsBufferId,
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


template <class BufferTypes>
ScaledDepthDeltaFeature<BufferTypes>::ScaledDepthDeltaFeature( const BufferId& indicesBufferId,
                                                                      const BufferId& pixelIndicesBufferId,
                                                                      const BufferId& depthsDataBufferId )
: mFloatParamsBufferId(GetBufferId("floatParams"))
, mIntParamsBufferId(GetBufferId("intParams"))
, mIndicesBufferId(indicesBufferId)
, mPixelIndicesBufferId(pixelIndicesBufferId)
, mScalesBufferId(NullKey)
, mDepthsImgsBufferId(depthsDataBufferId)
{}

template <class BufferTypes>
ScaledDepthDeltaFeature<BufferTypes>::~ScaledDepthDeltaFeature()
{}

template <class BufferTypes>
ScaledDepthDeltaFeatureBinding<BufferTypes> ScaledDepthDeltaFeature<BufferTypes>::Bind(const BufferCollectionStack& readCollection) const
{
    MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> const* floatParams = 
        readCollection.GetBufferPtr< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mFloatParamsBufferId);
    MatrixBufferTemplate<typename BufferTypes::ParamsInteger> const* intParams = 
        readCollection.GetBufferPtr< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mIntParamsBufferId);
    VectorBufferTemplate<typename BufferTypes::Index> const* indices = 
        readCollection.GetBufferPtr< VectorBufferTemplate<typename BufferTypes::Index> >(mIndicesBufferId);
    MatrixBufferTemplate<typename BufferTypes::Index> const* pixelIndices = 
        readCollection.GetBufferPtr< MatrixBufferTemplate<typename BufferTypes::Index> >(mPixelIndicesBufferId);
    Tensor3BufferTemplate<typename BufferTypes::SourceContinuous> const* depthImgs = 
        readCollection.GetBufferPtr< Tensor3BufferTemplate<typename BufferTypes::SourceContinuous> >(mDepthsImgsBufferId);
    
    MatrixBufferTemplate<typename BufferTypes::SourceContinuous> const* scales = NULL;
    if( mScalesBufferId != NullKey )
    {
        scales = readCollection.GetBufferPtr< MatrixBufferTemplate<typename BufferTypes::SourceContinuous> >(mScalesBufferId);
    }

    ASSERT_ARG_DIM_1D(floatParams->GetN(), intParams->GetN());

    return ScaledDepthDeltaFeatureBinding<BufferTypes>(floatParams, intParams, indices, pixelIndices, depthImgs, scales);
}

