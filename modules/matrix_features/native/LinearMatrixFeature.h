#pragma once

#include "asserts.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "UniqueBufferId.h"
#include "LinearMatrixFeatureBinding.h"

enum
{
    MATRIX_FEATURES = 10 // Move this to a more soucefile
};

// ----------------------------------------------------------------------------
//
// LinearMatrixFeature is a linear combination of dimensions (columns) for
// each sample (row).
//
// ----------------------------------------------------------------------------
template <class DataMatrixType, class FloatType, class IntType>
class LinearMatrixFeature
{
public:
    LinearMatrixFeature( const UniqueBufferId::BufferId floatParamsBufferId,
                    const UniqueBufferId::BufferId intParamsBufferId,
                    const UniqueBufferId::BufferId indicesBufferId,
                    const UniqueBufferId::BufferId matrixDataBufferId );

    LinearMatrixFeature( const UniqueBufferId::BufferId indicesBufferId,
                        const UniqueBufferId::BufferId matrixDataBufferId );

    ~LinearMatrixFeature();

    LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType> Bind(const BufferCollectionStack& readCollection) const;


    typedef FloatType Float;
    typedef IntType Int;
    typedef LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType> FeatureBinding;

    const UniqueBufferId::BufferId mFloatParamsBufferId;
    const UniqueBufferId::BufferId mIntParamsBufferId;
    const UniqueBufferId::BufferId mIndicesBufferId;
    const UniqueBufferId::BufferId mDataMatrixBufferId;
};

template <class DataMatrixType, class FloatType, class IntType>
LinearMatrixFeature<DataMatrixType, FloatType, IntType>::LinearMatrixFeature( const UniqueBufferId::BufferId floatParamsBufferId,
                                                              const UniqueBufferId::BufferId intParamsBufferId,
                                                              const UniqueBufferId::BufferId indicesBufferId,
                                                              const UniqueBufferId::BufferId matrixDataBufferId )
: mFloatParamsBufferId(floatParamsBufferId)
, mIntParamsBufferId(intParamsBufferId)
, mIndicesBufferId(indicesBufferId)
, mDataMatrixBufferId(matrixDataBufferId)
{}

template <class DataMatrixType, class FloatType, class IntType>
LinearMatrixFeature<DataMatrixType, FloatType, IntType>::LinearMatrixFeature( const UniqueBufferId::BufferId indicesBufferId,
                                                              const UniqueBufferId::BufferId matrixDataBufferId )
: mFloatParamsBufferId(UniqueBufferId::GetBufferId("floatParams"))
, mIntParamsBufferId(UniqueBufferId::GetBufferId("intParams"))
, mIndicesBufferId(indicesBufferId)
, mDataMatrixBufferId(matrixDataBufferId)
{}

template <class DataMatrixType, class FloatType, class IntType>
LinearMatrixFeature<DataMatrixType, FloatType, IntType>::~LinearMatrixFeature()
{}

template <class DataMatrixType, class FloatType, class IntType>
LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType> LinearMatrixFeature<DataMatrixType, FloatType, IntType>::Bind(const BufferCollectionStack& readCollection) const
{
    ASSERT(readCollection.HasBuffer< MatrixBufferTemplate<FloatType> >(mFloatParamsBufferId));
    ASSERT(readCollection.HasBuffer< MatrixBufferTemplate<IntType> >(mIntParamsBufferId));
    ASSERT(readCollection.HasBuffer< VectorBufferTemplate<IntType> >(mIndicesBufferId));
    ASSERT(readCollection.HasBuffer< DataMatrixType >(mDataMatrixBufferId));

    MatrixBufferTemplate<FloatType> const* floatParams = readCollection.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mFloatParamsBufferId);
    MatrixBufferTemplate<IntType> const* intParams = readCollection.GetBufferPtr< MatrixBufferTemplate<IntType> >(mIntParamsBufferId);
    VectorBufferTemplate<IntType> const* indices = readCollection.GetBufferPtr< VectorBufferTemplate<IntType> >(mIndicesBufferId);
    DataMatrixType const* dataMatrix = readCollection.GetBufferPtr< DataMatrixType >(mDataMatrixBufferId);

    ASSERT_ARG_DIM_1D(floatParams->GetN(), intParams->GetN());

    return LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType>(floatParams, intParams, indices, dataMatrix);
}

