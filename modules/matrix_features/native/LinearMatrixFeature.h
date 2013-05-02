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
    LinearMatrixFeature( const BufferId& floatParamsBufferId,
                        const BufferId& intParamsBufferId,
                        const BufferId& indicesBufferId,
                        const BufferId& matrixDataBufferId );

    LinearMatrixFeature( const BufferId& indicesBufferId,
                        const BufferId& matrixDataBufferId );

    ~LinearMatrixFeature();

    LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType> Bind(const BufferCollectionStack& readCollection) const;


    typedef FloatType Float;
    typedef IntType Int;
    typedef LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType> FeatureBinding;

    const BufferId mFloatParamsBufferId;
    const BufferId mIntParamsBufferId;
    const BufferId mIndicesBufferId;
    const BufferId mDataMatrixBufferId;
};

template <class DataMatrixType, class FloatType, class IntType>
LinearMatrixFeature<DataMatrixType, FloatType, IntType>::LinearMatrixFeature( const BufferId& floatParamsBufferId,
                                                              const BufferId& intParamsBufferId,
                                                              const BufferId& indicesBufferId,
                                                              const BufferId& matrixDataBufferId )
: mFloatParamsBufferId(floatParamsBufferId)
, mIntParamsBufferId(intParamsBufferId)
, mIndicesBufferId(indicesBufferId)
, mDataMatrixBufferId(matrixDataBufferId)
{}

template <class DataMatrixType, class FloatType, class IntType>
LinearMatrixFeature<DataMatrixType, FloatType, IntType>::LinearMatrixFeature( const BufferId& indicesBufferId,
                                                              const BufferId& matrixDataBufferId )
: mFloatParamsBufferId(GetBufferId("floatParams"))
, mIntParamsBufferId(GetBufferId("intParams"))
, mIndicesBufferId(indicesBufferId)
, mDataMatrixBufferId(matrixDataBufferId)
{}

template <class DataMatrixType, class FloatType, class IntType>
LinearMatrixFeature<DataMatrixType, FloatType, IntType>::~LinearMatrixFeature()
{}

template <class DataMatrixType, class FloatType, class IntType>
LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType> LinearMatrixFeature<DataMatrixType, FloatType, IntType>::Bind(const BufferCollectionStack& readCollection) const
{
    MatrixBufferTemplate<FloatType> const* floatParams = readCollection.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mFloatParamsBufferId);
    MatrixBufferTemplate<IntType> const* intParams = readCollection.GetBufferPtr< MatrixBufferTemplate<IntType> >(mIntParamsBufferId);
    VectorBufferTemplate<IntType> const* indices = readCollection.GetBufferPtr< VectorBufferTemplate<IntType> >(mIndicesBufferId);
    DataMatrixType const* dataMatrix = readCollection.GetBufferPtr< DataMatrixType >(mDataMatrixBufferId);

    ASSERT_ARG_DIM_1D(floatParams->GetN(), intParams->GetN());

    return LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType>(floatParams, intParams, indices, dataMatrix);
}

