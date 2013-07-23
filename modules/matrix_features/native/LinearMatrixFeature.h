#pragma once

#include "asserts.h"
#include "BufferTypes.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "SparseMatrixBuffer.h"
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
template <class BufferTypes, class DataMatrixType>
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

    LinearMatrixFeatureBinding<BufferTypes, DataMatrixType> Bind(const BufferCollectionStack& readCollection) const;


    typedef typename BufferTypes::FeatureValue Float;
    typedef typename BufferTypes::Index Int;
    typedef LinearMatrixFeatureBinding<BufferTypes, DataMatrixType> FeatureBinding;

    const BufferId mFloatParamsBufferId;
    const BufferId mIntParamsBufferId;
    const BufferId mIndicesBufferId;
    const BufferId mDataMatrixBufferId;
};

template <class BufferTypes, class DataMatrixType>
LinearMatrixFeature<BufferTypes, DataMatrixType>::LinearMatrixFeature( const BufferId& floatParamsBufferId,
                                                              const BufferId& intParamsBufferId,
                                                              const BufferId& indicesBufferId,
                                                              const BufferId& matrixDataBufferId )
: mFloatParamsBufferId(floatParamsBufferId)
, mIntParamsBufferId(intParamsBufferId)
, mIndicesBufferId(indicesBufferId)
, mDataMatrixBufferId(matrixDataBufferId)
{}

template <class BufferTypes, class DataMatrixType>
LinearMatrixFeature<BufferTypes, DataMatrixType>::LinearMatrixFeature( const BufferId& indicesBufferId,
                                                              const BufferId& matrixDataBufferId )
: mFloatParamsBufferId(GetBufferId("floatParams"))
, mIntParamsBufferId(GetBufferId("intParams"))
, mIndicesBufferId(indicesBufferId)
, mDataMatrixBufferId(matrixDataBufferId)
{}

template <class BufferTypes, class DataMatrixType>
LinearMatrixFeature<BufferTypes, DataMatrixType>::~LinearMatrixFeature()
{}

template <class BufferTypes, class DataMatrixType>
LinearMatrixFeatureBinding<BufferTypes, DataMatrixType> LinearMatrixFeature<BufferTypes, DataMatrixType>::Bind(const BufferCollectionStack& readCollection) const
{
    MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> const* floatParams = 
            readCollection.GetBufferPtr< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(mFloatParamsBufferId);
    MatrixBufferTemplate<typename BufferTypes::ParamsInteger> const* intParams = 
            readCollection.GetBufferPtr< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(mIntParamsBufferId);
    VectorBufferTemplate<typename BufferTypes::Index> const* indices = 
            readCollection.GetBufferPtr< VectorBufferTemplate<typename BufferTypes::Index> >(mIndicesBufferId);
    DataMatrixType const* dataMatrix = readCollection.GetBufferPtr< DataMatrixType >(mDataMatrixBufferId);

    ASSERT_ARG_DIM_1D(floatParams->GetN(), intParams->GetN());

    return LinearMatrixFeatureBinding<BufferTypes, DataMatrixType>(floatParams, intParams, indices, dataMatrix);
}

// template <class BufferTypes>
// class LinearDenseMatrixFeature: public LinearMatrixFeature<BufferTypes, MatrixBufferTemplate<typename BufferTypes::SourceContinuous> >
// {};

// template <class BufferTypes>
// class LinearSparseMatrixFeature: public LinearMatrixFeature<BufferTypes, SparseMatrixBufferTemplate<typename BufferTypes::SourceContinuous> >
// {};