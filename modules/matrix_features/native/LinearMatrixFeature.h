#pragma once

#include "asserts.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "UniqueBufferId.h"

enum
{
    MATRIX_FEATURES = 1 // Move this to a more soucefile
};

const int FEATURE_TYPE_INDEX = 0;     // Move this to a more soucefile
const int SPLIT_POINT_INDEX = FEATURE_TYPE_INDEX;  // Move this to a more soucefile
const int NUMBER_OF_DIMENSIONS = FEATURE_TYPE_INDEX + 1;
const int PARAM_START_INDEX = NUMBER_OF_DIMENSIONS + 1;  // Move this to a more soucefile

// ----------------------------------------------------------------------------
// 
// LinearMatrixFeature is a linear combination of dimensions (columns) for
// each sample (row).
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class LinearMatrixFeature
{
public:
    LinearMatrixFeature( const UniqueBufferId::BufferId floatParamsBufferId,
                    const UniqueBufferId::BufferId intParamsBufferId,
                    const UniqueBufferId::BufferId indicesBufferId,
                    const UniqueBufferId::BufferId matrixDataBufferId );
    ~LinearMatrixFeature();

    void Bind(const BufferCollectionStack& readCollection);

    FloatType FeatureValue( const int featureIndex, const int relativeSampleIndex) const;

    IntType GetNumberOfFeatures() const;
    IntType GetNumberOfDatapoints() const;

    typedef FloatType Float;
    typedef IntType Int;
    
private:
    const UniqueBufferId::BufferId mFloatParamsBufferId;
    const UniqueBufferId::BufferId mIntParamsBufferId;
    const UniqueBufferId::BufferId mIndicesBufferId;
    const UniqueBufferId::BufferId mDataMatrixBufferId;

    MatrixBufferTemplate<FloatType> const* mFloatParams;
    MatrixBufferTemplate<IntType> const* mIntParams;
    VectorBufferTemplate<IntType> const* mIndices;
    MatrixBufferTemplate<FloatType> const* mDataMatrix;

};

template <class FloatType, class IntType>
LinearMatrixFeature<FloatType, IntType>::LinearMatrixFeature( const UniqueBufferId::BufferId floatParamsBufferId,
                                                              const UniqueBufferId::BufferId intParamsBufferId,
                                                              const UniqueBufferId::BufferId indicesBufferId,
                                                              const UniqueBufferId::BufferId matrixDataBufferId )
: mFloatParamsBufferId(floatParamsBufferId)
, mIntParamsBufferId(intParamsBufferId)
, mIndicesBufferId(indicesBufferId)
, mDataMatrixBufferId(matrixDataBufferId)
, mFloatParams(NULL)
, mIntParams(NULL)
, mIndices(NULL)
, mDataMatrix(NULL)
{}

template <class FloatType, class IntType>
LinearMatrixFeature<FloatType, IntType>::~LinearMatrixFeature()
{}

template <class FloatType, class IntType>
void LinearMatrixFeature<FloatType, IntType>::Bind(const BufferCollectionStack& readCollection)
{
    ASSERT(readCollection.HasBuffer< MatrixBufferTemplate<FloatType> >(mFloatParamsBufferId));
    ASSERT(readCollection.HasBuffer< MatrixBufferTemplate<IntType> >(mIntParamsBufferId));
    ASSERT(readCollection.HasBuffer< VectorBufferTemplate<IntType> >(mIndicesBufferId));
    ASSERT(readCollection.HasBuffer< MatrixBufferTemplate<FloatType> >(mDataMatrixBufferId));

    mFloatParams = readCollection.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mFloatParamsBufferId);
    mIntParams = readCollection.GetBufferPtr< MatrixBufferTemplate<IntType> >(mIntParamsBufferId);
    mIndices = readCollection.GetBufferPtr< VectorBufferTemplate<IntType> >(mIndicesBufferId);
    mDataMatrix = readCollection.GetBufferPtr< MatrixBufferTemplate<FloatType> >(mDataMatrixBufferId);

    ASSERT_ARG_DIM_1D(mFloatParams->GetN(), mIntParams->GetN());
}

template <class FloatType, class IntType>
FloatType LinearMatrixFeature<FloatType, IntType>::FeatureValue( const int featureIndex, const int relativeSampleIndex) const
{
    FloatType featureValue = static_cast<FloatType>(0.0);
    const IntType numberOfDimensions = mIntParams->Get(featureIndex, NUMBER_OF_DIMENSIONS);
    for(int i=PARAM_START_INDEX; i<numberOfDimensions + PARAM_START_INDEX; i++)
    {
        const IntType dimension = mIntParams->Get(featureIndex, i);
        const IntType matrixIndex = mIndices->Get(relativeSampleIndex);
        featureValue += mFloatParams->Get(featureIndex, i) * mDataMatrix->Get(matrixIndex, dimension);
    }
    return featureValue;
}

template <class FloatType, class IntType>
IntType LinearMatrixFeature<FloatType, IntType>::GetNumberOfFeatures() const
{
    return (mIntParams != NULL) ? mIntParams->GetM() : 0;
}

template <class FloatType, class IntType>
IntType LinearMatrixFeature<FloatType, IntType>::GetNumberOfDatapoints() const
{
    return (mIndices != NULL) ? mIndices->GetN() : 0;
}
