#pragma once

#include "asserts.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Constants.h"

const int NUMBER_OF_DIMENSIONS = FEATURE_TYPE_INDEX + 1;
const int PARAM_START_INDEX = NUMBER_OF_DIMENSIONS + 1;  // Move this to a more soucefile

// ----------------------------------------------------------------------------
//
// LinearMatrixFeature is a linear combination of dimensions (columns) for
// each sample (row).
//
// ----------------------------------------------------------------------------
template <class DataMatrixType, class FloatType, class IntType>
class LinearMatrixFeatureBinding
{
public:
    LinearMatrixFeatureBinding( MatrixBufferTemplate<FloatType> const* floatParams,
                                 MatrixBufferTemplate<IntType> const* intParams,
                                 VectorBufferTemplate<IntType> const* indices,
                                 DataMatrixType const* dataMatrix);

    ~LinearMatrixFeatureBinding();

    FloatType FeatureValue( const int featureIndex, const int relativeSampleIndex) const;

    IntType GetNumberOfFeatures() const;
    IntType GetNumberOfDatapoints() const;

private:
    MatrixBufferTemplate<FloatType> const* mFloatParams;
    MatrixBufferTemplate<IntType> const* mIntParams;
    VectorBufferTemplate<IntType> const* mIndices;
    DataMatrixType const* mDataMatrix;

};

template <class DataMatrixType, class FloatType, class IntType>
LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType>::LinearMatrixFeatureBinding( MatrixBufferTemplate<FloatType> const* floatParams,
                                                                                         MatrixBufferTemplate<IntType> const* intParams,
                                                                                         VectorBufferTemplate<IntType> const* indices,
                                                                                         DataMatrixType const* dataMatrix )
: mFloatParams(floatParams)
, mIntParams(intParams)
, mIndices(indices)
, mDataMatrix(dataMatrix)
{}


template <class DataMatrixType, class FloatType, class IntType>
LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType>::~LinearMatrixFeatureBinding()
{}


//TODO: move to another class
template <class DataMatrixType, class FloatType, class IntType>
FloatType LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType>::FeatureValue( const int featureIndex, const int relativeSampleIndex) const
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

template <class DataMatrixType, class FloatType, class IntType>
IntType LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType>::GetNumberOfFeatures() const
{
    return mIntParams->GetM();
}

template <class DataMatrixType, class FloatType, class IntType>
IntType LinearMatrixFeatureBinding<DataMatrixType, FloatType, IntType>::GetNumberOfDatapoints() const
{
    return mIndices->GetN();
}
