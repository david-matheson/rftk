#pragma once

#include "asserts.h"
#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "Constants.h"

const int NUMBER_OF_DIMENSIONS_INDEX = FEATURE_TYPE_INDEX + 1;
const int PARAM_START_INDEX = NUMBER_OF_DIMENSIONS_INDEX + 1;  // Move this to a more soucefile

// ----------------------------------------------------------------------------
//
// LinearMatrixFeature is a linear combination of dimensions (columns) for
// each sample (row).
//
// ----------------------------------------------------------------------------
template <class BufferTypes, class DataMatrixType>
class LinearMatrixFeatureBinding
{
public:
    LinearMatrixFeatureBinding( MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> const* floatParams,
                                 MatrixBufferTemplate<typename BufferTypes::ParamsInteger> const* intParams,
                                 VectorBufferTemplate<typename BufferTypes::Index> const* indices,
                                 DataMatrixType const* dataMatrix);
    LinearMatrixFeatureBinding();
    ~LinearMatrixFeatureBinding();

    LinearMatrixFeatureBinding(const LinearMatrixFeatureBinding& other);
    LinearMatrixFeatureBinding & operator=(const LinearMatrixFeatureBinding & other);

    typename BufferTypes::FeatureValue FeatureValue( const typename BufferTypes::Index featureIndex, const typename BufferTypes::Index relativeSampleIndex) const;

    typename BufferTypes::Index GetNumberOfFeatures() const;
    typename BufferTypes::Index GetNumberOfDatapoints() const;

private:
    MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> const* mFloatParams;
    MatrixBufferTemplate<typename BufferTypes::ParamsInteger> const* mIntParams;
    VectorBufferTemplate<typename BufferTypes::Index> const* mIndices;
    DataMatrixType const* mDataMatrix;

};

template <class BufferTypes, class DataMatrixType>
LinearMatrixFeatureBinding<BufferTypes, DataMatrixType>::LinearMatrixFeatureBinding( MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> const* floatParams,
                                                                                         MatrixBufferTemplate<typename BufferTypes::ParamsInteger> const* intParams,
                                                                                         VectorBufferTemplate<typename BufferTypes::Index> const* indices,
                                                                                         DataMatrixType const* dataMatrix )
: mFloatParams(floatParams)
, mIntParams(intParams)
, mIndices(indices)
, mDataMatrix(dataMatrix)
{}

template <class BufferTypes, class DataMatrixType>
LinearMatrixFeatureBinding<BufferTypes, DataMatrixType>::LinearMatrixFeatureBinding()
: mFloatParams(NULL)
, mIntParams(NULL)
, mIndices(NULL)
, mDataMatrix(NULL)
{}

template <class BufferTypes, class DataMatrixType>
LinearMatrixFeatureBinding<BufferTypes, DataMatrixType>::LinearMatrixFeatureBinding( const LinearMatrixFeatureBinding& other )
: mFloatParams(other.mFloatParams)
, mIntParams(other.mIntParams)
, mIndices(other.mIndices)
, mDataMatrix(other.mDataMatrix)
{}

template <class BufferTypes, class DataMatrixType>
LinearMatrixFeatureBinding<BufferTypes, DataMatrixType>& LinearMatrixFeatureBinding<BufferTypes, DataMatrixType>::operator=(const LinearMatrixFeatureBinding & other)
{
    mFloatParams = other.mFloatParams;
    mIntParams = other.mIntParams;
    mIndices = other.mIndices;
    mDataMatrix = other.mDataMatrix;
    return *this;
}

template <class BufferTypes, class DataMatrixType>
LinearMatrixFeatureBinding<BufferTypes, DataMatrixType>::~LinearMatrixFeatureBinding()
{}


template <class BufferTypes, class DataMatrixType>
typename BufferTypes::FeatureValue LinearMatrixFeatureBinding<BufferTypes, DataMatrixType>::FeatureValue( const typename BufferTypes::Index featureIndex, const typename BufferTypes::Index relativeSampleIndex) const
{
    typename BufferTypes::FeatureValue featureValue = static_cast<typename BufferTypes::FeatureValue>(0.0);
    const typename BufferTypes::Index numberOfDimensions = mIntParams->Get(featureIndex, NUMBER_OF_DIMENSIONS_INDEX);
    for(int i=PARAM_START_INDEX; i<numberOfDimensions + PARAM_START_INDEX; i++)
    {
        const typename BufferTypes::Index dimension = mIntParams->Get(featureIndex, i);
        const typename BufferTypes::Index matrixIndex = mIndices->Get(relativeSampleIndex);
        featureValue += mFloatParams->Get(featureIndex, i) * mDataMatrix->Get(matrixIndex, dimension);
    }
    return featureValue;
}

template <class BufferTypes, class DataMatrixType>
typename BufferTypes::Index LinearMatrixFeatureBinding<BufferTypes, DataMatrixType>::GetNumberOfFeatures() const
{
    return mIntParams->GetM();
}

template <class BufferTypes, class DataMatrixType>
typename BufferTypes::Index LinearMatrixFeatureBinding<BufferTypes, DataMatrixType>::GetNumberOfDatapoints() const
{
    return mIndices->GetN();
}
