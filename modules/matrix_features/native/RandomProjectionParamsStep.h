#pragma once

#include <boost/random/uniform_int.hpp>

#include <vector>
#include <algorithm>

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "bootstrap.h"
#include "Constants.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"
#include "LinearMatrixFeature.h"

// ----------------------------------------------------------------------------
//
// RandomProjectionParamsStep constructs a float_params and int_params matrix
// for extracting features from a matrix.  Each feature is a random projection
// in a subspace of the dimensions.
//
// ----------------------------------------------------------------------------
template <class BufferTypes, class DataMatrixType>
class RandomProjectionParamsStep: public PipelineStepI
{
public:
    RandomProjectionParamsStep( const BufferId& numberOfFeatures,
                                    const BufferId& matrixData,
                                    const int subspaceDimension );
    virtual ~RandomProjectionParamsStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    // Read only output buffers
    const BufferId FloatParamsBufferId;
    const BufferId IntParamsBufferId;
private:
    enum { DIMENSION_OF_PARAMETERS = PARAM_START_INDEX + 1 };
    void SampleParams(typename BufferTypes::ParamsInteger numberOfFeatures,
                    const DataMatrixType& matrixBuffer,
                    MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                    MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams,
                    boost::mt19937& gen ) const;

    const BufferId mNumberOfFeaturesBufferId;
    const BufferId mMatrixDataBufferId;
    const int mSubspaceDimension;
};


template <class BufferTypes, class DataMatrixType>
RandomProjectionParamsStep<BufferTypes, DataMatrixType>::RandomProjectionParamsStep(  const BufferId& numberOfFeatures,
                                                                            const BufferId& matrixData,
                                                                            const int subspaceDimension )
: PipelineStepI("RandomProjectionParamsStep")
, FloatParamsBufferId(GetBufferId("FloatParams"))
, IntParamsBufferId(GetBufferId("IntParams"))
, mNumberOfFeaturesBufferId(numberOfFeatures)
, mMatrixDataBufferId(matrixData)
, mSubspaceDimension(subspaceDimension)
{}

template <class BufferTypes, class DataMatrixType>
RandomProjectionParamsStep<BufferTypes, DataMatrixType>::~RandomProjectionParamsStep()
{}

template <class BufferTypes, class DataMatrixType>
PipelineStepI* RandomProjectionParamsStep<BufferTypes, DataMatrixType>::Clone() const
{
    RandomProjectionParamsStep* clone = new RandomProjectionParamsStep<BufferTypes, DataMatrixType>(*this);
    return clone;
}

template <class BufferTypes, class DataMatrixType>
void RandomProjectionParamsStep<BufferTypes, DataMatrixType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                          BufferCollection& writeCollection,
                                                          boost::mt19937& gen,
                                                          BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(gen);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);

    const VectorBufferTemplate<typename BufferTypes::SourceInteger>& numberOfFeaturesBuffer =
            readCollection.GetBuffer< VectorBufferTemplate<typename BufferTypes::SourceInteger> >(mNumberOfFeaturesBufferId);
    ASSERT_ARG_DIM_1D(numberOfFeaturesBuffer.GetN(), 1)

    const DataMatrixType& matrixBuffer =
            readCollection.GetBuffer< DataMatrixType >(mMatrixDataBufferId);

    const typename BufferTypes::SourceInteger numberOfFeatures = numberOfFeaturesBuffer.Get(0);

    MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(FloatParamsBufferId);

    MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(IntParamsBufferId);

    SampleParams(numberOfFeatures, matrixBuffer, floatParams, intParams, gen);

}

template <class BufferTypes, class DataMatrixType>
void RandomProjectionParamsStep<BufferTypes, DataMatrixType>::SampleParams(typename BufferTypes::ParamsInteger numberOfFeatures,
                                                            const DataMatrixType& matrixBuffer,
                                                            MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                                                            MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams,
                                                            boost::mt19937& gen ) const
{
    const int numberOfDimensions =  matrixBuffer.GetN();
    floatParams.Resize(numberOfFeatures, PARAM_START_INDEX + numberOfDimensions);
    intParams.Resize(numberOfFeatures, PARAM_START_INDEX + numberOfDimensions);

    std::vector< int > dimensionsSubspace(numberOfDimensions);
    for(int i=0; i<numberOfDimensions; i++)
    {
        dimensionsSubspace[i] = i;
    }
    // Don't let the subspace be larger than actual dimension
    int subspaceDimension = mSubspaceDimension;
    if( numberOfDimensions < subspaceDimension)
    {
        subspaceDimension = numberOfDimensions;
    }
    boost::uniform_real<> uniform_weight(-1.0, 1.0);
    boost::variate_generator<boost::mt19937&,boost::uniform_real<> > var_uniform_weight(gen, uniform_weight);

    for(int f=0; f<numberOfFeatures; f++)
    {
        intParams.Set(f, FEATURE_TYPE_INDEX, MATRIX_FEATURES); // feature type
        intParams.Set(f, NUMBER_OF_DIMENSIONS_INDEX, subspaceDimension); // how many dimensions in projection

        std::random_shuffle ( dimensionsSubspace.begin(), dimensionsSubspace.end() );
        std::sort( dimensionsSubspace.begin(), dimensionsSubspace.begin() + subspaceDimension );

        for(int i=0; i<subspaceDimension; i++)
        {
            const int d = dimensionsSubspace[i];
            intParams.Set(f, PARAM_START_INDEX+i, d); // dimension index
            floatParams.Set(f, PARAM_START_INDEX+i, var_uniform_weight());
        }
    }
}
