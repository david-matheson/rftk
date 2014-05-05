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
// ClassPairDifferenceParamsStep constructs a float_params and int_params matrix
// for extracting features from a matrix.  Each feature is the difference in x
// between a pair of datapoints of different classes
//
// ----------------------------------------------------------------------------
template <class BufferTypes, class DataMatrixType>
class ClassPairDifferenceParamsStep: public PipelineStepI
{
public:
    ClassPairDifferenceParamsStep( const BufferId& numberOfFeatures,
                                    const BufferId& matrixData,
                                    const BufferId& classes,
                                    const BufferId& indices,
                                    const int subspaceDimension );
    virtual ~ClassPairDifferenceParamsStep();

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
                    const VectorBufferTemplate<typename BufferTypes::SourceInteger>& classesBuffer,
                    const VectorBufferTemplate<typename BufferTypes::Index>& indicesBuffer,
                    MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                    MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams,
                    boost::mt19937& gen ) const;

    const BufferId mNumberOfFeaturesBufferId;
    const BufferId mMatrixDataBufferId;
    const BufferId mClassesBufferId;
    const BufferId mIndicesBufferId;
    const int mSubspaceDimension;
};


template <class BufferTypes, class DataMatrixType>
ClassPairDifferenceParamsStep<BufferTypes, DataMatrixType>::ClassPairDifferenceParamsStep(  const BufferId& numberOfFeatures,
                                                                            const BufferId& matrixData,
                                                                            const BufferId& classes,
                                                                            const BufferId& indices,
                                                                            const int subspaceDimension )
: PipelineStepI("ClassPairDifferenceParamsStep")
, FloatParamsBufferId(GetBufferId("FloatParams"))
, IntParamsBufferId(GetBufferId("IntParams"))
, mNumberOfFeaturesBufferId(numberOfFeatures)
, mMatrixDataBufferId(matrixData)
, mClassesBufferId(classes)
, mIndicesBufferId(indices)
, mSubspaceDimension(subspaceDimension)
{}

template <class BufferTypes, class DataMatrixType>
ClassPairDifferenceParamsStep<BufferTypes, DataMatrixType>::~ClassPairDifferenceParamsStep()
{}

template <class BufferTypes, class DataMatrixType>
PipelineStepI* ClassPairDifferenceParamsStep<BufferTypes, DataMatrixType>::Clone() const
{
    ClassPairDifferenceParamsStep* clone = new ClassPairDifferenceParamsStep<BufferTypes, DataMatrixType>(*this);
    return clone;
}

template <class BufferTypes, class DataMatrixType>
void ClassPairDifferenceParamsStep<BufferTypes, DataMatrixType>::ProcessStep(const BufferCollectionStack& readCollection,
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

    const VectorBufferTemplate<typename BufferTypes::SourceInteger>& classesBuffer =
            readCollection.GetBuffer< VectorBufferTemplate<typename BufferTypes::SourceInteger> >(mClassesBufferId);

    const VectorBufferTemplate<typename BufferTypes::Index>& indicesBuffer =
            readCollection.GetBuffer< VectorBufferTemplate<typename BufferTypes::Index> >(mIndicesBufferId);

    const typename BufferTypes::SourceInteger numberOfFeatures = std::min(numberOfFeaturesBuffer.Get(0), indicesBuffer.GetN());

    MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(FloatParamsBufferId);

    MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(IntParamsBufferId);

    SampleParams(numberOfFeatures, matrixBuffer, classesBuffer, indicesBuffer, floatParams, intParams, gen);

}

template <class BufferTypes, class DataMatrixType>
void ClassPairDifferenceParamsStep<BufferTypes, DataMatrixType>::SampleParams(typename BufferTypes::ParamsInteger numberOfFeatures,
                                                            const DataMatrixType& matrixBuffer,
                                                            const VectorBufferTemplate<typename BufferTypes::SourceInteger>& classesBuffer,
                                                            const VectorBufferTemplate<typename BufferTypes::Index>& indicesBuffer,
                                                            MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                                                            MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams,
                                                            boost::mt19937& gen ) const
{
    UNUSED_PARAM(classesBuffer)

    const int numberOfDimensions =  matrixBuffer.GetN();
    floatParams.Resize(numberOfFeatures, PARAM_START_INDEX + numberOfDimensions);
    intParams.Resize(numberOfFeatures, PARAM_START_INDEX + numberOfDimensions);

    const int numberOfSamples = indicesBuffer.GetN();

    boost::uniform_int<> uniform_samples(0,numberOfSamples-1);
    boost::variate_generator<boost::mt19937&,boost::uniform_int<> > var_uniform_samples(gen, uniform_samples);

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

    for(int f=0; f<numberOfFeatures; f++)
    {
        const int relativeIndex1 = var_uniform_samples();
        const int index1 = indicesBuffer.Get(relativeIndex1);

        // Continue to sample the second sample until it's of a different class
        int relativeIndex2 = var_uniform_samples();
        int maxTries = 100;
        while(classesBuffer.Get(index1) == classesBuffer.Get(indicesBuffer.Get(relativeIndex2))
              && maxTries > 0)
        {
            relativeIndex2 = var_uniform_samples();
            maxTries--;
        }
        const int index2 = indicesBuffer.Get(relativeIndex2);

        intParams.Set(f, FEATURE_TYPE_INDEX, MATRIX_FEATURES); // feature type
        intParams.Set(f, NUMBER_OF_DIMENSIONS_INDEX, subspaceDimension); // how many dimensions in projection

        std::random_shuffle ( dimensionsSubspace.begin(), dimensionsSubspace.end() );
        std::sort( dimensionsSubspace.begin(), dimensionsSubspace.begin() + subspaceDimension );

        for(int i=0; i<subspaceDimension; i++)
        {
            const int d = dimensionsSubspace[i];
            intParams.Set(f, PARAM_START_INDEX+i, d); // dimension index
            const typename BufferTypes::ParamsContinuous componentDiff = matrixBuffer.Get(index1, d) - matrixBuffer.Get(index2, d);
            floatParams.Set(f, PARAM_START_INDEX+i, componentDiff);
        }
    }
}
