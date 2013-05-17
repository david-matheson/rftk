#pragma once

#include <boost/random/uniform_int.hpp>

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
template <class FloatType, class IntType>
class ClassPairDifferenceParamsStep: public PipelineStepI
{
public:
    ClassPairDifferenceParamsStep( const BufferId& numberOfFeatures,
                                    const BufferId& matrixData,
                                    const BufferId& classes,
                                    const BufferId& indices );
    virtual ~ClassPairDifferenceParamsStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    // Read only output buffers
    const BufferId FloatParamsBufferId;
    const BufferId IntParamsBufferId;
private:
    enum { DIMENSION_OF_PARAMETERS = PARAM_START_INDEX + 1 };
    void SampleParams(IntType numberOfFeatures,
                    const MatrixBufferTemplate<FloatType>& matrixBuffer,
                    const VectorBufferTemplate<IntType>& classesBuffer,
                    const VectorBufferTemplate<IntType>& indicesBuffer,
                    MatrixBufferTemplate<FloatType>& floatParams,
                    MatrixBufferTemplate<IntType>& intParams,
                    boost::mt19937& gen ) const;

    const BufferId mNumberOfFeaturesBufferId;
    const BufferId mMatrixDataBufferId;
    const BufferId mClassesBufferId;
    const BufferId mIndicesBufferId;
};


template <class FloatType, class IntType>
ClassPairDifferenceParamsStep<FloatType,IntType>::ClassPairDifferenceParamsStep(  const BufferId& numberOfFeatures,
                                                                                    const BufferId& matrixData,
                                                                                    const BufferId& classes,
                                                                                    const BufferId& indices )
: FloatParamsBufferId(GetBufferId("FloatParams"))
, IntParamsBufferId(GetBufferId("IntParams"))
, mNumberOfFeaturesBufferId(numberOfFeatures)
, mMatrixDataBufferId(matrixData)
, mClassesBufferId(classes)
, mIndicesBufferId(indices)
{}

template <class FloatType, class IntType>
ClassPairDifferenceParamsStep<FloatType,IntType>::~ClassPairDifferenceParamsStep()
{}

template <class FloatType, class IntType>
PipelineStepI* ClassPairDifferenceParamsStep<FloatType,IntType>::Clone() const
{
    ClassPairDifferenceParamsStep* clone = new ClassPairDifferenceParamsStep<FloatType,IntType>(*this);
    return clone;
}

template <class FloatType, class IntType>
void ClassPairDifferenceParamsStep<FloatType,IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                          BufferCollection& writeCollection,
                                                          boost::mt19937& gen) const
{
    UNUSED_PARAM(gen)

    const VectorBufferTemplate<IntType>& numberOfFeaturesBuffer =
            readCollection.GetBuffer< VectorBufferTemplate<IntType> >(mNumberOfFeaturesBufferId);
    ASSERT_ARG_DIM_1D(numberOfFeaturesBuffer.GetN(), 1)

    const MatrixBufferTemplate<FloatType>& matrixBuffer =
            readCollection.GetBuffer< MatrixBufferTemplate<FloatType> >(mMatrixDataBufferId);

    const VectorBufferTemplate<IntType>& classesBuffer =
            readCollection.GetBuffer< VectorBufferTemplate<IntType> >(mClassesBufferId);

    const VectorBufferTemplate<IntType>& indicesBuffer =
            readCollection.GetBuffer< VectorBufferTemplate<IntType> >(mIndicesBufferId);

    const IntType numberOfFeatures = std::min(numberOfFeaturesBuffer.Get(0), indicesBuffer.GetN());

    MatrixBufferTemplate<FloatType>& floatParams =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<FloatType> >(FloatParamsBufferId);

    MatrixBufferTemplate<IntType>& intParams =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<IntType> >(IntParamsBufferId);

    SampleParams(numberOfFeatures, matrixBuffer, classesBuffer, indicesBuffer, floatParams, intParams, gen);

}

template <class FloatType, class IntType>
void ClassPairDifferenceParamsStep<FloatType,IntType>::SampleParams(IntType numberOfFeatures,
                                                            const MatrixBufferTemplate<FloatType>& matrixBuffer,
                                                            const VectorBufferTemplate<IntType>& classesBuffer,
                                                            const VectorBufferTemplate<IntType>& indicesBuffer,
                                                            MatrixBufferTemplate<FloatType>& floatParams,
                                                            MatrixBufferTemplate<IntType>& intParams,
                                                            boost::mt19937& gen ) const
{
    UNUSED_PARAM(classesBuffer)

    const int numberOfDimensions =  matrixBuffer.GetN();
    floatParams.Resize(numberOfFeatures, PARAM_START_INDEX + numberOfDimensions);
    intParams.Resize(numberOfFeatures, PARAM_START_INDEX + numberOfDimensions);

    const int numberOfSamples = indicesBuffer.GetN();

    boost::uniform_int<> uniform_samples(0,numberOfSamples-1);
    boost::variate_generator<boost::mt19937&,boost::uniform_int<> > var_uniform_samples(gen, uniform_samples);

    for(int f=0; f<numberOfFeatures; f++)
    {
        const int relativeIndex1 = var_uniform_samples();
        const int index1 = indicesBuffer.Get(relativeIndex1);

        // Continue to sample the second sample until it's of a different class
        int relativeIndex2 = var_uniform_samples();
        int maxTries = 1000;
        while(classesBuffer.Get(index1) == classesBuffer.Get(indicesBuffer.Get(relativeIndex2))
              && maxTries > 0)
        {
            relativeIndex2 = var_uniform_samples();
            maxTries--;
        }
        const int index2 = indicesBuffer.Get(relativeIndex2);

        intParams.Set(f, FEATURE_TYPE_INDEX, MATRIX_FEATURES); // feature type
        intParams.Set(f, NUMBER_OF_DIMENSIONS_INDEX, numberOfDimensions); // how many dimensions in projection

        for(int d=0; d<numberOfDimensions; d++)
        {
            intParams.Set(f, PARAM_START_INDEX+d, d); // dimension index
            const FloatType componentDiff = matrixBuffer.Get(index1, d) - matrixBuffer.Get(index2, d);
            floatParams.Set(f, PARAM_START_INDEX+d, componentDiff);
        }
    }
}
