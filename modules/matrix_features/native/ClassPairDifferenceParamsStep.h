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
template <class BufferTypes>
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
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    // Read only output buffers
    const BufferId FloatParamsBufferId;
    const BufferId IntParamsBufferId;
private:
    enum { DIMENSION_OF_PARAMETERS = PARAM_START_INDEX + 1 };
    void SampleParams(typename BufferTypes::ParamsInteger numberOfFeatures,
                    const MatrixBufferTemplate<typename BufferTypes::SourceContinuous>& matrixBuffer,
                    const VectorBufferTemplate<typename BufferTypes::SourceInteger>& classesBuffer,
                    const VectorBufferTemplate<typename BufferTypes::Index>& indicesBuffer,
                    MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                    MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams,
                    boost::mt19937& gen ) const;

    const BufferId mNumberOfFeaturesBufferId;
    const BufferId mMatrixDataBufferId;
    const BufferId mClassesBufferId;
    const BufferId mIndicesBufferId;
};


template <class BufferTypes>
ClassPairDifferenceParamsStep<BufferTypes>::ClassPairDifferenceParamsStep(  const BufferId& numberOfFeatures,
                                                                            const BufferId& matrixData,
                                                                            const BufferId& classes,
                                                                            const BufferId& indices )
: PipelineStepI("ClassPairDifferenceParamsStep")
, FloatParamsBufferId(GetBufferId("FloatParams"))
, IntParamsBufferId(GetBufferId("IntParams"))
, mNumberOfFeaturesBufferId(numberOfFeatures)
, mMatrixDataBufferId(matrixData)
, mClassesBufferId(classes)
, mIndicesBufferId(indices)
{}

template <class BufferTypes>
ClassPairDifferenceParamsStep<BufferTypes>::~ClassPairDifferenceParamsStep()
{}

template <class BufferTypes>
PipelineStepI* ClassPairDifferenceParamsStep<BufferTypes>::Clone() const
{
    ClassPairDifferenceParamsStep* clone = new ClassPairDifferenceParamsStep<BufferTypes>(*this);
    return clone;
}

template <class BufferTypes>
void ClassPairDifferenceParamsStep<BufferTypes>::ProcessStep(const BufferCollectionStack& readCollection,
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

    const MatrixBufferTemplate<typename BufferTypes::SourceContinuous>& matrixBuffer =
            readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::SourceContinuous> >(mMatrixDataBufferId);

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

template <class BufferTypes>
void ClassPairDifferenceParamsStep<BufferTypes>::SampleParams(typename BufferTypes::ParamsInteger numberOfFeatures,
                                                            const MatrixBufferTemplate<typename BufferTypes::SourceContinuous>& matrixBuffer,
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
            const typename BufferTypes::ParamsContinuous componentDiff = matrixBuffer.Get(index1, d) - matrixBuffer.Get(index2, d);
            floatParams.Set(f, PARAM_START_INDEX+d, componentDiff);
        }
    }
}
