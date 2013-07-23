#pragma once

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
// DimensionPairDifferenceParamsStep constructs a float_params and int_params 
// matrix for extracting features from a matrix.  Each feature is the difference 
// between two dimensions choosen uniformily at random.
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class DimensionPairDifferenceParamsStep: public PipelineStepI
{
public:
    DimensionPairDifferenceParamsStep( const BufferId& numberOfFeaturesBufferId,
                            const BufferId& matrixDataBufferId );
    virtual ~DimensionPairDifferenceParamsStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen) const;

    // Read only output buffers
    const BufferId FloatParamsBufferId;
    const BufferId IntParamsBufferId;
private:
    enum { DIMENSION_OF_PARAMETERS = PARAM_START_INDEX + 2 };
    void SampleParams(  typename BufferTypes::ParamsInteger numberOfFeatures,
                        typename BufferTypes::ParamsInteger numberOfDimensions,
                        MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                        MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams) const;

    const BufferId mNumberOfFeaturesBufferId;
    const BufferId mMatrixDataBufferId;
};


template <class BufferTypes>
DimensionPairDifferenceParamsStep<BufferTypes>::DimensionPairDifferenceParamsStep(  const BufferId& numberOfFeaturesBufferId,
                                                                  const BufferId& matrixDataBufferId )
: FloatParamsBufferId(GetBufferId("FloatParams"))
, IntParamsBufferId(GetBufferId("IntParams"))
, mNumberOfFeaturesBufferId(numberOfFeaturesBufferId)
, mMatrixDataBufferId(matrixDataBufferId)
{}

template <class BufferTypes>
DimensionPairDifferenceParamsStep<BufferTypes>::~DimensionPairDifferenceParamsStep()
{}

template <class BufferTypes>
PipelineStepI* DimensionPairDifferenceParamsStep<BufferTypes>::Clone() const
{
    DimensionPairDifferenceParamsStep* clone = new DimensionPairDifferenceParamsStep<BufferTypes>(*this);
    return clone;
}

template <class BufferTypes>
void DimensionPairDifferenceParamsStep<BufferTypes>::ProcessStep(const BufferCollectionStack& readCollection,
                                                          BufferCollection& writeCollection,
                                                          boost::mt19937& gen) const
{
    UNUSED_PARAM(gen)

    const MatrixBufferTemplate<typename BufferTypes::SourceContinuous>& matrixBuffer =
            readCollection.GetBuffer< MatrixBufferTemplate<typename BufferTypes::SourceContinuous> >(mMatrixDataBufferId);
    const typename BufferTypes::ParamsInteger numberOfDimensions = matrixBuffer.GetN();

    const VectorBufferTemplate<typename BufferTypes::SourceInteger>& numberOfFeaturesBuffer =
            readCollection.GetBuffer< VectorBufferTemplate<typename BufferTypes::SourceInteger> >(mNumberOfFeaturesBufferId);
    ASSERT_ARG_DIM_1D(numberOfFeaturesBuffer.GetN(), 1)
    const typename BufferTypes::ParamsInteger numberOfFeatures = std::min( std::max(1, numberOfFeaturesBuffer.Get(0)), numberOfDimensions);

    MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(FloatParamsBufferId);

    MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(IntParamsBufferId);

    SampleParams(numberOfFeatures, numberOfDimensions, floatParams, intParams);

}

template <class BufferTypes>
void DimensionPairDifferenceParamsStep<BufferTypes>::SampleParams(typename BufferTypes::ParamsInteger numberOfFeatures,
                                                            typename BufferTypes::ParamsInteger numberOfDimensions,
                                                            MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                                                            MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams ) const
{
    floatParams.Resize(numberOfFeatures, DIMENSION_OF_PARAMETERS);
    intParams.Resize(numberOfFeatures, DIMENSION_OF_PARAMETERS);

    // Sample without replacement so a dimension is not choosen multiple times
    std::vector<typename BufferTypes::Index> candidateDimensions(numberOfFeatures*2);
    sampleIndicesWithOutReplacement(&candidateDimensions[0], numberOfFeatures*2, numberOfDimensions);

    for(typename BufferTypes::ParamsInteger i=0; i<numberOfFeatures; i++)
    {
        intParams.Set(i, FEATURE_TYPE_INDEX, MATRIX_FEATURES); // feature type
        intParams.Set(i, NUMBER_OF_DIMENSIONS_INDEX, 2); // how many dimensions in projection
        intParams.Set(i, PARAM_START_INDEX, candidateDimensions[i*2]); // dimension index
        intParams.Set(i, PARAM_START_INDEX+1, candidateDimensions[i*2+1]); // dimension index
        floatParams.Set(i, PARAM_START_INDEX, static_cast<typename BufferTypes::ParamsContinuous>(1.0)); // use a weight of 1.0
        floatParams.Set(i, PARAM_START_INDEX+1, static_cast<typename BufferTypes::ParamsContinuous>(-1.0)); // use a weight of -1.0
    }
}
