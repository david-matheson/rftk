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
template <class FloatType, class IntType>
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
    void SampleParams(IntType numberOfFeatures,
                      IntType numberOfDimensions,
                      MatrixBufferTemplate<FloatType>& floatParams,
                      MatrixBufferTemplate<IntType>& intParams ) const;

    const BufferId mNumberOfFeaturesBufferId;
    const BufferId mMatrixDataBufferId;
};


template <class FloatType, class IntType>
DimensionPairDifferenceParamsStep<FloatType,IntType>::DimensionPairDifferenceParamsStep(  const BufferId& numberOfFeaturesBufferId,
                                                                  const BufferId& matrixDataBufferId )
: FloatParamsBufferId(GetBufferId("FloatParams"))
, IntParamsBufferId(GetBufferId("IntParams"))
, mNumberOfFeaturesBufferId(numberOfFeaturesBufferId)
, mMatrixDataBufferId(matrixDataBufferId)
{}

template <class FloatType, class IntType>
DimensionPairDifferenceParamsStep<FloatType,IntType>::~DimensionPairDifferenceParamsStep()
{}

template <class FloatType, class IntType>
PipelineStepI* DimensionPairDifferenceParamsStep<FloatType,IntType>::Clone() const
{
    DimensionPairDifferenceParamsStep* clone = new DimensionPairDifferenceParamsStep<FloatType,IntType>(*this);
    return clone;
}

template <class FloatType, class IntType>
void DimensionPairDifferenceParamsStep<FloatType,IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                          BufferCollection& writeCollection,
                                                          boost::mt19937& gen) const
{
    UNUSED_PARAM(gen)

    const MatrixBufferTemplate<FloatType>& matrixBuffer =
            readCollection.GetBuffer< MatrixBufferTemplate<FloatType> >(mMatrixDataBufferId);
    const IntType numberOfDimensions = matrixBuffer.GetN();

    const VectorBufferTemplate<IntType>& numberOfFeaturesBuffer =
            readCollection.GetBuffer< VectorBufferTemplate<IntType> >(mNumberOfFeaturesBufferId);
    ASSERT_ARG_DIM_1D(numberOfFeaturesBuffer.GetN(), 1)
    const IntType numberOfFeatures = std::min( std::max(1, numberOfFeaturesBuffer.Get(0)), numberOfDimensions);

    MatrixBufferTemplate<FloatType>& floatParams =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<FloatType> >(FloatParamsBufferId);

    MatrixBufferTemplate<IntType>& intParams =
            writeCollection.GetOrAddBuffer< MatrixBufferTemplate<IntType> >(IntParamsBufferId);

    SampleParams(numberOfFeatures, numberOfDimensions, floatParams, intParams);

}

template <class FloatType, class IntType>
void DimensionPairDifferenceParamsStep<FloatType,IntType>::SampleParams(IntType numberOfFeatures,
                                                            IntType numberOfDimensions,
                                                            MatrixBufferTemplate<FloatType>& floatParams,
                                                            MatrixBufferTemplate<IntType>& intParams ) const
{
    floatParams.Resize(numberOfFeatures, DIMENSION_OF_PARAMETERS);
    intParams.Resize(numberOfFeatures, DIMENSION_OF_PARAMETERS);

    // Sample without replacement so a dimension is not choosen multiple times
    std::vector<IntType> candidateDimensions(numberOfFeatures*2);
    sampleIndicesWithOutReplacement(&candidateDimensions[0], numberOfFeatures*2, numberOfDimensions);

    for(int i=0; i<numberOfFeatures; i++)
    {
        intParams.Set(i, FEATURE_TYPE_INDEX, MATRIX_FEATURES); // feature type
        intParams.Set(i, NUMBER_OF_DIMENSIONS_INDEX, 2); // how many dimensions in projection
        intParams.Set(i, PARAM_START_INDEX, candidateDimensions[i*2]); // dimension index
        intParams.Set(i, PARAM_START_INDEX+1, candidateDimensions[i*2+1]); // dimension index
        floatParams.Set(i, PARAM_START_INDEX, static_cast<FloatType>(1.0)); // use a weight of 1.0
        floatParams.Set(i, PARAM_START_INDEX+1, static_cast<FloatType>(-1.0)); // use a weight of -1.0
    }
}
