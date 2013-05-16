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
// AxisAlignedParamsStep constructs a float_params and int_params matrix for
// extracting features from a matrix.  Each feature is a single dimension of
// the input matrix and is choosen uniformily from all dimensions.
//
// ----------------------------------------------------------------------------
template <class FloatType, class IntType>
class AxisAlignedParamsStep: public PipelineStepI
{
public:
    AxisAlignedParamsStep( const BufferId numberOfFeaturesBufferId,
                            const BufferId matrixDataBufferId );
    virtual ~AxisAlignedParamsStep();

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
                      IntType numberOfDimensions,
                      MatrixBufferTemplate<FloatType>& floatParams,
                      MatrixBufferTemplate<IntType>& intParams ) const;

    const BufferId mNumberOfFeaturesBufferId;
    const BufferId mMatrixDataBufferId;
};


template <class FloatType, class IntType>
AxisAlignedParamsStep<FloatType,IntType>::AxisAlignedParamsStep(  const BufferId numberOfFeaturesBufferId,
                                                                  const BufferId matrixDataBufferId )
: FloatParamsBufferId(GetBufferId("FloatParams"))
, IntParamsBufferId(GetBufferId("IntParams"))
, mNumberOfFeaturesBufferId(numberOfFeaturesBufferId)
, mMatrixDataBufferId(matrixDataBufferId)
{}

template <class FloatType, class IntType>
AxisAlignedParamsStep<FloatType,IntType>::~AxisAlignedParamsStep()
{}

template <class FloatType, class IntType>
PipelineStepI* AxisAlignedParamsStep<FloatType,IntType>::Clone() const
{
    AxisAlignedParamsStep* clone = new AxisAlignedParamsStep<FloatType,IntType>(*this);
    return clone;
}

template <class FloatType, class IntType>
void AxisAlignedParamsStep<FloatType,IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                          BufferCollection& writeCollection,
                                                          boost::mt19937& gen) const
{
    UNUSED_PARAM(gen)
    if(!writeCollection.HasBuffer< MatrixBufferTemplate<FloatType> >(FloatParamsBufferId)
        || !writeCollection.HasBuffer< MatrixBufferTemplate<IntType> >(IntParamsBufferId))
    {
        const MatrixBufferTemplate<FloatType>& matrixBuffer =
                readCollection.GetBuffer< MatrixBufferTemplate<FloatType> >(mMatrixDataBufferId);
        const IntType numberOfDimensions = matrixBuffer.GetN();

        const VectorBufferTemplate<IntType>& numberOfFeaturesBuffer =
                readCollection.GetBuffer< VectorBufferTemplate<IntType> >(mNumberOfFeaturesBufferId);
        ASSERT_ARG_DIM_1D(numberOfFeaturesBuffer.GetN(), 1)
        const IntType numberOfFeatures = std::min(numberOfFeaturesBuffer.Get(0), numberOfDimensions);

        MatrixBufferTemplate<FloatType>& floatParams =
                writeCollection.GetOrAddBuffer< MatrixBufferTemplate<FloatType> >(FloatParamsBufferId);

        MatrixBufferTemplate<IntType>& intParams =
                writeCollection.GetOrAddBuffer< MatrixBufferTemplate<IntType> >(IntParamsBufferId);

        SampleParams(numberOfFeatures, numberOfDimensions, floatParams, intParams);
    }
}

template <class FloatType, class IntType>
void AxisAlignedParamsStep<FloatType,IntType>::SampleParams(IntType numberOfFeatures,
                                                            IntType numberOfDimensions,
                                                            MatrixBufferTemplate<FloatType>& floatParams,
                                                            MatrixBufferTemplate<IntType>& intParams ) const
{
    floatParams.Resize(numberOfFeatures, DIMENSION_OF_PARAMETERS);
    intParams.Resize(numberOfFeatures, DIMENSION_OF_PARAMETERS);

    // Sample without replacement so a dimension is not choosen multiple times
    std::vector<IntType> candidateDimensions(numberOfFeatures);
    sampleIndicesWithOutReplacement(&candidateDimensions[0], numberOfFeatures, numberOfDimensions);

    for(int i=0; i<numberOfFeatures; i++)
    {
        floatParams.Set(i, PARAM_START_INDEX, static_cast<FloatType>(1.0)); // use a weight of 1.0 since there is 1 component
        intParams.Set(i, FEATURE_TYPE_INDEX, MATRIX_FEATURES); // feature type
        intParams.Set(i, NUMBER_OF_DIMENSIONS_INDEX, 1); // how many dimensions in projection
        intParams.Set(i, PARAM_START_INDEX, candidateDimensions[i]); // dimension index
    }
}
