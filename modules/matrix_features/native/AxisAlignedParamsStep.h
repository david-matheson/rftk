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
template <class BufferTypes, class DataMatrixType>
class AxisAlignedParamsStep: public PipelineStepI
{
public:
    AxisAlignedParamsStep( const BufferId& numberOfFeaturesBufferId,
                            const BufferId& matrixDataBufferId );
    virtual ~AxisAlignedParamsStep();

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
    void SampleParams(typename BufferTypes::Index numberOfFeatures,
                      typename BufferTypes::Index numberOfDimensions,
                        MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                        MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams) const;

    const BufferId mNumberOfFeaturesBufferId;
    const BufferId mMatrixDataBufferId;
};


template <class BufferTypes, class DataMatrixType>
AxisAlignedParamsStep<BufferTypes, DataMatrixType>::AxisAlignedParamsStep(  const BufferId& numberOfFeaturesBufferId,
                                                            const BufferId& matrixDataBufferId )
: PipelineStepI("AxisAlignedParamsStep")
, FloatParamsBufferId(GetBufferId("FloatParams"))
, IntParamsBufferId(GetBufferId("IntParams"))
, mNumberOfFeaturesBufferId(numberOfFeaturesBufferId)
, mMatrixDataBufferId(matrixDataBufferId)
{}

template <class BufferTypes, class DataMatrixType>
AxisAlignedParamsStep<BufferTypes, DataMatrixType>::~AxisAlignedParamsStep()
{}

template <class BufferTypes, class DataMatrixType>
PipelineStepI* AxisAlignedParamsStep<BufferTypes, DataMatrixType>::Clone() const
{
    AxisAlignedParamsStep* clone = new AxisAlignedParamsStep<BufferTypes, DataMatrixType>(*this);
    return clone;
}

template <class BufferTypes, class DataMatrixType>
void AxisAlignedParamsStep<BufferTypes, DataMatrixType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                          BufferCollection& writeCollection,
                                                          boost::mt19937& gen,
                                                          BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(gen);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);
    
    if(!writeCollection.HasBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(FloatParamsBufferId)
        || !writeCollection.HasBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(IntParamsBufferId))
    {
        const DataMatrixType& matrixBuffer =
                readCollection.GetBuffer< DataMatrixType >(mMatrixDataBufferId);
        const typename BufferTypes::Index numberOfDimensions = matrixBuffer.GetN();

        const VectorBufferTemplate<typename BufferTypes::SourceInteger>& numberOfFeaturesBuffer =
                readCollection.GetBuffer< VectorBufferTemplate<typename BufferTypes::SourceInteger> >(mNumberOfFeaturesBufferId);
        ASSERT_ARG_DIM_1D(numberOfFeaturesBuffer.GetN(), 1)
        const typename BufferTypes::Index numberOfFeatures = std::min( std::max(1, numberOfFeaturesBuffer.Get(0)), numberOfDimensions);

        MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams =
                writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(FloatParamsBufferId);

        MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams =
                writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(IntParamsBufferId);

        SampleParams(numberOfFeatures, numberOfDimensions, floatParams, intParams);
    }
}

template <class BufferTypes, class DataMatrixType>
void AxisAlignedParamsStep<BufferTypes, DataMatrixType>::SampleParams(typename BufferTypes::Index numberOfFeatures,
                                                        typename BufferTypes::Index numberOfDimensions,
                                                        MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                                                        MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams ) const
{
    floatParams.Resize(numberOfFeatures, DIMENSION_OF_PARAMETERS);
    intParams.Resize(numberOfFeatures, DIMENSION_OF_PARAMETERS);

    // Sample without replacement so a dimension is not choosen multiple times
    std::vector<typename BufferTypes::Index> candidateDimensions(numberOfFeatures);
    sampleIndicesWithOutReplacement(&candidateDimensions[0], numberOfFeatures, numberOfDimensions);

    for(int i=0; i<numberOfFeatures; i++)
    {
        floatParams.Set(i, PARAM_START_INDEX, static_cast<typename BufferTypes::ParamsContinuous>(1.0)); // use a weight of 1.0 since there is 1 component
        intParams.Set(i, FEATURE_TYPE_INDEX, MATRIX_FEATURES); // feature type
        intParams.Set(i, NUMBER_OF_DIMENSIONS_INDEX, 1); // how many dimensions in projection
        intParams.Set(i, PARAM_START_INDEX, candidateDimensions[i]); // dimension index
    }
}
