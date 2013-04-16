#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "bootstrap.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"

// ----------------------------------------------------------------------------
// 
// AxisAlignedParamsStep constructs a float_params and int_params matrix for
// extracting features from a matrix.  Each feature is a single dimension of
// the input matrix and is choosen uniformily from all dimensions.
//
// ----------------------------------------------------------------------------
enum
{
    MATRIX_FEATURES = 1 // Move this to a more soucefile
};

const int SPLIT_POINT_INDEX = 0;  // Move this to a more soucefile
const int FEATURE_TYPE_INDEX = SPLIT_POINT_INDEX;     // Move this to a more soucefile
const int PARAM_START_INDEX = SPLIT_POINT_INDEX + 1;  // Move this to a more soucefile


template <class FloatType, class IntType>
class AxisAlignedParamsStep: public PipelineStepI
{
public:
    AxisAlignedParamsStep( const UniqueBufferId::BufferId numberOfFeaturesBufferId,
                            const UniqueBufferId::BufferId matrixDataBufferId );
    virtual ~AxisAlignedParamsStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const VectorBufferTemplate<long long> indices,
                                const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection) const;

    // Read only output buffers
    const UniqueBufferId::BufferId FloatParamsBufferId;
    const UniqueBufferId::BufferId IntParamsBufferId;
private:
    enum { DIMENSION_OF_PARAMETERS = 2 };
    void SampleParams(IntType numberOfFeatures, 
                      IntType numberOfDimensions,
                      MatrixBufferTemplate<FloatType>& floatParams,
                      MatrixBufferTemplate<IntType>& intParams ) const;

    const UniqueBufferId::BufferId mNumberOfFeaturesBufferId;
    const UniqueBufferId::BufferId mMatrixDataBufferId;
};


template <class FloatType, class IntType>
AxisAlignedParamsStep<FloatType,IntType>::AxisAlignedParamsStep(  const UniqueBufferId::BufferId numberOfFeaturesBufferId,
                                                                  const UniqueBufferId::BufferId matrixDataBufferId )
: FloatParamsBufferId(UniqueBufferId::GetBufferId("FloatParams"))
, IntParamsBufferId(UniqueBufferId::GetBufferId("IntParams"))
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
void AxisAlignedParamsStep<FloatType,IntType>::ProcessStep(const VectorBufferTemplate<long long> indices,
                                                const BufferCollectionStack& readCollection,
                                                BufferCollection& writeCollection) const
{
    UNUSED_PARAM(indices);
    if(!writeCollection.HasBuffer< MatrixBufferTemplate<FloatType> >(FloatParamsBufferId)
        || !writeCollection.HasBuffer< MatrixBufferTemplate<FloatType> >(IntParamsBufferId))
    {
        ASSERT(readCollection.HasBuffer< MatrixBufferTemplate<FloatType> >(mMatrixDataBufferId))
        const MatrixBufferTemplate<FloatType>& matrixBuffer =
                readCollection.GetBuffer< MatrixBufferTemplate<FloatType> >(mMatrixDataBufferId);
        const IntType numberOfDimensions = matrixBuffer.GetN();

        ASSERT(readCollection.HasBuffer< VectorBufferTemplate<IntType> >(mNumberOfFeaturesBufferId))
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
        floatParams.Set(i, SPLIT_POINT_INDEX, static_cast<FloatType>(0.0)); // this will be replaced by best splitpoint
        floatParams.Set(i, PARAM_START_INDEX, static_cast<FloatType>(1.0)); // use a weight of 1.0 since there is 1 component
        intParams.Set(i, FEATURE_TYPE_INDEX, MATRIX_FEATURES); // feature type
        intParams.Set(i, PARAM_START_INDEX, candidateDimensions[i]); // dimension index
    }
}
