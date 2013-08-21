#pragma once

#include "VectorBuffer.h"
#include "MatrixBuffer.h"
#include "BufferCollection.h"
#include "BufferCollectionStack.h"
#include "bootstrap.h"
#include "Constants.h"
#include "PipelineStepI.h"
#include "UniqueBufferId.h"



// ----------------------------------------------------------------------------
//
// PixelPairGaussianOffsetsStep constructs a float_params and int_params matrix for
// extracting features from images.  Each feature is a pair of offsets in pixel
// space.
//
// ----------------------------------------------------------------------------
template <class BufferTypes>
class PixelPairGaussianOffsetsStep: public PipelineStepI
{
public:
    PixelPairGaussianOffsetsStep( const BufferId numberOfFeaturesBufferId,
                            const typename BufferTypes::ParamsContinuous ux,
                            const typename BufferTypes::ParamsContinuous uy,
                            const typename BufferTypes::ParamsContinuous vx,
                            const typename BufferTypes::ParamsContinuous vy  );
    virtual ~PixelPairGaussianOffsetsStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection,
                                boost::mt19937& gen,
                                BufferCollection& extraInfo, int nodeIndex) const;

    // Read only output buffers
    const BufferId FloatParamsBufferId;
    const BufferId IntParamsBufferId;
private:
    enum { DIMENSION_OF_PARAMETERS = FEATURE_SPECIFIC_PARAMS_START + 4 };
    void SampleParams(typename BufferTypes::ParamsInteger numberOfFeatures,
                      MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                      MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams,
                      boost::mt19937& gen ) const;

    const BufferId mNumberOfFeaturesBufferId;
    const typename BufferTypes::ParamsContinuous mUx;
    const typename BufferTypes::ParamsContinuous mUy;
    const typename BufferTypes::ParamsContinuous mVx;
    const typename BufferTypes::ParamsContinuous mVy;
    mutable boost::mt19937 mGen;
};


template <class BufferTypes>
PixelPairGaussianOffsetsStep<BufferTypes>::PixelPairGaussianOffsetsStep(  const BufferId numberOfFeaturesBufferId,
                                                                const typename BufferTypes::ParamsContinuous ux,
                                                                const typename BufferTypes::ParamsContinuous uy,
                                                                const typename BufferTypes::ParamsContinuous vx,
                                                                const typename BufferTypes::ParamsContinuous vy )
: FloatParamsBufferId(GetBufferId("FloatParams"))
, IntParamsBufferId(GetBufferId("IntParams"))
, mNumberOfFeaturesBufferId(numberOfFeaturesBufferId)
, mUx(ux)
, mUy(uy)
, mVx(vx)
, mVy(vy)
{}

template <class BufferTypes>
PixelPairGaussianOffsetsStep<BufferTypes>::~PixelPairGaussianOffsetsStep()
{}

template <class BufferTypes>
PipelineStepI* PixelPairGaussianOffsetsStep<BufferTypes>::Clone() const
{
    PixelPairGaussianOffsetsStep* clone = new PixelPairGaussianOffsetsStep<BufferTypes>(*this);
    return clone;
}


template <class BufferTypes>
void PixelPairGaussianOffsetsStep<BufferTypes>::ProcessStep(const BufferCollectionStack& readCollection,
                                                          BufferCollection& writeCollection,
                                                          boost::mt19937& gen,
                                                          BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);
    if(!writeCollection.HasBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(FloatParamsBufferId)
        || !writeCollection.HasBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(IntParamsBufferId))
    {
        const VectorBufferTemplate<typename BufferTypes::ParamsInteger>& numberOfFeaturesBuffer =
                readCollection.GetBuffer< VectorBufferTemplate<typename BufferTypes::ParamsInteger> >(mNumberOfFeaturesBufferId);
        ASSERT_ARG_DIM_1D(numberOfFeaturesBuffer.GetN(), 1)
        const typename BufferTypes::ParamsInteger numberOfFeatures = std::max(1, numberOfFeaturesBuffer.Get(0));

        MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams =
                writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsContinuous> >(FloatParamsBufferId);

        MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams =
                writeCollection.GetOrAddBuffer< MatrixBufferTemplate<typename BufferTypes::ParamsInteger> >(IntParamsBufferId);

        SampleParams(numberOfFeatures, floatParams, intParams, gen);
    }
}


template <class BufferTypes>
void PixelPairGaussianOffsetsStep<BufferTypes>::SampleParams(typename BufferTypes::ParamsInteger numberOfFeatures,
                                                            MatrixBufferTemplate<typename BufferTypes::ParamsContinuous>& floatParams,
                                                            MatrixBufferTemplate<typename BufferTypes::ParamsInteger>& intParams,
                                                            boost::mt19937& gen ) const
{
    floatParams.Resize(numberOfFeatures, DIMENSION_OF_PARAMETERS);
    intParams.Resize(numberOfFeatures, DIMENSION_OF_PARAMETERS);

    boost::normal_distribution<> ux_normal(0.0, mUx);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_ux_normal(gen, ux_normal);
    boost::normal_distribution<> uy_normal(0.0, mUy);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_uy_normal(gen, uy_normal);
    boost::normal_distribution<> vx_normal(0.0, mVx);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_vx_normal(gen, vx_normal);
    boost::normal_distribution<> vy_normal(0.0, mVy);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_vy_normal(gen, vy_normal);

    for(int i=0; i<numberOfFeatures; i++)
    {
        floatParams.Set(i, FEATURE_SPECIFIC_PARAMS_START, var_ux_normal());
        floatParams.Set(i, FEATURE_SPECIFIC_PARAMS_START+1, var_uy_normal());
        floatParams.Set(i, FEATURE_SPECIFIC_PARAMS_START+2, var_vx_normal());
        floatParams.Set(i, FEATURE_SPECIFIC_PARAMS_START+3, var_vy_normal());
    }
}
