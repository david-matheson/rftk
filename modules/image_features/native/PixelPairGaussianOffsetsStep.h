#pragma once

#include <ctime>

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

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
template <class FloatType, class IntType>
class PixelPairGaussianOffsetsStep: public PipelineStepI
{
public:
    PixelPairGaussianOffsetsStep( const BufferId numberOfFeaturesBufferId,
                            const FloatType ux,
                            const FloatType uy, 
                            const FloatType vx, 
                            const FloatType vy,
                            IntType seed = static_cast<unsigned int>(std::time(NULL)) );
    virtual ~PixelPairGaussianOffsetsStep();

    virtual PipelineStepI* Clone() const;

    virtual void ProcessStep(   const BufferCollectionStack& readCollection,
                                BufferCollection& writeCollection) const;

    // Read only output buffers
    const BufferId FloatParamsBufferId;
    const BufferId IntParamsBufferId;
private:
    enum { DIMENSION_OF_PARAMETERS = FEATURE_SPECIFIC_PARAMS_START + 4 };
    void SampleParams(IntType numberOfFeatures,
                      MatrixBufferTemplate<FloatType>& floatParams,
                      MatrixBufferTemplate<IntType>& intParams,
                      boost::mt19937& gen ) const;

    const BufferId mNumberOfFeaturesBufferId;
    const FloatType mUx;
    const FloatType mUy;
    const FloatType mVx;
    const FloatType mVy;
    mutable boost::mt19937 mGen;
};


template <class FloatType, class IntType>
PixelPairGaussianOffsetsStep<FloatType,IntType>::PixelPairGaussianOffsetsStep(  const BufferId numberOfFeaturesBufferId,
                                                                const FloatType ux,
                                                                const FloatType uy, 
                                                                const FloatType vx, 
                                                                const FloatType vy,
                                                                IntType seed )
: FloatParamsBufferId(GetBufferId("FloatParams"))
, IntParamsBufferId(GetBufferId("IntParams"))
, mNumberOfFeaturesBufferId(numberOfFeaturesBufferId)
, mUx(ux)
, mUy(uy)
, mVx(vx)
, mVy(vy)
, mGen(seed)
{}

template <class FloatType, class IntType>
PixelPairGaussianOffsetsStep<FloatType,IntType>::~PixelPairGaussianOffsetsStep()
{}

template <class FloatType, class IntType>
PipelineStepI* PixelPairGaussianOffsetsStep<FloatType,IntType>::Clone() const
{
    PixelPairGaussianOffsetsStep* clone = new PixelPairGaussianOffsetsStep<FloatType,IntType>(*this);
    return clone;
}


template <class FloatType, class IntType>
void PixelPairGaussianOffsetsStep<FloatType,IntType>::ProcessStep(const BufferCollectionStack& readCollection,
                                                          BufferCollection& writeCollection) const
{
    if(!writeCollection.HasBuffer< MatrixBufferTemplate<FloatType> >(FloatParamsBufferId)
        || !writeCollection.HasBuffer< MatrixBufferTemplate<FloatType> >(IntParamsBufferId))
    {
        ASSERT(readCollection.HasBuffer< VectorBufferTemplate<IntType> >(mNumberOfFeaturesBufferId))
        const VectorBufferTemplate<IntType>& numberOfFeaturesBuffer =
                readCollection.GetBuffer< VectorBufferTemplate<IntType> >(mNumberOfFeaturesBufferId);
        ASSERT_ARG_DIM_1D(numberOfFeaturesBuffer.GetN(), 1)
        const IntType numberOfFeatures = numberOfFeaturesBuffer.Get(0);

        MatrixBufferTemplate<FloatType>& floatParams =
                writeCollection.GetOrAddBuffer< MatrixBufferTemplate<FloatType> >(FloatParamsBufferId);

        MatrixBufferTemplate<IntType>& intParams =
                writeCollection.GetOrAddBuffer< MatrixBufferTemplate<IntType> >(IntParamsBufferId);

        SampleParams(numberOfFeatures, floatParams, intParams, mGen);
    }
}


template <class FloatType, class IntType>
void PixelPairGaussianOffsetsStep<FloatType,IntType>::SampleParams(IntType numberOfFeatures,
                                                            MatrixBufferTemplate<FloatType>& floatParams,
                                                            MatrixBufferTemplate<IntType>& intParams,
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
