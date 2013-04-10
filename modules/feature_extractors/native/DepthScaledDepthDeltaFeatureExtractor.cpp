#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

#include <cstdlib>
#include <ctime>
#include <vector>

#include <asserts.h>
#include <bootstrap.h>
#include <ImgFeatures.h>

#include "DepthScaledDepthDeltaFeatureExtractor.h"

DepthScaledDepthDeltaFeatureExtractor::DepthScaledDepthDeltaFeatureExtractor(float ux, float uy, float vx, float vy, int numberOfFeatures, bool usePoisson )
: mUx(ux)
, mUy(uy)
, mVx(vx)
, mVy(vy)
, mNumberOfFeatures(numberOfFeatures)
, mUsePoisson(usePoisson)
, mGen( static_cast<unsigned int>(std::time(NULL)) )
{
}

DepthScaledDepthDeltaFeatureExtractor::~DepthScaledDepthDeltaFeatureExtractor()
{}

FeatureExtractorI* DepthScaledDepthDeltaFeatureExtractor::Clone() const
{
    return new DepthScaledDepthDeltaFeatureExtractor(*this);
}

int DepthScaledDepthDeltaFeatureExtractor::GetNumberOfFeatures() const
{
    int numberOfFeatures = mNumberOfFeatures;
    if(mUsePoisson)
    {
        boost::poisson_distribution<> poisson(static_cast<double>(mNumberOfFeatures));
        boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(mGen, poisson);
        numberOfFeatures = std::max(1, var_poisson()); //can't have more features than components
    }
    return numberOfFeatures;
}

Float32MatrixBuffer DepthScaledDepthDeltaFeatureExtractor::CreateFloatParams(const int numberOfFeatures) const
{
    Float32MatrixBuffer floatParams = Float32MatrixBuffer(numberOfFeatures, GetFloatParamsDim());

    boost::normal_distribution<> ux_normal(0.0, mUx);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_ux_normal(mGen, ux_normal);
    boost::normal_distribution<> uy_normal(0.0, mUy);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_uy_normal(mGen, uy_normal);
    boost::normal_distribution<> vx_normal(0.0, mVx);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_vx_normal(mGen, vx_normal);
    boost::normal_distribution<> vy_normal(0.0, mVy);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_vy_normal(mGen, vy_normal);

    for(int i=0; i<numberOfFeatures; i++)
    {
        floatParams.Set(i, 1, var_ux_normal());
        floatParams.Set(i, 2, var_uy_normal());
        floatParams.Set(i, 3, var_vx_normal());
        floatParams.Set(i, 4, var_vy_normal());
    }
    return floatParams;
}

Int32MatrixBuffer DepthScaledDepthDeltaFeatureExtractor::CreateIntParams(const int numberOfFeatures) const
{
    Int32MatrixBuffer intParams = Int32MatrixBuffer(numberOfFeatures, GetIntParamsDim());
    for(int i=0; i<numberOfFeatures; i++)
    {
        intParams.Set(i, 0, GetUID());
    }
    return intParams;
}

//Includes the threshold
int DepthScaledDepthDeltaFeatureExtractor::GetFloatParamsDim() const { return 5; }

//Includes FeatureExtractor id
int DepthScaledDepthDeltaFeatureExtractor::GetIntParamsDim() const { return 1; }

void DepthScaledDepthDeltaFeatureExtractor::Extract(  const BufferCollection& data,
                                            const Int32VectorBuffer& sampleIndices,
                                            const Int32MatrixBuffer& intFeatureParams,
                                            const Float32MatrixBuffer& floatFeatureParams,
                                            Float32MatrixBuffer& featureValuesOUT) const// #features X #samples
{
    ASSERT( data.HasFloat32Tensor3Buffer(DEPTH_IMAGES) )
    const Float32Tensor3Buffer& depths = data.GetFloat32Tensor3Buffer(DEPTH_IMAGES);
    ASSERT( data.HasInt32MatrixBuffer(PIXEL_INDICES) )
    const Int32MatrixBuffer& pixelIndices = data.GetInt32MatrixBuffer(PIXEL_INDICES);
    ASSERT( data.HasFloat32MatrixBuffer(OFFSET_SCALES) )
    const Float32MatrixBuffer& offsetScales = data.GetFloat32MatrixBuffer(OFFSET_SCALES);


    const int numberOfSamples = sampleIndices.GetN();
    const int numberOfFeatures = intFeatureParams.GetM();

    featureValuesOUT.Resize(numberOfSamples, numberOfFeatures);

    // Extract component features
    for(int s=0; s<numberOfSamples; s++)
    {
        const int index = sampleIndices.Get(s);
        const int imgIndex = pixelIndices.Get(index, 0);
        const int pixelM = pixelIndices.Get(index, 1);
        const int pixelN = pixelIndices.Get(index, 2);

        const float scaleM = offsetScales.Get(index, 0);
        const float scaleN = offsetScales.Get(index, 1);

        for(int t=0; t<numberOfFeatures; t++)
        {
            const float um = floatFeatureParams.Get(t,1);
            const float un = floatFeatureParams.Get(t,2);
            const float vm = floatFeatureParams.Get(t,3);
            const float vn = floatFeatureParams.Get(t,4);
            const float delta = pixelDepthDelta(depths, imgIndex, pixelM, pixelN, scaleM*um, scaleN*un, scaleM*vm, scaleN*vn);
            featureValuesOUT.Set(s, t, delta);
        }
    }
}