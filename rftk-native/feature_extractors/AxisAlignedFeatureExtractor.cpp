#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>

#include <cstdlib>
#include <ctime>
#include <vector>

#include "assert_util.h"
#include "bootstrap.h"

#include "AxisAlignedFeatureExtractor.h"

AxisAlignedFeatureExtractor::AxisAlignedFeatureExtractor(int numberOfFeatures, int numberOfComponents, bool usePoisson)
: mNumberOfFeatures(numberOfFeatures)
, mNumberOfComponents(numberOfComponents)
, mUsePoisson(usePoisson)
, mGen( static_cast<unsigned int>(std::time(NULL)) )
{
}

AxisAlignedFeatureExtractor::~AxisAlignedFeatureExtractor()
{}

FeatureExtractorI* AxisAlignedFeatureExtractor::Clone() const
{
    return new AxisAlignedFeatureExtractor(*this);
}

int AxisAlignedFeatureExtractor::GetNumberOfFeatures() const
{
    int numberOfFeatures = mNumberOfFeatures;
    if(mUsePoisson)
    {
        boost::poisson_distribution<> poisson(static_cast<double>(mNumberOfFeatures));
        boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(mGen, poisson);
        numberOfFeatures = std::min(mNumberOfComponents, std::max(1, var_poisson())); //can't have more features than components
    }
    return numberOfFeatures;
}

Float32MatrixBuffer AxisAlignedFeatureExtractor::CreateFloatParams(const int numberOfFeatures) const
{
    Float32MatrixBuffer floatParams = Float32MatrixBuffer(numberOfFeatures, GetFloatParamsDim());
    return floatParams;
}

Int32MatrixBuffer AxisAlignedFeatureExtractor::CreateIntParams(const int numberOfFeatures) const
{
    Int32MatrixBuffer intParams = Int32MatrixBuffer(numberOfFeatures, GetIntParamsDim());
    std::vector<int> axes(numberOfFeatures);
    sampleIndicesWithOutReplacement(&axes[0], numberOfFeatures, mNumberOfComponents);

    for(int i=0; i<numberOfFeatures; i++)
    {
        intParams.Set(i, 0, GetUID());
        intParams.Set(i, 1, axes[i]);
    }
    return intParams;
}

//Includes the threshold
int AxisAlignedFeatureExtractor::GetFloatParamsDim() const { return 1; }

//Includes FeatureExtractor id
int AxisAlignedFeatureExtractor::GetIntParamsDim() const { return 2; }

void AxisAlignedFeatureExtractor::Extract(  const BufferCollection& data,
                                            const Int32MatrixBuffer& sampleIndices,
                                            const Int32MatrixBuffer& intFeatureParams,
                                            const Float32MatrixBuffer& floatFeatureParams,
                                            Float32MatrixBuffer& featureValuesOUT) const// #features X #samples
{
    // printf("AxisAlignedFeatureExtractor::Extract\n");
    ASSERT_ARG_DIM_1D(sampleIndices.GetN(), 1)
    ASSERT_ARG_DIM_1D(intFeatureParams.GetM(), floatFeatureParams.GetM())
    ASSERT( data.HasFloat32MatrixBuffer(X_FLOAT_DATA) )
    ASSERT_ARG_DIM_1D(data.GetFloat32MatrixBuffer(X_FLOAT_DATA).GetN(), mNumberOfComponents)

    const Float32MatrixBuffer& Xs = data.GetFloat32MatrixBuffer(X_FLOAT_DATA);

    const int numberOfSamples = sampleIndices.GetM();
    const int numberOfFeatures = intFeatureParams.GetM();
    const int numberOfDataPoints = Xs.GetM();

    featureValuesOUT.Resize(numberOfSamples, numberOfFeatures);

    // Extract component features
    for(int s=0; s<numberOfSamples; s++)
    {
        const int index = sampleIndices.Get(s, 0);
        ASSERT_VALID_RANGE(index, 0, numberOfDataPoints)

        for(int t=0; t<numberOfFeatures; t++)
        {
            const int componentId = intFeatureParams.Get(t, 1);
            const float value =  Xs.Get(index, componentId);
            featureValuesOUT.Set(s, t, value);
        }
    }
    // printf("AxisAlignedFeatureExtractor::Extract End\n");
}