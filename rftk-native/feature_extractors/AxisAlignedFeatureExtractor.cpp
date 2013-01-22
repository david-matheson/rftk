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
{
}

AxisAlignedFeatureExtractor::~AxisAlignedFeatureExtractor()
{}

int AxisAlignedFeatureExtractor::GetNumberOfFeatures() const
{
    boost::mt19937 gen( std::time(NULL) );
    boost::poisson_distribution<> poisson(static_cast<int>(mNumberOfFeatures));
    const int numberOfFeatures = mUsePoisson ? poisson(gen) : mNumberOfFeatures;
    return numberOfFeatures;
}

MatrixBufferFloat AxisAlignedFeatureExtractor::CreateFloatParams(const int numberOfFeatures) const
{
    MatrixBufferFloat floatParams = MatrixBufferFloat(numberOfFeatures, GetFloatParamsDim());
    return floatParams;
}

MatrixBufferInt AxisAlignedFeatureExtractor::CreateIntParams(const int numberOfFeatures) const
{
    MatrixBufferInt intParams = MatrixBufferInt(numberOfFeatures, GetIntParamsDim());
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

void AxisAlignedFeatureExtractor::Extract(  BufferCollection& data,
                                            const MatrixBufferInt& sampleIndices,
                                            const MatrixBufferInt& intFeatureParams,
                                            const MatrixBufferFloat& floatFeatureParams,
                                            MatrixBufferFloat& featureValuesOUT) const// #features X #samples
{
    // printf("AxisAlignedFeatureExtractor::Extract\n");
    ASSERT_ARG_DIM_1D(sampleIndices.GetN(), 1)
    ASSERT_ARG_DIM_1D(intFeatureParams.GetM(), floatFeatureParams.GetM())
    ASSERT( data.HasMatrixBufferFloat(X_FLOAT_DATA) )
    ASSERT_ARG_DIM_1D(data.GetMatrixBufferFloat(X_FLOAT_DATA).GetN(), mNumberOfComponents)

    const MatrixBufferFloat Xs = data.GetMatrixBufferFloat(X_FLOAT_DATA);

    const int numberOfSamples = sampleIndices.GetM();
    const int numberOfFeatures = intFeatureParams.GetM();
    const int numberOfDataPoints = Xs.GetM();

    // Create new results buffer if it's not the right dimensions
    if( featureValuesOUT.GetM() != numberOfSamples || featureValuesOUT.GetN() != numberOfFeatures )
    {
        featureValuesOUT = MatrixBufferFloat(numberOfSamples,numberOfFeatures);
    }

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