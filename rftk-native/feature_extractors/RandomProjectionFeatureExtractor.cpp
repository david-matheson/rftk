#include <boost/random/poisson_distribution.hpp>
#include <boost/random/uniform_real.hpp>

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <vector>

#include "assert_util.h"
#include "bootstrap.h"

#include "RandomProjectionFeatureExtractor.h"

RandomProjectionFeatureExtractor::RandomProjectionFeatureExtractor( int numberOfFeatures,
                                                                    int numberOfComponents,
                                                                    int numberOfComponentsInSubspace,
                                                                    bool usePoisson )
: mNumberOfFeatures(numberOfFeatures)
, mNumberOfComponents(numberOfComponents)
, mNumberOfComponentsInSubspace(numberOfComponentsInSubspace)
, mUsePoisson(usePoisson)
, mGen( static_cast<unsigned int>(std::time(NULL)) )
{
}

RandomProjectionFeatureExtractor::~RandomProjectionFeatureExtractor()
{}

int RandomProjectionFeatureExtractor::GetNumberOfFeatures() const
{
    int numberOfFeatures = mNumberOfFeatures;
    if(mUsePoisson)
    {
        boost::poisson_distribution<> poisson(static_cast<double>(mNumberOfFeatures));
        boost::variate_generator<boost::mt19937&,boost::poisson_distribution<> > var_poisson(mGen, poisson);
        numberOfFeatures = std::max(1, var_poisson());
    }
    return numberOfFeatures;
}

MatrixBufferFloat RandomProjectionFeatureExtractor::CreateFloatParams(const int numberOfFeatures) const
{
    MatrixBufferFloat floatParams = MatrixBufferFloat(numberOfFeatures, GetFloatParamsDim());

    boost::uniform_real<float> uniform(-1.0f, 1.0f);
    boost::variate_generator<boost::mt19937&,boost::uniform_real<float> > var_uniform_float(mGen, uniform);

    for(int i=0; i<numberOfFeatures; i++)
    {
        for(int c=0; c<mNumberOfComponentsInSubspace; c++)
        {
            const float componentProjection = var_uniform_float();
            floatParams.Set(i, c+2, componentProjection);
        }
    }
    return floatParams;
}

MatrixBufferInt RandomProjectionFeatureExtractor::CreateIntParams(const int numberOfFeatures) const
{
    MatrixBufferInt intParams = MatrixBufferInt(numberOfFeatures, GetIntParamsDim());
    for(int i=0; i<numberOfFeatures; i++)
    {
        // intParams[0,:] is reserved for feature type
        intParams.Set(i, 0, GetUID());
        intParams.Set(i, 1, mNumberOfComponentsInSubspace);

        std::vector<int> subspace(mNumberOfComponentsInSubspace);
        sampleIndicesWithOutReplacement(&subspace[0], mNumberOfComponentsInSubspace, mNumberOfComponents);
        for(int c=0; c<mNumberOfComponentsInSubspace; c++)
        {
            intParams.Set(i, c+2, subspace[c]);
        }
    }
    return intParams;
}

//Includes the threshold
int RandomProjectionFeatureExtractor::GetFloatParamsDim() const { return mNumberOfComponentsInSubspace + 2; }

//Includes FeatureExtractor id
int RandomProjectionFeatureExtractor::GetIntParamsDim() const { return mNumberOfComponentsInSubspace + 2; }

void RandomProjectionFeatureExtractor::Extract(  BufferCollection& data,
                                            const MatrixBufferInt& sampleIndices,
                                            const MatrixBufferInt& intFeatureParams,
                                            const MatrixBufferFloat& floatFeatureParams,
                                            MatrixBufferFloat& featureValuesOUT) const// #features X #samples
{
    // printf("RandomProjectionFeatureExtractor::Extract\n");
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

        for(int f=0; f<numberOfFeatures; f++)
        {
            float projectionValue = 0.0f;
            const int numberOfComponentsInProjection = intFeatureParams.Get(f, 1);
            for(int c=2; c<numberOfComponentsInProjection+2; c++)
            {
                // printf("FeatureExtractorStandardProjection %d %d %d %d\n", numberOfComponentsInProjection, s, t, p );
                const int componentId = intFeatureParams.Get(f, c);
                const float componentProjection = floatFeatureParams.Get(f, c);
                projectionValue += Xs.Get(index, componentId) * componentProjection;
            }
            featureValuesOUT.Set(s, f, projectionValue);
        }
    }
    // printf("RandomProjectionFeatureExtractor::Extract End\n");
}