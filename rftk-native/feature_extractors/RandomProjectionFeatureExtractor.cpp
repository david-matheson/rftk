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

FeatureExtractorI* RandomProjectionFeatureExtractor::Clone() const
{
    return new RandomProjectionFeatureExtractor(*this);
}

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

Float32MatrixBuffer RandomProjectionFeatureExtractor::CreateFloatParams(const int numberOfFeatures) const
{
    Float32MatrixBuffer floatParams = Float32MatrixBuffer(numberOfFeatures, GetFloatParamsDim());

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

Int32MatrixBuffer RandomProjectionFeatureExtractor::CreateIntParams(const int numberOfFeatures) const
{
    Int32MatrixBuffer intParams = Int32MatrixBuffer(numberOfFeatures, GetIntParamsDim());
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

void RandomProjectionFeatureExtractor::Extract(  const BufferCollection& data,
                                            const Int32VectorBuffer& sampleIndices,
                                            const Int32MatrixBuffer& intFeatureParams,
                                            const Float32MatrixBuffer& floatFeatureParams,
                                            Float32MatrixBuffer& featureValuesOUT) const// #features X #samples
{
    // printf("RandomProjectionFeatureExtractor::Extract\n");
    ASSERT_ARG_DIM_1D(intFeatureParams.GetM(), floatFeatureParams.GetM())
    ASSERT( data.HasFloat32MatrixBuffer(X_FLOAT_DATA) )
    ASSERT_ARG_DIM_1D(data.GetFloat32MatrixBuffer(X_FLOAT_DATA).GetN(), mNumberOfComponents)

    const Float32MatrixBuffer& Xs = data.GetFloat32MatrixBuffer(X_FLOAT_DATA);

    const int numberOfSamples = sampleIndices.GetN();
    const int numberOfFeatures = intFeatureParams.GetM();
    const int numberOfDataPoints = Xs.GetM();

    featureValuesOUT.Resize(numberOfSamples, numberOfFeatures);

    // Extract component features
    for(int s=0; s<numberOfSamples; s++)
    {
        const int index = sampleIndices.Get(s);
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
