#include <cstdlib>
#include <time.h>

#include "assert_util.h"

#include "AxisAlignedFeatureExtractor.h"

AxisAlignedFeatureExtractor::AxisAlignedFeatureExtractor(int numberOfFeatures, int numberOfComponents)
: mNumberOfFeatures(numberOfFeatures)
, mNumberOfComponents(numberOfComponents)
{
    /* initialize random seed: */
    srand ( time(NULL) );
}

AxisAlignedFeatureExtractor::~AxisAlignedFeatureExtractor() 
{}

MatrixBufferFloat AxisAlignedFeatureExtractor::CreateFloatParams() const 
{
    MatrixBufferFloat floatParams = MatrixBufferFloat(mNumberOfFeatures, GetFloatParamsDim()); 
    return floatParams;
}

MatrixBufferInt AxisAlignedFeatureExtractor::CreateIntParams() const 
{
    MatrixBufferInt intParams = MatrixBufferInt(mNumberOfFeatures, GetIntParamsDim()); 
    for(int i=0; i<mNumberOfFeatures; i++)
    {
        intParams.Set(i, 0, GetUID());
        intParams.Set(i, 1, std::rand() % mNumberOfComponents);
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
                                            MatrixBufferFloat& featureValuesOUT) // #features X #samples
{
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
}