#include "assert_util.h"

#include "AxisAlignedFeatureExtractor.h"

AxisAlignedFeatureExtractor::AxisAlignedFeatureExtractor()
{}

void AxisAlignedFeatureExtractor::Extract(  BufferCollection& bufferCollection,
                                            const MatrixBufferInt& sampleIndices,
                                            const MatrixBufferInt& intFeatureParams,
                                            const MatrixBufferFloat& floatFeatureParams,
                                            MatrixBufferFloat& featureValuesOUT) // #features X #samples
{
    ASSERT_ARG_DIM_1D(sampleIndices.GetN(), 1)
    ASSERT_ARG_DIM_1D(intFeatureParams.GetM(), floatFeatureParams.GetM())
    ASSERT( bufferCollection.HasMatrixBufferFloat("X") )

    const MatrixBufferFloat Xs = bufferCollection.GetMatrixBufferFloat("X");

    const int numberOfSamples = sampleIndices.GetM();
    const int numberOfFeatures = intFeatureParams.GetM();
    const int numberOfDataPoints = Xs.GetM();

    // Create new results buffer if it's not the right dimensions
    if( featureValuesOUT.GetM() != numberOfFeatures || featureValuesOUT.GetN() != numberOfSamples )
    {
        featureValuesOUT = MatrixBufferFloat(numberOfFeatures, numberOfSamples);
    }

    // Extract component features
    for(int s=0; s<numberOfSamples; s++)
    {
        const int index = sampleIndices.Get(s, 0);
        ASSERT_VALID_RANGE(index, 0, numberOfDataPoints)

        for(int t=0; t<numberOfFeatures; t++)
        {
            const int componentId = intFeatureParams.Get(t, 0);
            const float value =  Xs.Get(index, componentId);
            featureValuesOUT.Set(t, s, value);
        }
    }
}