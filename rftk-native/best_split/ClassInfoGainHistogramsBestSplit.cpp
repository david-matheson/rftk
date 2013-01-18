#include <math.h>
#include <stdio.h>
#include <stdexcept>

#include "assert_util.h"
#include "ImgBuffer.h"
#include "ClassInfoGainHistogramsBestSplit.h"



// float calculateDiscreteEntropy(const float* classLabelCounts, const float* logClassLabelCounts, const int numberOfClasses, float totalCounts)
// {
//     const float inverseTotalCounts = 1.0f / totalCounts;
//     const float logTotalCounts = log(totalCounts);

//     float entropy = 0.0f;
//     for(int i=0; i<numberOfClasses && totalCounts > 0.0f; i++)
//     {
//         const float prob = inverseTotalCounts * classLabelCounts[i];
//         entropy -= prob * (logClassLabelCounts[i] - logTotalCounts);

//         // printf("calculateDiscreteEntropy prob=%0.2f entropy=%0.2f classLabelCounts[%d]=%0.2f\n", prob, entropy, i, classLabelCounts[i]);
//     }
//     // printf("calculateDiscreteEntropy entropy=%0.2f\n", entropy);

//     return entropy;
// }


ClassInfoGainHistogramsBestSplit::ClassInfoGainHistogramsBestSplit(int maxClass)
: mMaxClass(maxClass+1) //+1 is because 0 is also a valid class
{
}


int ClassInfoGainHistogramsBestSplit::GetYDim() const
{
    return mMaxClass;
}

ClassInfoGainHistogramsBestSplit::~ClassInfoGainHistogramsBestSplit()
{
}


void ClassInfoGainHistogramsBestSplit::BestSplits(   BufferCollection& data,
                                                        MatrixBufferFloat& impurityOut,
                                                        MatrixBufferFloat& thresholdOut,
                                                        MatrixBufferFloat& childCountsOut,
                                                        MatrixBufferFloat& leftYsOut,
                                                        MatrixBufferFloat& rightYsOut) const
{
    ASSERT( data.HasMatrixBufferInt(HISTOGRAM_LEFT) )
    ASSERT( data.HasMatrixBufferFloat(HISTOGRAM_RIGHT) )

    const ImgBufferFloat histogramLeft = data.GetImgBufferFloat(HISTOGRAM_LEFT);
    const ImgBufferFloat histogramRight = data.GetImgBufferFloat(HISTOGRAM_RIGHT);

    const int numberOfFeatures = histogramLeft.GetNumberOfImgs();

    // Create new results buffer if they're not the right dimensions
    if( impurityOut.GetM() != numberOfFeatures || impurityOut.GetN() != 1 )
    {
        impurityOut = MatrixBufferFloat(numberOfFeatures, 1);
    }
    if( thresholdOut.GetM() != numberOfFeatures || thresholdOut.GetN() != 1 )
    {
        thresholdOut = MatrixBufferFloat(numberOfFeatures, 1);
    }
    if( childCountsOut.GetM() != numberOfFeatures || childCountsOut.GetN() != 1 )
    {
        childCountsOut = MatrixBufferFloat(numberOfFeatures, 2);
    }
    if( leftYsOut.GetM() != numberOfFeatures || leftYsOut.GetN() != mMaxClass )
    {
        leftYsOut = MatrixBufferFloat(numberOfFeatures, mMaxClass);
    }
    if( rightYsOut.GetM() != numberOfFeatures || rightYsOut.GetN() != mMaxClass )
    {
        rightYsOut = MatrixBufferFloat(numberOfFeatures, mMaxClass);
    }


    // for(int t=0; t<histogramLeft.GetM(); t++)
    // {
    //     impurityOut.Set(testIndex, 0, bestGainInEntropy);
    //     thresholdOut.Set(testIndex, 0, bestThreshold);
    //     childCountsOut.Set(testIndex, 0, bestLeftWeight);
    //     childCountsOut.Set(testIndex, 1, bestRightWeight);

    //     for(int c=0; c<mMaxClass; c++)
    //     {
    //         leftYsOut.Set(testIndex, c, bestLeftClassLabelCounts[c] / bestLeftWeight);
    //         rightYsOut.Set(testIndex, c, bestRightClassLabelCounts[c] / bestRightWeight);
    //     }
    // }
}