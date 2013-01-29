#include <math.h>
#include <stdio.h>
#include <stdexcept>
#include <cfloat>
#include <cstdio>

#include "assert_util.h"
#include "Tensor3Buffer.h"
#include "ClassInfoGainHistogramsBestSplit.h"

float sum(const float* array, int len)
{
    float sum = 0.0f;
    for(int i=0; i<len; i++)
    {
        sum += array[i];
    }
    return sum;
}


float calculateDiscreteEntropy(const float* classHistogramCounts, int numberOfClasses)
{
    const float total = sum(classHistogramCounts, numberOfClasses);
    if ( total < FLT_EPSILON )
    {
        return 0.0f;
    }

    float entropy = 0.0f;
    for(int c=0; c<numberOfClasses; c++)
    {
        const float prob = classHistogramCounts[c] / total;
        entropy -= (prob > FLT_EPSILON) ? prob*log(prob) : 0.0f;
    }
    return entropy;
}


ClassInfoGainHistogramsBestSplit::ClassInfoGainHistogramsBestSplit(int numberOfClasses)
: mNumberOfClasses(numberOfClasses)
{
}

BestSplitI* ClassInfoGainHistogramsBestSplit::Clone() const
{
    return new ClassInfoGainHistogramsBestSplit(*this);
}


int ClassInfoGainHistogramsBestSplit::GetYDim() const
{
    return mNumberOfClasses;
}

ClassInfoGainHistogramsBestSplit::~ClassInfoGainHistogramsBestSplit()
{
}


void ClassInfoGainHistogramsBestSplit::BestSplits(   const BufferCollection& data,
                                                        Float32MatrixBuffer& impurityOut,
                                                        Float32MatrixBuffer& thresholdOut,
                                                        Float32MatrixBuffer& childCountsOut,
                                                        Float32MatrixBuffer& leftYsOut,
                                                        Float32MatrixBuffer& rightYsOut) const
{
    ASSERT( data.HasFloat32Tensor3Buffer(HISTOGRAM_LEFT) )
    ASSERT( data.HasFloat32Tensor3Buffer(HISTOGRAM_RIGHT) )
    ASSERT( data.HasFloat32MatrixBuffer(THRESHOLDS) )

    const Float32Tensor3Buffer& histogramLeft = data.GetFloat32Tensor3Buffer(HISTOGRAM_LEFT);
    const Float32Tensor3Buffer& histogramRight = data.GetFloat32Tensor3Buffer(HISTOGRAM_RIGHT);
    const Float32MatrixBuffer& thresholds = data.GetFloat32MatrixBuffer(THRESHOLDS);

    ASSERT_ARG_DIM_3D(  histogramLeft.GetL(), histogramLeft.GetM(), histogramLeft.GetN(),
                        histogramRight.GetL(), histogramRight.GetM(), histogramRight.GetN() )
    ASSERT_ARG_DIM_1D( histogramLeft.GetN(), mNumberOfClasses )

    const int numberOfFeatures = histogramLeft.GetL();

    // Create new results buffer if they're not the right dimensions
    if( impurityOut.GetM() != numberOfFeatures || impurityOut.GetN() != 1 )
    {
        impurityOut = Float32MatrixBuffer(numberOfFeatures, 1);
    }
    if( thresholdOut.GetM() != numberOfFeatures || thresholdOut.GetN() != 1 )
    {
        thresholdOut = Float32MatrixBuffer(numberOfFeatures, 1);
    }
    if( childCountsOut.GetM() != numberOfFeatures || childCountsOut.GetN() != 1 )
    {
        childCountsOut = Float32MatrixBuffer(numberOfFeatures, 2);
    }
    if( leftYsOut.GetM() != numberOfFeatures || leftYsOut.GetN() != mNumberOfClasses )
    {
        leftYsOut = Float32MatrixBuffer(numberOfFeatures, mNumberOfClasses);
    }
    if( rightYsOut.GetM() != numberOfFeatures || rightYsOut.GetN() != mNumberOfClasses )
    {
        rightYsOut = Float32MatrixBuffer(numberOfFeatures, mNumberOfClasses);
    }

    std::vector<float> initialClassLabelCounts(mNumberOfClasses);
    for(int c=0; c<mNumberOfClasses; c++)
    {
        initialClassLabelCounts[c] += histogramLeft.Get(0,0,c);
        initialClassLabelCounts[c] += histogramRight.Get(0,0,c);
    }
    const float totalWeight = sum(&initialClassLabelCounts[0], mNumberOfClasses);
    const float entropyStart = calculateDiscreteEntropy(&initialClassLabelCounts[0], mNumberOfClasses);

    for(int f=0; f<histogramLeft.GetL(); f++)
    {
        float bestGainInEntropy = FLT_MIN;

        for(int t=0; t<histogramLeft.GetM(); t++)
        {
            const float* left = histogramLeft.GetRowPtrUnsafe(f,t);
            const float leftWeight = sum(left, mNumberOfClasses);
            const float leftRatio = (leftWeight > FLT_EPSILON) ? leftWeight / totalWeight : 0.0f;
            const float leftEntropy = leftRatio * calculateDiscreteEntropy(left, mNumberOfClasses);

            const float* right = histogramRight.GetRowPtrUnsafe(f,t);
            const float rightWeight = sum(right, mNumberOfClasses);
            const float rightRatio = (rightWeight > FLT_EPSILON) ? rightWeight / totalWeight : 0.0f;
            const float rightEntropy = rightRatio * calculateDiscreteEntropy(right, mNumberOfClasses);

            if( (entropyStart - leftEntropy - rightEntropy) > bestGainInEntropy)
            {
                bestGainInEntropy = (entropyStart - leftEntropy - rightEntropy);
                // printf("New best entropy f=%d t=%d impurity=%0.2f leftWeight=%0.f rightWeight=%0.2f\n",  f, t, bestGainInEntropy, leftWeight, rightWeight);
                impurityOut.Set(f, 0, bestGainInEntropy);
                thresholdOut.Set(f, 0, thresholds.Get(f,t));
                childCountsOut.Set(f, 0, leftWeight);
                childCountsOut.Set(f, 1, rightWeight);

                for(int c=0; c<mNumberOfClasses; c++)
                {
                    leftYsOut.Set(f, c, left[c] / leftWeight);
                    rightYsOut.Set(f, c, right[c] / rightWeight);
                }
            }
        }
    }
}