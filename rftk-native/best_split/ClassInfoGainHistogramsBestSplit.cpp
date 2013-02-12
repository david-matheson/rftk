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


ClassInfoGainHistogramsBestSplit::ClassInfoGainHistogramsBestSplit(int numberOfClasses,
                                      const std::string& leftImpurityHistrogramBufferName,
                                      const std::string& rightImpurityHistrogramBufferName,
                                      const std::string& leftYsHistogramBufferName,
                                      const std::string& rightYsHistrogramBufferName)
: mNumberOfClasses(numberOfClasses)
, mLeftImpurityHistrogramBufferName(leftImpurityHistrogramBufferName)
, mRightImpurityHistrogramBufferName(rightImpurityHistrogramBufferName)
, mLeftYsHistogramBufferName(leftYsHistogramBufferName)
, mRightYsHistogramBufferName(rightYsHistrogramBufferName)
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
                                                        Float32VectorBuffer& impurityOut,
                                                        Float32VectorBuffer& thresholdOut,
                                                        Float32MatrixBuffer& childCountsOut,
                                                        Float32MatrixBuffer& leftYsOut,
                                                        Float32MatrixBuffer& rightYsOut) const
{
    ASSERT( data.HasFloat32Tensor3Buffer(mLeftImpurityHistrogramBufferName) )
    ASSERT( data.HasFloat32Tensor3Buffer(mRightImpurityHistrogramBufferName) )
    ASSERT( data.HasFloat32Tensor3Buffer(mLeftYsHistogramBufferName) )
    ASSERT( data.HasFloat32Tensor3Buffer(mRightYsHistogramBufferName) )
    ASSERT( data.HasFloat32MatrixBuffer(THRESHOLDS) )
    ASSERT( data.HasInt32VectorBuffer(THRESHOLD_COUNTS) )

    const Float32Tensor3Buffer& impurityHistogramLeft = data.GetFloat32Tensor3Buffer(mLeftImpurityHistrogramBufferName);
    const Float32Tensor3Buffer& impurityHistogramRight = data.GetFloat32Tensor3Buffer(mRightImpurityHistrogramBufferName);
    const Float32Tensor3Buffer& ysHistogramLeft = data.GetFloat32Tensor3Buffer(mLeftYsHistogramBufferName);
    const Float32Tensor3Buffer& ysHistogramRight = data.GetFloat32Tensor3Buffer(mRightYsHistogramBufferName);
    const Float32MatrixBuffer& thresholds = data.GetFloat32MatrixBuffer(THRESHOLDS);
    const Int32VectorBuffer& thresholdCounts = data.GetInt32VectorBuffer(THRESHOLD_COUNTS);

    ASSERT_ARG_DIM_3D(  impurityHistogramLeft.GetL(), impurityHistogramLeft.GetM(), impurityHistogramLeft.GetN(),
                        impurityHistogramRight.GetL(), impurityHistogramRight.GetM(), impurityHistogramRight.GetN() )
    ASSERT_ARG_DIM_3D(  impurityHistogramLeft.GetL(), impurityHistogramLeft.GetM(), impurityHistogramLeft.GetN(),
                        ysHistogramLeft.GetL(), ysHistogramLeft.GetM(), ysHistogramLeft.GetN() )
    ASSERT_ARG_DIM_3D(  impurityHistogramLeft.GetL(), impurityHistogramLeft.GetM(), impurityHistogramLeft.GetN(),
                        ysHistogramRight.GetL(), ysHistogramRight.GetM(), ysHistogramRight.GetN() )
    ASSERT_ARG_DIM_1D( impurityHistogramLeft.GetN(), mNumberOfClasses )
    ASSERT_ARG_DIM_1D( impurityHistogramLeft.GetL(), thresholds.GetM() )
    ASSERT_ARG_DIM_1D( impurityHistogramLeft.GetM(), thresholds.GetN() )
    ASSERT_ARG_DIM_1D( thresholds.GetM(), thresholdCounts.GetN() )

    const int numberOfFeatures = impurityHistogramLeft.GetL();

    // Create new results buffer if they're not the right dimensions
    impurityOut.Resize(numberOfFeatures);
    thresholdOut.Resize(numberOfFeatures);
    childCountsOut.Resize(numberOfFeatures, 2);
    leftYsOut.Resize(numberOfFeatures, mNumberOfClasses);
    rightYsOut.Resize(numberOfFeatures, mNumberOfClasses);

    std::vector<float> initialClassLabelCounts(mNumberOfClasses);
    for(int c=0; c<mNumberOfClasses; c++)
    {
        initialClassLabelCounts[c] += impurityHistogramLeft.Get(0,0,c);
        initialClassLabelCounts[c] += impurityHistogramRight.Get(0,0,c);
    }
    const float totalWeight = sum(&initialClassLabelCounts[0], mNumberOfClasses);
    const float entropyStart = calculateDiscreteEntropy(&initialClassLabelCounts[0], mNumberOfClasses);

    for(int f=0; f<impurityHistogramLeft.GetL(); f++)
    {
        float bestGainInEntropy = FLT_MIN;

        for(int t=0; t<thresholdCounts.Get(f); t++)
        {
            const float* left = impurityHistogramLeft.GetRowPtrUnsafe(f,t);
            const float leftWeight = sum(left, mNumberOfClasses);
            const float leftRatio = (leftWeight > FLT_EPSILON) ? leftWeight / totalWeight : 0.0f;
            const float leftEntropy = leftRatio * calculateDiscreteEntropy(left, mNumberOfClasses);

            const float* right = impurityHistogramRight.GetRowPtrUnsafe(f,t);
            const float rightWeight = sum(right, mNumberOfClasses);
            const float rightRatio = (rightWeight > FLT_EPSILON) ? rightWeight / totalWeight : 0.0f;
            const float rightEntropy = rightRatio * calculateDiscreteEntropy(right, mNumberOfClasses);

            if( (entropyStart - leftEntropy - rightEntropy) > bestGainInEntropy)
            {
                bestGainInEntropy = (entropyStart - leftEntropy - rightEntropy);
                // printf("New best entropy f=%d t=%d impurity=%0.2f leftWeight=%0.f rightWeight=%0.2f\n",  f, t, bestGainInEntropy, leftWeight, rightWeight);
                impurityOut.Set(f, bestGainInEntropy);
                thresholdOut.Set(f, thresholds.Get(f,t));

                const float* leftYs = ysHistogramLeft.GetRowPtrUnsafe(f,t);
                const float leftYsTotal = sum(leftYs, mNumberOfClasses);
                const float* rightYs = ysHistogramRight.GetRowPtrUnsafe(f,t);
                const float rightYsTotal = sum(rightYs, mNumberOfClasses);
                for(int c=0; c<mNumberOfClasses; c++)
                {
                    leftYsOut.Set(f, c, leftYs[c] / leftYsTotal);
                    rightYsOut.Set(f, c, rightYs[c] / rightYsTotal);
                }

                childCountsOut.Set(f, 0, leftYsTotal);
                childCountsOut.Set(f, 1, rightYsTotal);
            }
        }
    }
}