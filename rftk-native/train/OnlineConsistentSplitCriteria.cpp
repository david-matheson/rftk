#include "float.h"
#include <math.h>

#include "assert_util.h"
#include "OnlineConsistentSplitCriteria.h"


OnlineConsistentSplitCriteria::OnlineConsistentSplitCriteria(float growthRate,
                                                            float minImpurity,
                                                            float minNumberOfSamplesFirstSplit,
                                                            float maxNumberOfSamplesFirstSplit)
: mGrowthRate(growthRate)
, mMinImpurity(minImpurity)
, mMinNumberOfSamplesFirstSplit(minNumberOfSamplesFirstSplit)
, mMaxNumberOfSamplesFirstSplit(maxNumberOfSamplesFirstSplit)
{}

OnlineConsistentSplitCriteria::~OnlineConsistentSplitCriteria()
{
}

SPLT_CRITERIA OnlineConsistentSplitCriteria::ShouldSplit(int treeDepth,
                                                const MatrixBufferFloat& impurityValues,
                                                const MatrixBufferFloat& childCounts) const
{
    ASSERT_ARG_DIM_1D(impurityValues.GetM(), childCounts.GetM())
    ASSERT_ARG_DIM_1D(impurityValues.GetN(), 1)
    ASSERT_ARG_DIM_1D(childCounts.GetN(), 2)

    const int bestSplit = BestSplit(treeDepth, impurityValues, childCounts);
    const bool shouldSplit = ((bestSplit >= 0) &&
                                ((impurityValues.Get(bestSplit,0) > mMinImpurity)
                                    || (childCounts.Get(bestSplit,0) + childCounts.Get(bestSplit,1) > mMaxNumberOfSamplesFirstSplit)));

    return (shouldSplit) ?
                 SPLT_CRITERIA_READY_TO_SPLIT : SPLT_CRITERIA_MORE_DATA_REQUIRED;
}

int OnlineConsistentSplitCriteria::BestSplit(int treeDepth,
                                    const MatrixBufferFloat& impurityValues,
                                    const MatrixBufferFloat& childCounts ) const
{
    ASSERT_ARG_DIM_1D(impurityValues.GetM(), childCounts.GetM())
    ASSERT_ARG_DIM_1D(impurityValues.GetN(), 1)
    ASSERT_ARG_DIM_1D(childCounts.GetN(), 2)

    const float minNumberOfSamples = mMinNumberOfSamplesFirstSplit * pow(mGrowthRate, treeDepth);
    const float maxNumberOfSamples = mMaxNumberOfSamplesFirstSplit * pow(mGrowthRate, treeDepth);

    int maxIndex = -1;
    float maxImpurity = FLT_MIN;
    float maxCounts = 0;
    for(int i=0; i<impurityValues.GetM(); i++)
    {
        const float leftCounts = childCounts.Get(i, 0);
        const float rightCounts = childCounts.Get(i, 1);
        maxCounts = ((leftCounts + rightCounts) > maxCounts) ? (leftCounts + rightCounts) : maxCounts;

        bool isBest = ((impurityValues.Get(i,0) > maxImpurity)
                && (leftCounts > minNumberOfSamples)
                && (rightCounts > minNumberOfSamples));
        if (isBest)
        {
            maxIndex = i;
            maxImpurity = impurityValues.Get(i,0);
        }
    }

    // Force a split if past the maximum number of samples
    if(maxIndex == -1 && maxCounts > mMaxNumberOfSamplesFirstSplit)
    {
        for(int i=0; i<impurityValues.GetM(); i++)
        {
            const float leftCounts = childCounts.Get(i, 0);
            const float rightCounts = childCounts.Get(i, 1);

            if (leftCounts + rightCounts > mMaxNumberOfSamplesFirstSplit)
            {
                maxIndex = i;
            }
        }
    }
    return maxIndex;
}
