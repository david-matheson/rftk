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

SplitCriteriaI* OnlineConsistentSplitCriteria::Clone() const
{
    return new OnlineConsistentSplitCriteria(*this);
}

SPLT_CRITERIA OnlineConsistentSplitCriteria::ShouldSplit(int treeDepth,
                                                const Float32VectorBuffer& impurityValues,
                                                const Float32MatrixBuffer& childCounts) const
{
    ASSERT_ARG_DIM_1D(impurityValues.GetN(), childCounts.GetM())
    ASSERT_ARG_DIM_1D(childCounts.GetN(), 2)

    const int bestSplit = BestSplit(treeDepth, impurityValues, childCounts);
    const bool shouldSplit = ((bestSplit >= 0) &&
                                ((impurityValues.Get(bestSplit) > mMinImpurity)
                                    || (childCounts.Get(bestSplit,0) + childCounts.Get(bestSplit,1) > mMaxNumberOfSamplesFirstSplit)));

    return (shouldSplit) ?
                 SPLT_CRITERIA_READY_TO_SPLIT : SPLT_CRITERIA_MORE_DATA_REQUIRED;
}

int OnlineConsistentSplitCriteria::BestSplit(int treeDepth,
                                    const Float32VectorBuffer& impurityValues,
                                    const Float32MatrixBuffer& childCounts ) const
{
    ASSERT_ARG_DIM_1D(impurityValues.GetN(), childCounts.GetM())
    ASSERT_ARG_DIM_1D(childCounts.GetN(), 2)

    const float minNumberOfSamples = mMinNumberOfSamplesFirstSplit * pow(mGrowthRate, treeDepth);
    const float maxNumberOfSamples = mMaxNumberOfSamplesFirstSplit * pow(mGrowthRate, treeDepth);

    int maxIndex = -1;
    float maxImpurity = FLT_MIN;
    float maxCounts = 0;
    for(int i=0; i<impurityValues.GetN(); i++)
    {
        const float leftCounts = childCounts.Get(i, 0);
        const float rightCounts = childCounts.Get(i, 1);
        maxCounts = ((leftCounts + rightCounts) > maxCounts) ? (leftCounts + rightCounts) : maxCounts;

        bool isBest = ((impurityValues.Get(i) > maxImpurity)
                && (leftCounts > minNumberOfSamples)
                && (rightCounts > minNumberOfSamples));
        if (isBest)
        {
            maxIndex = i;
            maxImpurity = impurityValues.Get(i);
        }
    }

    // Force a split if past the maximum number of samples
    if(maxIndex == -1 && maxCounts > maxNumberOfSamples)
    {
        for(int i=0; i<impurityValues.GetN(); i++)
        {
            const float leftCounts = childCounts.Get(i, 0);
            const float rightCounts = childCounts.Get(i, 1);

            if (leftCounts + rightCounts > maxNumberOfSamples)
            {
                maxIndex = i;
            }
        }
    }
    return maxIndex;
}
