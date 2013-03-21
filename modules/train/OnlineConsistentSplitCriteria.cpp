#include <stdio.h>
#include <math.h>
#include "float.h"

#include "asserts/asserts.h"
#include "OnlineConsistentSplitCriteria.h"


OnlineConsistentSplitCriteria::OnlineConsistentSplitCriteria(float growthRate,
                                                            float minImpurity,
                                                            float minNumberOfSamplesFirstSplit,
                                                            float maxNumberOfSamplesFirstSplit,
                                                            int maxDepth)
: mGrowthRate(growthRate)
, mMinImpurity(minImpurity)
, mMinNumberOfSamplesFirstSplit(minNumberOfSamplesFirstSplit)
, mMaxNumberOfSamplesFirstSplit(maxNumberOfSamplesFirstSplit)
, mMaxDepth(maxDepth)
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

    const float maxNumberOfSamples = mMaxNumberOfSamplesFirstSplit * pow(mGrowthRate, treeDepth);

    const int bestSplit = BestSplit(treeDepth, impurityValues, childCounts);
    const bool shouldSplit = ((bestSplit >= 0) &&
                                (treeDepth < mMaxDepth) &&
                                ((impurityValues.Get(bestSplit) > mMinImpurity)
                                    || (childCounts.Get(bestSplit,0) + childCounts.Get(bestSplit,1) > maxNumberOfSamples)));

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

    int maxIndex = -1;
    float maxImpurity = FLT_MIN;
    for(int i=0; i<impurityValues.GetN(); i++)
    {
        const float leftCounts = childCounts.Get(i, 0);
        const float rightCounts = childCounts.Get(i, 1);

        bool isBest = ((impurityValues.Get(i) > maxImpurity)
                && (leftCounts > minNumberOfSamples)
                && (rightCounts > minNumberOfSamples));
        if (isBest)
        {
            maxIndex = i;
            maxImpurity = impurityValues.Get(i);
        }
    }

    return maxIndex;
}

int OnlineConsistentSplitCriteria::MinTotalSamples( int treeDepth ) const
{
    const float minNumberOfSamples = 2.0f * mMinNumberOfSamplesFirstSplit * pow(mGrowthRate, treeDepth);
    return static_cast<int>(minNumberOfSamples);
}