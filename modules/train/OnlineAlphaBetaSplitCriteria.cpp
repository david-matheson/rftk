#include "float.h"

#include "assert_util.h"
#include "OnlineAlphaBetaSplitCriteria.h"


OnlineAlphaBetaSplitCriteria::OnlineAlphaBetaSplitCriteria( int maxDepth,
                                                            float minImpurity,
                                                            float minNumberOfSamples )
: mMaxDepth(maxDepth)
, mMinImpurity(minImpurity)
, mMinNumberOfSamples(minNumberOfSamples)
{}

OnlineAlphaBetaSplitCriteria::~OnlineAlphaBetaSplitCriteria()
{
}

SplitCriteriaI* OnlineAlphaBetaSplitCriteria::Clone() const
{
    return new OnlineAlphaBetaSplitCriteria(*this);
}

SPLT_CRITERIA OnlineAlphaBetaSplitCriteria::ShouldSplit(int treeDepth,
                                                const Float32VectorBuffer& impurityValues,
                                                const Float32MatrixBuffer& childCounts) const
{
    ASSERT_ARG_DIM_1D(impurityValues.GetN(), childCounts.GetM())
    ASSERT_ARG_DIM_1D(childCounts.GetN(), 2)

    const int bestSplit = BestSplit(treeDepth, impurityValues, childCounts);
    const bool shouldSplit = (bestSplit >= 0
                            && treeDepth < mMaxDepth
                            && impurityValues.Get(bestSplit) > mMinImpurity);

    return (shouldSplit) ?
                 SPLT_CRITERIA_READY_TO_SPLIT : SPLT_CRITERIA_MORE_DATA_REQUIRED;
}

int OnlineAlphaBetaSplitCriteria::BestSplit(int treeDepth,
                                    const Float32VectorBuffer& impurityValues,
                                    const Float32MatrixBuffer& childCounts ) const
{
    UNUSED_PARAM(treeDepth) // Suppress the unused warning because treeDepth is part of the interface
    ASSERT_ARG_DIM_1D(impurityValues.GetN(), childCounts.GetM())
    ASSERT_ARG_DIM_1D(impurityValues.GetN(), 1)
    ASSERT_ARG_DIM_1D(childCounts.GetN(), 2)

    int maxIndex = -1;
    float maxImpurity = FLT_MIN;
    for(int i=0; i<impurityValues.GetN(); i++)
    {
        const float leftCounts = childCounts.Get(i, 0);
        const float rightCounts = childCounts.Get(i, 1);

        bool isBest = ((impurityValues.Get(i) > maxImpurity)
                && ((leftCounts+rightCounts) > mMinNumberOfSamples));
        if (isBest)
        {
            maxIndex = i;
            maxImpurity = impurityValues.Get(i);
        }
    }
    return maxIndex;
}

int OnlineAlphaBetaSplitCriteria::MinTotalSamples( int treeDepth ) const
{
    UNUSED_PARAM(treeDepth) // Suppress the unused warning because treeDepth is part of the interface
    return mMinNumberOfSamples;
}