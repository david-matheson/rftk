#include "float.h"

#include "assert_util.h"
#include "OfflineSplitCriteria.h"

OfflineSplitCriteria::OfflineSplitCriteria( int maxDepth, 
                                            float minImpurity, 
                                            float minSampleCounts,
                                            float minChildSampleCounts)
: mMaxDepth(maxDepth)
, mMinImpurity(minImpurity)
, mMinSampleCounts(minSampleCounts)
, mMinChildSampleCounts(minChildSampleCounts)
{}

OfflineSplitCriteria::~OfflineSplitCriteria()
{
}

SplitCriteriaI* OfflineSplitCriteria::Clone() const
{
    return new OfflineSplitCriteria(*this);
}

SPLT_CRITERIA OfflineSplitCriteria::ShouldSplit(int treeDepth,
                                                const Float32MatrixBuffer& impurityValues,
                                                const Float32MatrixBuffer& childCounts) const
{
    ASSERT_ARG_DIM_1D(impurityValues.GetM(), childCounts.GetM())
    ASSERT_ARG_DIM_1D(impurityValues.GetN(), 1)
    ASSERT_ARG_DIM_1D(childCounts.GetN(), 2)

    const int bestSplit = BestSplit(treeDepth, impurityValues, childCounts);
    const bool shouldSplit = (bestSplit >= 0
                            && treeDepth < mMaxDepth 
                            && impurityValues.Get(bestSplit, 0) > mMinImpurity);

    return (shouldSplit) ?
                 SPLT_CRITERIA_READY_TO_SPLIT : SPLT_CRITERIA_MORE_DATA_REQUIRED;
}

int OfflineSplitCriteria::BestSplit(int treeDepth,
                                    const Float32MatrixBuffer& impurityValues,
                                    const Float32MatrixBuffer& childCounts ) const
{
    ASSERT_ARG_DIM_1D(impurityValues.GetM(), childCounts.GetM())
    ASSERT_ARG_DIM_1D(impurityValues.GetN(), 1)
    ASSERT_ARG_DIM_1D(childCounts.GetN(), 2)

    int maxIndex = -1;
    float maxImpurity = FLT_MIN;
    for(int i=0; i<impurityValues.GetM(); i++)
    {
        const float leftCounts = childCounts.Get(i, 0);
        const float rightCounts = childCounts.Get(i, 1);

        bool isBest = ((impurityValues.Get(i,0) > maxImpurity)
                && ((leftCounts+rightCounts) > mMinSampleCounts) 
                && (leftCounts > mMinChildSampleCounts)
                && (rightCounts > mMinChildSampleCounts));
        if (isBest)
        {
            maxIndex = i;
            maxImpurity = impurityValues.Get(i,0);
        }
    }
    return maxIndex;
}
