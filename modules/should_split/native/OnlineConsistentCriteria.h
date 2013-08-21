#pragma once

#include "ShouldSplitCriteriaI.h"

// ----------------------------------------------------------------------------
//
// OnlineConsistentCriteria follows Consistency of Online Random Forests
// http://arxiv.org/pdf/1302.4853.pdf
//
// ----------------------------------------------------------------------------
class OnlineConsistentCriteria: public ShouldSplitCriteriaI
{
public:
    OnlineConsistentCriteria(float minImpurity,
                            float minNumberOfSamplesFirstSplit,
                            float maxNumberOfSamplesFirstSplit,
                            float growthRate);
    virtual ~OnlineConsistentCriteria();

    virtual ShouldSplitCriteriaI* Clone() const;

    virtual bool ShouldSplit(int depth, float impurity,
                            int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints,
                            BufferCollection& extraInfo, int nodeIndex) const;
private:
    const float mMinImpurity;
    const float mMinNumberOfSamplesFirstSplit;
    const float mMaxNumberOfSamplesFirstSplit;
    const float mGrowthRate;
};