#pragma once

#include "ShouldSplitCriteriaI.h"
// ----------------------------------------------------------------------------
//
// MinChildSizeSumCriteria checks if the number of datapoints of both children
// are above the minimum
//
// ----------------------------------------------------------------------------
class MinChildSizeSumCriteria: public ShouldSplitCriteriaI
{
public:
    MinChildSizeSumCriteria(int minNumberOfChildDatapointsSum);
    virtual ~MinChildSizeSumCriteria();

    virtual ShouldSplitCriteriaI* Clone() const;

    virtual bool ShouldSplit(int depth, float impurity,
                            int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints) const;
private:
    int mMinNumberOfChildDatapointsSum;
};