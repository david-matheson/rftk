#pragma once

#include "ShouldSplitCriteriaI.h"
// ----------------------------------------------------------------------------
//
// MinChildSizeCriteria checks if the number of datapoints of both children
// are above the minimum
//
// ----------------------------------------------------------------------------
class MinChildSizeCriteria: public ShouldSplitCriteriaI
{
public:
    MinChildSizeCriteria(int minNumberOfChildDatapoints);
    virtual ~MinChildSizeCriteria();

    virtual ShouldSplitCriteriaI* Clone() const;

    virtual bool ShouldSplit(int depth, float impurity,
                            int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints) const;
private:
    int mMinNumberOfChildDatapoints;
};