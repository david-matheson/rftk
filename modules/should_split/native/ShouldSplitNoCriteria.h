#pragma once

#include "ShouldSplitCriteriaI.h"
// ----------------------------------------------------------------------------
//
// ShouldSplitNoCriteria has no criteria and always accepts the split with the
// highest impurity
//
// ----------------------------------------------------------------------------
class ShouldSplitNoCriteria: public ShouldSplitCriteriaI
{
public:
    ShouldSplitNoCriteria();
    virtual ~ShouldSplitNoCriteria();

    virtual ShouldSplitCriteriaI* Clone() const;

    virtual bool ShouldSplit(int depth, float impurity,
                            int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints) const;
};