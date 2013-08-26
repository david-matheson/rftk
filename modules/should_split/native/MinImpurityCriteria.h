#pragma once

#include "ShouldSplitCriteriaI.h"
// ----------------------------------------------------------------------------
//
// MinImpurityCriteria checks if the impurity is above a minimum
//
// ----------------------------------------------------------------------------
class MinImpurityCriteria: public ShouldSplitCriteriaI
{
public:
    MinImpurityCriteria(float minImpurity);
    virtual ~MinImpurityCriteria();

    virtual ShouldSplitCriteriaI* Clone() const;

    virtual bool ShouldSplit(int depth, float impurity,
                            int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints,
                            BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const;
private:
    float mMinImpurity;
};