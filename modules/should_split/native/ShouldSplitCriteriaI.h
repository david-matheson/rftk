#pragma once

#include "BufferCollection.h"

// ----------------------------------------------------------------------------
//
// ShouldSplitCriteriaI determines whether a split is valid
//
// ----------------------------------------------------------------------------
class ShouldSplitCriteriaI
{
public:
    virtual ~ShouldSplitCriteriaI() {}

    virtual ShouldSplitCriteriaI* Clone() const = 0;

    virtual bool ShouldSplit(int depth, float impurity,
                            int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints,
                            BufferCollection& extraInfo, int nodeIndex) const = 0;
};