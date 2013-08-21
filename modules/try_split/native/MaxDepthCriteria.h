#pragma once

#include "TrySplitCriteriaI.h"

// ----------------------------------------------------------------------------
//
// MaxDepthCriteria checks if past max depth
//
// ----------------------------------------------------------------------------
class MaxDepthCriteria: public TrySplitCriteriaI
{
public:
    MaxDepthCriteria(int maxDepth);
    virtual ~MaxDepthCriteria();

    virtual TrySplitCriteriaI* Clone() const;

    virtual bool TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex) const;
private:
    const int mMaxDepth;
};