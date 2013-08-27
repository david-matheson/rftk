#pragma once

#include "TrySplitCriteriaI.h"

// ----------------------------------------------------------------------------
//
// TrySplitNoCriteria has no criteria and always tries to split the node
//
// ----------------------------------------------------------------------------
class TrySplitNoCriteria: public TrySplitCriteriaI
{
public:
    TrySplitNoCriteria();
    virtual ~TrySplitNoCriteria();

    virtual TrySplitCriteriaI* Clone() const;

    virtual bool TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const;
};