#pragma once

#include "BufferCollection.h"

// ----------------------------------------------------------------------------
//
// TrySplitCriteriaI determines whether to try to split a node
//
// ----------------------------------------------------------------------------
class TrySplitCriteriaI
{
public:
    virtual ~TrySplitCriteriaI() {}

    virtual TrySplitCriteriaI* Clone() const = 0;

    virtual bool TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex) const = 0;
};