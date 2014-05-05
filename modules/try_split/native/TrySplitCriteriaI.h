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

    virtual bool TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const = 0;
};


const int TRY_SPLIT_FALSE = 1;
const int TRY_SPLIT_TRUE = 2;