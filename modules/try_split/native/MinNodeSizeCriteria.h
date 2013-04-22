#pragma once

#include "TrySplitCriteriaI.h"

// ----------------------------------------------------------------------------
//
// MinNodeSizeCriteria checks if number of datapoints is passed the min
//
// ----------------------------------------------------------------------------
class MinNodeSizeCriteria: public TrySplitCriteriaI
{
public:
    MinNodeSizeCriteria(int minNumberOfDatapoints);
    virtual ~MinNodeSizeCriteria();

    virtual TrySplitCriteriaI* Clone() const;

    virtual bool TrySplit(int depth, int numberOfDatapoints) const;
private:
    const int mMinNumberOfDatapoints;
};