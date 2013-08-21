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
    MinNodeSizeCriteria(double minNumberOfDatapoints);
    virtual ~MinNodeSizeCriteria();

    virtual TrySplitCriteriaI* Clone() const;

    virtual bool TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex) const;
private:
    const double mMinNumberOfDatapoints;
};