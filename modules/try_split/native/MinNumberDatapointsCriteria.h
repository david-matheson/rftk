#pragma once

#include "TrySplitCriteriaI.h"

// ----------------------------------------------------------------------------
//
// MinNumberDatapointsCriteria checks if number of datapoints is passed the min
//
// ----------------------------------------------------------------------------
class MinNumberDatapointsCriteria: public TrySplitCriteriaI
{
public:
    MinNumberDatapointsCriteria(int minNumberOfDatapoints);
    virtual ~MinNumberDatapointsCriteria();

    virtual TrySplitCriteriaI* Clone() const;

    virtual bool TrySplit(int depth, int numberOfDatapoints) const;
private:
    const int mMinNumberOfDatapoints;
};