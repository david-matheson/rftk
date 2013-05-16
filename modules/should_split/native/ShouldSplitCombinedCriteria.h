#pragma once

#include <vector>
#include "ShouldSplitCriteriaI.h"

// ----------------------------------------------------------------------------
//
// ShouldSplitCombinedCriteria checks if all criteria pass
//
// ----------------------------------------------------------------------------
class ShouldSplitCombinedCriteria: public ShouldSplitCriteriaI
{
public:
    ShouldSplitCombinedCriteria(std::vector<ShouldSplitCriteriaI*> criterias);
    virtual ~ShouldSplitCombinedCriteria();

    virtual ShouldSplitCriteriaI* Clone() const;

    virtual bool ShouldSplit(int depth, float impurity,
                            int numberOfDatapoints, int leftNumberOfDataponts, int rightNumberOfDatapoints) const;
private:
    std::vector<ShouldSplitCriteriaI*> mCriterias;
};