#pragma once

#include <vector>
#include "TrySplitCriteriaI.h"

// ----------------------------------------------------------------------------
//
// TrySplitCombinedCriteria checks if all criteria pass
//
// ----------------------------------------------------------------------------
class TrySplitCombinedCriteria: public TrySplitCriteriaI
{
public:
    TrySplitCombinedCriteria(std::vector<TrySplitCriteriaI*> criterias);
    virtual ~TrySplitCombinedCriteria();

    virtual TrySplitCriteriaI* Clone() const;

    virtual bool TrySplit(int depth, int numberOfDatapoints) const;
private:
    std::vector<TrySplitCriteriaI*> mCriterias;
};