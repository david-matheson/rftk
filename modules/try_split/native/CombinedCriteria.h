#pragma once

#include <vector>
#include "TrySplitCriteriaI.h"

// ----------------------------------------------------------------------------
//
// CombinedCriteria checks if all criteria pass
//
// ----------------------------------------------------------------------------
class CombinedCriteria: public TrySplitCriteriaI
{
public:
    CombinedCriteria(std::vector<TrySplitCriteriaI*> criterias);
    virtual ~CombinedCriteria();

    virtual TrySplitCriteriaI* Clone() const;

    virtual bool TrySplit(int depth, int numberOfDatapoints) const;
private:
    std::vector<TrySplitCriteriaI*> mCriterias;
};