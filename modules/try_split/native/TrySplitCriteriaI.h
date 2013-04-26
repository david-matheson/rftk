#pragma once

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

    virtual bool TrySplit(int depth, double numberOfDatapoints) const = 0;
};