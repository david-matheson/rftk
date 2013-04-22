#include "asserts.h" // for UNUSED_PARAM
#include "TrySplitNoCriteria.h"


TrySplitNoCriteria::TrySplitNoCriteria()
{}

TrySplitNoCriteria::~TrySplitNoCriteria()
{}

TrySplitCriteriaI* TrySplitNoCriteria::Clone() const
{
    TrySplitNoCriteria* clone = new TrySplitNoCriteria(*this);
    return clone;
}

bool TrySplitNoCriteria::TrySplit(int depth, int numberOfDatapoints) const
{
    UNUSED_PARAM(depth);
    UNUSED_PARAM(numberOfDatapoints);
    return true;
}
