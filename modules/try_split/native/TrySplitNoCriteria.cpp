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

bool TrySplitNoCriteria::TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(depth);
    UNUSED_PARAM(numberOfDatapoints);
    UNUSED_PARAM(extraInfo)
    UNUSED_PARAM(nodeIndex)
    return true;
}
