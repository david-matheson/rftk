#include "asserts.h" // for UNUSED_PARAM
#include "MinNodeSizeCriteria.h"


MinNodeSizeCriteria::MinNodeSizeCriteria(int minNumberOfDatapoints)
: mMinNumberOfDatapoints(minNumberOfDatapoints)
{}

MinNodeSizeCriteria::~MinNodeSizeCriteria()
{}

TrySplitCriteriaI* MinNodeSizeCriteria::Clone() const
{
    MinNodeSizeCriteria* clone = new MinNodeSizeCriteria(*this);
    return clone;
}

bool MinNodeSizeCriteria::TrySplit(int depth, int numberOfDatapoints) const
{
    UNUSED_PARAM(depth);
    return (numberOfDatapoints >= mMinNumberOfDatapoints);
}
