#include "asserts.h" // for UNUSED_PARAM
#include "MinNodeSizeCriteria.h"

// remove print
#include <stdio.h>

MinNodeSizeCriteria::MinNodeSizeCriteria(double minNumberOfDatapoints)
: mMinNumberOfDatapoints(minNumberOfDatapoints)
{}

MinNodeSizeCriteria::~MinNodeSizeCriteria()
{}

TrySplitCriteriaI* MinNodeSizeCriteria::Clone() const
{
    MinNodeSizeCriteria* clone = new MinNodeSizeCriteria(*this);
    return clone;
}

bool MinNodeSizeCriteria::TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(depth);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);
    return (numberOfDatapoints >= mMinNumberOfDatapoints);
}
