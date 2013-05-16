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

bool MinNodeSizeCriteria::TrySplit(int depth, double numberOfDatapoints) const
{
    UNUSED_PARAM(depth);
    return (numberOfDatapoints >= mMinNumberOfDatapoints);
}
