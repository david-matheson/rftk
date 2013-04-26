#include "asserts.h" // for UNUSED_PARAM
#include "MaxDepthCriteria.h"

MaxDepthCriteria::MaxDepthCriteria(int maxDepth)
: mMaxDepth(maxDepth)
{}

MaxDepthCriteria::~MaxDepthCriteria()
{}

TrySplitCriteriaI* MaxDepthCriteria::Clone() const
{
    MaxDepthCriteria* clone = new MaxDepthCriteria(*this);
    return clone;
}

bool MaxDepthCriteria::TrySplit(int depth, double numberOfDatapoints) const
{
    UNUSED_PARAM(numberOfDatapoints);
    return (depth < mMaxDepth);
}