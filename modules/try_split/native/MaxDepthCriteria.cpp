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

bool MaxDepthCriteria::TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex) const
{
    UNUSED_PARAM(numberOfDatapoints);
    UNUSED_PARAM(extraInfo);
    UNUSED_PARAM(nodeIndex);
    return (depth < mMaxDepth);
}