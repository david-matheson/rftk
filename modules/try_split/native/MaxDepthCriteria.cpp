#include "unused.h" 
#include "BufferCollectionUtils.h"
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

bool MaxDepthCriteria::TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const
{
    UNUSED_PARAM(numberOfDatapoints);

    const bool result = (depth < mMaxDepth);

    if(recordInfo)
    {
        WriteValue<int>(extraInfo, "TrySplit-MaxDepthCriteria", nodeIndex, result ? TRY_SPLIT_TRUE : TRY_SPLIT_FALSE);      
    }

    return result;
}