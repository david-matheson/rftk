#include "unused.h" 
#include "BufferCollectionUtils.h"
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

bool MinNodeSizeCriteria::TrySplit(int depth, double numberOfDatapoints, BufferCollection& extraInfo, int nodeIndex, bool recordInfo) const
{
    UNUSED_PARAM(depth);

    const bool result = (numberOfDatapoints >= mMinNumberOfDatapoints);

    if(recordInfo)
    {
        WriteValue<int>(extraInfo, "TrySplit-MinNodeSizeCriteria", nodeIndex, result ? TRY_SPLIT_TRUE : TRY_SPLIT_FALSE);      
    }

    return result;
}
